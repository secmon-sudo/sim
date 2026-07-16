"""
Tests for TokenBucket rate limiter.
Blueprint V20.1 §4.5.3
"""

import datetime
import time

import pytest

from src.core.token_bucket import TokenBucket


class TestTokenBucket:
    def test_acquire_basic(self):
        """Basic token acquisition should succeed."""
        bucket = TokenBucket(rate_per_minute=10, daily_limit=100)
        assert bucket.acquire(timeout=1) is True
        assert bucket.daily_used == 1

    def test_acquire_exhausts_rpm(self):
        """Should raise TimeoutError when RPM is exhausted."""
        bucket = TokenBucket(rate_per_minute=2, daily_limit=100)
        bucket.acquire(timeout=0)
        bucket.acquire(timeout=0)
        with pytest.raises(TimeoutError):
            bucket.acquire(timeout=0)

    def test_daily_limit_exhaustion(self):
        """Should raise RuntimeError when daily limit is reached."""
        bucket = TokenBucket(rate_per_minute=100, daily_limit=3)
        bucket.acquire(timeout=0)
        bucket.acquire(timeout=0)
        bucket.acquire(timeout=0)
        with pytest.raises(RuntimeError, match="Daily LLM quota exhausted"):
            bucket.acquire(timeout=0)

    def test_refill_over_time(self):
        """Tokens should refill over time."""
        bucket = TokenBucket(rate_per_minute=60, daily_limit=1000)
        # Use all tokens
        for _ in range(60):
            bucket.acquire(timeout=0)
        # Wait for refill (1 second = 1 token at 60 RPM)
        time.sleep(1.1)
        assert bucket.acquire(timeout=0) is True

    def test_day_reset(self):
        """Daily counter should reset on new day."""
        bucket = TokenBucket(rate_per_minute=100, daily_limit=5)
        for _ in range(5):
            bucket.acquire(timeout=0)

        # Simulate next day
        bucket._current_day = datetime.date.today() - datetime.timedelta(days=1)
        # Should succeed — day reset
        assert bucket.acquire(timeout=1) is True
        assert bucket.daily_used == 1

    def test_remaining_daily(self):
        bucket = TokenBucket(rate_per_minute=10, daily_limit=100)
        assert bucket.remaining_daily == 100
        bucket.acquire(timeout=0)
        assert bucket.remaining_daily == 99

    def test_utilization_pct(self):
        bucket = TokenBucket(rate_per_minute=10, daily_limit=100)
        assert bucket.utilization_pct == 0.0
        for _ in range(10):
            bucket.acquire(timeout=0)
        assert bucket.utilization_pct == 10.0

    def test_unlimited_daily(self):
        """None daily_limit should allow unlimited usage and return None for remaining."""
        bucket = TokenBucket(rate_per_minute=100, daily_limit=None)
        for _ in range(50):
            bucket.acquire(timeout=0)
        assert bucket.remaining_daily is None
        assert bucket.daily_used == 50

    def test_tpm_ceiling_blocks_burst(self):
        """TPM budget should block a burst even when RPM tokens remain."""
        # High RPM/burst so requests aren't the constraint; tight 3K TPM is.
        bucket = TokenBucket(rate_per_minute=60, daily_limit=1000, burst=60, tpm_limit=3000)
        # Two 1200-token calls fit (2400 <= 3000); the third (would be 3600) must block
        # on TPM despite plenty of RPM tokens still available.
        assert bucket.acquire(est_tokens=1200, timeout=0) is True
        assert bucket.acquire(est_tokens=1200, timeout=0) is True
        with pytest.raises(TimeoutError):
            bucket.acquire(est_tokens=1200, timeout=0)

    def test_tpm_refills_over_time(self):
        """TPM window should refill so throughput resumes after a short pause."""
        bucket = TokenBucket(rate_per_minute=60, daily_limit=1000, burst=60, tpm_limit=600)
        bucket.acquire(est_tokens=600, timeout=0)  # drain the TPM window
        with pytest.raises(TimeoutError):
            bucket.acquire(est_tokens=600, timeout=0)
        time.sleep(1.1)  # 600 TPM → 10 tokens/sec → ~11 tokens back
        assert bucket.acquire(est_tokens=10, timeout=0) is True

    def test_oversized_prompt_not_deadlocked(self):
        """A request larger than the whole TPM window should still pass on a full window."""
        bucket = TokenBucket(rate_per_minute=60, daily_limit=1000, burst=60, tpm_limit=1000)
        # est_tokens exceeds tpm_limit; clamped requirement means a full window suffices.
        assert bucket.acquire(est_tokens=5000, timeout=0) is True

    def test_no_tpm_limit_ignores_est_tokens(self):
        """With tpm_limit=None, est_tokens is ignored (back-compat)."""
        bucket = TokenBucket(rate_per_minute=60, daily_limit=1000)
        for _ in range(10):
            assert bucket.acquire(est_tokens=99999, timeout=0) is True
