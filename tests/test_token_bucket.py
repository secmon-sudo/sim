"""
Tests for TokenBucket rate limiter.
Blueprint V20.1 §4.5.3
"""

import datetime
import time
from unittest.mock import patch

import pytest

from src.core.token_bucket import TokenBucket


class TestTokenBucket:
    def test_acquire_basic(self):
        """Basic token acquisition should succeed."""
        bucket = TokenBucket(rate_per_minute=10, daily_limit=100)
        assert bucket.acquire(timeout=1) is True
        assert bucket._daily_used == 1

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
        assert bucket._daily_used == 1

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
        """None daily_limit should allow unlimited usage."""
        bucket = TokenBucket(rate_per_minute=100, daily_limit=None)
        for _ in range(50):
            bucket.acquire(timeout=0)
        assert bucket.remaining_daily == 999_999
