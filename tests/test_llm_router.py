"""
Tests for LLMRouter multi-provider failover.
Blueprint V20.1 QA-09
"""

import time
from unittest.mock import patch

import pytest

from src.core.llm_router import LLMAccount, LLMRouter, ProviderStatus
from src.core.token_bucket import TokenBucket


def make_account(provider="groq", account_id="A", model="test-model", rpd=1000, rpm=30):
    """Helper to create a test LLMAccount."""
    return LLMAccount(
        provider=provider,
        account_id=account_id,
        model=model,
        api_key="test-key-123",
        rpm=rpm,
        rpd=rpd,
        bucket=TokenBucket(rate_per_minute=rpm, daily_limit=rpd),
    )


class TestLLMRouter:
    def test_get_first_available(self):
        """Should return the first account in priority order."""
        a1 = make_account(model="model-primary", rpd=1000)
        a2 = make_account(model="model-fallback", rpd=1000)
        router = LLMRouter([a1, a2])
        acct = router.get_available_account()
        assert acct.model == "model-primary"

    def test_failover_on_rate_limit(self):
        """After rate-limit failure, should rotate to next account."""
        a1 = make_account(model="openai/gpt-oss-120b", rpd=1000)
        a2 = make_account(model="llama-3.3-70b-versatile", rpd=1000)
        router = LLMRouter([a1, a2])

        # Get first account and report rate limit
        acct1 = router.get_available_account()
        assert acct1.model == "openai/gpt-oss-120b"
        router.report_failure(acct1, is_rate_limit=True)

        # Should rotate to second
        acct2 = router.get_available_account()
        assert acct2.model == "llama-3.3-70b-versatile"

    def test_all_exhausted_returns_none(self):
        """When all accounts exhausted, should return None."""
        a1 = make_account(rpd=0)  # Already exhausted
        router = LLMRouter([a1])
        assert router.get_available_account() is None

    def test_cooldown_recovery(self):
        """Rate-limited accounts should recover after cooldown."""
        a1 = make_account(rpd=1000)
        router = LLMRouter([a1])

        acct = router.get_available_account()
        router.report_failure(acct, is_rate_limit=True)

        # Immediately after → should be None (in cooldown)
        assert router.get_available_account() is None

        # After cooldown → should recover
        acct.cooldown_until = time.monotonic() - 1
        recovered = router.get_available_account()
        assert recovered is not None
        assert recovered.model == "test-model"

    def test_error_threshold(self):
        """After 10 consecutive errors, account should be marked ERROR."""
        a1 = make_account(rpd=1000)
        router = LLMRouter([a1])

        acct = router.get_available_account()
        for _ in range(10):
            router.report_failure(acct, is_rate_limit=False)

        assert acct.status == ProviderStatus.ERROR
        assert acct.daily_errors == 10

    def test_report_success_resets(self):
        """Successful call should reset error count and status."""
        a1 = make_account(rpd=1000)
        router = LLMRouter([a1])

        acct = router.get_available_account()
        router.report_failure(acct, is_rate_limit=False)
        router.report_failure(acct, is_rate_limit=False)
        assert acct.daily_errors == 2

        router.report_success(acct)
        assert acct.daily_errors == 0
        assert acct.status == ProviderStatus.ACTIVE

    def test_total_daily_quota(self):
        """total_daily_quota should sum all account RPDs."""
        a1 = make_account(rpd=1000)
        a2 = make_account(rpd=200)
        router = LLMRouter([a1, a2])
        assert router.total_daily_quota == 1200

    def test_status_snapshot(self):
        """get_status_snapshot should return serializable dict."""
        a1 = make_account(provider="groq", account_id="A", model="test", rpd=100)
        router = LLMRouter([a1])
        snap = router.get_status_snapshot()
        assert "groq/A/test" in snap
        assert snap["groq/A/test"]["status"] == "active"

    def test_cross_provider_failover(self):
        """Groq exhausted → should failover to OpenRouter."""
        groq = make_account(provider="groq", model="groq-model", rpd=1)
        openrouter = make_account(provider="openrouter", model="or-model", rpd=1000)
        router = LLMRouter([groq, openrouter])

        # Use groq's single daily request
        a1 = router.get_available_account()
        assert a1.provider == "groq"

        # Next request should go to openrouter
        a2 = router.get_available_account()
        assert a2.provider == "openrouter"
