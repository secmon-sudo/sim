"""
SIM — Multi-Provider LLM Router
Blueprint V20.1 §4.5.4 + §4.5.5

Priority-ordered failover across Groq and OpenRouter accounts.
Each event uses exactly ONE model — the first available in the cascade.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.core.token_bucket import TokenBucket

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    ERROR = "error"


@dataclass
class LLMAccount:
    """Represents a single model slot on a specific provider account."""

    provider: str        # "groq" | "openrouter"
    account_id: str      # "A" | "B"
    model: str           # e.g. "openai/gpt-oss-120b"
    api_key: str
    rpm: int
    rpd: int
    bucket: TokenBucket
    status: ProviderStatus = ProviderStatus.ACTIVE
    cooldown_until: float = 0.0
    daily_errors: int = 0

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.account_id}/{self.model}"


class LLMRouter:
    """
    Priority-ordered failover across multiple provider accounts.

    Usage:
        router = build_llm_router()
        acct = router.get_available_account()
        if acct:
            # make the call
            router.report_success(acct)
        else:
            # all accounts exhausted
    """

    def __init__(self, accounts: list[LLMAccount]):
        self._accounts = accounts
        self._lock = threading.Lock()

    @property
    def total_daily_quota(self) -> int:
        """Sum of all account RPD limits."""
        return sum(a.rpd for a in self._accounts)

    @property
    def total_daily_used(self) -> int:
        """Sum of all account daily usage."""
        return sum(a.bucket._daily_used for a in self._accounts)

    @property
    def accounts(self) -> list[LLMAccount]:
        return self._accounts

    def get_available_account(self) -> Optional[LLMAccount]:
        """Return the highest-priority account that can accept a request.

        Returns None if all accounts are exhausted or in cooldown.
        """
        with self._lock:
            now = time.monotonic()
            for acct in self._accounts:
                # Active and not in cooldown → try to acquire
                if acct.status == ProviderStatus.ACTIVE and acct.cooldown_until <= now:
                    try:
                        acct.bucket.acquire(timeout=0)
                        return acct
                    except TimeoutError:
                        acct.status = ProviderStatus.RATE_LIMITED
                        acct.cooldown_until = now + 60
                        logger.info("Account %s RPM limited, cooldown 60s", acct.display_name)
                    except RuntimeError:
                        acct.status = ProviderStatus.QUOTA_EXHAUSTED
                        logger.warning("Account %s daily quota exhausted", acct.display_name)

                # Auto-recover rate-limited accounts after cooldown
                elif acct.status == ProviderStatus.RATE_LIMITED and acct.cooldown_until <= now:
                    acct.status = ProviderStatus.ACTIVE
                    try:
                        acct.bucket.acquire(timeout=0)
                        return acct
                    except (TimeoutError, RuntimeError):
                        acct.cooldown_until = now + 120

            return None

    def report_success(self, acct: LLMAccount):
        """Mark an account as healthy after a successful call."""
        with self._lock:
            acct.status = ProviderStatus.ACTIVE
            acct.daily_errors = 0

    def report_failure(self, acct: LLMAccount, is_rate_limit: bool = False):
        """Mark an account as degraded after a failed call."""
        with self._lock:
            if is_rate_limit:
                acct.status = ProviderStatus.RATE_LIMITED
                acct.cooldown_until = time.monotonic() + 120
                logger.warning("Account %s rate-limited (429), cooldown 120s", acct.display_name)
            else:
                acct.daily_errors += 1
                if acct.daily_errors >= 10:
                    acct.status = ProviderStatus.ERROR
                    acct.cooldown_until = time.monotonic() + 600
                    logger.error(
                        "Account %s marked ERROR after %d failures, cooldown 600s",
                        acct.display_name, acct.daily_errors
                    )

    def get_status_snapshot(self) -> dict:
        """Returns serializable status for telemetry logging."""
        return {
            acct.display_name: {
                "status": acct.status.value,
                "daily_used": acct.bucket._daily_used,
                "daily_limit": acct.rpd,
                "errors": acct.daily_errors,
            }
            for acct in self._accounts
        }


def build_llm_router() -> LLMRouter:
    """
    Initialize all LLM accounts from environment variables.
    Cascade order: Groq-A (smart) → Groq-B (throughput) → OpenRouter → Groq bulk.
    """
    accounts = [
        # ① Groq-A Primary — en akıllı
        LLMAccount(
            provider="groq", account_id="A",
            model="openai/gpt-oss-120b",
            api_key=os.environ.get("GROQ_API_KEY_A", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000),
        ),
        # ② Groq-A Secondary — kanıtlanmış kalite
        LLMAccount(
            provider="groq", account_id="A",
            model="llama-3.3-70b-versatile",
            api_key=os.environ.get("GROQ_API_KEY_A", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000),
        ),
        # ③ Groq-B Throughput — en yüksek TPM
        LLMAccount(
            provider="groq", account_id="B",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.environ.get("GROQ_API_KEY_B", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000),
        ),
        # ④ Groq-B Burst — en yüksek RPM
        LLMAccount(
            provider="groq", account_id="B",
            model="qwen/qwen3-32b",
            api_key=os.environ.get("GROQ_API_KEY_B", ""),
            rpm=60, rpd=1000,
            bucket=TokenBucket(rate_per_minute=60, daily_limit=1000),
        ),
        # ⑤ OpenRouter-A Emergency — 405B
        LLMAccount(
            provider="openrouter", account_id="A",
            model="nousresearch/hermes-3-llama-3.1-405b:free",
            api_key=os.environ.get("OPENROUTER_API_KEY_A", ""),
            rpm=20, rpd=200,
            bucket=TokenBucket(rate_per_minute=20, daily_limit=200),
        ),
        # ⑥ OpenRouter-B Mirror — cross-provider yedek
        LLMAccount(
            provider="openrouter", account_id="B",
            model="openai/gpt-oss-120b:free",
            api_key=os.environ.get("OPENROUTER_API_KEY_B", ""),
            rpm=20, rpd=200,
            bucket=TokenBucket(rate_per_minute=20, daily_limit=200),
        ),
        # ⑧ Groq Bulk Fallback — son çare, 14.4K RPD
        LLMAccount(
            provider="groq", account_id="A",
            model="llama-3.1-8b-instant",
            api_key=os.environ.get("GROQ_API_KEY_A", ""),
            rpm=30, rpd=14400,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=14400),
        ),
    ]
    # Filter out accounts with empty API keys
    active = [a for a in accounts if a.api_key]
    if not active:
        logger.critical("No LLM API keys configured! Set GROQ_API_KEY_A/B and/or OPENROUTER_API_KEY_A/B")
    return LLMRouter(active)
