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

# --- Cooldown tuning ------------------------------------------------------
# Local RPM throttle (our own bucket ran dry): tokens refill at rate_per_minute,
# so a short pause is enough — don't idle the account for a full minute.
RPM_COOLDOWN_SECONDS = 15
# Provider returned HTTP 429: back off, but free-tier RPM windows reset within
# ~a minute. Used only when the response carries no Retry-After header.
RATE_LIMIT_COOLDOWN_SECONDS = 30
# Upper bound so a bogus Retry-After can't park an account for hours.
MAX_RATE_LIMIT_COOLDOWN_SECONDS = 300
# Repeated hard (non-429) errors: likely a real outage/bad key — back off long.
ERROR_COOLDOWN_SECONDS = 600
ERROR_THRESHOLD = 10
# Max tokens held at once per model slot — smooths the opening burst.
DEFAULT_BURST = 8


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
        return sum(a.bucket.daily_used for a in self._accounts)

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
                        acct.cooldown_until = now + RPM_COOLDOWN_SECONDS
                        logger.info(
                            "Account %s RPM limited, cooldown %ds",
                            acct.display_name, RPM_COOLDOWN_SECONDS,
                        )
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
                        acct.cooldown_until = now + RPM_COOLDOWN_SECONDS

            return None

    def report_success(self, acct: LLMAccount):
        """Mark an account as healthy after a successful call."""
        with self._lock:
            acct.status = ProviderStatus.ACTIVE
            acct.daily_errors = 0

    def report_failure(
        self,
        acct: LLMAccount,
        is_rate_limit: bool = False,
        retry_after: float | None = None,
    ):
        """Mark an account as degraded after a failed call.

        retry_after: seconds from the provider's Retry-After header (429), if any.
        Honored over the default cooldown, clamped to MAX_RATE_LIMIT_COOLDOWN_SECONDS.
        """
        with self._lock:
            if is_rate_limit:
                cooldown = RATE_LIMIT_COOLDOWN_SECONDS
                if retry_after is not None and retry_after > 0:
                    cooldown = min(retry_after, MAX_RATE_LIMIT_COOLDOWN_SECONDS)
                acct.status = ProviderStatus.RATE_LIMITED
                acct.cooldown_until = time.monotonic() + cooldown
                logger.warning(
                    "Account %s rate-limited (429), cooldown %.0fs%s",
                    acct.display_name, cooldown,
                    " (Retry-After)" if retry_after else "",
                )
            else:
                acct.daily_errors += 1
                if acct.daily_errors >= ERROR_THRESHOLD:
                    acct.status = ProviderStatus.ERROR
                    acct.cooldown_until = time.monotonic() + ERROR_COOLDOWN_SECONDS
                    logger.error(
                        "Account %s marked ERROR after %d failures, cooldown %ds",
                        acct.display_name, acct.daily_errors, ERROR_COOLDOWN_SECONDS,
                    )

    def get_status_snapshot(self) -> dict:
        """Returns serializable status for telemetry logging."""
        return {
            acct.display_name: {
                "status": acct.status.value,
                "daily_used": acct.bucket.daily_used,
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
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST),
        ),
        # ② Groq-A Secondary — kalite yedeği (eski llama-3.3-70b-versatile yerine)
        LLMAccount(
            provider="groq", account_id="A",
            model="qwen/qwen3.6-27b",
            api_key=os.environ.get("GROQ_API_KEY_A", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST),
        ),
        # ③ Groq-B Throughput — en yüksek TPM (eski llama-4-scout yerine)
        LLMAccount(
            provider="groq", account_id="B",
            model="openai/gpt-oss-120b",
            api_key=os.environ.get("GROQ_API_KEY_B", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST),
        ),
        # ④ Groq-B Burst — model çeşitliliği (eski qwen3-32b yerine)
        LLMAccount(
            provider="groq", account_id="B",
            model="qwen/qwen3.6-27b",
            api_key=os.environ.get("GROQ_API_KEY_B", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST),
        ),
        # ⑤ OpenRouter-A Emergency — 405B
        LLMAccount(
            provider="openrouter", account_id="A",
            model="nousresearch/hermes-3-llama-3.1-405b:free",
            api_key=os.environ.get("OPENROUTER_API_KEY_A", ""),
            rpm=20, rpd=200,
            bucket=TokenBucket(rate_per_minute=20, daily_limit=200, burst=DEFAULT_BURST),
        ),
        # ⑥ OpenRouter-B Mirror — cross-provider yedek
        LLMAccount(
            provider="openrouter", account_id="B",
            model="openai/gpt-oss-120b:free",
            api_key=os.environ.get("OPENROUTER_API_KEY_B", ""),
            rpm=20, rpd=200,
            bucket=TokenBucket(rate_per_minute=20, daily_limit=200, burst=DEFAULT_BURST),
        ),
        # ⑦ Reserved for future model slot (Blueprint V20.1 §4.5.2)
        # ⑧ Groq Bulk Fallback — son çare (eski llama-3.1-8b-instant yerine gpt-oss-20b)
        # NOT: 8b-instant 14.4K RPD sundu; ücretsiz katmanda hiçbir sohbet modeli
        # artık 1K RPD üstüne çıkmıyor (2026-06-17 Groq deprecation).
        LLMAccount(
            provider="groq", account_id="A",
            model="openai/gpt-oss-20b",
            api_key=os.environ.get("GROQ_API_KEY_A", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST),
        ),
    ]
    # Filter out accounts with empty API keys
    active = [a for a in accounts if a.api_key]
    if not active:
        logger.critical("No LLM API keys configured! Set GROQ_API_KEY_A/B and/or OPENROUTER_API_KEY_A/B")
    return LLMRouter(active)


def build_bulk_router() -> LLMRouter:
    """Router for low-stakes, high-volume work (e.g. storyline narrative prose).

    Uses gpt-oss-20b — deliberately a model that is NOT in the main cascade — so bulk
    narrative work keeps its own separate quota buckets and never competes with Pass C
    classification for the scarce smart-model quota. Stacking the slot across both Groq
    keys yields ~2K RPD combined, isolated from classification.

    History: this used to run on llama-3.1-8b-instant (14.4K RPD), but Groq deprecated it
    on 2026-06-17 and no free-tier chat model exceeds 1K RPD anymore, so capacity is
    reconstructed by pooling per-key slots instead. Falls back to the full router only if
    no Groq key is set at all.
    """
    slots = [
        LLMAccount(
            provider="groq", account_id=account_id,
            model="openai/gpt-oss-20b",
            api_key=os.environ.get(f"GROQ_API_KEY_{account_id}", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST),
        )
        for account_id in ("A", "B")
    ]
    active = [s for s in slots if s.api_key]
    if not active:
        logger.warning("Bulk router: no GROQ_API_KEY_A/B set, falling back to full router")
        return build_llm_router()
    return LLMRouter(active)
