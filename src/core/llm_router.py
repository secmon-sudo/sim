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
from dataclasses import dataclass
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
# Deterministic client error (HTTP 4xx other than 429): the request is structurally
# rejected (bad model, unsupported param, bad key). Sideline the slot briefly so the
# router stops re-selecting it within the same rotation loop — where no cooldown means
# a broken slot gets picked again on its remaining burst tokens (the double-400 we saw).
CLIENT_ERROR_COOLDOWN_SECONDS = 120
# Max tokens held at once per model slot — smooths the opening burst.
DEFAULT_BURST = 8
# Groq free-tier tokens-per-minute ceiling (gpt-oss-120b/20b, qwen3.6-27b all list 8K).
# This — not RPM — is the binding constraint; modeling it stops a burst from tripping 429.
GROQ_TPM = 8000
# OpenRouter free-model limits are ACCOUNT-wide, not per model: 20 RPM across all
# :free models, and 1000 requests/day when the account holds ≥$10 in credits
# (key A, funded 2026-07-09) vs 50/day unfunded (key B). All :free slots on one
# key must therefore share a single TokenBucket — see build_llm_router().
OPENROUTER_FREE_RPM = 20
OPENROUTER_FREE_RPD_FUNDED = 1000
OPENROUTER_FREE_RPD_UNFUNDED = 50


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
        """Sum of RPD limits over distinct buckets (slots sharing an account-wide
        bucket, e.g. OpenRouter :free models on one key, count once)."""
        return sum({id(a.bucket): a.rpd for a in self._accounts}.values())

    @property
    def total_daily_used(self) -> int:
        """Sum of daily usage over distinct buckets."""
        seen: dict[int, int] = {id(a.bucket): a.bucket.daily_used for a in self._accounts}
        return sum(seen.values())

    @property
    def accounts(self) -> list[LLMAccount]:
        return self._accounts

    def get_available_account(self, est_tokens: int = 0,
                              predicate=None) -> Optional[LLMAccount]:
        """Return the highest-priority account that can accept a request.

        est_tokens: estimated tokens for this call (prompt + max_tokens), charged
        against each account's TPM window so a burst can't blow the per-minute token
        ceiling. predicate: optional per-call filter — accounts it rejects are passed
        over without any state change or bucket spend (e.g. the request-size guard).
        Returns None if all accounts are exhausted, in cooldown, or filtered out.
        """
        with self._lock:
            now = time.monotonic()
            for acct in self._accounts:
                if predicate is not None and not predicate(acct):
                    continue
                # Active and not in cooldown → try to acquire
                if acct.status == ProviderStatus.ACTIVE and acct.cooldown_until <= now:
                    try:
                        acct.bucket.acquire(est_tokens=est_tokens, timeout=0)
                        return acct
                    except TimeoutError:
                        acct.status = ProviderStatus.RATE_LIMITED
                        acct.cooldown_until = now + RPM_COOLDOWN_SECONDS
                        logger.info(
                            "Account %s RPM/TPM limited, cooldown %ds",
                            acct.display_name, RPM_COOLDOWN_SECONDS,
                        )
                    except RuntimeError:
                        acct.status = ProviderStatus.QUOTA_EXHAUSTED
                        logger.warning("Account %s daily quota exhausted", acct.display_name)

                # Auto-recover rate-limited accounts after cooldown
                elif acct.status == ProviderStatus.RATE_LIMITED and acct.cooldown_until <= now:
                    acct.status = ProviderStatus.ACTIVE
                    try:
                        acct.bucket.acquire(est_tokens=est_tokens, timeout=0)
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
        hard_error: bool = False,
    ):
        """Mark an account as degraded after a failed call.

        retry_after: seconds from the provider's Retry-After header (429), if any.
        Honored over the default cooldown, clamped to MAX_RATE_LIMIT_COOLDOWN_SECONDS.
        hard_error: the slot is returning unusable responses right now — a deterministic
        client 4xx (not 429), or an empty/error body inside an HTTP 200 (OpenRouter free
        upstream failures). Sideline it on a short cooldown so it leaves the rotation
        instead of being re-picked on burst tokens.
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
            elif hard_error:
                acct.daily_errors += 1
                acct.status = ProviderStatus.RATE_LIMITED
                acct.cooldown_until = time.monotonic() + CLIENT_ERROR_COOLDOWN_SECONDS
                logger.warning(
                    "Account %s returning unusable responses (4xx or empty-200), cooldown %ds",
                    acct.display_name, CLIENT_ERROR_COOLDOWN_SECONDS,
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

    def penalize_model_slot(self, provider: str, account_id: str, model: str):
        """Sideline the slot matching a routing triple, for callers that only have
        the call_llm result dict.

        Used on response-CONTENT failures the client can't detect (e.g. a batch of
        structurally broken JSON with finish_reason=stop — OpenRouter :free routes
        across upstreams of varying quality, and a degraded upstream keeps emitting
        garbage). Sidelining shifts the next call to the following cascade slot.
        """
        for acct in self._accounts:
            if (acct.provider, acct.account_id, acct.model) == (provider, account_id, model):
                self.report_failure(acct, hard_error=True)
                return

    def seconds_until_available(self) -> Optional[float]:
        """Seconds until the soonest account can serve again.

        Returns 0.0 if an account is ready now, a positive wait if all serviceable
        accounts are merely on cooldown, or None if none can recover today (every account
        is daily-quota-exhausted). Lets a caller pace through a throttle instead of aborting
        the moment the per-minute token windows drain.
        """
        with self._lock:
            now = time.monotonic()
            soonest: Optional[float] = None
            for acct in self._accounts:
                if acct.status == ProviderStatus.QUOTA_EXHAUSTED:
                    continue  # won't recover until tomorrow's day-boundary reset
                wait = max(0.0, acct.cooldown_until - now)
                if soonest is None or wait < soonest:
                    soonest = wait
            return soonest

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


# Process-wide bucket registry: providers enforce rate limits server-side per
# (API key, model). Multiple router instances can target the same pair — the main
# router (Pass C), the bulk router used by the storyline adjudicator and narratives,
# and the main router's own gpt-oss-20b fallback slot all share gpt-oss-20b on key A.
# Giving each its own TokenBucket would let them collectively issue N× the real quota
# (split-brain). Sharing one bucket per (provider, key, model) keeps accounting truthful.
_BUCKET_REGISTRY: dict[tuple[str, str, str], TokenBucket] = {}
_REGISTRY_LOCK = threading.Lock()


def _share_buckets(accounts: list[LLMAccount]) -> list[LLMAccount]:
    """Replace each account's bucket with the one shared for its (provider, key, model).

    Accounts with the same server-side rate limit converge on a single bucket; the first
    one constructed for a pair wins (rate params are identical for a given key+model).
    """
    for a in accounts:
        key = (a.provider, a.api_key, a.model)
        with _REGISTRY_LOCK:
            shared = _BUCKET_REGISTRY.setdefault(key, a.bucket)
        a.bucket = shared
    return accounts


def reset_bucket_registry() -> None:
    """Clear the shared-bucket registry (test isolation only)."""
    with _REGISTRY_LOCK:
        _BUCKET_REGISTRY.clear()


def build_llm_router() -> LLMRouter:
    """
    Initialize all LLM accounts from environment variables.
    Cascade order: OpenRouter-A free (smartest) → Groq A/B → OpenRouter-B → Gemini → Groq bulk.
    """
    # OpenRouter :free limits are account-wide (20 RPM / 1000 RPD funded), so BOTH
    # key-A free slots below must drain this one bucket — separate buckets would let
    # them jointly issue 2× the real quota. No TPM ceiling on OpenRouter free tier.
    openrouter_a_free_bucket = TokenBucket(
        rate_per_minute=OPENROUTER_FREE_RPM,
        daily_limit=OPENROUTER_FREE_RPD_FUNDED,
        burst=DEFAULT_BURST,
    )
    accounts = [
        # ① OpenRouter-A Primary — Nemotron 3 Super: free listedeki en iyi
        # zekâ/güvenilirlik dengesi (468B haftalık token, 1M ctx). Hesap fonlu
        # (≥$10) olduğu için 1000 istek/gün. Slug models API'den doğrulandı
        # (2026-07-09); çıplak "nemotron-3-super" yok, boyut ekli kimlik gerekiyor.
        LLMAccount(
            provider="openrouter", account_id="A",
            model="nvidia/nemotron-3-super-120b-a12b:free",
            api_key=os.environ.get("OPENROUTER_API_KEY_A", ""),
            rpm=OPENROUTER_FREE_RPM, rpd=OPENROUTER_FREE_RPD_FUNDED,
            bucket=openrouter_a_free_bucket,
        ),
        # ② OpenRouter-A Secondary — gpt-oss ailesi: prompt'larımızın Groq'ta
        # kanıtlandığı model ailesi; Nemotron endpoint'i tökezlerse sıfır uyum
        # maliyetiyle devralır. ① ile AYNI hesap kotasını (bucket) paylaşır.
        # NOT: gpt-oss-120b:free 2026-07-17'de OpenRouter'dan kaldırıldı (HTTP
        # 404) — free katmanda ailenin kalan tek üyesi 20b.
        LLMAccount(
            provider="openrouter", account_id="A",
            model="openai/gpt-oss-20b:free",
            api_key=os.environ.get("OPENROUTER_API_KEY_A", ""),
            rpm=OPENROUTER_FREE_RPM, rpd=OPENROUTER_FREE_RPD_FUNDED,
            bucket=openrouter_a_free_bucket,
        ),
        # ③ Groq-A — en akıllı Groq slotu
        LLMAccount(
            provider="groq", account_id="A",
            model="openai/gpt-oss-120b",
            api_key=os.environ.get("GROQ_API_KEY_A", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST, tpm_limit=GROQ_TPM),
        ),
        # ④ Groq-A Secondary — kalite yedeği (eski llama-3.3-70b-versatile yerine)
        LLMAccount(
            provider="groq", account_id="A",
            model="qwen/qwen3.6-27b",
            api_key=os.environ.get("GROQ_API_KEY_A", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST, tpm_limit=GROQ_TPM),
        ),
        # ⑤ Groq-B Throughput — en yüksek TPM (eski llama-4-scout yerine)
        LLMAccount(
            provider="groq", account_id="B",
            model="openai/gpt-oss-120b",
            api_key=os.environ.get("GROQ_API_KEY_B", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST, tpm_limit=GROQ_TPM),
        ),
        # ⑥ Groq-B Burst — model çeşitliliği (eski qwen3-32b yerine)
        LLMAccount(
            provider="groq", account_id="B",
            model="qwen/qwen3.6-27b",
            api_key=os.environ.get("GROQ_API_KEY_B", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST, tpm_limit=GROQ_TPM),
        ),
        # ⑦ OpenRouter-B Mirror — cross-key yedek. Hesap fonsuz → 50 istek/gün.
        # (Eski Hermes-3-405B slotu kaldırıldı: key A'nın kotası artık hesap
        # genelinde paylaşıldığından üçüncü bir key-A free slotu kota eklemiyordu.
        # 120b:free'nin kaldırılmasıyla (2026-07-17) 20b:free'ye düşürüldü.)
        LLMAccount(
            provider="openrouter", account_id="B",
            model="openai/gpt-oss-20b:free",
            api_key=os.environ.get("OPENROUTER_API_KEY_B", ""),
            rpm=OPENROUTER_FREE_RPM, rpd=OPENROUTER_FREE_RPD_UNFUNDED,
            bucket=TokenBucket(
                rate_per_minute=OPENROUTER_FREE_RPM,
                daily_limit=OPENROUTER_FREE_RPD_UNFUNDED,
                burst=DEFAULT_BURST,
            ),
        ),
        # ⑧ Gemini — üçüncü bağımsız sağlayıcı (AI Studio free tier, OpenAI-compat).
        # Groq/OpenRouter kesintilerinden etkilenmez; 250K TPM ile Groq'un 8K TPM
        # duvarı burada yok. RPD değerleri hesabın AI Studio rate-limit panelinden
        # doğrulandı (2026-07-09) — web kaynaklarının yazdığı 1000-1500 RPD gerçek
        # değil; metin modellerinde kota çoğunlukla 20 RPD, istisnası 3.1-flash-lite
        # (500 RPD). Kotalar Pasifik gece yarısında sıfırlanır.
        LLMAccount(
            provider="gemini", account_id="A",
            model="gemini-3.1-flash-lite",
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            rpm=15, rpd=500,
            bucket=TokenBucket(rate_per_minute=15, daily_limit=500, burst=DEFAULT_BURST),
        ),
        LLMAccount(
            provider="gemini", account_id="A",
            # Acil yedek: yalnızca 20 RPD — bucket yerelde durdurur, 429'a sürmez.
            # Çıplak "gemini-3-flash" 404 döndürüyor; doğru kimlik "-preview" ekli
            # (deprecated ama kapanış tarihi yok).
            model="gemini-3-flash-preview",
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            rpm=5, rpd=20,
            bucket=TokenBucket(rate_per_minute=5, daily_limit=20, burst=DEFAULT_BURST),
        ),
        # ⑨ Groq Bulk Fallback — son çare (eski llama-3.1-8b-instant yerine gpt-oss-20b)
        # NOT: 8b-instant 14.4K RPD sundu; ücretsiz katmanda hiçbir sohbet modeli
        # artık 1K RPD üstüne çıkmıyor (2026-06-17 Groq deprecation).
        LLMAccount(
            provider="groq", account_id="A",
            model="openai/gpt-oss-20b",
            api_key=os.environ.get("GROQ_API_KEY_A", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST, tpm_limit=GROQ_TPM),
        ),
    ]
    # Filter out accounts with empty API keys
    active = [a for a in accounts if a.api_key]
    if not active:
        logger.critical("No LLM API keys configured! Set GROQ_API_KEY_A/B, OPENROUTER_API_KEY_A/B and/or GEMINI_API_KEY")
    return LLMRouter(_share_buckets(active))


def build_quality_router() -> LLMRouter:
    """Router for LOW-VOLUME, quality-sensitive prose/judgment work: the SITREP
    narrator, storyline narratives, and the weekly forecast — every text a human
    actually reads. Deliberately NOT used for Pass A–E bulk scoring: these slots'
    rate limits can't carry bulk volume, and swapping bulk models would shift the
    severity-score calibration the alert thresholds are tuned to.

    Cascade: Mistral large (best Turkish of the free options) → Cerebras
    gpt-oss-120b (fast, 1M tokens/day) → the full main cascade as fallback, so a
    missing key or provider outage degrades to exactly the pre-2026-07-17 behavior.

    Limits read off the providers' dashboards (2026-07-17):
      - Mistral free tier: per-model rate limits only — mistral-large-2512 at
        250K TPM / 0.07 RPS (~4 RPM). No daily request cap shown; rpd is set to a
        generous bound just to keep the TokenBucket day-accounting meaningful.
      - Cerebras free tier: 5 requests/min, 2400/day; 30K tokens/min, 1M/day.
        The 30K TPM window also acts as the per-request ceiling (model_profiles).
    """
    quality_slots = [
        LLMAccount(
            provider="mistral", account_id="A",
            model="mistral-large-2512",
            api_key=os.environ.get("MISTRAL_API_KEY", ""),
            rpm=4, rpd=2000,
            bucket=TokenBucket(rate_per_minute=4, daily_limit=2000, burst=2,
                               tpm_limit=250_000),
        ),
        LLMAccount(
            provider="cerebras", account_id="A",
            model="gpt-oss-120b",
            api_key=os.environ.get("CEREBRAS_API_KEY", ""),
            rpm=5, rpd=2400,
            bucket=TokenBucket(rate_per_minute=5, daily_limit=2400, burst=2,
                               tpm_limit=30_000),
        ),
    ]
    active = [s for s in quality_slots if s.api_key]
    if not active:
        logger.warning("Quality router: no MISTRAL_API_KEY/CEREBRAS_API_KEY set, "
                       "falling back to full router")
        return build_llm_router()
    return LLMRouter(_share_buckets(active) + build_llm_router().accounts)


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
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST, tpm_limit=GROQ_TPM),
        )
        for account_id in ("A", "B")
    ]
    active = [s for s in slots if s.api_key]
    if not active:
        logger.warning("Bulk router: no GROQ_API_KEY_A/B set, falling back to full router")
        return build_llm_router()
    return LLMRouter(_share_buckets(active))
