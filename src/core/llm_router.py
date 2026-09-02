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
# Was 10 until 2026-08-27. A 5xx costs a round trip when the provider answers fast and
# a full request_timeout when it does not, so ten strikes is a budget, not a guard:
# Daily Country SITREP #48 spent 29 of its 34 minutes collecting nine mistral-large
# 503s and six 180-second read timeouts before the tenth failure finally sidelined the
# account. Three is enough evidence that a provider is down for this run, and routers
# are rebuilt per run so a wrong call costs one run's worth of a single slot.
ERROR_THRESHOLD = 3
# A 5xx below the threshold is still treated as transient — a short sideline, not the
# 600s outage cooldown — but it must be a sideline. Leaving the slot SERVING is what let
# the same dead account be re-picked on the very next selection, nine times in a row.
SOFT_ERROR_COOLDOWN_SECONDS = 30
# Deterministic client error (HTTP 4xx other than 429): the request is structurally
# rejected (bad model, unsupported param, bad key). Sideline the slot briefly so the
# router stops re-selecting it within the same rotation loop — where no cooldown means
# a broken slot gets picked again on its remaining burst tokens (the double-400 we saw).
CLIENT_ERROR_COOLDOWN_SECONDS = 120
# A slot that blew its wall-clock ceiling is sidelined for longer than a 4xx one: a
# broken slot answers a re-probe instantly, a slow slot charges another full ceiling
# for the same information. Routers are rebuilt per run, so a cooldown this long means
# "for the rest of this run" and nothing more — the next run re-probes from scratch.
SLOW_SLOT_COOLDOWN_SECONDS = 1800
# Max tokens held at once per model slot — smooths the opening burst.
DEFAULT_BURST = 8
# Groq free-tier tokens-per-minute ceiling (gpt-oss-120b/20b, qwen3.8-27b all list 8K).
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
        cooldown: float | None = None,
    ):
        """Mark an account as degraded after a failed call.

        retry_after: seconds from the provider's Retry-After header (429), if any.
        Honored over the default cooldown, clamped to MAX_RATE_LIMIT_COOLDOWN_SECONDS.
        hard_error: the slot is unusable right now — a deterministic client 4xx (not
        429), an empty/error body inside an HTTP 200 (OpenRouter free upstream
        failures), or a response so slow it blew the profile's wall-clock ceiling.
        Sideline it on a short cooldown so it leaves the rotation instead of being
        re-picked on burst tokens.
        cooldown: overrides the default hard-error cooldown, for callers that know
        re-probing this slot is expensive (see SLOW_SLOT_COOLDOWN_SECONDS).
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
                sideline = cooldown if cooldown is not None else CLIENT_ERROR_COOLDOWN_SECONDS
                acct.daily_errors += 1
                acct.status = ProviderStatus.RATE_LIMITED
                acct.cooldown_until = time.monotonic() + sideline
                logger.warning(
                    "Account %s unusable (4xx, empty-200, or too slow), cooldown %.0fs",
                    acct.display_name, sideline,
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
                else:
                    acct.status = ProviderStatus.RATE_LIMITED
                    acct.cooldown_until = time.monotonic() + SOFT_ERROR_COOLDOWN_SECONDS
                    logger.warning(
                        "Account %s soft error #%d, sidelined %ds",
                        acct.display_name, acct.daily_errors, SOFT_ERROR_COOLDOWN_SECONDS,
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
        # ② OpenRouter-A Secondary — Nemotron endpoint'i tökezlerse devralan
        # bağımsız aile. ① ile AYNI hesap kotasını (bucket) paylaşır.
        # NOT: bu slot önce gpt-oss-120b:free idi (2026-07-17'de kaldırıldı, HTTP
        # 404), sonra gpt-oss-20b:free. 2026-09-02'de OpenRouter kataloğunda free
        # gpt-oss ailesinin SON üyesi de kalktı — 421 modellik listede ne
        # 20b:free ne 120b:free var, yalnız ücretli slug'lar duruyor. Slot her
        # koşuda 38 kez 404 alıyordu. Yerine önce GLM 5.2 kondu (acil, doğrulanmamış),
        # 2026-09-02'de MiniMax M3'e yükseltildi: free katmanda response_format
        # destekleyen 7 modelden en çok kullanılanı (4,15T token vs GLM'in 20,6B'si)
        # ve 1M bağlam. Reasoning modeli → model_profiles'ta reasoning kilidi ŞART.
        # NOT: nemotron-3-ultra (550B, kataloğun en güçlüsü) BİLEREK seçilmedi —
        # response_format desteği yok ve JSON mode olmadan Pass C batch'lerinin
        # ~%7'si cümle ortasında bozuluyordu (2026-08-05/06).
        LLMAccount(
            provider="openrouter", account_id="A",
            model="minimax/minimax-m3:free",
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
        # ④ Groq-A Secondary — kalite yedeği (eski llama-3.3-70b-versatile yerine).
        # qwen3.6-27b 2026-09-02'de deprecate edildi, 14 Eylül'de kapanıyor ve
        # sonrasında istekler 3.8'e OTOMATİK yönlendirilecekti. Beklemek yerine
        # kontrollü geçildi: otomatik yönlendirme, Pass C'nin sınıflandırma modelini
        # bizim haberimiz olmadan değiştirmek demekti ve model değişimi severity
        # dağılımını kaydırır. `scripts/probe_models.py` ile ikisi aynı 4 raporda
        # yan yana ölçüldü: 16 alanın 16'sı BİREBİR aynı (relevance skorları dahil),
        # 3.8 biraz da hızlı (1718ms vs 1840ms). reasoning_effort="none" kabul
        # ediliyor, yani model_profiles'taki qwen kuralı aynen geçerli.
        LLMAccount(
            provider="groq", account_id="A",
            model="qwen/qwen3.8-27b",
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
            model="qwen/qwen3.8-27b",
            api_key=os.environ.get("GROQ_API_KEY_B", ""),
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000, burst=DEFAULT_BURST, tpm_limit=GROQ_TPM),
        ),
        # ⑦ OpenRouter-B Mirror — cross-key yedek. Hesap fonsuz → 50 istek/gün.
        # (Eski Hermes-3-405B slotu kaldırıldı: key A'nın kotası artık hesap
        # genelinde paylaşıldığından üçüncü bir key-A free slotu kota eklemiyordu.
        # 120b:free'nin kaldırılmasıyla (2026-07-17) 20b:free'ye, o da katalogdan
        # kalkınca (2026-09-02) ② ile aynı modele düşürüldü — bugün MiniMax M3.
        # Key B fonsuz olduğu için buraya ücretli bir slug konamaz — free kalmak
        # zorunda.)
        LLMAccount(
            provider="openrouter", account_id="B",
            model="minimax/minimax-m3:free",
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
        # ⑧b Gemini genişlemesi — kota PROJE ve MODEL başına ayrı verilir.
        #
        # 3.5-flash-lite, 3.1-flash-lite ile aynı projede olmasına rağmen kendi
        # 500 RPD'sini taşıyor (ayrı model kovası). GEMINI_API_KEY_2 ise ayrı bir
        # projeye ait, dolayısıyla her iki modelde de sıfırdan 500 RPD getiriyor.
        # Bu anahtar 2026-07-23'e kadar YALNIZCA SITREP grounding'inde kullanılıyordu;
        # 2.5-flash-lite'ın emekli olmasıyla o yol tamamen öldü ve anahtar boşta
        # kaldı. Metin tarafı sapasağlam — beş modelde de 200 döndüğü, OpenAI-uyumlu
        # uçta json_mode'un temiz JSON verdiği gerçek çağrıyla doğrulandı.
        #
        # Toplam Gemini kapasitesi: 500 → 2000 RPD.
        LLMAccount(
            provider="gemini", account_id="A",
            model="gemini-3.5-flash-lite",
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            rpm=15, rpd=500,
            bucket=TokenBucket(rate_per_minute=15, daily_limit=500, burst=DEFAULT_BURST),
        ),
        LLMAccount(
            provider="gemini", account_id="B",
            model="gemini-3.1-flash-lite",
            api_key=os.environ.get("GEMINI_API_KEY_2", ""),
            rpm=15, rpd=500,
            bucket=TokenBucket(rate_per_minute=15, daily_limit=500, burst=DEFAULT_BURST),
        ),
        LLMAccount(
            provider="gemini", account_id="B",
            model="gemini-3.5-flash-lite",
            api_key=os.environ.get("GEMINI_API_KEY_2", ""),
            rpm=15, rpd=500,
            bucket=TokenBucket(rate_per_minute=15, daily_limit=500, burst=DEFAULT_BURST),
        ),
        LLMAccount(
            provider="gemini", account_id="A",
            # Acil yedek: yalnızca 20 RPD — bucket yerelde durdurur, 429'a sürmez.
            # Çıplak "gemini-3-flash" 404 döndürüyor; doğru kimlik "-preview" ekli
            # (deprecated ama kapanış tarihi yok). Sırası kasıtlı: 20 RPD'lik bu
            # slot 500'lüklerin ARKASINDA durmalı, yoksa dakikalar içinde tükenip
            # arkasındaki 1500 RPD'ye hiç sıra gelmeden cascade'i aşağı iter.
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

    Cascade: Mistral medium (best Turkish still reachable on this account) → LLM7
    minimax-m2.7 → Pollinations laguna-s-2.1 → the full main cascade as fallback,
    so a missing key or provider outage degrades to exactly the pre-2026-07-17
    behavior.

    The two rungs between exist because Cerebras' removal (below) left Mistral
    ALONE here, and the fall-through is not a real safety net for this router's
    work: the main cascade's Groq slots are skipped by the request-size guard at a
    6K-token budget, so a Mistral outage meant no narrative at all. That is not
    hypothetical — Daily Country SITREP #47 died on the workflow timeout and #48
    survived it by 42 seconds during the 2026-08-27 ReadTimeout storm. Both rungs
    are measured against the real Pass C prompt (scripts/probe_models.py), not
    chosen from catalog copy:
      - llm7/minimax-m2.7 — PASS at 14.1s, first-party slot on LLM7's token
        allowance, which is a TOKEN budget (~1M/day) and so is spent by this
        router's low volume far more slowly than by anything in Pass A-E.
      - pollinations/poolside-laguna-s-2.1:free — PASS at 15.9s, and the only one
        of six free Pollinations models that was usable: glm-fast returned HTTP 200
        with an empty completion after 96s, nemotron-3-ultra-550b read-timed out
        twice, two more 503'd. Turkish prose checked separately (11.4s, grammatical,
        no inverted sentences) — but its usage came back with reasoning_tokens at
        163 of 386 completion tokens, i.e. hidden thinking takes ~42% of whatever
        max_tokens the caller asked for. At the 6K SITREP budget that is a real cut
        in usable narrative, and no reasoning knob is declared for this provider
        (see model_profiles), so treat its output length as roughly half the ask.
        It sits LAST on purpose: every free model there is an individual's upstream
        key registered into a community router and flagged alpha, so it is a
        zero-cost safety net and never a slot to depend on.
    Neither is a quality upgrade over Mistral — both are cheaper prose. They are
    ordered behind it so they only ever run when the rung above is already dead.

    Cerebras (gpt-oss-120b) sat between the two until 2026-09-02, when its free
    tier ended: every run answered HTTP 402 (payment required) exactly once before
    rotating away, so the slot had stopped being a quality tier and become a fixed
    per-run round-trip. Removed rather than re-pointed at the paid tier — the whole
    router exists to buy prose quality at zero cost, and Mistral already covers it.

    mistral-large-2512 was the slot until the same day, when it began answering
    403 "This model is not available in your subscription tier". The account is
    fine and the slug is current (Mistral Large 3, v25.12) — the MODEL is outside
    the tier, and mistral-large-2411 retired 2026-05-31, so no "large" is
    reachable. NB: the rate-limit dashboard still lists mistral-large-2512 with
    limits, so a model appearing there is NOT evidence of access; only a live call
    is. That is what the 4xx error-body logging in llm_client is for.

    Limits read off the account's rate-limit dashboard (2026-09-02):
      - mistral-medium-latest: 25K TPM / 0.83 RPS. TPM is the binding constraint
        and it is 10x TIGHTER than large's 250K — one 6K-token SITREP narrative
        plus its prompt is a large fraction of a single minute's budget, so burst
        is 1: two concurrent full-size calls would exceed the window and 429.
        rpd is a generous bound, only to keep day-accounting meaningful.
      - Deliberately NOT ministral-14b/8b/3b: they have 25-50x the TPM headroom
        (937K/625K/1.3M) but they are small edge models, and prose quality is the
        only reason this router exists. Throughput was never the constraint here.
    """
    quality_slots = [
        LLMAccount(
            provider="mistral", account_id="A",
            model="mistral-medium-latest",
            api_key=os.environ.get("MISTRAL_API_KEY", ""),
            rpm=2, rpd=2000,
            bucket=TokenBucket(rate_per_minute=2, daily_limit=2000, burst=1,
                               tpm_limit=25_000),
        ),
        # LLM7 publishes no RPM/RPD; the documented allowance is ~1M tokens/day and
        # this router spends tokens in narrative-sized lumps, not in request counts.
        # So the limits below are a self-imposed bound to keep day-accounting honest
        # and to stop a retry storm from draining the allowance the rest of the day,
        # NOT a mirror of a published server-side limit. burst=1 for the same reason
        # as Mistral: two concurrent 6K-token narratives are not a load this tier is
        # sized for. Lowest measured availability of LLM7's four reachable models
        # (94.4%), which is acceptable for a slot that only runs when Mistral is out.
        LLMAccount(
            provider="llm7", account_id="A",
            model="minimax-m2.7",
            api_key=os.environ.get("LLM7_KEY", ""),
            rpm=6, rpd=300,
            bucket=TokenBucket(rate_per_minute=6, daily_limit=300, burst=1),
        ),
        # per_user_rpm=30 is published in the Pollinations catalogue; halved here
        # because the number describes the community router's own ceiling, not the
        # upstream key behind it, and that key's owner gets no warning from us.
        LLMAccount(
            provider="pollinations", account_id="A",
            model="YoannDev90/poolside-laguna-s-2.1:free",
            api_key=os.environ.get("POLLINATIONS_API_KEY", ""),
            rpm=15, rpd=300,
            bucket=TokenBucket(rate_per_minute=15, daily_limit=300, burst=1),
        ),
    ]
    active = [s for s in quality_slots if s.api_key]
    if not active:
        logger.warning("Quality router: no MISTRAL_API_KEY/LLM7_KEY/"
                       "POLLINATIONS_API_KEY set, falling back to full router")
        return build_llm_router()
    return LLMRouter(_share_buckets(active) + build_llm_router().accounts)


def build_bulk_router() -> LLMRouter:
    """Router for low-stakes, high-volume work (e.g. storyline narrative prose).

    Uses gpt-oss-20b, which the main cascade touches ONLY as its last-resort rung
    (build_llm_router's final Groq slot). So bulk narrative work never competes with
    Pass C for the scarce smart-model quota — gpt-oss-120b and qwen3.8-27b have their
    own per-(key, model) buckets — but it does share the bucket of that final rung,
    which is the same sharing _BUCKET_REGISTRY exists to account for honestly. An
    earlier version of this docstring claimed the model was NOT in the main cascade
    at all, which the registry comment above already contradicted. Stacking the slot across both Groq
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
