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
    # Overrides PROVIDER_ENDPOINTS for this slot. Every provider so far has one
    # fixed URL for everybody; Cloudflare Workers AI puts the ACCOUNT ID in the
    # path, so the endpoint is a property of the account rather than of the
    # provider and cannot be looked up from a flat table.
    endpoint: str = ""

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
        # ⑧c Üçüncü proje (2026-09-03). Aynı gerekçenin devamı: kotanın proje başına
        # olduğu tahmin değil, kendi 429'umuzun quotaId'siyle sabit —
        # "GenerateRequestsPerDayPerProjectPerModel-FreeTier". İki model × 500 RPD
        # ile toplam Gemini kapasitesi 2000 → 3000 RPD.
        #
        # GEMINI_API_KEY_3 tanımlı değilse bu iki slot aşağıdaki `if a.api_key`
        # süzgecinde düşer, yani anahtar oluşturulmadan önce merge etmek güvenli.
        #
        # DİKKAT — her kota proje başına DEĞİL: Gemini-3 ailesinin Search grounding
        # kotası iki mevcut projede de 0/0 çıkmıştı (2026-07-18). Proje çoğaltmak
        # MODEL RPD'sini çarpar, tier geneli kısıtları çarpmaz.
        LLMAccount(
            provider="gemini", account_id="C",
            model="gemini-3.1-flash-lite",
            api_key=os.environ.get("GEMINI_API_KEY_3", ""),
            rpm=15, rpd=500,
            bucket=TokenBucket(rate_per_minute=15, daily_limit=500, burst=DEFAULT_BURST),
        ),
        LLMAccount(
            provider="gemini", account_id="C",
            model="gemini-3.5-flash-lite",
            api_key=os.environ.get("GEMINI_API_KEY_3", ""),
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


# Slots measured against the bulletin's REAL extraction prompt
# (scripts/probe_models.py --bulletin). Re-measured 2026-09-04 on the sample
# grown from 8 rows to 16, after the 4 Sep bulletin filed 15 real strikes as
# "regional" and printed a police arrest between two missile strikes:
#
#   gemini-3.5-flash-lite   actor 16/16  target 16/16  standing 16/16  war 16/16  ← PASS
#   qwen/qwen3.8-27b        actor 16/16  target 16/16  standing 16/16  war 15/16  (see below)
#   nemotron-3-super        actor 14/16  target 16/16  standing 13/16  war 16/16  ← EXCLUDED
#   openai/gpt-oss-20b      actor  6/8   standing  7/8                            ← EXCLUDED
#
# qwen's figures are from the round before the last three prompt clarifications,
# and its one war miss is a reference label the probe then talked me out of. Groq
# has answered 429 on the first call of every probe round since, so it has NOT
# been re-measured under the final prompt — the changes since only tighten
# definitions in directions it already answered correctly, but that is an argument
# and not a measurement. Re-run it.
#
# Both exclusions are for a specific reproducible failure, never a general
# impression, and both are the same class: asserting a direction the text does not
# carry.
#   * gpt-oss-20b returns actor=iran on "Iran says 18 killed, 142 injured in US
#     strikes", filing an American strike as an Iranian one.
#   * nemotron returns actor=iran on "Kuwait Air Defences Responding To Missile,
#     Drone Strikes: Kuwaiti Army", which names no attacker at all. It held that
#     answer across four probe rounds including the one where a concrete
#     counterweight in the prompt fixed exactly this in gemini. Filling the actor
#     in from what a model knows about who is fighting in the Gulf is the failure
#     assign_section's own docstring calls the worst outcome available, and that
#     headline shape appeared six times in the 4 Sep window alone.
# Losing the third rung costs nothing observed: over the bulletin's first two days
# extraction ran 27 times on qwen and 12 on gemini, and never once reached
# nemotron. And the failure it leaves behind is the honest one — no slot means
# every event stays unattributed and lands in the regional section, which is the
# bulletin saying it could not read the direction rather than guessing it.
BULLETIN_MEASURED_MODELS = (
    "qwen/qwen3.8-27b",
    "gemini-3.5-flash-lite",
)


def build_bulletin_router() -> LLMRouter:
    """Router for the Iran bulletin's direction extraction.

    Filtered from build_llm_router() rather than re-declared, so rate limits,
    shared buckets and key wiring can never drift from the bulk definitions — the
    only thing this function decides is WHICH slots are allowed.

    Returns an empty router when none of the measured models has a key. That is
    deliberate: extraction then fails, every event keeps the unattributed default
    and falls to the regional section, which reports that direction could not be
    established. Falling back to the full cascade instead would silently reach the
    one slot that inverts it.
    """
    order = {m: i for i, m in enumerate(BULLETIN_MEASURED_MODELS)}
    accounts = [a for a in build_llm_router().accounts if a.model in order]
    accounts.sort(key=lambda a: order[a.model])
    if not accounts:
        logger.warning("Bulletin router has no measured slot with a key; "
                       "direction extraction will report unattributed")
    return LLMRouter(accounts)


def build_quality_router() -> LLMRouter:
    """Router for LOW-VOLUME, quality-sensitive prose/judgment work: the SITREP
    narrator, storyline narratives, and the weekly forecast — every text a human
    actually reads. Deliberately NOT used for Pass A–E bulk scoring: these slots'
    rate limits can't carry bulk volume, and swapping bulk models would shift the
    severity-score calibration the alert thresholds are tuned to.

    Cascade: OpenRouter gemini-3.1-flash-lite (paid, the floor) → Cloudflare Workers AI
    gpt-oss-120b → Cloudflare mistral-small-3.1-24b (both only when the
    CLOUDFLARE_* vars are set) → LLM7 minimax-m2.7 → Pollinations gpt-oss → the
    full main cascade as fallback, so a missing key or provider outage degrades to
    exactly the pre-2026-07-17 behavior.

    One paid rung on top, everything free beneath it. That shape is the point: a
    free tier dying stops being an incident and becomes a log line, because the
    reports were never being written by it. The free slots stay because they cost
    nothing and now answer to contracts (see call_llm's `accept`) — they are a
    fallback, not the plan.

    Cloudflare leads from 2026-09-04, and the reason is a measurement rather than
    a preference. Two things happened that day. Mistral answered HTTP 429 to every
    one of the five country SITREP calls, and then to two more cold single calls
    45 seconds apart in the regression probe — this is not our own burst against
    the 25K TPM ceiling, it is the slot being unavailable, and a dead slot at the
    front of a cascade is a fixed per-run round-trip paid before any work starts.
    That is exactly why Cerebras was removed two days earlier when its free tier
    began answering 402. And the two Cloudflare slots PASSED the same probe on the
    real SITREP prompt while the two rungs below them failed it.

    Mistral keeps its place in the cascade rather than being deleted: nothing says
    the 429s are permanent, its Turkish is the best this project has measured, and
    a slot that costs one round-trip when it is down costs nothing when it is up.
    It is demoted, not judged.

    A word on minimax-m2.7, which sits third and passed its Pass C probe: on
    2026-09-04 it narrated all five SITREPs and shortened EVERY citation URL to a
    bare domain, so all 108 were blanked by the allowlist and the reports shipped
    with no working link — while six other models over the preceding 21 days
    averaged 0.3 blanked per report. It is not demoted for that, because the fix
    belongs one layer up: run_sitrep_llm now holds every slot to the citation
    contract and rotates past whichever one fails it. Ordering is about quality;
    contracts are about correctness, and a cascade should not encode a contract.

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
      - pollinations/gpt-oss (GPT-OSS 20B, official) — PASS at 37.5s on the real
        SITREP prompt, 2026-09-04, replacing laguna-s-2.1 which failed the same
        probe by inventing an FIR code. Still LAST: two of the four official models
        probed alongside it returned an empty HTTP 200, so "official" raises this
        provider's floor without making it dependable.
    Neither is a quality upgrade on the rungs above — both are cheaper prose, and
    both failed the 2026-09-04 regression probe (minimax truncates citation URLs
    to bare domains, laguna invents FIR codes). They are last because they only
    ever run when everything above them is already dead, and a degraded report is
    better than no report.

    Cerebras (gpt-oss-120b) sat here until 2026-09-02, when its free tier ended:
    every run answered HTTP 402 (payment required) exactly once before rotating
    away, so the slot had stopped being a quality tier and become a fixed per-run
    round-trip. The MODEL came back on 2026-09-04 on Cloudflare, which is the
    point of naming the host and the weights separately: what ended was Cerebras'
    free tier, not gpt-oss-120b's suitability for this work.

    ── Mistral, removed 2026-09-04 ──────────────────────────────────────────
    Mistral was this router's reason for existing: it wrote the SITREP prose from
    2026-07-17 and its Turkish is still the best this project has measured, over
    290 successful calls in the preceding month. It is gone because the ACCOUNT
    stopped being able to make requests, in two steps that only made sense once
    the 429 bodies were finally logged:

      * 2026-09-02, mistral-large-2512 → HTTP 403 "This model is not available in
        your subscription tier". Read at the time as one model leaving the tier.
      * 2026-09-04 03:16 the last call succeeded; by 07:32 every call answered
        HTTP 429 carrying `x-ratelimit-limit-req-minute=0`.

    The limit was not exceeded. The limit IS zero — the workspace is entitled to
    no requests at all. A new API key changed nothing, and it could not: Mistral
    sets rate tiers per WORKSPACE, so a fresh key in the same workspace inherits
    the same zero. Billing was never attached to that workspace, which makes the
    two dates one event rather than two: a free entitlement expiring in stages,
    expensive models first.

    Deleted rather than demoted, after being demoted for half a day. A slot that
    cannot answer is a fixed per-run round-trip paid before any work starts —
    exactly the reason Cerebras went — and this one cannot come back without a
    billing decision. If that decision is ever made, note that Mistral Medium 3.5
    is $8.22/month for this router's measured volume, against $0 for the two
    Cloudflare slots that now do the job and passed the same probe.
    """
    quality_slots = _quality_slots()
    active = [s for s in quality_slots if s.api_key]
    if not active:
        logger.warning("Quality router: no CLOUDFLARE_API_TOKEN/LLM7_KEY/"
                       "POLLINATIONS_API_KEY set, falling back to full router")
        return build_llm_router()
    return LLMRouter(_share_buckets(active) + build_llm_router().accounts)


def quality_slot_models() -> tuple:
    """The (provider, model) pairs this router DELIBERATELY chose, without the
    main cascade it falls through to.

    Exists so the weekly regression probe tests what production actually uses.
    A hand-kept list beside the router would drift from it, and a drifted list is
    worse than none: it reports green about slots nothing runs while the slots
    that do run go unwatched.
    """
    return tuple((s.provider, s.model) for s in _quality_slots() if s.api_key)


def _quality_slots() -> list:
    """The quality cascade's own slots, in order. See build_quality_router."""
    quality_slots = [
        # ── The floor, added 2026-09-04 ──────────────────────────────────────
        #
        # This is the first slot in this router that is not a free tier, and the
        # reason is a rate, not an incident. Free tiers have no SLA, no version
        # pin and no deprecation policy, so the RATE at which this cascade has to
        # be rebuilt is set by other people: Cerebras' free tier ended 2 Sep,
        # gemini-2.5-flash-lite retired early, OpenRouter's free gpt-oss family
        # vanished, Mistral's workspace went to zero requests 4 Sep, laguna's
        # family started posting five-day shutdown notices. Every guard built
        # today lowers the cost of NOTICING that; none of them lowers the rate.
        # Only a contract does.
        #
        # It costs nothing new. OPENROUTER_API_KEY_A was funded with $10 in July
        # 2026 to lift the free-model request cap, and that credit has been idle
        # ever since because this project only ever used its `:free` slots. At
        # this router's measured volume — 2.21M prompt + 0.55M completion tokens
        # a month, 8.2% of SIM's total and 100% of every LLM incident it has had
        # — Haiku 4.5 is $4.96/month, so the credit already sitting there is
        # about two months, and gpt-5-mini at $1.65 would be six.
        #
        # Chosen on the real SITREP prompt, priced against this router's MEASURED
        # monthly volume (2.27M prompt + 0.50M completion, from seven days of
        # telemetry). Every candidate below passed; the choice was made on
        # Turkish, speed and price in that order (probes, 2026-09-04):
        #
        #   gemini-3.1-flash-lite  PASS   3.3s  278 words  $1.31/mo  ← this one
        #   claude-haiku-4.5       PASS   9.8s  256 words  $4.96/mo
        #   gemini-2.5-flash-lite  PASS   2.9s  266 words  $0.43/mo
        #   gpt-5-mini             PASS  38.3s  427 words  $1.65/mo
        #   gemini-3-flash-preview PASS   6.2s  262 words  $2.75/mo
        #
        # gemini-3.1-flash-lite wrote the cleanest Turkish of anything probed all
        # day — "kritik bir eşiğe ulaşmıştır", "ikinci bir duyuruya kadar askıya
        # aldığını açıklamıştır" — and it alone handled the unnamed carriers
        # correctly ("ismi belirtilmeyen iki farklı havayolu şirketi") instead of
        # inventing names. It is three times faster than Haiku and a quarter of
        # the price.
        #
        # And it is not a guess dressed as a measurement: this model already
        # narrated 14 production SITREPs between 29 Aug and 2 Sep, every one of
        # them with working citations and 0.3 blanked per report. Fourteen real
        # reports outrank any probe.
        #
        # gpt-5-mini and Haiku both wrote publisher names as bare domains, which
        # the prompt forbids. gemini-3-flash was excluded on principle rather than
        # output — it is a `-preview` id, and binding the floor of the cascade to
        # a preview reintroduces the instability the slot exists to end. The
        # 2.5-flash-lite line is kept as the cheaper fallback if this ever needs
        # to get cheaper still.
        #
        # rpd is a SPEND cap, not a provider limit. A call costs $0.0055 here, the
        # real load is 8 a day, and 15 leaves room for rotations while capping a
        # runaway month at $2.47. OpenRouter is prepaid, so the balance is itself
        # a hard ceiling and nothing can overspend it — which makes the danger not
        # a surprise bill but a surprise SILENCE, since the floor would simply
        # drop away and the free rungs below would quietly take over.
        # output_health.check_openrouter_credit is the other half of that fix.
        LLMAccount(
            provider="openrouter", account_id="A",
            model="google/gemini-3.1-flash-lite",
            api_key=os.environ.get("OPENROUTER_API_KEY_A", ""),
            rpm=6, rpd=15,
            bucket=TokenBucket(rate_per_minute=6, daily_limit=15, burst=1),
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
        # because the number describes the gateway's own ceiling, not the upstream
        # capacity behind it.
        #
        # Was YoannDev90/poolside-laguna-s-2.1:free until 2026-09-04. Two things
        # replaced it, both from reading the LIVE catalogue rather than the note
        # written about it two days earlier:
        #
        #   * laguna FAILED that day's regression probe — it wrote the FIR code
        #     OAKWX where the payload said OAKX, a fabricated aviation identifier
        #     in a report whose hardest rule is that codes come only from the
        #     computed airspace block.
        #   * The catalogue now lists 228 text models, 39 of them OFFICIAL rather
        #     than community. That distinction is the exact objection that put
        #     laguna last: a community entry is an individual's upstream key
        #     registered into a shared router and flagged alpha, and the laguna
        #     family is visibly churning — a sibling entry carries a five-day
        #     shutdown notice naming our own slot as the migration target.
        #
        # gpt-oss is the only one of four probed officials that passed. Nemotron
        # 3.5 Lightning ran to 3002 words, hit the length ceiling and rewrote its
        # URLs; glm-5.3-flash and GPT-5 Nano both returned an empty HTTP 200. Two
        # failures in four is why this stays LAST despite being official.
        #
        # It is GPT-OSS 20B — the small sibling of the 120B leading this cascade
        # on Cloudflare. At the bottom of a fallback chain that is a feature: the
        # same weights family the report was written around, reached through a
        # completely different host, so a Cloudflare outage does not take the
        # prose style with it.
        LLMAccount(
            provider="pollinations", account_id="A",
            model="gpt-oss",
            api_key=os.environ.get("POLLINATIONS_API_KEY", ""),
            rpm=15, rpd=300,
            bucket=TokenBucket(rate_per_minute=15, daily_limit=300, burst=1),
        ),
    ]
    # Cloudflare Workers AI, added 2026-09-04 because Mistral is the only slot here
    # that has ever met the SITREP's citation contract and it 429'd every one of the
    # five country calls that morning — one narrative is ~13.5K tokens against a 25K
    # TPM ceiling, so two countries in a minute is already over. A second competent
    # slot is what that run needed, not a deeper stack of cheap ones.
    #
    # The free allocation is 10,000 Neurons/day and it is the whole budget, so the
    # model is chosen by what a SITREP-shaped call COSTS, not by what a model card
    # promises. This router makes 7 calls on a full day (5 countries + the digest +
    # the bulletin narrative), and a call is ~11K prompt against the 6K completion
    # ceiling. Against Cloudflare's published Neuron rates:
    #
    #   @cf/openai/gpt-oss-120b                    759 N/call → 5,313/day   53%
    #   @cf/mistralai/mistral-small-3.1-24b       654 N/call → 4,578/day   46%
    #   @cf/meta/llama-3.3-70b-instruct-fp8-fast 1,522 N/call → 10,654/day 107%  ✗
    #
    # llama-3.3-70b was this slot for one afternoon on an estimate that assumed a
    # 3K completion. At the ceiling the narrator is actually allowed, it does not
    # fit the day — 70b output is 204,805 Neurons/M against gpt-oss-120b's 68,182,
    # and output is what a narrator spends.
    #
    # gpt-oss-120b is the pick for a second reason: it is not a new model here. It
    # WAS this router's quality rung on Cerebras until 2026-09-02, writing the
    # SITREP, storyline and forecast prose, and it left because its host's free tier
    # ended rather than because it stopped being good at the job. Cloudflare hosts
    # the same weights.
    #
    # Probed on the host 2026-09-04 with the real SITREP prompt
    # (scripts/probe_models.py --prose). No hidden reasoning leaked and no slot
    # altered a URL, which was the specific worry — Cloudflare documents no
    # reasoning knob for the OpenAI-compatible route, so model_profiles sends none
    # where Cerebras had reasoning_effort=low:
    #
    #   gpt-oss-120b            PASS  22s  759 N  Turkish good, labels canonical.
    #                                             Wrote publisher names as bare
    #                                             domains ("reuters.com"), which the
    #                                             prompt forbids and the source chips
    #                                             render badly — cosmetic, the URL
    #                                             beside it was correct.
    #   mistral-small-3.1-24b   PASS  18s  654 N  Format exactly right; clumsier
    #                                             Turkish, and it printed one event
    #                                             twice in different sections.
    #   llama-4-scout-17b       PASS  17s  734 N  Good Turkish, but dropped the
    #                                             "Doğruluk Durumu:" prefix, which
    #                                             _normalize_label_line keys on — a
    #                                             deviant label it cannot repair.
    #   gemma-4-26b-a4b-it      PASS  83s  264 N  Best Turkish of the five and
    #                                             perfect format. FIVE TIMES the
    #                                             latency, on a payload a fraction of
    #                                             a real one. Not taken; the same
    #                                             slowness got Gemma 4 rejected on
    #                                             Gemini (57.7s, 2026-09-02).
    #   glm-5.3-flash           HTTP 403 — not reachable on this account. The other
    #                                             four answered on the same token, so
    #                                             it is the model, not the token.
    #
    # Two slots, not four. Every Cloudflare model draws on the SAME 10,000 Neuron
    # account allowance, so a third and fourth rung buy no capacity — only another
    # 13.5K-token call to waste when a model misbehaves. What a second slot DOES buy
    # is a different set of weights when the first one fails for its own reasons
    # (a 5xx, or the citation contract), which is exactly the minimax case. Ordered
    # by Turkish; mistral-small is second because its format compliance was perfect
    # and llama-4-scout's was not, and a fallback should be structurally safe.
    #
    # rpd below is a self-imposed bound to stop a retry storm spending the day's
    # Neurons in a minute, NOT a published request limit. The two share a bucket
    # for the same reason they share the allowance.
    cf_account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    cf_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    if cf_account and cf_token:
        cf_endpoint = ("https://api.cloudflare.com/client/v4/accounts/"
                       f"{cf_account}/ai/v1/chat/completions")
        for offset, cf_model in enumerate(("@cf/openai/gpt-oss-120b",
                                           "@cf/mistralai/mistral-small-3.1-24b-instruct")):
            # After the paid floor, not before it: the Cloudflare slots are free
            # and excellent, but "free and excellent" is what every rung in this
            # cascade has been on the day it was added.
            quality_slots.insert(1 + offset, LLMAccount(
                provider="cloudflare", account_id="A",
                model=cf_model,
                api_key=cf_token,
                endpoint=cf_endpoint,
                # 20/day, not 12. On the free allowance the bound existed to stop
                # a retry storm eating the day's Neurons; on Workers Paid the
                # cliff is gone and the bound's job changes to capping SPEND. The
                # router makes 7 calls on a full day, so 20 leaves room for a
                # rotation or two and still cannot run away.
                rpm=6, rpd=20,
                bucket=TokenBucket(rate_per_minute=6, daily_limit=20, burst=1),
            ))

    return quality_slots


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
