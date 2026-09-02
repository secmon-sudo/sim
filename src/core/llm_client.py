"""
SIM — Unified LLM Call Wrapper
Blueprint V20.1 §4.5.6 + §4.5.8

Sends classification requests to the first available LLM provider.
Handles retries, failover, and telemetry logging.
"""

import json
import logging
import threading
import time
from typing import Any

import httpx
import tenacity

from src.core.llm_router import SLOW_SLOT_COOLDOWN_SECONDS, LLMAccount, LLMRouter
from src.core.model_profiles import get_profile, wall_clock_ceiling

logger = logging.getLogger(__name__)

PROVIDER_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    # Google AI Studio's OpenAI-compatibility layer — same chat/completions shape.
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    # Quality-tier provider (2026-07-17): OpenAI-compatible chat/completions.
    # Cerebras removed 2026-09-02 when its free tier ended (HTTP 402 every run).
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    # Aggregator added 2026-09-02 to refill the quality tier Cerebras left empty.
    # OpenAI-compatible; bearer auth like the rest. Its daily allowance is a TOKEN
    # budget (~1M), not a request count, which is why it is aimed at the quality
    # tier and not at Pass C: classify_batch alone is ~81% of SIM's ~1.2M/day and
    # would drain the whole allowance in a single run.
    "llm7": "https://api.llm7.io/v1/chat/completions",
}


class LLMAllThrottled(RuntimeError):
    """Every account is on cooldown/rate-limited; no request was even attempted.

    Expected flow under free-tier TPM pacing (per-minute token windows drained) —
    callers like run_pass_c wait for the soonest refill and retry. Distinct from
    the generic "exhausted after real attempts" RuntimeError, which signals actual
    request failures. Subclasses RuntimeError so existing catchers keep working.
    """


class LLMWallClockExceeded(RuntimeError):
    """One request outran its profile's end-to-end ceiling.

    NOT a subclass of httpx.TimeoutException, deliberately: _send_request's tenacity
    policy retries timeouts, and re-sending a prompt to a slot that is already too slow
    would multiply the wait it was raised to prevent. call_llm sidelines the slot and
    rotates instead. Subclasses RuntimeError so unaware catchers stay safe.
    """


class LLMRequestTooLarge(RuntimeError):
    """THIS request exceeds every account's per-request size ceiling.

    A fault of the request, not of the accounts: retrying the same payload can
    never succeed, so callers must drop/shrink the item and move on — waiting
    (LLMAllThrottled) or aborting the whole stage (generic RuntimeError) are both
    wrong responses. Subclasses RuntimeError so unaware catchers stay safe.
    """


# (provider, model) pairs whose profile claims response_format support but whose
# endpoint proved otherwise at runtime. Keyed by model, not API key: the capability
# belongs to the model/provider pair, so one slot's discovery spares its twin on the
# other key. Process-wide and never cleared — a provider that 400s on response_format
# will keep doing so for the rest of the run.
_JSON_MODE_SIDELINED: set[tuple[str, str]] = set()
_JSON_MODE_LOCK = threading.Lock()


def _json_mode_sidelined(acct: LLMAccount) -> bool:
    with _JSON_MODE_LOCK:
        return (acct.provider, acct.model) in _JSON_MODE_SIDELINED


def _sideline_json_mode(acct: LLMAccount) -> None:
    with _JSON_MODE_LOCK:
        _JSON_MODE_SIDELINED.add((acct.provider, acct.model))


def reset_json_mode_sidelines() -> None:
    """Clear the runtime json-mode denylist (test isolation only)."""
    with _JSON_MODE_LOCK:
        _JSON_MODE_SIDELINED.clear()


#: Cap on the provider error text copied into the log — enough for any real
#: message ("Model not found or no access to it"), short enough that a provider
#: echoing the prompt back cannot flood the run log.
ERROR_DETAIL_MAX_CHARS = 300


def _error_detail(response: httpx.Response) -> str:
    """One-line reason a provider rejected a request, for the rotation log.

    Providers agree on the shape more than the wording: OpenAI-compatible APIs
    nest it under `error.message`, some return a bare `message`, and a proxy in
    front of any of them can return plain text or HTML. Try the structured forms,
    fall back to the raw body, and never let a diagnostic raise — this runs on a
    path that is already handling a failure.
    """
    try:
        body = response.json()
        if isinstance(body, dict):
            err = body.get("error")
            detail = None
            if isinstance(err, dict):
                detail = err.get("message") or err.get("code") or err.get("type")
            elif isinstance(err, str):
                detail = err
            detail = detail or body.get("message") or body.get("detail")
            if detail:
                return " ".join(str(detail).split())[:ERROR_DETAIL_MAX_CHARS]
        text = json.dumps(body)
    except Exception:
        try:
            text = response.text or ""
        except Exception:
            # Reading the body can fail too (stream consumed, decode error). A
            # diagnostic that raises would replace a rotation with a crash, so the
            # fallback has a fallback.
            return "<unreadable body>"
    return " ".join(text.split())[:ERROR_DETAIL_MAX_CHARS] or "<empty body>"


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Extract a backoff hint (seconds) from a 429 response.

    Honors the standard `Retry-After` header (delta-seconds form) and Groq/OpenAI's
    `x-ratelimit-reset-requests` (e.g. "2.5s", "1m30s"). Returns None if absent/unparsable.
    """
    ra = response.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass  # HTTP-date form is not worth parsing for a sub-minute reset window
    reset = response.headers.get("x-ratelimit-reset-requests")
    if reset:
        try:
            total, num = 0.0, ""
            for ch in reset:
                if ch.isdigit() or ch == ".":
                    num += ch
                elif ch == "m":
                    total += float(num or 0) * 60
                    num = ""
                elif ch == "s":
                    total += float(num or 0)
                    num = ""
            if num:  # bare number, assume seconds
                total += float(num)
            return total or None
        except ValueError:
            pass
    return None


def _post_within(send, body: dict, ceiling: float, acct: LLMAccount) -> httpx.Response:
    """Run one blocking POST, giving up on it after `ceiling` seconds.

    httpx has no total-request timeout — its timeout is per read — so the deadline is
    enforced from the outside: the request runs on a daemon thread and the caller stops
    waiting when the ceiling passes. The abandoned thread cannot be cancelled (no
    interruptible I/O primitive here), but it is a daemon, so it neither blocks process
    exit nor outlives the run; its socket closes when the provider finally answers.
    Abandoning a request wastes one already-spent bucket token, which is exactly the
    trade the ceiling exists to make.

    Enforced per HTTP attempt, so a tenacity retry gets a fresh ceiling: retries only
    fire on connection errors, where the previous attempt sent nothing at all.
    """
    outcome: dict[str, Any] = {}

    def _run() -> None:
        try:
            outcome["response"] = send(body)
        except BaseException as exc:  # re-raised on the calling thread below
            outcome["error"] = exc

    worker = threading.Thread(
        target=_run, daemon=True, name=f"llm-post-{acct.provider}-{acct.account_id}"
    )
    worker.start()
    worker.join(ceiling)
    if worker.is_alive():
        raise LLMWallClockExceeded(
            f"{acct.display_name} exceeded its {ceiling:.0f}s wall-clock ceiling"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["response"]


def _worth_retrying(retry_state) -> bool:
    """Retry a connection that never got established; never re-pay a read timeout.

    The two failures inside httpx.TimeoutException cost wildly different amounts and
    carry different information. A ConnectTimeout/ConnectError fails before the request
    is on the wire — cheap, usually transient, worth another go. A ReadTimeout means the
    server ACCEPTED the request and then went silent for the whole request_timeout, and
    retrying buys the same answer at the same price: with mistral-large's 180s timeout,
    three attempts is nine minutes spent learning one fact.

    Measured 2026-08-27 on Daily Country SITREP #48: six read timeouts, ~180s apiece,
    inside a 33.8-minute run whose normal duration is 8-12 minutes. Run #47 the day
    before died on the workflow's 35-minute timeout and #48 survived it by 42 seconds.
    This is the same reasoning the router already applies to slow slots, where a longer
    cooldown is justified because "a slow slot charges another full ceiling for the same
    information" — it just was not applied to the retry that precedes it.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))


@tenacity.retry(
    retry=_worth_retrying,
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
    stop=tenacity.stop_after_attempt(3),
    before_sleep=lambda rs: logger.warning(
        "LLM connection retry #%d: %s",
        rs.attempt_number,
        rs.outcome.exception(),
    ),
)
def _send_request(acct: LLMAccount, messages: list[dict], max_tokens: int = 1024,
                  json_mode: bool = True) -> httpx.Response:
    """Single request to a specific account. Retries on connection errors only.

    json_mode=True forces a JSON-object response (for classifiers/forecasters that
    json.loads the reply). Prose callers (e.g. the storyline narrator) MUST pass
    json_mode=False: Groq's json_object validator requires the word "json" in the
    conversation, so a prose prompt without it returns HTTP 400.
    """
    headers = {
        "Authorization": f"Bearer {acct.api_key}",
        "Content-Type": "application/json",
    }
    if acct.provider == "openrouter":
        headers["HTTP-Referer"] = "https://sim-osint.app"
        headers["X-Title"] = "SIM-OSINT-Pipeline"

    payload = {
        "model": acct.model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    # All per-model quirks (json_mode support, reasoning-minimizing params) are
    # declared in src/core/model_profiles.py — see its checklist before adding a
    # model slot. json_mode forces a JSON object so reasoning models can't return
    # an empty or prose-wrapped message that then fails json.loads. (The prompt
    # already instructs "Respond ONLY with valid JSON", satisfying the OpenAI-compat
    # requirement that the word "json" appear in the conversation.)
    profile = get_profile(acct.provider, acct.model)
    sending_json_mode = profile.supports_json_mode and json_mode and not _json_mode_sidelined(acct)
    if sending_json_mode:
        payload["response_format"] = {"type": "json_object"}
    payload.update(profile.payload_extras)

    def _send(body: dict) -> httpx.Response:
        return httpx.post(
            PROVIDER_ENDPOINTS[acct.provider],
            headers=headers,
            json=body,
            timeout=profile.request_timeout,
        )

    def _post(body: dict) -> httpx.Response:
        return _post_within(_send, body, wall_clock_ceiling(profile, max_tokens), acct)

    response = _post(payload)

    # A profile can only CLAIM response_format support; the endpoint is the authority
    # and providers have revoked it before (OpenRouter free, 2026-07-08). Without this
    # probe a revocation turns every call into a hard 4xx and takes the whole slot out
    # of the cascade. Re-sending without response_format is the experiment: if the bare
    # request succeeds, response_format was the culprit and json mode is disabled for
    # this model process-wide; if it 400s too, the fault lies elsewhere and the error
    # propagates untouched. Costs one extra request on a rare path — and only once per
    # model, since the sideline is sticky. (The retry isn't charged to the account's
    # token bucket, which is spent per call_llm attempt, not per HTTP request.)
    if sending_json_mode and response.status_code == 400:
        bare = {k: v for k, v in payload.items() if k != "response_format"}
        retry = _post(bare)
        if retry.is_success:
            _sideline_json_mode(acct)
            logger.warning(
                "LLM %s rejected response_format (HTTP 400) but succeeded without it — "
                "json mode disabled for this model for the rest of the process; "
                "drop it from OPENROUTER_JSON_MODE_MODELS if this persists",
                acct.display_name,
            )
        response = retry

    response.raise_for_status()
    return response


# The stages that spend LLM calls. Kept as one closed list so spend can be summed
# by stage without string-matching drift; add here before using a new name.
PURPOSES = frozenset({
    "classify_single",        # pass_c, one event per call (fallback path)
    "classify_batch",         # pass_c, chunk of events per call (normal path)
    "dedup_adjudication",     # duplicate-card judgement at dispatch
    "storyline_adjudication", # storyline linking judgement
    "storyline_narrative",    # prose narrative per storyline
    "sitrep_country",         # per-country daily SITREP prose
    "sitrep_digest",          # run-level executive briefing
    "forecast_g1_selection",  # weekly: country shortlist
    "forecast_g2_country",    # weekly: per-country assessment
    "forecast_g3_global",     # weekly: global assessment
    "vocab_audit_judge",      # weekly gate audit, scripts/vocab_audit.py
})


def call_llm(router: LLMRouter, prompt: str, system_prompt: str | None = None, max_tokens: int = 1024,
             json_mode: bool = True) -> dict[str, Any]:
    """
    Try all available accounts in priority order.

    Returns dict with keys:
        - response: parsed JSON from LLM
        - provider: "groq" | "openrouter"
        - account: "A" | "B"
        - model: model ID string
        - latency_ms: int
        - content: extracted text content

    Raises LLMAllThrottled if every account is on cooldown before any attempt,
    or RuntimeError if all accounts were tried and failed.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Estimate tokens for TPM accounting: ~4 chars/token for the prompt, plus the
    # completion budget. Charged against each account's per-minute token window so a
    # burst of requests can't blow the (much tighter than RPM) TPM ceiling.
    est_tokens = sum(len(m["content"]) for m in messages) // 4 + max_tokens

    # Pre-send size guard: skip accounts whose per-request ceiling this call would
    # blow (Groq answers HTTP 413) — without spending the account's bucket tokens
    # or, worse, its 4xx cooldown. Accounts that DID answer 413 anyway (the ~4
    # chars/token estimate undershot) join the same skip set so one oversized
    # payload can't be re-sent to the slot that just rejected it.
    skip_for_size: set[str] = set()

    def _fits(acct: LLMAccount) -> bool:
        if acct.display_name in skip_for_size:
            return False
        limit = get_profile(acct.provider, acct.model).max_request_tokens
        if limit is not None and est_tokens > limit:
            skip_for_size.add(acct.display_name)
            return False
        return True

    last_error = None
    attempted = False

    for _ in range(len(router.accounts)):
        acct = router.get_available_account(est_tokens=est_tokens, predicate=_fits)
        if acct is None:
            break
        attempted = True

        try:
            t0 = time.monotonic()
            resp = _send_request(acct, messages, max_tokens=max_tokens, json_mode=json_mode)
            latency_ms = int((time.monotonic() - t0) * 1000)

            data = resp.json()
            content = ""
            finish_reason = ""
            if data.get("choices"):
                choice = data["choices"][0]
                content = choice.get("message", {}).get("content", "")
                finish_reason = choice.get("finish_reason") or ""

            # OpenRouter free endpoints fail INSIDE a 200 when the upstream
            # provider chokes: the body carries an "error" object or an empty
            # completion (blank content, blank finish_reason, ~2s latency).
            # Returning that as success surfaces downstream as a bogus parse
            # error and skips the 8 healthy fallback slots — treat it as a
            # provider failure instead: sideline the slot and rotate.
            api_error = data.get("error")
            if api_error or not content.strip():
                router.report_failure(acct, hard_error=True)
                last_error = RuntimeError(
                    f"unusable HTTP 200 from {acct.display_name}: "
                    f"{'error body: ' + str(api_error)[:200] if api_error else 'empty completion'}"
                )
                logger.warning(
                    "LLM %s returned %s inside HTTP 200 (finish_reason=%r), rotating...",
                    acct.display_name,
                    f"error body {str(api_error)[:200]!r}" if api_error else "an empty completion",
                    finish_reason,
                )
                continue

            router.report_success(acct)
            return {
                "response": data,
                "provider": acct.provider,
                "account": acct.account_id,
                "model": acct.model,
                "latency_ms": latency_ms,
                "content": content,
                "finish_reason": finish_reason,
            }

        except LLMWallClockExceeded as e:
            # hard_error with the long cooldown: a slot that blew the ceiling once will
            # blow it on the next chunk too, and Pass C is sequential — re-probing it
            # every two minutes would spend most of what the ceiling just saved. The
            # sideline lasts the rest of this run only; routers are rebuilt per run, so
            # a transient upstream stall never demotes the slot beyond it.
            router.report_failure(acct, hard_error=True,
                                  cooldown=SLOW_SLOT_COOLDOWN_SECONDS)
            last_error = e
            logger.warning(
                "LLM %s too slow (>%.0fs wall clock for a %d-token budget), "
                "sidelining and rotating...",
                acct.display_name,
                wall_clock_ceiling(get_profile(acct.provider, acct.model), max_tokens),
                max_tokens,
            )

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 413:
                # Payload too large is THIS request's fault, not the slot's: don't
                # cooldown the account (that starves every later, normal-sized call —
                # narrator outage of 2026-07-16). Just never re-send this payload here.
                skip_for_size.add(acct.display_name)
                last_error = e
                logger.warning(
                    "LLM %s rejected request as too large (HTTP 413, est %d tokens), "
                    "skipping slot for this call only",
                    acct.display_name, est_tokens,
                )
                continue
            is_429 = status == 429
            # Other 4xx are deterministic (bad model/param/key) — sideline the slot so it
            # isn't re-picked this loop. 5xx stays on the soft error path (transient).
            is_hard_error = 400 <= status < 500 and not is_429
            retry_after = _parse_retry_after(e.response) if is_429 else None
            router.report_failure(
                acct,
                is_rate_limit=is_429,
                retry_after=retry_after,
                hard_error=is_hard_error,
            )
            last_error = e
            # Log the provider's own explanation on deterministic 4xx. Without it a
            # dead slot is indistinguishable from a dead key: Cerebras' 402 and
            # OpenRouter's 404 on a retired slug both read as a bare status code, and
            # diagnosing Mistral's 403 (2026-09-02) meant guessing between "no model
            # access", "no billing plan" and "bad key" from no evidence at all. Only
            # on hard errors — 429s are ordinary and their bodies would be noise —
            # and truncated, since a provider error body can carry the whole prompt.
            logger.warning(
                "LLM %s failed (HTTP %d)%s, rotating...",
                acct.display_name,
                status,
                f": {_error_detail(e.response)}" if is_hard_error else "",
            )

        except Exception as e:
            router.report_failure(acct)
            last_error = e
            logger.exception("LLM %s unexpected error", acct.display_name)

    if not attempted:
        if router.accounts and len(skip_for_size) >= len(router.accounts):
            raise LLMRequestTooLarge(
                f"Request (~{est_tokens} tokens) exceeds every account's size ceiling"
            )
        raise LLMAllThrottled("All LLM accounts on cooldown/rate-limited; no request attempted")
    if skip_for_size and isinstance(last_error, httpx.HTTPStatusError) \
            and last_error.response.status_code == 413:
        raise LLMRequestTooLarge(
            f"Request (~{est_tokens} tokens) rejected as too large by all remaining accounts"
        )
    raise RuntimeError(f"All LLM accounts exhausted. Last error: {last_error}")


def log_llm_telemetry(db_conn, result: dict, router: LLMRouter, success: bool,
                      *, purpose: str):
    """Write telemetry record after every LLM call (success or failure).

    `purpose` names the pipeline stage that spent the call and is keyword-only and
    mandatory on purpose: the column it feeds was absent until 2026-08-24, and
    without it the table answered "how many calls" but never "on what". Two thirds
    of the stages did not log at all, so the rows that did exist described only
    classification and quietly implied the prose stages were free.

    Use the stage names in PURPOSES — free-form strings would fragment the rollup.
    """
    if purpose not in PURPOSES:
        # Not fatal: a mislabelled row is worth more than a lost one, and this
        # function must never be the reason a run dies.
        logger.warning("Unknown LLM telemetry purpose %r — recording as-is", purpose)
    try:
        usage = result.get("response", {}).get("usage", {})
        db_conn.execute(
            "INSERT INTO system_telemetry(event_type, value_json) VALUES ('llm_call', %s)",
            (json.dumps({
                "purpose": purpose,
                "provider": result.get("provider", "unknown"),
                "account": result.get("account", "unknown"),
                "model": result.get("model", "unknown"),
                "tokens_used": usage.get("total_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_ms": result.get("latency_ms", 0),
                "success": success,
                "daily_used": router.total_daily_used,
                "daily_quota": router.total_daily_quota,
                "accounts": router.get_status_snapshot(),
            }),),
        )
        db_conn.commit()
    except Exception:
        logger.exception("Failed to log LLM telemetry")
