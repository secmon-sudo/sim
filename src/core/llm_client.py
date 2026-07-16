"""
SIM — Unified LLM Call Wrapper
Blueprint V20.1 §4.5.6 + §4.5.8

Sends classification requests to the first available LLM provider.
Handles retries, failover, and telemetry logging.
"""

import json
import logging
import time
from typing import Any

import httpx
import tenacity

from src.core.llm_router import LLMAccount, LLMRouter
from src.core.model_profiles import get_profile

logger = logging.getLogger(__name__)

PROVIDER_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    # Google AI Studio's OpenAI-compatibility layer — same chat/completions shape.
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}


class LLMAllThrottled(RuntimeError):
    """Every account is on cooldown/rate-limited; no request was even attempted.

    Expected flow under free-tier TPM pacing (per-minute token windows drained) —
    callers like run_pass_c wait for the soonest refill and retry. Distinct from
    the generic "exhausted after real attempts" RuntimeError, which signals actual
    request failures. Subclasses RuntimeError so existing catchers keep working.
    """


class LLMRequestTooLarge(RuntimeError):
    """THIS request exceeds every account's per-request size ceiling.

    A fault of the request, not of the accounts: retrying the same payload can
    never succeed, so callers must drop/shrink the item and move on — waiting
    (LLMAllThrottled) or aborting the whole stage (generic RuntimeError) are both
    wrong responses. Subclasses RuntimeError so unaware catchers stay safe.
    """


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


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(
        (httpx.ConnectError, httpx.TimeoutException)
    ),
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
    if profile.supports_json_mode and json_mode:
        payload["response_format"] = {"type": "json_object"}
    payload.update(profile.payload_extras)

    response = httpx.post(
        PROVIDER_ENDPOINTS[acct.provider],
        headers=headers,
        json=payload,
        timeout=30.0,
    )
    response.raise_for_status()
    return response


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
            logger.warning(
                "LLM %s failed (HTTP %d), rotating...",
                acct.display_name,
                status,
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


def log_llm_telemetry(db_conn, result: dict, router: LLMRouter, success: bool):
    """Write telemetry record after every LLM call (success or failure)."""
    try:
        usage = result.get("response", {}).get("usage", {})
        db_conn.execute(
            "INSERT INTO system_telemetry(event_type, value_json) VALUES ('llm_call', %s)",
            (json.dumps({
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
