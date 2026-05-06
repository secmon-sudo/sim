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

logger = logging.getLogger(__name__)

PROVIDER_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


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
def _send_request(acct: LLMAccount, messages: list[dict], max_tokens: int = 1024) -> httpx.Response:
    """Single request to a specific account. Retries on connection errors only."""
    headers = {
        "Authorization": f"Bearer {acct.api_key}",
        "Content-Type": "application/json",
    }
    if acct.provider == "openrouter":
        headers["HTTP-Referer"] = "https://sim-osint.app"
        headers["X-Title"] = "SIM-OSINT-Pipeline"

    response = httpx.post(
        PROVIDER_ENDPOINTS[acct.provider],
        headers=headers,
        json={
            "model": acct.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response


def call_llm(router: LLMRouter, prompt: str, system_prompt: str | None = None, max_tokens: int = 1024) -> dict[str, Any]:
    """
    Try all available accounts in priority order.

    Returns dict with keys:
        - response: parsed JSON from LLM
        - provider: "groq" | "openrouter"
        - account: "A" | "B"
        - model: model ID string
        - latency_ms: int
        - content: extracted text content

    Raises RuntimeError if all accounts are exhausted.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_error = None

    for _ in range(len(router.accounts)):
        acct = router.get_available_account()
        if acct is None:
            break

        try:
            t0 = time.monotonic()
            resp = _send_request(acct, messages, max_tokens=max_tokens)
            latency_ms = int((time.monotonic() - t0) * 1000)

            data = resp.json()
            content = ""
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content", "")

            router.report_success(acct)
            return {
                "response": data,
                "provider": acct.provider,
                "account": acct.account_id,
                "model": acct.model,
                "latency_ms": latency_ms,
                "content": content,
            }

        except httpx.HTTPStatusError as e:
            is_429 = e.response.status_code == 429
            router.report_failure(acct, is_rate_limit=is_429)
            last_error = e
            logger.warning(
                "LLM %s failed (HTTP %d), rotating...",
                acct.display_name,
                e.response.status_code,
            )

        except Exception as e:
            router.report_failure(acct)
            last_error = e
            logger.exception("LLM %s unexpected error", acct.display_name)

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
