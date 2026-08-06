"""call_llm response-handling tests — in-200 failure rotation.

OpenRouter free endpoints can fail INSIDE an HTTP 200: the body carries an
"error" object or an empty completion when the upstream provider chokes
(observed 2026-07-10: instant 200s with blank content and blank finish_reason
from nemotron-3-super:free). call_llm must treat those as provider failures
and rotate to the next cascade slot, not return them as success.
"""
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.core import llm_client
from src.core.llm_router import LLMAccount, LLMRouter, ProviderStatus
from src.core.token_bucket import TokenBucket


def _acct(model, provider="openrouter", account_id="A"):
    return LLMAccount(
        provider=provider, account_id=account_id, model=model, api_key="k",
        rpm=60, rpd=1000,
        bucket=TokenBucket(rate_per_minute=60, daily_limit=1000, burst=8),
    )


def _resp(payload):
    r = MagicMock()
    r.json.return_value = payload
    return r


_GOOD = {"choices": [{"message": {"content": '{"ok": 1}'}, "finish_reason": "stop"}]}


def test_call_llm_rotates_on_empty_200():
    router = LLMRouter([
        _acct("nvidia/nemotron-3-super-120b-a12b:free"),
        _acct("openai/gpt-oss-20b:free"),
    ])
    empty = _resp({"choices": [{"message": {"content": ""}, "finish_reason": None}]})
    with patch.object(llm_client, "_send_request", side_effect=[empty, _resp(_GOOD)]):
        result = llm_client.call_llm(router, "prompt")
    assert result["content"] == '{"ok": 1}'
    assert result["model"] == "openai/gpt-oss-20b:free"
    # The flaky slot must be sidelined so it isn't re-picked immediately.
    assert router.accounts[0].status == ProviderStatus.RATE_LIMITED


def test_call_llm_rotates_on_error_body_200():
    router = LLMRouter([
        _acct("nvidia/nemotron-3-super-120b-a12b:free"),
        _acct("openai/gpt-oss-20b:free"),
    ])
    err = _resp({"error": {"code": 502, "message": "upstream failure"}, "choices": []})
    with patch.object(llm_client, "_send_request", side_effect=[err, _resp(_GOOD)]):
        result = llm_client.call_llm(router, "prompt")
    assert result["content"] == '{"ok": 1}'
    assert result["model"] == "openai/gpt-oss-20b:free"


def test_call_llm_all_empty_raises_runtime_error():
    router = LLMRouter([_acct("m1"), _acct("m2", account_id="B")])
    empty = _resp({"choices": []})
    with patch.object(llm_client, "_send_request", side_effect=[empty, empty]):
        with pytest.raises(RuntimeError, match="exhausted"):
            llm_client.call_llm(router, "prompt")


def test_call_llm_good_response_passes_through():
    router = LLMRouter([_acct("nvidia/nemotron-3-super-120b-a12b:free")])
    with patch.object(llm_client, "_send_request", return_value=_resp(_GOOD)):
        result = llm_client.call_llm(router, "prompt")
    assert result["content"] == '{"ok": 1}'
    assert result["finish_reason"] == "stop"
    assert router.accounts[0].status == ProviderStatus.ACTIVE


# ── Request-size guard + 413 taxonomy (Groq narrator outage, 2026-07-16) ──

_OVERSIZED = "x" * 40_000  # ~10K tokens at 4 chars/token — above Groq's 8K ceiling


def _http_error(status):
    import httpx
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.headers = {}
    return httpx.HTTPStatusError("err", request=MagicMock(), response=resp)


def test_oversized_request_skips_groq_without_sending_or_cooldown():
    router = LLMRouter([_acct("openai/gpt-oss-20b", provider="groq")])
    with patch.object(llm_client, "_send_request") as send:
        with pytest.raises(llm_client.LLMRequestTooLarge):
            llm_client.call_llm(router, _OVERSIZED)
    send.assert_not_called()
    # The account must stay healthy — the request was the problem.
    assert router.accounts[0].status == ProviderStatus.ACTIVE
    assert router.accounts[0].cooldown_until == 0.0


def test_oversized_request_falls_through_to_unlimited_provider():
    router = LLMRouter([
        _acct("openai/gpt-oss-20b", provider="groq"),
        _acct("openai/gpt-oss-20b:free", provider="openrouter"),
    ])
    with patch.object(llm_client, "_send_request", return_value=_resp(_GOOD)) as send:
        result = llm_client.call_llm(router, _OVERSIZED)
    assert result["model"] == "openai/gpt-oss-20b:free"
    assert send.call_count == 1  # Groq slot never attempted


def test_http_413_rotates_without_cooldown():
    router = LLMRouter([
        _acct("openai/gpt-oss-20b", provider="groq", account_id="A"),
        _acct("openai/gpt-oss-20b", provider="groq", account_id="B"),
    ])
    # Estimate says it fits, provider disagrees: rotate, but do NOT sideline the slot.
    with patch.object(llm_client, "_send_request",
                      side_effect=[_http_error(413), _resp(_GOOD)]):
        result = llm_client.call_llm(router, "prompt")
    assert result["account"] == "B"
    assert router.accounts[0].status == ProviderStatus.ACTIVE
    assert router.accounts[0].cooldown_until == 0.0


def test_all_413_raises_request_too_large():
    router = LLMRouter([
        _acct("openai/gpt-oss-20b", provider="groq", account_id="A"),
        _acct("openai/gpt-oss-20b", provider="groq", account_id="B"),
    ])
    with patch.object(llm_client, "_send_request",
                      side_effect=[_http_error(413), _http_error(413)]):
        with pytest.raises(llm_client.LLMRequestTooLarge):
            llm_client.call_llm(router, "prompt")
    # Both slots stay in rotation for the NEXT (normal-sized) call.
    assert all(a.status == ProviderStatus.ACTIVE for a in router.accounts)


def test_send_request_payload_uses_model_profile():
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json)
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        return resp

    with patch.object(llm_client.httpx, "post", side_effect=fake_post):
        llm_client._send_request(
            _acct("openai/gpt-oss-120b", provider="groq"),
            [{"role": "user", "content": "hi"}],
        )
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["reasoning_effort"] == "low"


# ── json-mode self-heal ────────────────────────────────────────────────────
# A profile only CLAIMS response_format support; providers have revoked it before
# (OpenRouter free, 2026-07-08). Without the probe below a revocation would 4xx every
# call and drop the slot — which for Nemotron means losing 73% of Pass C capacity.

def _post_stub(*statuses):
    """Sequence of fake httpx responses, capturing each payload sent."""
    sent = []

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        resp = MagicMock()
        status = statuses[len(sent) - 1]
        resp.status_code = status
        resp.is_success = 200 <= status < 300
        if status >= 400:
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"HTTP {status}", request=MagicMock(), response=resp)
        else:
            resp.raise_for_status.return_value = None
        return resp

    return fake_post, sent


@pytest.fixture(autouse=True)
def _clean_json_mode_state():
    llm_client.reset_json_mode_sidelines()
    yield
    llm_client.reset_json_mode_sidelines()


def test_json_mode_400_retries_bare_and_sidelines_model():
    acct = _acct("nvidia/nemotron-3-super-120b-a12b:free")
    fake_post, sent = _post_stub(400, 200)
    with patch.object(llm_client.httpx, "post", side_effect=fake_post):
        llm_client._send_request(acct, [{"role": "user", "content": "hi"}])
    assert "response_format" in sent[0]
    assert "response_format" not in sent[1]   # the probe proves the culprit
    assert "reasoning" in sent[1]             # other profile extras survive the retry
    assert llm_client._json_mode_sidelined(acct)


def test_sidelined_model_skips_json_mode_on_later_calls():
    acct = _acct("nvidia/nemotron-3-super-120b-a12b:free")
    fake_post, sent = _post_stub(400, 200, 200)
    with patch.object(llm_client.httpx, "post", side_effect=fake_post):
        llm_client._send_request(acct, [{"role": "user", "content": "hi"}])
        # Same model on the OTHER key: the capability belongs to the model, so one
        # slot's discovery must spare its twin an extra 400.
        llm_client._send_request(
            _acct("nvidia/nemotron-3-super-120b-a12b:free", account_id="B"),
            [{"role": "user", "content": "hi"}],
        )
    assert len(sent) == 3
    assert "response_format" not in sent[2]


def test_non_json_mode_400_propagates_without_sidelining():
    # A 400 that persists without response_format is about something else (bad model
    # id, bad param) — json mode must not take the blame, and the slot's normal
    # hard-error path must still fire.
    acct = _acct("nvidia/nemotron-3-super-120b-a12b:free")
    fake_post, _ = _post_stub(400, 400)
    with patch.object(llm_client.httpx, "post", side_effect=fake_post):
        with pytest.raises(httpx.HTTPStatusError):
            llm_client._send_request(acct, [{"role": "user", "content": "hi"}])
    assert not llm_client._json_mode_sidelined(acct)


def test_json_mode_probe_not_triggered_for_prose_callers():
    # json_mode=False callers never sent response_format, so a 400 is never its fault.
    acct = _acct("nvidia/nemotron-3-super-120b-a12b:free")
    fake_post, sent = _post_stub(400)
    with patch.object(llm_client.httpx, "post", side_effect=fake_post):
        with pytest.raises(httpx.HTTPStatusError):
            llm_client._send_request(acct, [{"role": "user", "content": "hi"}], json_mode=False)
    assert len(sent) == 1
    assert not llm_client._json_mode_sidelined(acct)
