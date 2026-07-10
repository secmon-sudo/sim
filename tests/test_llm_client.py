"""call_llm response-handling tests — in-200 failure rotation.

OpenRouter free endpoints can fail INSIDE an HTTP 200: the body carries an
"error" object or an empty completion when the upstream provider chokes
(observed 2026-07-10: instant 200s with blank content and blank finish_reason
from nemotron-3-super:free). call_llm must treat those as provider failures
and rotate to the next cascade slot, not return them as success.
"""
from unittest.mock import MagicMock, patch

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
        _acct("openai/gpt-oss-120b:free"),
    ])
    empty = _resp({"choices": [{"message": {"content": ""}, "finish_reason": None}]})
    with patch.object(llm_client, "_send_request", side_effect=[empty, _resp(_GOOD)]):
        result = llm_client.call_llm(router, "prompt")
    assert result["content"] == '{"ok": 1}'
    assert result["model"] == "openai/gpt-oss-120b:free"
    # The flaky slot must be sidelined so it isn't re-picked immediately.
    assert router.accounts[0].status == ProviderStatus.RATE_LIMITED


def test_call_llm_rotates_on_error_body_200():
    router = LLMRouter([
        _acct("nvidia/nemotron-3-super-120b-a12b:free"),
        _acct("openai/gpt-oss-120b:free"),
    ])
    err = _resp({"error": {"code": 502, "message": "upstream failure"}, "choices": []})
    with patch.object(llm_client, "_send_request", side_effect=[err, _resp(_GOOD)]):
        result = llm_client.call_llm(router, "prompt")
    assert result["content"] == '{"ok": 1}'
    assert result["model"] == "openai/gpt-oss-120b:free"


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
