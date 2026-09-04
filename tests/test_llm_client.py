"""call_llm response-handling tests — in-200 failure rotation.

OpenRouter free endpoints can fail INSIDE an HTTP 200: the body carries an
"error" object or an empty completion when the upstream provider chokes
(observed 2026-07-10: instant 200s with blank content and blank finish_reason
from nemotron-3-super:free). call_llm must treat those as provider failures
and rotate to the next cascade slot, not return them as success.
"""
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.core import llm_client, llm_router, model_profiles
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


# ── Caller acceptance check (SITREP citation contract, 2026-09-04) ──
#
# On 2026-09-04 minimax-m2.7 narrated all five country SITREPs after Mistral 429'd
# every call, and shortened every one of its 108 citation URLs to a bare domain.
# The transport layer saw five perfectly healthy HTTP 200s. `accept` is how a
# caller says what "healthy" means for its own work.


def test_call_llm_rotates_when_the_caller_rejects_the_content():
    router = LLMRouter([
        _acct("llm7-ish", provider="llm7"),
        _acct("openai/gpt-oss-20b:free"),
    ])
    bad = _resp({"choices": [{"message": {"content": "no citations here"},
                              "finish_reason": "stop"}]})
    with patch.object(llm_client, "_send_request", side_effect=[bad, _resp(_GOOD)]):
        result = llm_client.call_llm(router, "prompt",
                                     accept=lambda text: "{" in text)
    assert result["model"] == "openai/gpt-oss-20b:free"
    # Sidelined, not merely skipped: a model that cannot honour the contract for
    # one country will not honour it for the next four either.
    assert router.accounts[0].status == ProviderStatus.RATE_LIMITED


def test_call_llm_without_an_accept_check_is_unchanged():
    router = LLMRouter([_acct("m1")])
    with patch.object(llm_client, "_send_request", return_value=_resp(_GOOD)):
        assert llm_client.call_llm(router, "prompt")["content"] == '{"ok": 1}'


def test_call_llm_raises_when_every_slot_is_rejected():
    router = LLMRouter([_acct("m1"), _acct("m2", account_id="B")])
    bad = _resp({"choices": [{"message": {"content": "nope"},
                              "finish_reason": "stop"}]})
    with patch.object(llm_client, "_send_request", side_effect=[bad, bad]):
        with pytest.raises(RuntimeError):
            llm_client.call_llm(router, "prompt", accept=lambda text: False)


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


# ── Wall-clock ceiling (2026-08-12) ────────────────────────────────────────
# httpx's timeout is per read, and OpenRouter drips keepalive bytes while an upstream
# generates, so a 30s read timeout never fired on 44-108s nemotron batches. Pass C is
# sequential: every one of those seconds was run duration. The ceiling below bounds
# slowness itself, and hands the work to the next cascade slot.

def _slow_post(delay, sent=None):
    """A POST that takes `delay` seconds, recording the accounts it was called for."""
    def fake_post(url, headers=None, json=None, timeout=None):
        if sent is not None:
            sent.append(json)
        time.sleep(delay)
        resp = MagicMock()
        resp.status_code = 200
        resp.is_success = True
        resp.raise_for_status.return_value = None
        resp.json.return_value = _GOOD
        return resp
    return fake_post


@pytest.fixture
def _tiny_ceiling():
    """Shrink the ceiling so tests measure the mechanism, not the wait."""
    with patch.object(model_profiles, "DEFAULT_WALL_CLOCK_SECONDS", 0.05):
        yield


def test_slow_slot_is_sidelined_and_call_rotates(_tiny_ceiling):
    router = LLMRouter([
        _acct("nvidia/nemotron-3-super-120b-a12b:free"),
        _acct("openai/gpt-oss-20b:free", account_id="B"),
    ])
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(headers["Authorization"])
        if len(calls) == 1:
            time.sleep(0.5)          # first slot outruns the ceiling
        resp = MagicMock()
        resp.status_code = 200
        resp.is_success = True
        resp.raise_for_status.return_value = None
        resp.json.return_value = _GOOD
        return resp

    with patch.object(llm_client.httpx, "post", side_effect=fake_post):
        result = llm_client.call_llm(router, "prompt")

    assert result["model"] == "openai/gpt-oss-20b:free"
    # Sidelined for the rest of the run, not merely error-counted: re-probing a slow
    # slot costs another full ceiling, so it must outlast the 120s client-error cooldown.
    assert router.accounts[0].status == ProviderStatus.RATE_LIMITED
    assert router.accounts[0].cooldown_until - time.monotonic() > llm_router.CLIENT_ERROR_COOLDOWN_SECONDS


def test_ceiling_does_not_retry_the_slow_slot(_tiny_ceiling):
    # tenacity retries httpx timeouts; re-sending to a slot that is already too slow
    # would multiply the very wait the ceiling exists to cut, so the exception must
    # stay outside that retry class — one attempt per slot, then rotate.
    router = LLMRouter([_acct("nvidia/nemotron-3-super-120b-a12b:free")])
    sent = []
    with patch.object(llm_client.httpx, "post", side_effect=_slow_post(0.5, sent)):
        with pytest.raises(RuntimeError, match="exhausted"):
            llm_client.call_llm(router, "prompt")
    assert len(sent) == 1


def test_fast_response_is_untouched_by_the_ceiling(_tiny_ceiling):
    router = LLMRouter([_acct("nvidia/nemotron-3-super-120b-a12b:free")])
    with patch.object(llm_client.httpx, "post", side_effect=_slow_post(0)):
        result = llm_client.call_llm(router, "prompt")
    assert result["content"] == '{"ok": 1}'
    assert router.accounts[0].status == ProviderStatus.ACTIVE


def test_transport_errors_still_reach_the_caller(_tiny_ceiling):
    # The ceiling runs the request on another thread; an exception raised there must
    # surface on the calling thread, not be swallowed into a timeout.
    acct = _acct("openai/gpt-oss-120b", provider="groq")
    fake_post, _ = _post_stub(400)
    with patch.object(llm_client.httpx, "post", side_effect=fake_post):
        with pytest.raises(httpx.HTTPStatusError):
            llm_client._send_request(acct, [{"role": "user", "content": "hi"}],
                                     json_mode=False)


def test_prose_slot_keeps_a_ceiling_long_enough_to_write():
    # mistral-large spends minutes on a 6K-token SITREP by design; policing it with the
    # classification ceiling would sideline the only slot that writes the prose well.
    mistral = model_profiles.get_profile("mistral", "mistral-large-2512")
    groq = model_profiles.get_profile("groq", "openai/gpt-oss-120b")
    assert model_profiles.wall_clock_ceiling(mistral, 6000) \
        > model_profiles.wall_clock_ceiling(groq, 2312)


def test_ceiling_scales_with_the_completion_budget():
    # The SITREP narrative (6K tokens) can fall through to the main cascade when the
    # quality slots are down. A flat classification ceiling would sideline every slot
    # able to write it — the Groq ones are already skipped at that size — and the
    # narrative would fail outright rather than merely run slow.
    profile = model_profiles.get_profile("openrouter", "nvidia/nemotron-3-super-120b-a12b:free")
    batch = model_profiles.wall_clock_ceiling(profile, 450 * 4 + 512)
    narrative = model_profiles.wall_clock_ceiling(profile, 6000)
    assert batch == model_profiles.DEFAULT_WALL_CLOCK_SECONDS
    assert narrative > 2 * batch


class TestReadTimeoutIsNotRetried:
    """A read timeout must not be re-paid; a failed connection may be retried.

    Measured 2026-08-27 on Daily Country SITREP #48: six read timeouts at ~180s each
    (mistral-large's request_timeout) inside a 33.8-minute run whose normal duration is
    8-12 minutes. Run #47 the day before died on the workflow's 35-minute timeout.
    """

    def _retry_state(self, exc):
        class _Outcome:
            def exception(self_inner):
                return exc
        class _State:
            outcome = _Outcome()
            attempt_number = 1
        return _State()

    def test_read_timeout_is_not_retried(self):
        from src.core.llm_client import _worth_retrying
        assert _worth_retrying(self._retry_state(httpx.ReadTimeout("timed out"))) is False

    def test_write_timeout_is_not_retried(self):
        from src.core.llm_client import _worth_retrying
        assert _worth_retrying(self._retry_state(httpx.WriteTimeout("timed out"))) is False

    def test_connection_failures_are_still_retried(self):
        from src.core.llm_client import _worth_retrying
        for exc in (httpx.ConnectError("refused"),
                    httpx.ConnectTimeout("timed out"),
                    httpx.PoolTimeout("no slot")):
            assert _worth_retrying(self._retry_state(exc)) is True, type(exc).__name__

    def test_unrelated_exception_is_not_retried(self):
        from src.core.llm_client import _worth_retrying
        assert _worth_retrying(self._retry_state(ValueError("nope"))) is False


class TestErrorDetail:
    """A dead slot must say WHY. Cerebras' 402, OpenRouter's 404 on a retired slug
    and Mistral's 403 were indistinguishable bare status codes in the run log
    (2026-09-02), which is what made diagnosing them guesswork.
    """

    def _resp(self, payload=None, text=None):
        r = MagicMock(spec=httpx.Response)
        if payload is None:
            r.json.side_effect = ValueError("not json")
            r.text = text or ""
        else:
            r.json.return_value = payload
            r.text = text or ""
        return r

    def test_openai_compatible_nested_message(self):
        detail = llm_client._error_detail(
            self._resp({"error": {"message": "Model not found or no access to it",
                                  "type": "invalid_request_error"}})
        )
        assert detail == "Model not found or no access to it"

    def test_bare_message_field(self):
        assert llm_client._error_detail(self._resp({"message": "Insufficient credits"})) \
            == "Insufficient credits"

    def test_error_as_plain_string(self):
        assert llm_client._error_detail(self._resp({"error": "quota exceeded"})) \
            == "quota exceeded"

    def test_falls_back_to_raw_body_when_not_json(self):
        assert "Forbidden" in llm_client._error_detail(
            self._resp(payload=None, text="<html>403 Forbidden</html>")
        )

    def test_collapses_whitespace_and_truncates(self):
        detail = llm_client._error_detail(
            self._resp({"error": {"message": "x " * 400}})
        )
        assert len(detail) <= llm_client.ERROR_DETAIL_MAX_CHARS
        assert "\n" not in detail

    def test_empty_body_is_labelled_not_blank(self):
        assert llm_client._error_detail(self._resp(payload=None, text="")) == "<empty body>"

    def test_never_raises(self):
        broken = MagicMock(spec=httpx.Response)
        broken.json.side_effect = RuntimeError("boom")
        type(broken).text = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        try:
            llm_client._error_detail(broken)
        except Exception as exc:  # pragma: no cover - the point is that this never runs
            pytest.fail(f"_error_detail raised on a broken response: {exc}")
