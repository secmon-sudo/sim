"""
Tests for Pass C batch classification.

One LLM call classifies a whole chunk: the ~2K-token system prompt is paid once
per call and one RPM slot covers N events. These tests cover response parsing,
per-item fallout, lock requeue on throttle, and the run_pass_c chunk loop.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import src.pipeline.pass_c_classify as pc
from src.core.llm_client import LLMAllThrottled
from src.pipeline.pass_c_classify import LLMParseError


def _event(i):
    return {
        "id": f"00000000-0000-0000-0000-00000000000{i}",
        "source_title": f"Missile strike on city {i}",
        "source_domain": "example.com",
        "canonical_text": f"Report {i}: explosion and airstrike killed several people.",
    }


def _batch_content(*reports):
    return json.dumps({"results": list(reports)})


# ── _parse_batch_response ──────────────────────────────────────────────────

def test_parse_batch_maps_by_report_number():
    content = _batch_content(
        {"report": 2, "event_type": "missile_strike"},
        {"report": 1, "event_type": "terrorism"},
    )
    items = pc._parse_batch_response(content, expected=2)
    assert items[1]["event_type"] == "terrorism"
    assert items[2]["event_type"] == "missile_strike"


def test_parse_batch_falls_back_to_position_and_bounds():
    content = _batch_content(
        {"event_type": "riot"},                       # no report number → position 1
        {"report": 99, "event_type": "out_of_range"}, # out of bounds → dropped
    )
    items = pc._parse_batch_response(content, expected=2)
    assert items[1]["event_type"] == "riot"
    assert 2 not in items and 99 not in items


def test_parse_batch_rejects_missing_results():
    with pytest.raises(LLMParseError):
        pc._parse_batch_response(json.dumps({"answers": []}), expected=2)


# ── per-object salvage ─────────────────────────────────────────────────────
# Free-tier models drop garbage tokens mid-object with finish_reason=stop; the outer
# json.loads then fails even though most report objects are intact. Losing the whole
# chunk over one bad token cost ~7% of Pass C batches (2026-08-05/06).

# Verbatim shape of the Nemotron corruption: a stray `",` after anchor_name.
_CORRUPT = '''{"results": [
  {"report": 1, "event_type": "missile_strike", "anchor_name": "Kyiv",",
   "occurred_at": "2026-08-05"},
  {"report": 2, "event_type": "drone_attack_critical_infra", "anchor_name": "Odesa"},
  {"report": 3, "event_type": "terrorism", "anchor_name": "Beirut"}
]}'''


def test_salvage_recovers_intact_objects_around_a_corrupt_one():
    items = pc._parse_batch_response(_CORRUPT, expected=3)
    assert set(items) == {2, 3}          # report 1 is the casualty, not the chunk
    assert items[2]["event_type"] == "drone_attack_critical_infra"
    assert items[3]["anchor_name"] == "Beirut"


def test_salvage_ignores_objects_without_an_explicit_report_number():
    # Position fallback is unsafe on a partial list: without "report" there is no way
    # to know which event an object describes, and a wrong guess mislabels an event.
    content = '{"results": [{"event_type": "riot"},,, {"report": 2, "event_type": "flood"}]}'
    items = pc._parse_batch_response(content, expected=2)
    assert set(items) == {2}


def test_salvage_is_not_fooled_by_braces_inside_strings():
    content = ('{"results": [{"report": 1, "summary": "shell hit {block 5} at dawn"},,'
               ' {"report": 2, "summary": "quiet"}]}')
    items = pc._parse_batch_response(content, expected=2)
    assert items[1]["summary"] == "shell hit {block 5} at dawn"
    assert set(items) == {1, 2}


def test_total_garbage_still_raises_so_the_slot_gets_penalized():
    with pytest.raises(LLMParseError):
        pc._parse_batch_response("I cannot classify these reports.", expected=3)


def test_out_of_bounds_report_numbers_dropped_during_salvage():
    content = '{"results": [{"report": 9, "event_type": "x"},,, {"report": 1, "event_type": "y"}]}'
    items = pc._parse_batch_response(content, expected=2)
    assert set(items) == {1}


# ── classify_event_batch ───────────────────────────────────────────────────

def _patch_batch(**overrides):
    defaults = dict(
        acquire_lock=MagicMock(return_value=True),
        release_lock=MagicMock(),
        deterministic_relevance=MagicMock(return_value={"score": 50, "has_high_signal": False}),
        _try_prescreen_archive=MagicMock(return_value=False),
        _apply_llm_classification=MagicMock(return_value={"event_type": "x"}),
        log_llm_telemetry=MagicMock(),
    )
    defaults.update(overrides)
    return {name: patch.object(pc, name, mock) for name, mock in defaults.items()}, defaults


def test_batch_classifies_all_events_with_one_call():
    events = [_event(1), _event(2), _event(3)]
    call = MagicMock(return_value={"content": _batch_content(
        {"report": 1, "event_type": "a"},
        {"report": 2, "event_type": "b"},
        {"report": 3, "event_type": "c"},
    )})
    patches, mocks = _patch_batch(call_llm=call)
    with patch.multiple(pc, **{n: m for n, m in mocks.items()}):
        stats = pc.classify_event_batch(MagicMock(), MagicMock(), events, "wid")
    assert stats == {"classified": 3, "failed": 0}
    assert call.call_count == 1
    prompt = call.call_args.kwargs["prompt"]
    assert "REPORT 1:" in prompt and "REPORT 3:" in prompt


def test_batch_logs_telemetry_once_per_call_not_once_per_event():
    """One LLM call covers the whole chunk, so system_telemetry must get exactly one
    llm_call row for it. Logging inside the per-event apply loop recorded the same
    call (identical latency/token counts) once per event and inflated the table ~4.7x,
    which silently corrupted every per-call metric derived from it."""
    events = [_event(1), _event(2), _event(3)]
    call = MagicMock(return_value={"content": _batch_content(
        {"report": 1, "event_type": "a"},
        {"report": 2, "event_type": "b"},
        {"report": 3, "event_type": "c"},
    )})
    patches, mocks = _patch_batch(call_llm=call)
    with patch.multiple(pc, **{n: m for n, m in mocks.items()}):
        pc.classify_event_batch(MagicMock(), MagicMock(), events, "wid")

    mocks["log_llm_telemetry"].assert_called_once()
    # ...and the apply path must be told not to log it a second time.
    for c in mocks["_apply_llm_classification"].call_args_list:
        assert c.kwargs.get("log_telemetry") is False


def test_batch_missing_item_left_queued():
    events = [_event(1), _event(2)]
    call = MagicMock(return_value={"content": _batch_content(
        {"report": 1, "event_type": "a"},
    )})
    patches, mocks = _patch_batch(call_llm=call)
    with patch.multiple(pc, **{n: m for n, m in mocks.items()}):
        stats = pc.classify_event_batch(MagicMock(), MagicMock(), events, "wid")
    assert stats == {"classified": 1, "failed": 1}
    # The missing event's lock must be released with requeue so it can retry.
    requeued = [c for c in mocks["release_lock"].call_args_list if c.kwargs.get("requeue")]
    assert len(requeued) == 1


def test_batch_throttle_requeues_and_propagates():
    events = [_event(1), _event(2)]
    call = MagicMock(side_effect=LLMAllThrottled("all slots throttled"))
    patches, mocks = _patch_batch(call_llm=call)
    with patch.multiple(pc, **{n: m for n, m in mocks.items()}):
        with pytest.raises(LLMAllThrottled):
            pc.classify_event_batch(MagicMock(), MagicMock(), events, "wid")
    requeued = [c for c in mocks["release_lock"].call_args_list if c.kwargs.get("requeue")]
    assert len(requeued) == 2


def test_batch_parse_error_leaves_events_queued():
    events = [_event(1), _event(2)]
    call = MagicMock(return_value={"content": "not json at all"})
    patches, mocks = _patch_batch(call_llm=call)
    with patch.multiple(pc, **{n: m for n, m in mocks.items()}):
        stats = pc.classify_event_batch(MagicMock(), MagicMock(), events, "wid")
    assert stats == {"classified": 0, "failed": 2, "parse_error": True}
    assert not mocks["_apply_llm_classification"].called


def test_batch_parse_error_penalizes_slot():
    # Garbage JSON must sideline the emitting slot so the next chunk rotates
    # to another cascade slot instead of re-feeding the degraded upstream.
    events = [_event(1)]
    call = MagicMock(return_value={
        "content": "not json at all", "provider": "openrouter",
        "account": "A", "model": "nvidia/nemotron-3-super-120b-a12b:free",
    })
    router = MagicMock()
    patches, mocks = _patch_batch(call_llm=call)
    with patch.multiple(pc, **{n: m for n, m in mocks.items()}):
        pc.classify_event_batch(MagicMock(), router, events, "wid")
    router.penalize_model_slot.assert_called_once_with(
        "openrouter", "A", "nvidia/nemotron-3-super-120b-a12b:free")


def test_batch_parse_error_logs_failure_telemetry():
    # Successes already log telemetry; the garbage-JSON path must log success=False
    # so a degrading :free slot's true failure rate becomes measurable (it was
    # previously invisible — only successes were recorded).
    events = [_event(1)]
    call = MagicMock(return_value={
        "content": "not json at all", "provider": "openrouter",
        "account": "A", "model": "nvidia/nemotron-3-super-120b-a12b:free",
    })
    patches, mocks = _patch_batch(call_llm=call)
    with patch.multiple(pc, **{n: m for n, m in mocks.items()}):
        pc.classify_event_batch(MagicMock(), MagicMock(), events, "wid")
    tel = mocks["log_llm_telemetry"]
    tel.assert_called_once()
    assert tel.call_args.kwargs.get("success") is False


def test_validate_and_parse_tolerates_control_chars_in_strings():
    # Raw newline inside a quoted value (seen from degraded :free upstreams)
    # must parse instead of failing with "Invalid control character".
    raw = '{"results": [{"report": 1, "event_type": "riot", "note": "line one\nline two"}]}'
    parsed = pc.validate_and_parse(raw)
    assert parsed["results"][0]["event_type"] == "riot"


def test_batch_prescreen_skips_llm_call():
    events = [_event(1)]
    call = MagicMock()
    patches, mocks = _patch_batch(
        call_llm=call,
        _try_prescreen_archive=MagicMock(return_value=True),
    )
    with patch.multiple(pc, **{n: m for n, m in mocks.items()}):
        stats = pc.classify_event_batch(MagicMock(), MagicMock(), events, "wid")
    assert stats == {"classified": 1, "failed": 0}
    assert not call.called


# ── run_pass_c chunking ────────────────────────────────────────────────────

def test_run_pass_c_chunks_events_through_batches():
    events = [_event(i) for i in range(1, 8)]  # 7 events, batch size 3 → 3 chunks
    seen_chunks = []

    def fake_batch(db, router, chunk, worker_id):
        seen_chunks.append(len(chunk))
        return {"classified": len(chunk), "failed": 0}

    router = MagicMock()
    db = MagicMock()
    with patch.object(pc, "BATCH_CLASSIFY_SIZE", 3), \
         patch.object(pc, "get_events_for_classification", return_value=events), \
         patch.object(pc, "classify_event_batch", side_effect=fake_batch):
        stats = pc.run_pass_c(db, router, limit=50)

    assert seen_chunks == [3, 3, 1]
    assert stats["events_classified"] == 7
    assert stats["llm_exhausted"] is False


def test_run_pass_c_aborts_after_consecutive_parse_errors():
    # 30 events / batch size 3 → 10 chunks, but every batch fails to parse:
    # the pass must stop after PASS_C_MAX_CONSECUTIVE_PARSE_ERRORS chunks
    # instead of grinding until the workflow timeout.
    events = [_event(i) for i in range(1, 31)]
    calls = []

    def fake_batch(db, router, chunk, worker_id):
        calls.append(len(chunk))
        return {"classified": 0, "failed": len(chunk), "parse_error": True}

    with patch.object(pc, "BATCH_CLASSIFY_SIZE", 3), \
         patch.object(pc, "get_events_for_classification", return_value=events), \
         patch.object(pc, "classify_event_batch", side_effect=fake_batch):
        stats = pc.run_pass_c(MagicMock(), MagicMock(), limit=50)

    assert len(calls) == pc.PASS_C_MAX_CONSECUTIVE_PARSE_ERRORS
    assert stats["aborted_on_parse_errors"] is True


def test_run_pass_c_parse_error_counter_resets_on_success():
    # parse-fail, success, parse-fail, ... never reaches the consecutive
    # threshold, so all chunks are attempted.
    events = [_event(i) for i in range(1, 31)]  # 10 chunks of 3
    outcomes = []

    def fake_batch(db, router, chunk, worker_id):
        if len(outcomes) % 2 == 0:
            result = {"classified": 0, "failed": len(chunk), "parse_error": True}
        else:
            result = {"classified": len(chunk), "failed": 0}
        outcomes.append(result)
        return result

    with patch.object(pc, "BATCH_CLASSIFY_SIZE", 3), \
         patch.object(pc, "get_events_for_classification", return_value=events), \
         patch.object(pc, "classify_event_batch", side_effect=fake_batch):
        stats = pc.run_pass_c(MagicMock(), MagicMock(), limit=50)

    assert len(outcomes) == 10
    assert "aborted_on_parse_errors" not in stats


# ── geographically-scoped event type ────────────────────────────────────────

def test_is_african_covers_the_corpus_countries():
    from src.core.geo import is_african

    # Countries that legitimately produced african_terrorism over 14 days.
    for iso in ("NG", "NE", "CD", "KE", "SO"):
        assert is_african(iso) is True
    # The ones that produced it wrongly.
    for iso in ("PK", "IN"):
        assert is_african(iso) is False
    # An unresolved country must not keep a geographic label by default.
    assert is_african(None) is False
    assert is_african("") is False
    assert is_african("ke") is True   # case-insensitive


def test_african_terrorism_outside_africa_is_demoted_to_terrorism():
    """A Balochistan counter-terrorism operation was filed as `african_terrorism`
    (6 Pakistani + 2 Indian events over 14 days). The prompt already scopes the type
    to Africa; free-tier models ignore that, so the guard is deterministic."""
    from src.pipeline.pass_c_classify import GEO_SCOPED_EVENT_TYPE, GEO_SCOPED_FALLBACK

    captured = {}

    class _DB:
        def execute(self, sql, params=None):
            r = MagicMock()
            if sql.strip().upper().startswith("UPDATE"):
                captured["params"] = params
            else:
                # every catalog lookup succeeds
                r.fetchone.return_value = (params[0],) if params else ("x",)
            return r

        def commit(self):
            pass

        def transaction(self):
            from contextlib import nullcontext
            return nullcontext()

    parsed = {
        "event_type": GEO_SCOPED_EVENT_TYPE,
        "relevance": 90,
        "confidence": 0.8,
        "country_iso": "PK",
        "anchor_name": "Mastung",
        "time_certainty": "same_day",
    }
    with patch.object(pc, "update_domain_penalty", MagicMock()):
        pc._apply_llm_classification(
            _DB(), MagicMock(),
            {"id": "e" * 32, "source_domain": "dawn.com"},
            {"score": 60, "has_high_signal": True},
            parsed, {"provider": "p", "model": "m", "response": {}}, "wid",
            log_telemetry=False,
        )

    assert parsed["event_type"] == GEO_SCOPED_FALLBACK
    assert GEO_SCOPED_EVENT_TYPE not in captured["params"]
