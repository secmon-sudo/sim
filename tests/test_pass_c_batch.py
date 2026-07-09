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
    assert stats == {"classified": 0, "failed": 2}
    assert not mocks["_apply_llm_classification"].called


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
