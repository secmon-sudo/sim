"""
Tests for Pass C throttle pacing.

TPM is far tighter than RPM on the free tier, so a backlog would otherwise stop after a
handful of events. run_pass_c should wait for the soonest token-window refill and retry
when every slot is momentarily throttled, but still abort promptly on a genuine outage.
"""

from unittest.mock import MagicMock, patch

import src.pipeline.pass_c_classify as pc


def _run(router, classify_side_effect, events=("e1", "e2", "e3")):
    with patch.object(pc, "get_events_for_classification", return_value=list(events)), \
         patch.object(pc, "classify_single_event", side_effect=classify_side_effect), \
         patch.object(pc, "log_llm_telemetry", lambda *a, **k: None), \
         patch("time.sleep", lambda s: None):
        return pc.run_pass_c(MagicMock(), router, limit=50)


def test_paces_through_transient_throttle():
    """A transient throttle on each event should be retried, not abort the whole pass."""
    calls = {"n": 0}

    def classify(db, router, event, worker_id):
        calls["n"] += 1
        if calls["n"] % 2 == 1:  # first attempt of each event is throttled
            raise RuntimeError("All LLM accounts on cooldown/rate-limited")
        return {"event_type": "x"}

    router = MagicMock()
    router.seconds_until_available.return_value = 0.0  # a slot will be ready shortly
    router.get_status_snapshot.return_value = {}

    stats = _run(router, classify)
    assert stats["events_classified"] == 3
    assert stats["llm_exhausted"] is False


def test_aborts_on_genuine_outage():
    """When no account can recover today, the pass must stop immediately."""
    router = MagicMock()
    router.seconds_until_available.return_value = None  # nothing recovers today
    router.get_status_snapshot.return_value = {}

    stats = _run(router, RuntimeError("exhausted"))
    assert stats["events_classified"] == 0
    assert stats["llm_exhausted"] is True


def test_aborts_when_wait_exceeds_cap():
    """A refill wait longer than the per-wait cap should abort rather than stall the run."""
    router = MagicMock()
    router.seconds_until_available.return_value = pc.PASS_C_PACING_MAX_WAIT + 1
    router.get_status_snapshot.return_value = {}

    stats = _run(router, RuntimeError("exhausted"))
    assert stats["events_classified"] == 0
    assert stats["llm_exhausted"] is True
