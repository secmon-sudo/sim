"""Tests for the dispatch-time duplicate-page adjudicator (suppression layer 3).

The two suppression keys collapse duplicates only when two reports of one incident agree
on a machine identity — a storyline_id, or a normalized location string. Measured
2026-08-11 (runs 1453/1456/1458) neither survives paraphrase: one Colombian earthquake
paged four times under four storyline_ids, twice inside a single run. This layer asks the
model the question the keys were standing in for, and these tests pin both halves of its
contract: it must mute genuine repeats, and it must fail toward SENDING.
"""

from unittest.mock import MagicMock

import pytest

import src.pipeline.pass_d_score as pd
from src.core.storyline_adjudicator import adjudicate_duplicate_page


def _reply(content: str):
    """A call_llm stand-in returning a canned model reply."""
    return lambda *a, **kw: {"content": content}


_QUAKE = {
    "source_title": "7.4M earthquake hits western Colombia: at least 110 killed",
    "storyline_hint": "colombia earthquake casualties",
    "anchor_name_raw": "western Colombia",
}
_PAGED = [
    {"id": "e1", "alert_tier": "ALERT",
     "source_title": "A 7.4-magnitude earthquake shakes western Colombia",
     "storyline_hint": "colombia quake", "anchor_name_raw": "San Jose Del Palmar, Chocó"},
    {"id": "e2", "alert_tier": "WATCH",
     "source_title": "Colombia car bomb tests new president's security crackdown",
     "storyline_hint": "colombia car bomb", "anchor_name_raw": None},
]


class TestAdjudicateDuplicatePage:
    def test_returns_matched_card(self):
        match = adjudicate_duplicate_page(
            _QUAKE, _PAGED, MagicMock(), call_llm_fn=_reply('{"match": 1}'))
        assert match["id"] == "e1"
        assert match["alert_tier"] == "ALERT"

    def test_new_verdict_returns_none(self):
        assert adjudicate_duplicate_page(
            _QUAKE, _PAGED, MagicMock(), call_llm_fn=_reply('{"match": "NEW"}')) is None

    def test_no_candidates_skips_the_llm(self):
        called = MagicMock()
        assert adjudicate_duplicate_page(
            _QUAKE, [], MagicMock(), call_llm_fn=called) is None
        called.assert_not_called()

    def test_llm_error_fails_safe_to_sending(self):
        def boom(*a, **kw):
            raise RuntimeError("router exhausted")

        assert adjudicate_duplicate_page(
            _QUAKE, _PAGED, MagicMock(), call_llm_fn=boom) is None

    def test_unparseable_reply_fails_safe_to_sending(self):
        assert adjudicate_duplicate_page(
            _QUAKE, _PAGED, MagicMock(), call_llm_fn=_reply("who knows")) is None

    def test_out_of_range_index_fails_safe_to_sending(self):
        assert adjudicate_duplicate_page(
            _QUAKE, _PAGED, MagicMock(), call_llm_fn=_reply('{"match": 9}')) is None

    def test_prompt_carries_every_candidate(self):
        seen = {}

        def capture(router, prompt, **kw):
            seen["prompt"] = prompt
            return {"content": '{"match": "NEW"}'}

        adjudicate_duplicate_page(_QUAKE, _PAGED, MagicMock(), call_llm_fn=capture)
        assert "[1]" in seen["prompt"] and "[2]" in seen["prompt"]
        assert "car bomb" in seen["prompt"]
        assert _QUAKE["source_title"][:40] in seen["prompt"]


def _base_event(**over):
    ev = {
        "severity_score": 90,
        "alert_tier": "ALERT",
        "storyline_id": "S1",
        "anchor_name_raw": "Kyiv",
        "country_iso": "UA",
    }
    ev.update(over)
    return ev


@pytest.fixture()
def open_keys(monkeypatch):
    """Both suppression keys free, and a send that always succeeds."""
    monkeypatch.setattr(pd, "suppression_blocks", lambda db, k, tier: False)
    monkeypatch.setattr(pd, "recent_paged_alerts", lambda db, iso, exc: _PAGED)
    monkeypatch.setattr(pd, "register_alert", lambda *a, **kw: None)
    monkeypatch.setattr(pd, "get_peak_tier", lambda db, sid: None)
    sent = MagicMock(return_value=True)
    monkeypatch.setattr(pd, "send_telegram_alert", sent)
    return sent


class TestDispatchWithDuplicateAdjudicator:
    def test_duplicate_at_same_tier_is_muted(self, open_keys, monkeypatch):
        recorded = []
        monkeypatch.setattr(pd, "record_suppression",
                            lambda db, k, tier, eid, **kw: recorded.append((k, tier)))

        ev = _base_event(alert_tier="ALERT")
        result = pd.dispatch_alert(MagicMock(), ev, "evt1",
                                   lambda e, paged: _PAGED[0])
        assert result == "suppressed_duplicate"
        open_keys.assert_not_called()
        # The verdict is cached into this event's own keys, so its next sibling is muted
        # by the cheap path instead of costing another call.
        assert recorded and all(tier == "ALERT" for _, tier in recorded)

    def test_escalation_still_pages(self, open_keys, monkeypatch):
        """Same incident, but it got worse — the rule that governs the keys governs the
        adjudicator's verdict too."""
        monkeypatch.setattr(pd, "record_suppression", lambda *a, **kw: None)

        ev = _base_event(alert_tier="CRITICAL")
        result = pd.dispatch_alert(MagicMock(), ev, "evt1",
                                   lambda e, paged: {"id": "e2", "alert_tier": "WATCH"})
        assert result == "sent"
        open_keys.assert_called_once()

    def test_new_verdict_pages(self, open_keys, monkeypatch):
        monkeypatch.setattr(pd, "record_suppression", lambda *a, **kw: None)
        assert pd.dispatch_alert(MagicMock(), _base_event(), "evt1",
                                 lambda e, paged: None) == "sent"
        open_keys.assert_called_once()

    def test_adjudicator_error_pages(self, open_keys, monkeypatch):
        monkeypatch.setattr(pd, "record_suppression", lambda *a, **kw: None)

        def boom(event, paged):
            raise RuntimeError("bulk router down")

        assert pd.dispatch_alert(MagicMock(), _base_event(), "evt1", boom) == "sent"
        open_keys.assert_called_once()

    def test_absent_adjudicator_pages(self, open_keys, monkeypatch):
        """The pre-existing call shape (no adjudicator) must behave exactly as before."""
        monkeypatch.setattr(pd, "record_suppression", lambda *a, **kw: None)
        assert pd.dispatch_alert(MagicMock(), _base_event(), "evt1") == "sent"
        open_keys.assert_called_once()

    def test_not_consulted_when_a_key_already_blocks(self, monkeypatch):
        """Cost guard: the LLM is the last resort, never a substitute for the keys."""
        monkeypatch.setattr(pd, "suppression_blocks", lambda db, k, tier: True)
        monkeypatch.setattr(pd, "send_telegram_alert", MagicMock())
        adj = MagicMock()

        assert pd.dispatch_alert(MagicMock(), _base_event(), "evt1", adj) == "suppressed"
        adj.assert_not_called()

    def test_not_consulted_for_untiered_events(self, monkeypatch):
        monkeypatch.setattr(pd, "send_telegram_alert", MagicMock())
        adj = MagicMock()

        ev = _base_event(alert_tier=None)
        assert pd.dispatch_alert(MagicMock(), ev, "evt1", adj) == "skipped"
        adj.assert_not_called()
