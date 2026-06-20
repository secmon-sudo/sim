"""
Tests for Faz 3 — hybrid storyline linking (anchor-assist + date-token removal)
and the zero-LLM narrative timeline.
"""

from datetime import datetime, timedelta

from src.core.storyline import jaccard_similarity, should_link_storyline
from src.core.storyline_narrative import (
    TREND_ESCALATING,
    TREND_STABLE,
    build_timeline,
    summarize_timeline,
)

_T0 = datetime(2026, 6, 8, 10, 0)


def _ev(hint, iso="AF", anchor=None, when=_T0, sev=50):
    return {
        "storyline_hint": hint,
        "country_iso": iso,
        "anchor_name_norm": anchor,
        "occurred_at_est": when,
        "severity_score": sev,
    }


class TestDateTokenRemoval:
    def test_date_hint_does_not_distort(self):
        # Same event reported on consecutive days must stay maximally similar.
        sim = jaccard_similarity(
            "istanbul ataturk bomb threat jun8", "istanbul ataturk bomb threat jun9"
        )
        assert sim == 1.0

    def test_flight_number_preserved(self):
        # "dl54" is a strong identifier, must NOT be stripped as a date token.
        sim = jaccard_similarity("delta dl54 emergency atlanta jun7", "delta dl54 emergency atlanta")
        assert sim > 0.5


class TestHybridAnchorAssist:
    def test_paraphrase_same_airport_same_day_links(self):
        a = _ev("kabul airport explosion terminal", anchor="KBL", when=_T0)
        b = _ev("blast rocks kabul international departures", anchor="KBL", when=_T0 + timedelta(hours=4))
        # Lexical similarity alone is far too low to link...
        assert jaccard_similarity(a["storyline_hint"], b["storyline_hint"]) < 0.2
        # ...but same anchor within the tight window rescues it.
        assert should_link_storyline(a, b) is True

    def test_same_airport_far_apart_low_sim_does_not_link(self):
        a = _ev("kabul airport explosion terminal", anchor="KBL", when=_T0)
        b = _ev("kabul airport security drill announced", anchor="KBL", when=_T0 + timedelta(days=10))
        assert should_link_storyline(a, b) is False

    def test_different_airport_low_sim_does_not_link(self):
        a = _ev("kabul airport explosion terminal", anchor="KBL", iso="AF", when=_T0)
        b = _ev("blast rocks departures hall", anchor="JFK", iso="US", when=_T0)
        assert should_link_storyline(a, b) is False

    def test_country_mismatch_hard_gate(self):
        a = _ev("identical attack hint here", iso="AF", when=_T0)
        b = _ev("identical attack hint here", iso="US", when=_T0)
        assert should_link_storyline(a, b) is False


class TestNarrative:
    def test_build_timeline_orders_and_sequences(self):
        evs = [
            _ev("c", when=_T0 + timedelta(hours=20)),
            _ev("a", when=_T0),
            _ev("b", when=_T0 + timedelta(hours=4)),
        ]
        tl = build_timeline(evs)
        assert [e["seq"] for e in tl] == [1, 2, 3]
        assert tl[0]["occurred_at_est"] < tl[1]["occurred_at_est"] < tl[2]["occurred_at_est"]

    def test_summary_detects_escalation(self):
        evs = [
            _ev("x", when=_T0, sev=40),
            _ev("x", when=_T0 + timedelta(hours=4), sev=70),
            _ev("x", when=_T0 + timedelta(days=1), sev=90),
        ]
        s = summarize_timeline(evs)
        assert s["event_count"] == 3
        assert s["peak_severity"] == 90
        assert s["severity_trend"] == TREND_ESCALATING

    def test_empty_summary_is_safe(self):
        s = summarize_timeline([])
        assert s["event_count"] == 0
        assert s["severity_trend"] == TREND_STABLE
