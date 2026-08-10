"""
The removed "LLM false-negative guard", and the measurement that replaced it.

Pass C used to floor relevance to 30 and keep any event whose title+body contained a
_HIGH_SIGNAL_TERMS word, so the LLM could never "silently archive" a real incident.
Measured over the 7 days to 2026-08-10 it kept 642 events and produced ZERO alerts —
and could not have produced one: rescuing a "noise" verdict also stamps
FALLBACK_EVENT_TYPE, whose catalog severity_base is 20, against an ALERT floor of 65
and a severity floor of 90. The rescue and the cap were the same line of code.

Two root causes worth keeping tests on:

1. _HIGH_SIGNAL_TERMS is tuned for RECALL — it exists to score relevance, so it holds
   "war", "conflict", "killed", "sanctions", "refugee", "nuclear". A recall vocabulary
   makes a hopeless precision veto: it fired on flood tolls, road-safety statistics and
   opinion columns.
2. It matched the article BODY, so any commentary mentioning an incident qualified
   ("Twin Falls gun sales spike after In-N-Out mass shooting").

Even the tightest variant (hostile acts, title only) left 38 rescues in that week, of
which 2 were real incidents. The other 36 included a tech blog on the domain
explosion.com, a metal band called Car Bomb, the 1933 Simele massacre and a kidnapped
Serbian eagle.
"""

from unittest.mock import MagicMock, patch

import pytest

import src.pipeline.pass_c_classify as pc
from src.pipeline.pass_c_classify import HOSTILE_ACT_PATTERN


class TestHostileActPattern:
    @pytest.mark.parametrize("title", [
        "AA Reports 4 Injured in Mrauk-U Night Airstrikes",
        "18 Injured in Shamsabad Industrial Explosion",
        "Car bomb hits Colombia highway",
        "Russian drone strikes passenger train travelling from Sumy to Kyiv",
        "Gunfire reported outside the embassy",
        "Beauty Influencer Assassinated by Hitman",
    ])
    def test_matches_hostile_acts(self, title):
        assert HOSTILE_ACT_PATTERN.search(title)

    @pytest.mark.parametrize("title", [
        # Outcome words without an act — these are what made the old guard fire on
        # floods and road-safety copy.
        "Kerala floods leave 21 dead, six missing as IMD warns of more rain",
        "Bangladesh's roads remain deadly; 416 killed in July alone",
        "Explosive Wildfire in British Columbia Leaves 1 Dead",
        # Ambient politics — present in nearly every geopolitical article.
        "Palestinian rejectionism, not poverty, remains the main obstacle for peace",
        "UK imposes new sanctions on 13 Russian entities",
        "Hungary comes 'within millimetres' of shutting down nuclear power plant",
        "Security Council LIVE: Warnings of famine and collapse in South Sudan",
    ])
    def test_ignores_outcome_and_ambient_vocabulary(self, title):
        assert HOSTILE_ACT_PATTERN.search(title) is None

    def test_it_is_a_strict_subset_of_the_relevance_vocabulary(self):
        """The point of the split: the scoring vocabulary must stay broad, and this
        one must not inherit its breadth."""
        from src.pipeline.ingest_filters import _HIGH_SIGNAL_TERMS

        for word in ("war", "conflict", "clashes", "killed", "dead", "casualties",
                     "sanctions", "refugee", "displaced", "nuclear", "evacuated"):
            assert word in _HIGH_SIGNAL_TERMS, f"{word} should still score relevance"
            assert HOSTILE_ACT_PATTERN.search(f"a {word} happened") is None, \
                f"{word} must not act as a hostile-act signal"


# ── the guard is gone ──────────────────────────────────────────────────────

class _FakeConn:
    def __init__(self):
        self.updates = []

    def execute(self, sql, params=None):
        if sql.strip().upper().startswith(("UPDATE", "INSERT")):
            self.updates.append((sql, params))
        result = MagicMock()
        result.fetchone.return_value = ("security_incident",)
        return result

    def transaction(self):
        from contextlib import nullcontext
        return nullcontext()

    def commit(self):
        pass

    def rollback(self):
        pass


def _apply(title, relevance, llm_type="noise"):
    conn = _FakeConn()
    event = {"id": "abcdef12-0000-0000-0000-000000000000",
             "source_title": title, "source_domain": "example.com"}
    det = {"has_high_signal": True, "has_flight_disruption": False, "score": 40}
    parsed = {"event_type": llm_type, "relevance_score": relevance}
    with patch.object(pc, "log_llm_telemetry", MagicMock()):
        out = pc._apply_llm_classification(
            conn, MagicMock(), event, det, parsed,
            {"provider": "groq", "model": "m", "response": {}}, "wid")
    archived = any("status = 'archived'" in sql for sql, _ in conn.updates)
    return out, archived


class TestOverrideRemoved:
    def test_high_signal_no_longer_rescues_a_noise_verdict(self):
        # The exact old behaviour: has_high_signal + relevance 15 used to survive.
        out, archived = _apply("Explosion reported somewhere", relevance=15)
        assert archived, "a relevance-15 noise verdict must now be archived"

    def test_the_domain_is_scored_honestly_when_archiving(self):
        """The keep path credited the domain (penalty 0), so explosion.com sat at
        penalty_score 0.000 across 3 rescued events."""
        conn = _FakeConn()
        event = {"id": "abcdef12-0000-0000-0000-000000000000",
                 "source_title": "Explosion reported", "source_domain": "explosion.com"}
        with patch.object(pc, "log_llm_telemetry", MagicMock()), \
             patch.object(pc, "update_domain_penalty", MagicMock()) as penalty:
            pc._apply_llm_classification(
                conn, MagicMock(), event,
                {"has_high_signal": True, "has_flight_disruption": False, "score": 40},
                {"event_type": "noise", "relevance_score": 15},
                {"provider": "groq", "model": "m", "response": {}}, "wid")
        penalty.assert_called_once_with(conn, "explosion.com", 1)

    def test_genuinely_relevant_events_are_untouched(self):
        out, archived = _apply("Missile strike kills 12 in Kyiv", relevance=85,
                               llm_type="missile_strike")
        assert not archived


class TestFalseNegativeTelemetry:
    def test_archived_hostile_act_headline_is_flagged(self):
        out, archived = _apply("AA Reports 4 Injured in Mrauk-U Night Airstrikes",
                               relevance=15)
        assert archived
        assert out["_high_signal_archived"] is True

    def test_archived_ordinary_noise_is_not_flagged(self):
        out, archived = _apply("Kerala floods leave 21 dead", relevance=10)
        assert archived
        assert "_high_signal_archived" not in out

    def test_marker_is_not_persisted_into_llm_parsed_output(self):
        """It is set after the row is written, so the stored JSON stays clean."""
        conn = _FakeConn()
        event = {"id": "abcdef12-0000-0000-0000-000000000000",
                 "source_title": "Explosion at the airport", "source_domain": "x.com"}
        with patch.object(pc, "log_llm_telemetry", MagicMock()):
            pc._apply_llm_classification(
                conn, MagicMock(), event,
                {"has_high_signal": True, "has_flight_disruption": False, "score": 40},
                {"event_type": "noise", "relevance_score": 15},
                {"provider": "groq", "model": "m", "response": {}}, "wid")
        written = [params for sql, params in conn.updates if "status = 'archived'" in sql]
        assert written and all("_high_signal_archived" not in str(p) for p in written)

    def test_batch_counts_flagged_events_into_stats(self):
        """The flag has to survive the batch path, which is what production runs."""
        import json

        events = [{"id": f"0000000{i}-0000-0000-0000-000000000000",
                   "source_title": f"Report {i}", "source_domain": "example.com",
                   "canonical_text": "text"} for i in (1, 2)]
        content = json.dumps({"results": [{"report": 1, "event_type": "noise"},
                                          {"report": 2, "event_type": "noise"}]})
        # First event archived with a hostile-act headline, second ordinary noise.
        applied = MagicMock(side_effect=[{"_high_signal_archived": True}, {"event_type": "x"}])
        mocks = dict(
            acquire_lock=MagicMock(return_value=True),
            release_lock=MagicMock(),
            deterministic_relevance=MagicMock(return_value={"score": 50,
                                                            "has_high_signal": True}),
            _try_prescreen_archive=MagicMock(return_value=False),
            _apply_llm_classification=applied,
            log_llm_telemetry=MagicMock(),
            call_llm=MagicMock(return_value={"content": content}),
        )
        with patch.multiple(pc, **mocks):
            stats = pc.classify_event_batch(MagicMock(), MagicMock(), events, "wid")

        assert stats["classified"] == 2
        assert stats["high_signal_archived"] == 1
