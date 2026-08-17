"""Tests for the report_kind alert gate — the classifier's read of what an ARTICLE is.

Measured 2026-08-11 over 7 days: the severity floor produced 296 of 561 ALERT-tier
events while applying no confidence requirement at all, and the junk inside it was not
low-confidence, it was the wrong KIND of article — arrests, charges, reopenings,
released footage, condemnations, "Day 1,625" war diaries. Two instruments were measured
and rejected before this one:

  - a confidence threshold on the floor: the floor bucket spans conf 0.21-0.50, so any
    bar at the ladder's 0.50 deletes it entirely, taking "13 killed in Ukrainian strike
    on Tatarstan" and "Saudi Arabia suspends Najran airport" with the junk;
  - title keywords for follow-up language: 9.5% recall on the floor bucket while also
    firing on 5.7% of the clean ladder ALERTs.

So the tests below pin the property that matters most: the gate withholds pages only on
an explicit known verdict, and every degraded path resolves to sending.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.core.alerts import evaluate_alert_tier, evaluate_alert_tier_verbose
from src.pipeline.pass_c_classify import REPORT_KIND_NEW, _safe_report_kind


def _event(**over):
    """An event that clears the ALERT ladder on its own merits."""
    ev = {
        "severity_score": 95,
        "system_confidence": 0.55,
        "anchor_confidence": "LOW",
        "time_certainty": "same_day",
        "event_type": "missile_strike",
        "anchor_name_norm": "KBP",
        "source_title": "Missile strike hits Kyiv district, three killed",
        "report_kind": REPORT_KIND_NEW,
    }
    ev.update(over)
    return ev


class TestSafeReportKind:
    @pytest.mark.parametrize("raw", ["followup", "roundup", "commentary", "new_incident"])
    def test_known_values_pass_through(self, raw):
        assert _safe_report_kind(raw) == raw

    @pytest.mark.parametrize("raw,expected", [
        ("follow-up", "followup"),
        ("Roundup ", "roundup"),
        ("  COMMENTARY", "commentary"),
        ("new incident", REPORT_KIND_NEW),
    ])
    def test_paraphrases_are_normalized(self, raw, expected):
        """Models reliably paraphrase enum values; that is worth normalising. Inventing
        new ones is not worth guessing at — see the test below."""
        assert _safe_report_kind(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "retrospective", "news", 3, {"a": 1}, []])
    def test_unknown_values_resolve_to_new_incident(self, raw):
        """The whole failure direction of this feature: a value the gate does not
        recognise must not be trusted to mean 'not news'."""
        assert _safe_report_kind(raw) == REPORT_KIND_NEW


class TestReportKindGate:
    @pytest.mark.parametrize("kind", ["followup", "roundup", "commentary"])
    def test_non_news_kinds_withhold_the_page(self, kind):
        assert evaluate_alert_tier(_event(report_kind=kind)) is None

    def test_new_incident_pages(self):
        assert evaluate_alert_tier(_event()) == "ALERT"

    def test_missing_field_pages(self):
        """Events classified before the field existed keep their old behaviour."""
        ev = _event()
        del ev["report_kind"]
        assert evaluate_alert_tier(ev) == "ALERT"

    def test_unrecognised_value_pages(self):
        assert evaluate_alert_tier(_event(report_kind="retrospective")) == "ALERT"

    def test_critical_is_exempt(self):
        """Same exemption as the aftermath title gate: a roundup is sometimes the only
        carrier of a genuinely major development."""
        ev = _event(report_kind="roundup", system_confidence=0.70)
        assert evaluate_alert_tier(ev) == "CRITICAL"

    def test_gate_also_vetoes_the_severity_floor(self):
        """The floor is the path this gate exists for: it promotes to ALERT without any
        confidence requirement, so it must not outrank the article-shape check."""
        ev = _event(report_kind="followup", severity_score=95, system_confidence=0.30)
        # Confidence 0.30 clears no ladder rung; only the floor could produce a tier.
        assert evaluate_alert_tier(_event(severity_score=95, system_confidence=0.30)) == "ALERT"
        assert evaluate_alert_tier(ev) is None

    def test_watch_tier_is_gated_too(self):
        ev = _event(report_kind="followup", severity_score=50, system_confidence=0.45)
        assert evaluate_alert_tier(_event(severity_score=50, system_confidence=0.45)) == "WATCH"
        assert evaluate_alert_tier(ev) is None

    def test_advisories_bypass_the_gate(self):
        """Advisories return before the article-shape gates — a standing advisory is not
        an incident report and would read as 'commentary' to any article classifier."""
        ev = _event(event_type="travel_advisory", severity_score=90, report_kind="commentary")
        assert evaluate_alert_tier(ev) == "ALERT"


class TestVetoReason:
    def test_report_kind_veto_is_named(self):
        tier, veto = evaluate_alert_tier_verbose(_event(report_kind="roundup"))
        assert tier is None
        assert veto == "report_kind_roundup"

    def test_title_gate_veto_is_named(self):
        tier, veto = evaluate_alert_tier_verbose(
            _event(source_title="Colombia earthquake update: death toll rises to 111"))
        assert tier is None
        assert veto == "aftermath_title"

    def test_title_gate_wins_when_both_fire(self):
        """Ordering is not arbitrary: the free check runs first, so the counter
        attributes a veto to the cheapest gate that could have produced it."""
        _, veto = evaluate_alert_tier_verbose(_event(
            source_title="Colombia earthquake update: death toll rises to 111",
            report_kind="followup"))
        assert veto == "aftermath_title"

    def test_no_veto_reported_for_a_clean_page(self):
        assert evaluate_alert_tier_verbose(_event()) == ("ALERT", None)

    def test_no_veto_reported_when_nothing_qualified(self):
        """A veto means a tier was taken away, not that none was earned."""
        tier, veto = evaluate_alert_tier_verbose(
            _event(severity_score=10, system_confidence=0.1, report_kind="followup"))
        assert tier is None
        assert veto is None


def _scoring_db(llm_parsed: dict):
    """A db_conn stub shaped for score_single_event, carrying `llm_parsed` on the event."""
    db_conn = MagicMock()
    cursor_event = MagicMock()
    cursor_event.fetchone.return_value = (
        "event_123", "missile_strike", "Kyiv", "UA",
        json.dumps(llm_parsed), "Kyiv missile strike",
        datetime.now(timezone.utc) - timedelta(hours=2),
        "Missile strike hits Kyiv district, three killed",
        "https://example.com/kyiv", datetime.now(timezone.utc) - timedelta(hours=1),
        "example.com",
        datetime.now(timezone.utc) - timedelta(hours=2),  # published_at
        True,  # date_verified — publisher's own date, so the gate is testing report_kind
        None,  # corroborating_sources — uncorroborated, so the floor stays out of it
    )
    cursor_counts = MagicMock(); cursor_counts.fetchone.return_value = (0, 0)
    cursor_cat = MagicMock(); cursor_cat.fetchone.return_value = (95,)
    # One source only (diversity 0.3) so system_confidence lands under CRITICAL's 0.62
    # bar — CRITICAL is exempt from the gate, so an over-confident fixture would test
    # nothing.
    cursor_div = MagicMock(); cursor_div.fetchone.return_value = (1,)
    cursor_supp = MagicMock(); cursor_supp.fetchone.return_value = None

    def side_effect(query, params=None):
        if "FROM events WHERE id =" in query:
            return cursor_event
        if "FROM event_type_catalog" in query:
            return cursor_cat
        if "COUNT(DISTINCT source_domain)" in query:
            return cursor_div
        if "FROM alert_suppression" in query:
            return cursor_supp
        if "COUNT(*)" in query:
            return cursor_counts
        return MagicMock()

    db_conn.execute.side_effect = side_effect
    return db_conn


class TestReportKindReachesTheGate:
    """The field is only useful if pass_d actually carries it from llm_parsed_output into
    the gate's input dict — a wiring gap here would be silent, since a missing value is
    indistinguishable from new_incident by design."""

    @patch("src.pipeline.pass_d_score.resolve_anchor_for_event")
    @patch("src.pipeline.pass_d_score.send_telegram_alert")
    def _score(self, llm_parsed, mock_send, mock_anchor):
        from src.pipeline.pass_d_score import score_single_event

        mock_anchor.return_value = {
            "norm": "KBP", "confidence": 0.5, "level": "MEDIUM", "czib_flag": False,
            "latitude": 50.34, "longitude": 30.89, "country_iso": "UA",
        }
        result = score_single_event(_scoring_db(llm_parsed), "event_123", [])
        return result, mock_send

    # 0.8*0.4 + 0.5*0.3 + 0.3*0.3 = 0.56 — clears ALERT's 0.50, under CRITICAL's 0.62.
    _LLM = {"confidence": 0.8, "time_certainty": "same_day"}

    def test_followup_is_not_paged(self):
        result, mock_send = self._score({**self._LLM, "report_kind": "followup"})
        assert result["alert_tier"] is None
        assert result["alert_veto"] == "report_kind_followup"
        mock_send.assert_not_called()

    def test_new_incident_is_paged(self):
        result, mock_send = self._score({**self._LLM, "report_kind": REPORT_KIND_NEW})
        assert result["alert_tier"] == "ALERT"
        assert result["alert_veto"] is None
        mock_send.assert_called_once()
