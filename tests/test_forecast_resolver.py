"""Tests for the forecast resolver (automated forecast verification)."""

import json
from datetime import date
from unittest.mock import MagicMock

from src.services.forecast_resolver import (
    _actual_direction,
    _grade_report,
    build_calibration_feedback,
    resolve_pending_reports,
)


# 1. Actual-direction thresholds mirror validate_g2_assessment / calculate_trajectory
def test_actual_direction_thresholds():
    assert _actual_direction(delta=10.0, z_score=1.5) == "Escalating"
    assert _actual_direction(delta=10.0, z_score=0.5) == "Stable"     # z too low
    assert _actual_direction(delta=5.0, z_score=2.0) == "Stable"      # delta too low
    assert _actual_direction(delta=-10.0, z_score=0.0) == "De-escalating"
    assert _actual_direction(delta=0.0, z_score=0.0) == "Stable"


def _assessment(country, direction, confidence="Medium"):
    return {
        "country": country,
        "forecast": {"risk_direction": direction, "confidence": confidence},
    }


# 2. Grading: accuracy, brier, recall, fp_rate over a mixed forecast set
def test_grade_report_mixed_outcomes():
    current_scores = {
        "UA": {"ti": 80.0, "delta": 12.0, "z_score": 1.5},   # actually Escalating
        "IL": {"ti": 40.0, "delta": 0.0, "z_score": 0.2},    # actually Stable
        "SY": {"ti": 55.0, "delta": 9.0, "z_score": 1.2},    # actually Escalating (unassessed!)
    }
    graded = _grade_report(
        report_scores={"UA": {"ti": 68.0}, "IL": {"ti": 40.0}},
        assessments=[
            _assessment("UA", "Escalating", "High"),      # correct
            _assessment("IL", "Escalating", "Low"),       # wrong (over-escalation)
        ],
        deteriorating=[],
        watchlist=[],
        current_scores=current_scores,
    )
    assert graded["accuracy"] == 0.5
    assert graded["direction_correct"] is True
    # Brier: correct High -> (0.9-1)^2=0.01; wrong Low -> (0.5-0)^2=0.25 -> mean 0.13
    assert abs(graded["brier"] - 0.13) < 1e-9
    # Recall: UA flagged of {UA, SY} actually escalated -> 0.5 (SY was a miss)
    assert graded["escalation_recall"] == 0.5
    # FP rate: 2 Escalating forecasts, 1 didn't escalate -> 0.5
    assert graded["fp_rate"] == 0.5


# 3. A country absent from current scores = quiet week = de-escalation from high TI
def test_grade_report_country_went_quiet():
    graded = _grade_report(
        report_scores={"YE": {"ti": 50.0}},
        assessments=[_assessment("YE", "De-escalating")],
        deteriorating=[],
        watchlist=[],
        current_scores={},  # no events anywhere this week
    )
    rec = graded["records"][0]
    assert rec["actual_direction"] == "De-escalating"
    assert rec["correct"] is True
    assert graded["escalation_recall"] is None  # nothing actually escalated
    assert graded["fp_rate"] is None            # no Escalating forecasts


# 4. Watchlist membership counts toward escalation recall
def test_grade_report_watchlist_counts_as_flagged():
    graded = _grade_report(
        report_scores={"IR": {"ti": 30.0}},
        assessments=[_assessment("IR", "Stable")],
        deteriorating=[],
        watchlist=["LB"],
        current_scores={
            "IR": {"ti": 30.0, "delta": 0.0, "z_score": 0.0},
            "LB": {"ti": 60.0, "delta": 15.0, "z_score": 2.0},  # escalated, but watchlisted
        },
    )
    assert graded["escalation_recall"] == 1.0


# 5. No gradable assessments -> None (report skipped, no row written)
def test_grade_report_empty_assessments():
    assert _grade_report({}, [], [], [], {"UA": {"ti": 1, "delta": 0, "z_score": 0}}) is None


class FakeConn:
    """Minimal db_conn stand-in: canned SELECT rows, records INSERTs."""

    def __init__(self, select_rows):
        self.select_rows = select_rows
        self.inserts = []
        self.committed = False

    def execute(self, sql, params=None):
        cursor = MagicMock()
        if sql.strip().upper().startswith("SELECT") or "SELECT r.id" in sql:
            cursor.fetchall.return_value = self.select_rows
        else:
            self.inserts.append((sql, params))
        return cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


# 6. End-to-end resolution: pending report graded and persisted once
def test_resolve_pending_reports_persists_validation():
    report_row = (
        "report-uuid-1",
        date(2026, 7, 1),
        {"UA": {"ti": 68.0}},  # scores_json
        {"country_assessments": [_assessment("UA", "Escalating", "High")]},
        [],   # deteriorating
        [],   # watchlist
    )
    conn = FakeConn([report_row])
    res = resolve_pending_reports(
        conn,
        current_scores={"UA": {"ti": 80.0, "delta": 12.0, "z_score": 1.5}},
        week_start=date(2026, 7, 1),
    )
    assert len(res) == 1
    assert res[0]["accuracy"] == 1.0
    assert len(conn.inserts) == 1
    sql, params = conn.inserts[0]
    assert "report_validations" in sql
    assert params[0] == "report-uuid-1"
    details = json.loads(params[-1])
    assert details["countries"][0]["country"] == "UA"
    assert conn.committed


# 7. Calibration feedback: over-escalation bias is detected and phrased
def test_calibration_feedback_overescalation_bias():
    details = {
        "countries": [
            {"country": "IL", "forecast_direction": "Escalating",
             "actual_direction": "Stable", "correct": False},
            {"country": "SY", "forecast_direction": "Escalating",
             "actual_direction": "Stable", "correct": False},
            {"country": "UA", "forecast_direction": "Escalating",
             "actual_direction": "Escalating", "correct": True},
        ]
    }
    conn = FakeConn([(0.33, details)])
    fb = build_calibration_feedback(conn)
    assert "1/3" in fb["note"]
    assert "over-escalating" in fb["note"]
    # Missed countries surface per-country notes; the correct one does not
    assert "IL" in fb["per_country"] and "SY" in fb["per_country"]
    assert "UA" not in fb["per_country"]


# 8. No history -> empty note, G2 prompt stays untouched
def test_calibration_feedback_empty_history():
    fb = build_calibration_feedback(FakeConn([]))
    assert fb == {"note": "", "per_country": {}}
