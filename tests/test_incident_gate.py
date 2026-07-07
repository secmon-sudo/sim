"""Tests for the severity incident-gate: a generic umbrella event type with no located
anchor and no casualties must be capped so an LLM mislabel can't become near-critical.
"""

from unittest.mock import MagicMock

from src.pipeline.pass_d_score import INCIDENT_GATE_CAP, _has_casualties, compute_severity


def _db(base):
    """db_conn whose event_type_catalog lookup returns the given severity_base."""
    db = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (base,)
    db.execute.return_value = cur
    return db


_NO_ANCHOR = {"confidence": 0.0}                 # Unknown location
_GOOD_ANCHOR = {"confidence": 1.0, "czib_flag": False}


class TestIncidentGate:
    def test_umbrella_no_location_no_casualties_is_capped(self):
        # geopolitical_conflict base 45 stays under the cap anyway; use an inflated base
        # to prove the cap itself fires (simulates any high-base umbrella mislabel).
        sev = compute_severity("geopolitical_conflict", _NO_ANCHOR, _db(85), {})
        assert sev == INCIDENT_GATE_CAP  # 85 -> capped to 50

    def test_umbrella_with_location_not_capped(self):
        # A located conflict earns the proximity bonus and escapes the gate.
        sev = compute_severity("geopolitical_conflict", _GOOD_ANCHOR, _db(45),
                               {"casualties": None})
        assert sev == 45 + 30  # base + proximity, no cap

    def test_umbrella_with_casualties_not_capped(self):
        sev = compute_severity("geopolitical_conflict", _NO_ANCHOR, _db(85),
                               {"casualties": {"deaths": 4}})
        assert sev > INCIDENT_GATE_CAP  # real incident signal -> gate does not apply

    def test_specific_type_never_gated(self):
        # A specific incident type (missile_strike) is NOT an umbrella; even without a
        # located anchor it keeps its full base.
        sev = compute_severity("missile_strike", _NO_ANCHOR, _db(100), {})
        assert sev == 100


class TestHasCasualties:
    def test_deaths(self):
        assert _has_casualties({"casualties": {"deaths": 1}}) is True

    def test_injuries_only(self):
        assert _has_casualties({"casualties": {"injuries": 3}}) is True

    def test_zeros(self):
        assert _has_casualties({"casualties": {"deaths": 0, "injuries": 0}}) is False

    def test_missing_field(self):
        assert _has_casualties({}) is False

    def test_none(self):
        assert _has_casualties(None) is False

    def test_malformed(self):
        assert _has_casualties({"casualties": "lots"}) is False
        assert _has_casualties({"casualties": {"deaths": "many"}}) is False
