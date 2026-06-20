"""
Tests for Faz 2.4 — safety (emniyet) vs security (güvenlik) separation.

Accidental safety events are tagged is_safety and de-prioritized (severity capped)
so they don't raise security alerts, while mass-casualty safety events still surface.
"""

from src.pipeline.pass_d_score import (
    SAFETY_SEVERITY_CAP,
    apply_safety_downrank,
)


class TestSafetyDownrank:
    def test_security_event_untouched(self):
        sev, is_safety = apply_safety_downrank("bomb_threat", 90, {})
        assert sev == 90
        assert is_safety is False

    def test_routine_safety_event_capped_and_tagged(self):
        # engine_failure base ~55 → capped below alert threshold, tagged safety.
        sev, is_safety = apply_safety_downrank("engine_failure", 55, {})
        assert sev == SAFETY_SEVERITY_CAP
        assert is_safety is True

    def test_safety_below_cap_unchanged_value(self):
        sev, is_safety = apply_safety_downrank("bird_strike", 30, {})
        assert sev == 30  # already below cap
        assert is_safety is True

    def test_mass_casualty_safety_not_capped(self):
        # A fatal emergency landing still surfaces (tagged safety, not capped).
        llm = {"casualties": {"deaths": 12, "injuries": 40}}
        sev, is_safety = apply_safety_downrank("emergency_landing", 85, llm)
        assert sev == 85
        assert is_safety is True

    def test_coverage_preserved(self):
        # Safety events are never dropped — they keep a (low) score and stay queryable.
        sev, is_safety = apply_safety_downrank("depressurization", 65, {})
        assert sev > 0
        assert is_safety is True
