"""
Tests for Faz 1.2 (military-bypass canceller) and Faz 1.3 (aviation-nexus bonus).
"""

from src.pipeline.pass_a_ingest import is_noise
from src.pipeline.pass_d_score import (
    AVIATION_NEXUS_BONUS,
    compute_aviation_bonus,
)


class TestMilitaryBypassCanceller:
    def test_documentary_with_military_term_is_noise(self):
        # Previously "missile" rescued this via the military bypass — now filtered.
        assert is_noise("A new documentary about the missile strike on the city") is True

    def test_film_with_airstrike_is_noise(self):
        assert is_noise("New film about the airstrike that changed the war") is True

    def test_live_military_event_not_noise(self):
        assert is_noise("Missile strike hits airbase, casualties reported") is False

    def test_live_airport_attack_not_noise(self):
        assert is_noise("Drone strike hits airport runway, flights suspended") is False


class TestAviationNexusBonus:
    def test_aviation_event_type_gets_bonus(self):
        assert compute_aviation_bonus({"event_type": "aviation_personnel_attack"}, None) == AVIATION_NEXUS_BONUS

    def test_generic_event_with_airport_text_gets_bonus(self):
        ev = {"event_type": "terrorism", "source_title": "Blast at Kabul airport terminal"}
        assert compute_aviation_bonus(ev, None) == AVIATION_NEXUS_BONUS

    def test_llm_direct_aviation_impact_gets_bonus(self):
        ev = {"event_type": "missile_strike", "llm_parsed": {"aviation_impact": "direct"}}
        assert compute_aviation_bonus(ev, None) == AVIATION_NEXUS_BONUS

    def test_pure_geopolitics_no_bonus(self):
        ev = {"event_type": "military_action", "source_title": "Tanks cross the border region"}
        assert compute_aviation_bonus(ev, None) == 0

    def test_broad_coverage_preserved(self):
        # A maritime/cyber/protest event without aviation nexus is still scored —
        # it just doesn't earn the aviation bonus (coverage unchanged, only ranking).
        ev = {"event_type": "civil_unrest", "source_title": "Mass protest grips the capital"}
        assert compute_aviation_bonus(ev, None) == 0
