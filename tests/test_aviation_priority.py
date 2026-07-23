"""
Tests for Faz 1.2 (military-bypass canceller) and Faz 1.3 (aviation-nexus bonus).
"""

from src.pipeline.pass_a_ingest import _matches_security_keywords, is_noise
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


class TestFlightDisruptionGate:
    """The gate that decides whether a flight-disruption headline is ingested.

    An airline is the end customer, so "which carrier stopped flying where" is
    the highest-value line in a SITREP — but the vocabulary of a security
    grounding is identical to that of a snowstorm cancellation. Coverage was
    measured against a live Google News feed on 2026-07-23: before these
    keywords, 10 of 14 genuine Gulf-conflict disruption headlines were dropped.
    """

    @staticmethod
    def _passes(title: str) -> bool:
        return _matches_security_keywords(title, "") and not is_noise(title)

    def test_carrier_suspension_passes(self):
        assert self._passes("Emirates suspends all flights to Tehran amid strikes")

    def test_gerund_form_passes(self):
        # "Airlines Suspending Flights" — the participle is as common in
        # headlines as the third-person verb.
        assert self._passes("Emirates and Etihad among airlines suspending flights to Kuwait")

    def test_route_suspension_passes(self):
        assert self._passes("Air France suspends routes to Riyadh, Dubai and Beirut")

    def test_airport_ceasing_operations_passes(self):
        assert self._passes("Kuwait International Airport temporarily suspends operations")

    def test_passive_voice_passes(self):
        assert self._passes("Flights suspended at Bahrain International Airport")

    def test_cancellation_with_security_cause_passes(self):
        assert self._passes("Jordan flight cancellations continue as Iranian attacks disrupt air travel")

    def test_weather_cancellation_filtered(self):
        # Safety, not security — the distinction the SITREP scope rests on.
        assert not self._passes("Delta cancels flights due to snowstorm in Chicago")

    def test_winter_storm_filtered(self):
        assert not self._passes("United cancels flights after winter storm hits Denver")

    def test_fog_disruption_filtered(self):
        assert not self._passes("Heathrow flight disruption caused by dense fog")

    def test_commercial_route_news_filtered(self):
        assert not self._passes("Ryanair launches new route to Malaga with fare sale")

    def test_maintenance_filtered(self):
        assert not self._passes("Airline cancels flights for scheduled maintenance")

    def test_verb_place_flights_passes(self):
        # "cancel <place> flights" defeats any fixed-phrase list; the
        # aviation-noun + disruption-verb conjunction is what catches it.
        assert self._passes("Etihad, Emirates cancel Kuwait flights as Gulf tensions disrupt travel")

    def test_extended_suspensions_passes(self):
        assert self._passes("UAE airlines extend Gulf flight suspensions amid conflict")


class TestDisruptionGateStaysAviationOnly:
    """The conjunction must not turn into a business-news firehose.

    "suspends operations" was briefly a standalone keyword; it admitted mines,
    factories, banks and telcos, all of which reach this gate through the
    general feeds (Reuters, Al Jazeera). Requiring an aviation noun in the same
    text is what keeps them out.
    """

    @staticmethod
    def _passes(title: str) -> bool:
        return _matches_security_keywords(title, "") and not is_noise(title)

    def test_mine_suspension_filtered(self):
        assert not self._passes("Gold mine suspends operations after workplace accident in Ghana")

    def test_telco_suspension_filtered(self):
        assert not self._passes("Vodafone suspends service in rural areas over billing dispute")

    def test_factory_suspension_filtered(self):
        assert not self._passes("Tesla factory suspends operations for annual retooling")

    def test_rail_route_cancellation_filtered(self):
        assert not self._passes("Amtrak cancels routes amid budget shortfall")


class TestSmugglingRouteNotNoise:
    def test_sanctions_evasion_route_not_noise(self):
        # "new route" was briefly a noise filter to block fare-sale PR; it also
        # deleted smuggling and sanctions-evasion reporting, which is signal.
        assert is_noise("Russia opens new route to bypass Western sanctions on oil exports") is False

    def test_commercial_route_launch_still_noise(self):
        assert is_noise("Ryanair launches new route to Malaga with fare sale") is True
