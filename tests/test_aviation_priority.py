"""
Tests for Faz 1.2 (military-bypass canceller) and Faz 1.3 (aviation-nexus bonus).
"""

from src.pipeline.ingest_filters import (
    _is_aviation_security_incident,
    _is_bare_security_incident,
    _is_flight_disruption,
    _is_screening_breach,
    priority_score,
)
from src.pipeline.pass_a_ingest import _matches_security_keywords, is_noise
from src.pipeline.pass_c_classify import (
    PRESCREEN_SKIP_FLOOR,
    deterministic_relevance,
)
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


class TestFlightDisruptionSurvivesPassC:
    """Fix A — a genuine flight disruption is never dropped in Pass C.

    Real cases (measured 2026-07-24) scored 0 on the relevance heuristic and had
    no other security keyword, so they were prescreen-archived before the LLM or
    LLM-archived at sev 0 — losing "which carrier stopped flying where", the
    highest-value line in an aviation SITREP. deterministic_relevance now flags
    them and lifts the score above the prescreen floor; weather stays filtered.
    """

    def test_carrier_suspension_flagged_and_survives_prescreen(self):
        det = deterministic_relevance(
            "Qatar Airways temporarily suspends passenger flights to Bahrain, Erbil and Kuwait", "")
        assert det["has_flight_disruption"] is True
        assert det["score"] >= PRESCREEN_SKIP_FLOOR  # would NOT be prescreen-archived

    def test_conflict_cancellation_flagged(self):
        det = deterministic_relevance("Several flights cancelled due to Iranian aggression", "")
        assert det["has_flight_disruption"] is True
        assert det["score"] >= PRESCREEN_SKIP_FLOOR

    def test_disruption_verb_in_body_only(self):
        det = deterministic_relevance("Dubai flight update", "All flights cancelled after strikes")
        assert det["has_flight_disruption"] is True

    def test_weather_cancellation_not_flagged(self):
        # Safety, not security — must remain droppable (is_noise catches it).
        det = deterministic_relevance("Delta cancels flights due to snowstorm in Chicago", "")
        assert det["has_flight_disruption"] is False

    def test_winter_storm_not_flagged(self):
        det = deterministic_relevance("United cancels flights after winter storm hits Denver", "")
        assert det["has_flight_disruption"] is False

    def test_non_aviation_not_flagged(self):
        det = deterministic_relevance("Gold mine suspends operations after workplace accident", "")
        assert det["has_flight_disruption"] is False


class TestWeakDisruptionVerbsNeedHeadlineAndNexus:
    """The strict vocabulary is noun-shaped: it has "disruption" but not
    "disrupted", and no "delayed", "diverted" or "stranded" at all — so "Flights
    diverted after Manchester Airport security breach" was not a disruption to
    this pipeline at all.

    Measured 2026-08-23 over 7 days: 660 events carry an aviation noun, the strict
    gate claims 120, and these verbs sit in another 77. Letting them in over the
    whole article is 45% junk — a war roundup names an airport in one paragraph
    and a delay in another. Requiring the verb and the noun to share the HEADLINE,
    plus a security nexus anywhere in the article, turns that into 19 additions,
    all real: the Manchester airfield breach (5 filings), the Houston Hobby bomb
    threat (3), Moscow's airports closing under drone attack (4), Moldovan
    airspace closed by a cruise missile, an unauthorised aircraft at Fort
    Lauderdale.
    """

    def test_security_caused_diversion_passes(self):
        t = "Flights diverted after Manchester Airport security breach"
        assert _is_flight_disruption(t, t)

    def test_bomb_threat_delay_passes(self):
        t = "Southwest Airlines Bomb Threat at Houston Hobby Airport Triggers Tarmac Delays"
        assert _is_flight_disruption(t, t)

    def test_drone_attack_closing_airports_passes(self):
        t = "New wave of drones in Russia: Airports closed"
        assert _is_flight_disruption(t, t)

    def test_technical_snag_is_not_a_disruption(self):
        t = "Indigo Delhi-Mangaluru Flight Delayed By 3 Hours Due To Technical Snag"
        assert not _is_flight_disruption(t, t)

    def test_wildlife_delay_is_not_a_disruption(self):
        t = "Endangered bearded vulture delays flight at Crete's Heraklion Airport"
        assert not _is_flight_disruption(t, t)

    def test_roundup_mentioning_both_in_the_body_is_not(self):
        """The whole point of the headline scope: co-occurrence across 4,000
        characters of war roundup says nothing about aviation."""
        title = "War in Ukraine: latest news. Missiles strike Kyiv: 15 dead"
        body = (title + " Elsewhere an airport reopened. A train was delayed. "
                "Drone attacks continued overnight.")
        assert not _is_flight_disruption(body, title)

    def test_strict_path_is_unchanged(self):
        t = "Emirates suspends all flights to Tehran amid strikes"
        assert _is_flight_disruption(t, t)
        assert _is_flight_disruption(t)

    def test_caller_without_a_headline_keeps_old_behaviour(self):
        t = "Flights diverted after Manchester Airport security breach"
        assert not _is_flight_disruption(t)


class TestScreeningBreachGate:
    """A prohibited item carried through aviation screening (added 2026-08-27).

    The class was invisible to all four gates: the general-feed admission filter,
    the ingest priority scorer, the prescreen, and the keyword lexicon.
    """

    HEADLINE = ("Businessman flies to Delhi with 31 live rounds after passing "
                "through Dhaka airport security")

    def test_detects_the_headline_that_prompted_the_gate(self):
        assert _is_screening_breach(self.HEADLINE) is True

    def test_reaches_the_general_feed_admission_filter(self):
        # Returned False before the conjunction was added, so on a general RSS feed
        # the item was rejected before ranking and left no row to notice.
        assert _matches_security_keywords(self.HEADLINE, "") is True

    def test_outranks_the_binding_insert_budget(self):
        # Scored 0 before. The budget always binds and the highest DROPPED item in
        # run #1740 scored 3, so anything at or below 3 is not reliably ingested.
        assert priority_score(self.HEADLINE, "") > 3

    def test_survives_the_prescreen(self):
        det = deterministic_relevance(self.HEADLINE, "")
        assert det["has_screening_breach"] is True
        assert det["has_security"] is True
        assert det["score"] >= PRESCREEN_SKIP_FLOOR

    def test_variants_of_the_same_incident(self):
        for title in (
            "Live bullet found aboard United flight",
            "Passenger's pistol accidentally fires at Varanasi Airport: 2 injured",
            "Bangladeshi trader carried ammunition on Delhi flight, probe ordered",
            "Dhaka airport security lapse: passenger boarded with live cartridges",
        ):
            assert _is_screening_breach(title) is True, title

    def test_conjunction_needs_all_three_vocabularies(self):
        # Item + aviation alone matched 46 headlines in 14 days, 36 of them off-class.
        assert _is_screening_breach("Ammunition depot struck by Ukrainian drone") is False
        assert _is_screening_breach("Airport expansion project gets $2bn funding") is False

    def test_engine_fan_blade_is_not_a_prohibited_item(self):
        # Why "blade" is deliberately absent from the item vocabulary.
        assert _is_screening_breach(
            "Passenger Partially Sucked Out of Plane After Fan Blade Shatters Window"
        ) is False

    def test_military_cargo_is_not_a_screening_breach(self):
        assert _is_screening_breach(
            "Turkey-Pakistan Military Flights: Is Ankara Sending Fresh Weapons to Islamabad"
        ) is False


class TestAviationSecurityIncidentGate:
    """Bomb threat, runway incursion, drone sighting, GNSS jamming, laser, stowaway.

    Added 2026-08-27. Measured over 15,429 headlines: 49 matches, all on-class, of
    which 10 were prescreen-archived at score 0 and the other 39 all scored priority
    1 — the median inserted item, i.e. first to be dropped whenever the budget binds.
    """

    ARCHIVED = (
        "African stowaway found frozen to death in plane's wheel compartment at Gatwick Airport",
        "ATSB Probes Third Runway Near Miss at Sydney Airport",
        "Police Didn't Notify Public Of G7 Bomb Scare At Calgary Airport",
        "Qantas Jets Avoids Collision at Sydney Airport Marks 4th Near Miss Incident",
    )

    def test_recovers_headlines_the_prescreen_had_archived(self):
        for title in self.ARCHIVED:
            det = deterministic_relevance(title, "")
            assert det["has_aviation_incident"] is True, title
            assert det["score"] >= PRESCREEN_SKIP_FLOOR, title

    def test_reaches_the_general_feed_admission_filter(self):
        for title in self.ARCHIVED:
            assert _matches_security_keywords(title, "") is True, title

    def test_outranks_the_binding_insert_budget(self):
        # The 39 that already passed all scored exactly 1 before this gate existed.
        for title in self.ARCHIVED:
            assert priority_score(title, "") > 3, title

    def test_covers_the_classes_that_never_reached_the_feed(self):
        for title in (
            "GPS jamming disrupts navigation for flights over the Baltic",
            "Drone sighting halts departures at Gatwick Airport",
            "Laser strike blinds pilot on approach, aircraft diverted",
            "Man attempts to breach cockpit door mid-flight on Delta aircraft",
        ):
            assert _is_aviation_security_incident(title) is True, title

    def test_needs_an_aviation_noun(self):
        # The class term alone is not enough: bomb threats and near misses happen
        # everywhere, and this pipeline's scope here is the aviation ones.
        assert _is_aviation_security_incident("Bomb threat closes city hall") is False
        assert _is_aviation_security_incident("Near miss as trains pass on same track") is False

    def test_unrelated_security_news_does_not_match(self):
        assert _is_aviation_security_incident("Gun violence in Chicago claims three lives") is False


class TestBareSecurityNounGate:
    """The bare security noun reporting harm (added 2026-08-31).

    Found by the weekly vocabulary audit, not by accident: the prescreen's judged miss
    rate went 10% -> 30% against 2267 weekly rejections. Every one of the 2273 events
    the prescreen archived in seven days scored exactly 0, and 102 of them name a
    security noun beside something killed, wounded or destroyed.
    """

    def test_recovers_the_headlines_the_prescreen_archived(self):
        for title in (
            "Ukraine Drones Attack Ozon Logistics Hubs, Killing Two Children in Dagestan",
            "Infant food warehouse in Gaza destroyed by Israeli strike",
            "Drone Hits Train in Kharkiv Region, Killing Passenger",
            "Russian attack damages Nova Poshta warehouses in Kyiv Oblast",
            "Four Palestinians martyred in Israeli attacks, three more succumb to wounds",
        ):
            assert _is_bare_security_incident(title) is True, title
            det = deterministic_relevance(title, "")
            assert det["has_bare_incident"] is True, title
            assert det["score"] >= PRESCREEN_SKIP_FLOOR, title

    def test_bare_noun_without_harm_is_not_enough(self):
        # 290 archived headlines carry a security noun and no harm clause. They stay
        # archived: the harm clause is what separates the military sense from the rest.
        assert _is_bare_security_incident("Russia and Ukraine discuss drone technology") is False
        assert _is_bare_security_incident("Defence firm unveils new attack drone design") is False

    def test_civilian_senses_stay_out(self):
        assert _is_bare_security_incident("Man survives heart attack after collapsing") is False
        assert _is_bare_security_incident("Panic attack left her injured in a fall") is False

    def test_lone_strike_with_labour_vocabulary_is_a_dispute(self):
        assert _is_bare_security_incident("Kenya Nurses Strike Hits Day 29") is False
        assert _is_bare_security_incident(
            "Bank Holiday rail chaos to hit thousands as new strike announced") is False

    def test_labour_veto_does_not_fire_when_another_noun_is_present(self):
        # "pilots" and "rail" collide with military reporting, so the veto applies only
        # when "strike" is the ONLY security noun in the headline.
        assert _is_bare_security_incident(
            "MiG-29 pilots destroyed command post for Russian drone operators") is True

    def test_nature_strikes_are_excluded(self):
        assert _is_bare_security_incident(
            "Bird Strikes: What Happens When a Bird Hits a Jet Engine") is False
        assert _is_bare_security_incident(
            "Lightning strikes near man, 11 flights diverted, extensive property damage") is False

