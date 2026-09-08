"""Prescreen vocabulary gaps found by the 2026-09-07 weekly audit.

scripts/vocab_audit.py samples what each gate REJECTED and has a model judge whether
the rejection was right. The prescreen's miss rate has run 2% → 10% → 30% → 20% over
four audits, twice in a row past the 10% bar that pages, against ~2,070 rejections a
week. Six misses were reported; every one scored 0 with no flag set at all.

What they had in common was not a missing subject but a missing FORM of a word the
vocabulary already knew:

  * "closure" as a noun, never "closed" as a verb — so "Eight Indonesian airports
    closed due to Anak Krakatau volcanic ash" was invisible on the day that story
    led the SITREP and paged CRITICAL;
  * "attacked" but not "attack" — "Russian Forces Attack Ukrnafta Facilities"
    scored 0 while "Russian forces attacked Ukrnafta facilities" scored 25;
  * "attack on" and "attack against" but not "attack at" or "attack near";
  * no word at all for an aviation-security event before anyone calls it an attack
    ("the Leipzig drone incident"), or for an embassy issuing a security alert.

Measured against the same week's 2,074 prescreen-archived headlines, the additions
below reach 81 of them — about 12 extra classification calls a day on ~800.
"""

import pytest

from src.pipeline.ingest_filters import (
    _is_aviation_security_incident,
    _is_flight_disruption,
    _is_official_security_alert,
)
from src.pipeline.pass_c_classify import deterministic_relevance

PRESCREEN_FLOOR = 15


def _survives(title: str, text: str = "") -> bool:
    """True when the prescreen would send this to an LLM rather than archive it."""
    return deterministic_relevance(title, text)["score"] >= PRESCREEN_FLOOR


class TestTheSixAuditMisses:
    """The six headlines the 2026-09-07 audit judged wrongly rejected.

    Three are recovered. The other three are named here rather than quietly
    dropped, because which misses a fix does NOT close is the part that decides
    whether the next audit reads as progress or as noise.
    """

    @pytest.mark.parametrize("title", [
        "US Embassy sounds security alert in Kuwait - arabtimesonline.com",
        "Germany accused the Russian Federation of a hybrid attack at Leipzig Airport "
        "and promised to strengthen border controls for Russians",
        "Eight Indonesian airports closed due to Anak Krakatau volcanic ash, "
        "170,000 travellers affected - Telegraph India",
    ])
    def test_the_three_recoverable_ones_now_reach_the_llm(self, title):
        assert _survives(title)

    def test_a_bare_drone_incident_is_still_archived(self):
        """"Drone incident in Leipzig" carries no aviation noun, so the aviation
        conjunction cannot reach it. That is the trade: the bare class term matches
        24 archived headlines a week and 20 of them are the diplomatic aftermath of
        one event ("Trump Reacts to…", "Russia rejects UK accusation over…"), which
        the report_kind gate would then veto anyway. Requiring an aviation noun
        keeps the 8 that describe the incident.
        """
        assert not _survives("Drone incident in Leipzig: the trail leads to Russia")
        assert not _is_aviation_security_incident("Drone incident: how should the EU respond?")

    def test_the_strikes_causing_chaos_miss_was_closed_from_the_other_side(self):
        """Left open on 2026-09-07 and closed on 2026-09-08, by a different route.

        The reading then was that "Iran-Kuwait strikes trigger fresh flight chaos in
        Gulf" needed "strike" to count as an aviation security nexus, which it cannot:
        the same word is a labour dispute, and "Air France strike causes flight chaos"
        would arrive on the same path. That is still true. What reached it instead is
        _is_state_actor_strike — two named states either side of a kinetic verb — a
        frame written for a different miss entirely. Worth pinning: the anchor that
        was unsafe to guess from the aviation side was already implied by the
        geopolitics.
        """
        assert _survives("Iran-Kuwait strikes trigger fresh flight chaos in Gulf")

    def test_a_labour_dispute_causing_the_same_chaos_stays_out(self):
        """Which is what made the aviation-side anchor unsafe, and still does."""
        assert not _survives("Air France strike causes flight chaos at Charles de Gaulle")

    def test_an_untranslated_headline_is_beyond_this_vocabulary(self):
        """The Albanian miss. Pass A translates a headline whose letters are ≥30%
        non-Latin, and Albanian is Latin script, so it arrives untranslated and no
        English vocabulary can read it. A real gap, and not one more English
        patterns can close."""
        assert not _survives("Identifikohen dy të dyshuarit për sulmin me dron në Leipzig")


class TestCloseIsAVerbToo:
    """_DISRUPTION_PATTERN carried "closure" and "closures" but no form of close."""

    @pytest.mark.parametrize("title", [
        "Eight Indonesian airports closed due to Anak Krakatau volcanic ash",
        "Anak Krakatau erupts twice, shuts Jakarta airport - Daily Pioneer",
        "Volcanic ash shuts Jakarta airport, disrupts flights across Indonesia",
        "Four more Indonesian airports closed as volcanic ash disrupts flight operations",
        "Anak Krakatau ash grounds flights across Indonesia, stranding 150,000",
    ])
    def test_an_airport_that_closed_is_a_disruption(self, title):
        assert _is_flight_disruption(title, title)
        assert _survives(title)

    @pytest.mark.parametrize("title", [
        "Terminal 3 closed for renovation at Heathrow",
        "Gatwick runway closed for scheduled maintenance overnight",
    ])
    def test_a_planned_closure_is_still_archived(self, title):
        """The strict path matches these — is_noise() is what keeps them out, and
        deterministic_relevance ANDs the two. Measured over a week of headlines,
        every status: zero aviation closures mention maintenance or renovation, so
        this is the guard being pinned, not a class being traded away."""
        assert not _survives(title)


class TestAttackIsAVerbInThePresentTense:
    """"attacked" was in the vocabulary; "attack" and "attacking" were not."""

    @pytest.mark.parametrize("title", [
        "Russian Forces Attack Ukrnafta Facilities Four Times Across Three Ukrainian Regions",
        "Ukrainian Drones Attack Oil Depots in Sochi Protected by Special Structures",
        "Laftagaren forces attack Somali army positions in Baidoa again",
        "Drones Attack Sanctioned Russian Vessel Lady Maria in Mediterranean Sea",
    ])
    def test_an_armed_subject_attacking_now_is_read(self, title):
        assert _survives(title)

    def test_the_past_tense_it_always_read_is_unchanged(self):
        assert _survives("Russian forces attacked Ukrnafta facilities")

    @pytest.mark.parametrize("title", [
        "Stocks attack record highs",
        "The film bombed at the box office",
        "Workers prepare for strike at the plant",
    ])
    def test_the_metaphors_the_anchor_exists_for_stay_out(self, title):
        # Bare "attacks"/"attacking" matches 73 archived headlines in a week; with
        # the armed-subject anchor it matches 12. The anchor is the whole rule.
        assert not _survives(title)


class TestTheMissingPrepositions:
    @pytest.mark.parametrize("title", [
        "Germany accused the Russian Federation of a hybrid attack at Leipzig Airport",
        "Russia attempted attack at airport last month using explosive-laden drone",
        "Following russian attack near Kyiv, massive fire breaks out at infrastructure facility",
        "Ukrainian Railways denies report of strike near Kyiv train station",
    ])
    def test_at_and_near_are_the_same_frame_as_on(self, title):
        assert _survives(title)

    def test_strikes_at_is_deliberately_excluded(self):
        """"strike" is the one noun in this frame with a live idiom in that
        position. Over a week: "attacks at" 18, "strikes at" 1, "strikes at the
        heart of" 0 — so excluding the single word costs nothing."""
        assert not _survives("Bill strikes at the heart of the housing crisis")

    @pytest.mark.parametrize("title", [
        "Man dies of heart attack at Delhi metro station",
        "Panic attack at work: how to cope",
    ])
    def test_medical_collocations_are_scrubbed_not_matched(self, title):
        # Admitting "attack at" put these in reach for the first time. They are
        # removed the way _is_bare_security_incident removes them — by scrubbing the
        # phrase, so a real incident in the same headline still matches.
        assert not _survives(title)

    def test_a_real_incident_beside_a_medical_collocation_still_matches(self):
        assert _survives("Man dies of heart attack as drones attack the depot")


class TestOfficialSecurityAlerts:
    @pytest.mark.parametrize("title", [
        "US Embassy sounds security alert in Kuwait",
        "Seven US embassies issue security alert warnings in quick succession - Newsweek",
        "U.S. Embassy Jerusalem Issues Security Alert Over Possible Iranian Threats",
        "Israel Issues Level 4 Travel Warning for Somaliland, Citing Terror Threat",
    ])
    def test_an_issued_alert_is_an_event(self, title):
        assert _is_official_security_alert(title)
        assert _survives(title)

    @pytest.mark.parametrize("title", [
        "Malta calls for stronger EU response to maritime security threats - MaltaToday",
        "Drone Threat Shapes Germany's Security Landscape - Devdiscourse",
        "India to establish unified intelligence grid to counter multi-front security threats",
    ])
    def test_discussing_threats_is_not_issuing_one(self, title):
        # The bare noun phrase matches 19 archived headlines a week and about half
        # are this; requiring an issuer and an issuing verb narrows it to 12.
        assert not _is_official_security_alert(title)


class TestFlightChaos:
    def test_chaos_with_a_security_nexus_is_a_disruption(self):
        t = "San Diego Airport Chaos: Drones, Balloon Trigger Ground Stop"
        assert _is_flight_disruption(t, t)

    @pytest.mark.parametrize("title", [
        "Delhi on Red Alert as Heavy Rain Triggers Waterlogging, Flight Chaos",
        "Europe Flight Chaos LIVE: 2,141 Delays Hit Airports Today",
        "Delhi Rain Alert: IndiGo Warns Of Delays Amid Airport Chaos",
    ])
    def test_a_wet_tuesday_is_not(self, title):
        """"chaos" sits in the WEAK list, which requires a security nexus in the
        text. That requirement — not is_noise(), which returns False for all three
        of these — is what separates them."""
        assert not _is_flight_disruption(title, title)
