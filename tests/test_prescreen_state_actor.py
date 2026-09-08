"""A named state, a kinetic verb, and somebody on the receiving end.

HOSTILE_ACT_PATTERN has an armed-subject frame (forces, troops, jets) and an
asset-object frame written, in its own comment, for "the headlines whose subject is
a bare country name no subject list can hold". The asset frame solves that from the
object side, which works only when the object is one of the listed assets — a naval
base, a refinery, an airport. Most of the time it is not.

Measured 2026-09-08 over seven days of production: 174 of 2074 prescreen-archived
headlines put a state actor in front of a kinetic verb, and 15 of 16 sampled still
scored ZERO after the 2026-09-07 vocabulary work. An Iran-US missile exchange and
strikes on Kyiv, archived without an LLM ever reading them.
"""

import pytest

from src.pipeline.ingest_filters import _is_state_actor_strike
from src.pipeline.pass_c_classify import deterministic_relevance

PRESCREEN_FLOOR = 15


def _survives(title: str) -> bool:
    return deterministic_relevance(title, "")["score"] >= PRESCREEN_FLOOR


class TestTheClassItWasWrittenFor:
    @pytest.mark.parametrize("title", [
        "Iran attacks US bases in Kuwait, UAE - The Daily Star",
        "US strikes Iran again, blasts heard in Bandar Abbas, Chabahar",
        "Russia attacks Kiev airport ahead of US special envoy's trip",
        "Russia targets security chief's office in daylight Kyiv strike: Zelensky",
        "US pounds Iran, Tehran strikes back at bases in biggest exchange since July",
        "Russia said to strike Coca-Cola factory near Kyiv",
        "Russia hits Ukraine's Odesa overnight, damaging high-rise and schools",
        "Iran Hit US Base In Jordan? New Satellite Images Reveal Damage",
        "Iran Targets Energy Facilities Across Gulf After Israel Struck its Key Gas Installations",
    ])
    def test_it_now_reaches_the_llm(self, title):
        assert _survives(title)


class TestTheConjunctionIsTheRule:
    @pytest.mark.parametrize("title", [
        "China targets 5% growth for next year",
        "India hits record high on foreign inflows",
        "Russia targets inflation of 4% by 2027",
        "Turkey targets tourism revenue of $60 billion",
    ])
    def test_one_country_and_an_economic_object_is_not_a_strike(self, title):
        """A negative lookahead on the object was tried first and is too fragile —
        "targets 5% growth" puts a number between the verb and the noun, and "shares
        hit new low" puts an adjective there. Two named parties is the honest rule:
        a strike has someone on the receiving end and a forecast does not."""
        assert not _is_state_actor_strike(title)

    @pytest.mark.parametrize("title", [
        "US strikes trade deal with Japan",
        "China and India strike a border agreement",
        "US and UK hit a tariffs accord",
    ])
    def test_two_countries_over_an_economic_object_is_still_not_a_strike(self, title):
        # Two parties alone would admit these, which is why the economic objects are
        # vetoed anywhere in the headline rather than only beside the verb.
        assert not _is_state_actor_strike(title)

    def test_a_single_named_party_is_not_enough(self):
        assert not _is_state_actor_strike("Russia strikes again")


class TestAdjectivesAreNotActors:
    """"Russian attacks" is a noun phrase, not a state acting.

    Admitting nationality adjectives on the subject side turned "Ukraine's churches
    sustain aid amid Russian attacks" into a hostile act. A state in subject position
    writes itself as a noun.
    """

    @pytest.mark.parametrize("title", [
        "Ukraine's churches sustain aid amid Russian attacks - Mission Network News",
        "Ukrainian Fashion Week Pushes On as Russian Attacks Intensify",
    ])
    def test_a_nationality_adjective_does_not_make_a_strike(self, title):
        assert not _is_state_actor_strike(title)

    def test_the_adjective_still_counts_as_the_second_party(self):
        # It is dropped from the ACTOR side only; naming Ukraine is still naming a
        # party to the incident.
        assert _is_state_actor_strike("Russia hits Ukrainian ports")


class TestItClearsTheFloorOnItsOwn:
    def test_a_state_strike_headline_carries_no_other_vocabulary(self):
        """15 of 16 sampled scored exactly 0 before this, which is why the flag adds
        its own score rather than relying on a keyword to be present too."""
        title = "Iran attacks US bases in Kuwait, UAE"
        d = deterministic_relevance(title, "")
        assert d["has_state_strike"] is True
        assert d["score"] >= PRESCREEN_FLOOR
