"""The prescreen must not archive an attack reported as a NOUN rather than a verb.

The 2026-08-11 fix taught `deterministic_relevance` the verb forms ("drones ATTACKED
the refinery"). Measured again on 2026-08-13 over 7 days, 168 of 2011 prescreen-archived
events still scored 0 because their headline names the act as a bare noun carrying a
preposition or a delivery verb — a shape no noun-phrase dictionary entry matches:

    "Ukraine launched a coordinated attack on the Russian naval base in Novorossiysk"
    "Ukraine Carries Out Major Strike on Russian Naval Base in Novorossiysk"
    "Ukraine Hits Russia's Novorossiysk Port, Damaging Grain Terminals"
    "Russian Forces Seize Vodyanoe in Kharkov Region"

All four are the SAME two incidents, and every one of them was archived without an LLM
call. What makes that expensive is not the lost article but the lost corroboration: the
one Novorossiysk report that happened to use a covered phrasing paged at
system_confidence 0.51, a number built from corroborating sources — so the five drops
depressed the very signal the ALERT (0.50) and CRITICAL (0.62) gates read.

Re-measured after the fix on the same 7-day corpus: 149 of 2011 now reach the LLM
(~21/day), of which 1 is a labour dispute and 2 are political commentary.
"""

import pytest

from src.pipeline.pass_c_classify import PRESCREEN_SKIP_FLOOR, deterministic_relevance


def _prescreened(title: str, text: str = "") -> bool:
    """True when the prescreen would archive this article without an LLM call."""
    return deterministic_relevance(title, text)["score"] < PRESCREEN_SKIP_FLOOR


# Real headlines, all taken from events the prescreen archived unseen in production.
NOUN_PLUS_PREPOSITION = [
    "Photos show Russian attack on a warehouse in Kyiv - Norwalk Hour",
    "Myanmar escalating aerial attacks on civilians - The Manila Times",
    "More than 60 arrested after looting, attacks on foreign nationals in KZN - News24",
    "Number of people injured in Russian strike on Kharkiv rises, including child",
    "Arab Parliament President condemns Iranian attack on ADNOC tanker - Sharjah24",
]

DELIVERY_VERB_PLUS_ACT = [
    "Ukraine launched a coordinated attack on the Russian naval base in Novorossiysk",
    "Ukraine Carries Out Major Strike on Russian Naval Base in Novorossiysk - uatv.ua",
    "Ukrainian drones launch massive strike on Tatarstan oil refineries - Around Prague",
    "Ukrainian drones carried out deadly strikes on industrial targets in Tatarstan",
]

ARMED_SUBJECT_PLUS_VERB = [
    "Russian Forces Seize Vodyanoe in Kharkov Region as Ukraine Suffers Losses",
    "Defense Forces strike two oil refineries in Yaroslavl and Ufa - General Staff",
]

KINETIC_VERB_PLUS_ASSET = [
    "Ukraine Hits Russia's Novorossiysk Port, Damaging Grain Terminals - whalesbook.com",
    "Strategic Strike: Ukraine Hits Russian Naval Base - Devdiscourse",
    "Russian strike hits children's hospital and warehouse in Kyiv - RBC-Ukraine",
]

# The frames must not turn ordinary civilian verbs into security signal. Each of these
# carries one of the four frames' verbs or nouns and none of them is an incident; they
# are the precision controls the frames were written against, and the first two are real
# prescreen-archived headlines from the same corpus.
NOT_INCIDENTS = [
    # "captures" with a civilian subject and an abstract object.
    "Trucker captures 21-year-old pilot's death-defying maneuver during landing",
    # bare "hit" with a person as object — no armed subject, no asset.
    "Police: Boys, ages 4 and 7, allegedly took parents' car and hit woman walking dog",
    # "Captured" with no armed subject, and the incident is 60 years old.
    "Military Digest | Captured Pak Captains exposed Op Gibraltar: How India crushed 1965",
    # Commercial senses of "strike" and "target".
    "Manchester United strikes deal with new shirt sponsor",
    "Company targets net zero emissions by 2035",
    "Grocery and drugstore retailer Metro reports Q3 profit down from year ago",
]


class TestReportFramesReachTheLLM:
    @pytest.mark.parametrize("title", NOUN_PLUS_PREPOSITION + DELIVERY_VERB_PLUS_ACT
                             + ARMED_SUBJECT_PLUS_VERB + KINETIC_VERB_PLUS_ASSET)
    def test_frame_headlines_are_not_archived_unseen(self, title):
        assert not _prescreened(title)

    @pytest.mark.parametrize("title", NOT_INCIDENTS)
    def test_civilian_senses_stay_archived(self, title):
        assert _prescreened(title)

    def test_frames_score_as_ordinary_security_vocabulary(self):
        """A frame match is +25 and stays subject to the is_noise() veto, exactly like a
        bare verb — it is not promoted to has_high_signal. The metaphors live in this
        shape too ("Trump's 'Jihadist' Attack on Democrats"), and the noise veto is what
        holds them; promoting the frame would suppress that veto."""
        det = deterministic_relevance("Ukraine launched a coordinated attack on the base", "")
        assert det["has_hostile_act"] is True
        assert det["has_high_signal"] is False
        assert det["score"] == 25

    def test_noise_veto_still_wins_over_a_frame(self):
        det = deterministic_relevance("Film review: the box office attack on comedy bombed", "")
        assert det["score"] < PRESCREEN_SKIP_FLOOR

    def test_bare_act_noun_alone_is_not_enough(self):
        """The frames key on the preposition/verb, not on the noun. Without one of them
        "attack" is as likely to be a metaphor as an incident, which is why the noun was
        never added to the dictionaries on its own."""
        det = deterministic_relevance("Under attack from critics, the minister resigned", "")
        assert det["has_hostile_act"] is False


class TestLabourStrikesAreAcceptedCost:
    def test_labour_strike_reaches_the_llm(self):
        """`general_strike` is a real event type in this catalog and transport//general
        strikes do reach the SITREP, so a strike headline is security vocabulary here —
        it costs one classification call and the LLM decides. Documented rather than
        regex'd away: 1 of the 149 rescued headlines over 7 days is this shape."""
        assert not _prescreened("Union launches strike over pay at three factories")
