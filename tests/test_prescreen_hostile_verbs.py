"""The deterministic prescreen must not archive a reported attack without an LLM call.

The prescreen exists to save tokens on articles with NO security vocabulary at all. Its
vocabularies are built from NOUN PHRASES ("drone attack", "air strike", "car bomb"),
which matches nothing in the verb construction most wire copy uses — so "Drones ATTACKED
the petrochemical center" scored 0 and was archived unseen.

Measured 2026-08-11 over 7 days: 64 of 1779 prescreen-archived events matched hostile-act
verb forms, including "A drone carrying explosives attacked a Ukrainian An-124 in
Germany" — an aviation incident, which is the pipeline's whole reason to exist.

The counter that was supposed to make this visible (pass_c.high_signal_archived) only
covers the LLM archive path, so nothing on this path was ever measured.
"""

import pytest

from src.pipeline.pass_c_classify import PRESCREEN_SKIP_FLOOR, deterministic_relevance


def _prescreened(title: str, text: str = "") -> bool:
    """True when the prescreen would archive this article without an LLM call."""
    return deterministic_relevance(title, text)["score"] < PRESCREEN_SKIP_FLOOR


# Real headlines, taken from events the prescreen archived in production.
ARCHIVED_IN_PROD = [
    "A drone carrying explosives attacked a Ukrainian An-124 in Germany - UA.NEWS",
    "The enemy attacked Kharkiv: a multi-story residential building in the Saltivskyi district",
    "Drones attacked the petrochemical center of the Russian Federation in Tatarstan",
    "ADNOC reports 15 vessels attacked while transiting Hormuz",
    "The enemy attacked Sumy with guided aerial bombs: five injured",
    "Moscow Region, Crimea, Nizhnekamsk, and Rostov Oblast Attacked by Ukrainian Drones",
]

# The reason the verb forms are scored as ordinary security vocabulary rather than as a
# high signal: bare verbs are where the metaphors live, and is_noise() already knows
# them. A verb-only match must stay subject to that veto.
METAPHORS = [
    "Movie review: the film bombed at the box office",
    "Stock market attacked by inflation fears, analysts say",
]

OFF_TOPIC = [
    "Airline reviews the new business class seat",
    "Flight simulator enthusiasts gather in Ohio",
    "Airport lounge review: is the new terminal worth it",
]


class TestHostileVerbsSurvivePrescreen:
    @pytest.mark.parametrize("title", ARCHIVED_IN_PROD)
    def test_reported_attacks_reach_the_llm(self, title):
        assert not _prescreened(title)

    @pytest.mark.parametrize("title", METAPHORS)
    def test_metaphors_stay_archived(self, title):
        """A verb-only hit does not override the noise veto — otherwise every 'the film
        bombed' costs a classification call."""
        assert _prescreened(title)

    @pytest.mark.parametrize("title", OFF_TOPIC)
    def test_off_topic_still_archived(self, title):
        """The prescreen's actual job, unchanged."""
        assert _prescreened(title)

    def test_noun_phrase_still_overrides_the_noise_veto(self):
        """The pre-existing contract: a real high-signal phrase outranks is_noise(),
        which is why it is scored higher than a bare verb."""
        det = deterministic_relevance("Car bomb explosion kills three in Baghdad market", "")
        assert det["has_high_signal"] is True
        assert det["score"] >= 45

    def test_verb_match_is_reported_separately(self):
        """The two vocabularies stay distinguishable, so the noise veto can treat them
        differently and so this is debuggable from the stored prescreen payload."""
        det = deterministic_relevance("The enemy attacked Kharkiv overnight", "")
        assert det["has_hostile_act"] is True
        assert det["has_high_signal"] is False
        assert det["has_security"] is True

    def test_verb_match_in_body_counts(self):
        """The prescreen reads title + text; a headline can be coy about the act."""
        assert not _prescreened(
            "Overnight developments in the region",
            "Officials said drones attacked the refinery shortly after midnight.",
        )
