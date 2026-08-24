"""The ingest keyword gate held "killed" but not "kills", 2026-08-24.

Headlines default to the present tense — "Strike kills 16", not "16 were killed" —
so a lexicon written from participles alone rejected the most canonical event shape
this pipeline exists to catch. Found by the weekly vocab audit, which put the
keyword gate's miss rate at 7.5% for W35 against 3.3% the week before.

The second half of the defect mattered more. priority_score ranks candidates for a
per-run insert budget that binds on EVERY run (events_inserted was exactly 100 in
every measured run), so an item that clears the gate but scores 0 is not merely
ranked low — it is dropped. Both the gate and the scorer read casualty counts in
one word order only, which made two reports of the same event score 6 or 0 on
sentence construction alone.
"""

from src.pipeline.ingest_filters import _matches_security_keywords, priority_score


def gate(title: str) -> bool:
    return _matches_security_keywords(title, "")


class TestPresentTenseCasualtyRecall:
    """Present-tense casualty verbs reach the classifier."""

    def test_the_headline_the_audit_caught(self):
        assert gate("Fireball in central Gaza as Israeli strike kills child")

    def test_strike_kills_count(self):
        assert gate("Russian strike on Kyiv kills 16")

    def test_gerund_form(self):
        assert gate("Russians strike gas station in Kharkiv, injuring two people")

    def test_spelled_out_count(self):
        assert gate("US strike in eastern Pacific kills two in anti-cartel campaign")

    def test_armed_subject_anchor(self):
        assert gate("Gang raid kills 30 in Haiti")

    def test_human_object_anchor(self):
        assert gate("Terrorists attack mosque, kill worshippers during Friday prayers")

    def test_past_tense_still_passes(self):
        """The forms that already worked must keep working."""
        assert gate("Russian strike on Kyiv killed 16")
        assert gate("Two soldiers were wounded in the shelling")


class TestMetaphorsStillRejected:
    """A bare casualty verb is where the metaphors live.

    pass_c's HOSTILE_ACT_PATTERN anchors these deliberately, and it scores off the
    same lexicon — so the fix adds an anchored pattern rather than the bare verb.
    Adding "kills" to the flat term list instead let both of these through at 45
    against a prescreen floor of 15.
    """

    def test_policy_metaphor(self):
        assert not gate("New tax deal kills jobs, say analysts")

    def test_sports_metaphor(self):
        assert not gate("Manchester United kills off Chelsea comeback")

    def test_product_metaphor(self):
        assert not gate("This feature kills the competition")

    def test_weather_is_not_an_attack(self):
        assert not gate("Frost kills crops in Spain")


class TestWordOrderSymmetry:
    """The same event must score the same in either construction."""

    def test_digit_count_either_order(self):
        assert (priority_score("Russian barrage kills 14 in Kyiv region", "")
                == priority_score("14 killed in Russian barrage on Kyiv region", ""))

    def test_injury_either_order(self):
        assert (priority_score("Blast injures 18", "")
                == priority_score("18 injured in blast", ""))

    def test_spelled_out_count_scores(self):
        assert priority_score("Strike kills at least two", "") > 0

    def test_large_count_escalates_when_spelled_out(self):
        """"dozens" must clear the >= 10 escalation that "30" clears."""
        assert (priority_score("Dozens killed in airstrike", "")
                >= priority_score("Three killed in airstrike", ""))

    def test_countless_claim_still_outranks_nothing(self):
        """An anchored claim with no number stated is still an incident report."""
        assert priority_score("Israeli strike kills child in Gaza", "") > 0

    def test_metaphor_scores_zero(self):
        assert priority_score("New tax deal kills jobs, say analysts", "") == 0


class TestHebrewCasualtyParity:
    """Hebrew was the one feed language with no casualty vocabulary in the scorer.

    Its incident reports could never earn the casualty bonus, so they sat below
    their English equivalents in the budget ranking — which surfaced the moment the
    English side was strengthened: real Hebrew incidents were displaced out of the
    top 100 by the same event reported in English.
    """

    def test_hebrew_fatalities_score_like_english(self):
        assert (priority_score("דיווח: לפחות 30 הרוגים במתקפה של כנופייה בהאיטי", "")
                == priority_score("Report: at least 30 killed in gang attack in Haiti", ""))

    def test_hebrew_wounded(self):
        assert priority_score("4 פצועים בפיגוע ירי", "") > 0
