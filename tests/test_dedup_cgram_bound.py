"""The character-n-gram bound in front of the char-ratio matcher.

content_dedup_cpu was Pass A's largest phase — 106.5s of a 309s pass on 2026-09-07.
Profiling put 96% of it in one place: title_similarity, and inside that
find_longest_match. difflib's own cheap bounds were not doing the work they look
like they do — of 120,000 comparisons, real_quick_ratio rejected ~11k and
quick_ratio ~24k, leaving 80,925 (67%) to run the full O(n*m) matcher. That is
arithmetic rather than luck: quick_ratio bounds the ratio by the multiset of
CHARACTERS two strings share, and two English headlines share most of the alphabet.

The bound added instead looks at character ORDER, which is what the ratio measures.
Calibrated against the 837 dropped-duplicate pairs production actually recorded:

                     min on real matches    rejects of the workload
  word jaccard             0.059                    89.6%
  plural-folded words      0.125                    97.1%
  character 4-grams        0.177                    99.2%

Replayed with scripts/replay_dedup against a pre-change baseline, both modes:
identical verdicts, 98.7s -> 4.3s (events) and 136.1s -> 9.9s (pairs).
"""

import pytest

from src.pipeline.ingest_filters import (
    _TITLE_CGRAM_FLOOR,
    _TITLE_SIM_THRESHOLD,
    _char_ngrams_cached,
    _jaccard,
    _word_set_cached,
    find_content_duplicate,
    title_similarity,
)


def _cgram_jaccard(a: str, b: str) -> float:
    return _jaccard(_char_ngrams_cached(a), _char_ngrams_cached(b))


# Real pairs production merged, taken from the low end of the token-overlap
# distribution — these are the ones a word-based bound would have thrown away.
LOW_OVERLAP_REAL_DUPLICATES = [
    ("Indonesia Keeps Several Airports Closed Though Volcanic Eruption Is Easing - Kiripost",
     "Indonesia extends airport closures as volcano disruptions linger - chinadailyhk"),
    ("Volcanic eruptions strand 170,000 at Indonesian airports - upi.com",
     "Volcano eruption triggers flight suspensions at Indonesia's main airport"),
    ("Volcano eruption suspends flights at Indonesia's main airport - Balkanweb.com",
     "Mount Anak Krakatau eruption grounds eight Indonesian airports - ANTARA News"),
]


class TestTheBoundDoesNotCostARealMatch:
    @pytest.mark.parametrize("a,b", LOW_OVERLAP_REAL_DUPLICATES)
    def test_a_real_duplicate_still_clears_the_bound(self, a, b):
        assert title_similarity(a, b) >= _TITLE_SIM_THRESHOLD, "fixture is not a match"
        assert _cgram_jaccard(a, b) >= _TITLE_CGRAM_FLOOR

    @pytest.mark.parametrize("a,b", LOW_OVERLAP_REAL_DUPLICATES)
    def test_a_word_based_bound_would_have_lost_it(self, a, b):
        """Why the bound counts characters and not words. These pairs differ by
        morphology — "airports closed" against "airport closures" — which costs a
        word-set metric nearly everything and a character metric almost nothing."""
        words = _jaccard(_word_set_cached(a), _word_set_cached(b))
        assert words < 0.20
        assert _cgram_jaccard(a, b) > words

    @pytest.mark.parametrize("a,b", LOW_OVERLAP_REAL_DUPLICATES)
    def test_the_matcher_still_returns_them_as_duplicates(self, a, b):
        assert find_content_duplicate([(b, "", "")], a, "") == 0


class TestTheBoundKeepsItsMargin:
    def test_the_floor_sits_well_under_the_lowest_real_match(self):
        """0.177 is the lowest character-n-gram similarity among the 837 recorded
        duplicates. A bound calibrated to the last observed match is one that breaks
        on the next corpus, so the floor is set at 0.10 — a real match would have to
        fall 43% below anything yet seen before this cost one."""
        assert _TITLE_CGRAM_FLOOR <= 0.12
        observed_floor = min(_cgram_jaccard(a, b) for a, b in LOW_OVERLAP_REAL_DUPLICATES)
        assert _TITLE_CGRAM_FLOOR < observed_floor

    @pytest.mark.parametrize("a,b", [
        ("Germany Plans Counter-Drone Units Across Nine Cities",
         "Three killed, including a child, after Russian drone hits home in Kharkiv region"),
        ("US Embassy sounds security alert in Kuwait",
         "Anak Krakatau erupts twice, shuts Jakarta airport"),
    ])
    def test_unrelated_headlines_do_not_reach_the_matcher(self, a, b):
        # The saving is only real if ordinary pairs are rejected. Unrelated pairs sit
        # at a median of 0.008 across the production workload.
        assert _cgram_jaccard(a, b) < _TITLE_CGRAM_FLOOR


class TestJaccardIdentity:
    @pytest.mark.parametrize("a,b", [
        ({1, 2, 3}, {2, 3, 4}),
        ({1}, {1}),
        (set(), {1}),
        ({1, 2}, set()),
        ({"a", "b", "c"}, {"d"}),
    ])
    def test_the_union_free_form_is_the_same_number(self, a, b):
        expected = len(a & b) / len(a | b) if a and b else 0.0
        assert _jaccard(a, b) == expected
