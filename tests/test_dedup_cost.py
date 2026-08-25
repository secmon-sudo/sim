"""Dedup hot-loop cost, added 2026-08-25.

content_dedup_cpu is 256 s/run — 30% of Pass A — measured over 11 production runs.
Profiling it on a real 800-event corpus put SequenceMatcher at 50 s of 71 s and
repeated normalization at 12 s. Two changes came out of that: difflib's own upper
bounds skip pairs no threshold could accept, and the pure text derivations are
cached because the stored side is identical across every candidate in a run.

Both are meant to be INVISIBLE: verified against HEAD on 800 candidates x 600
stored events (480K comparisons, 612 duplicates) with zero differing verdicts.
These tests pin the properties that equivalence rests on, since a future threshold
or lexicon change re-opens the question.
"""

import difflib

import pytest

from src.pipeline.ingest_filters import (
    _shingles,
    _shingles_cached,
    _word_set,
    _word_set_cached,
    find_content_duplicate,
    normalize_title,
    title_similarity,
)


class TestTitleSimilarityBound:
    def test_min_ratio_zero_returns_the_true_ratio(self):
        """The default path is untouched: callers that want the number get the number."""
        a, b = "Blast at Kabul airport kills 12", "Explosion near Kabul airport leaves 12 dead"
        expected = difflib.SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()
        assert title_similarity(a, b) == expected

    def test_bound_never_rejects_a_pair_the_matcher_would_accept(self):
        """The safety property. difflib documents real_quick_ratio() and quick_ratio()
        as >= ratio(), so gating on them can only skip pairs that were going to fail."""
        titles = [
            "Russian missile strike on Kharkiv region kills ten",
            "Russian Missile Attack on Kyiv Kills 12",
            "Drone attack halts flights at Sochi airport",
            "Sochi airport suspends flights after drone attack",
            "Cost of filming in Moscow three times lower than in LA",
            "Two killed in Baghdad market bombing",
            "Baghdad market bombing leaves two dead - Reuters",
        ]
        for threshold in (0.5, 0.65, 0.78, 0.9):
            for a in titles:
                for b in titles:
                    gated = title_similarity(a, b, threshold)
                    true_ratio = title_similarity(a, b)
                    assert (gated >= threshold) == (true_ratio >= threshold), (a, b, threshold)

    def test_short_circuit_value_is_an_upper_bound_not_the_ratio(self):
        """Documented contract: below min_ratio the return is only meaningful as
        'rejected'. Pinned so nobody starts reading it as a similarity score."""
        a, b = "Bird strike at Heathrow", "Ceasefire talks resume in Doha as mediators press both sides"
        gated = title_similarity(a, b, 0.78)
        assert gated < 0.78
        assert gated != title_similarity(a, b)

    def test_empty_after_normalization_is_zero_either_way(self):
        assert title_similarity("!!!", "Kabul blast", 0.78) == 0.0
        assert title_similarity("!!!", "Kabul blast") == 0.0


class TestDerivationCaches:
    def test_word_set_wrapper_hands_back_a_private_copy(self):
        """The cache holds a frozenset, but _word_set promises a set. Mutating what a
        caller got must not poison every later lookup of the same string."""
        title = "Drone attack halts flights at Sochi airport"
        first = _word_set(title)
        first.add("SENTINEL")
        assert "SENTINEL" not in _word_set(title)

    def test_shingles_wrapper_hands_back_a_private_copy(self):
        text = " ".join(f"word{i}" for i in range(40))
        first = _shingles(text)
        first.add("SENTINEL")
        assert "SENTINEL" not in _shingles(text)

    def test_cached_values_match_the_wrappers(self):
        text = "Explosion near Kabul airport leaves 12 dead, officials say"
        assert set(_word_set_cached(text)) == _word_set(text)
        assert set(_shingles_cached(text)) == _shingles(text)

    def test_short_text_shingles_fall_back_to_words(self):
        """Fewer words than the window: the n-gram set would be empty, so the words
        themselves stand in. Kept because the cached form rebuilt this branch."""
        assert _shingles("two words") == {"two", "words"}


class TestVerdictsUnchanged:
    @pytest.fixture
    def stored(self):
        return [
            ("Ceasefire talks resume in Doha", "Mediators pressed both sides in Doha.", ""),
            ("Russian Missile Attack on Kyiv Kills 12", "A missile struck a residential block.", "Kyiv"),
        ]

    def test_near_identical_headline_still_dedups(self, stored):
        assert find_content_duplicate(stored, "Ceasefire Talks Resume in Doha - Reuters", "") == 0

    def test_place_veto_still_survives_the_gate(self, stored):
        """The Kharkiv/Kyiv case from 2026-08-20: the char matcher scores wire
        scaffolding on its own, and only the place veto stops it. The bound must not
        change which entry the loop reaches."""
        assert find_content_duplicate(
            stored, "Russian missile strike on Kharkiv region kills ten", ""
        ) is None

    def test_unrelated_headline_is_not_a_duplicate(self, stored):
        assert find_content_duplicate(stored, "Bird strike at Heathrow", "") is None
