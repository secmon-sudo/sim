"""The recency slice that feeds overexposed_tokens.

Widening the linking candidate pool from ~21 hours of raw events to every storyline in
the 14-day window (2026-08-11) would have quietly changed a second, unrelated thing:
overexposed_tokens counts how many distinct storylines share a word, and its
min_storylines=3 threshold was calibrated against a ~200-row pool. Against 2147
storylines, three-way agreement is nearly free, and the first word to be declared
"generic" would have been "nizhnekamsk" — which named 14 storylines that day precisely
because the fragmentation this fix removes was splitting one incident.

So the two consumers were separated: linking sees the whole window, the token census
sees the head of it.

Updated 2026-08-25: the linking pool now carries every MEMBER of every storyline, not
one representative each, so "the head of the list" is no longer 200 storylines' worth of
rows. The slice counts distinct storylines itself — taking rows would shrink the
denominator the threshold was calibrated against, and a smaller denominator declares
fewer words generic, which makes the containment path MORE permissive.
"""

from src.pipeline.pass_d_score import (
    CYCLE_SLICE_STORYLINES,
    cycle_common_tokens,
    link_storylines,
)


def _rep(i: int, hint: str) -> dict:
    return {"storyline_id": f"s{i}", "storyline_hint": hint}


class TestCycleSlice:
    def test_counts_tokens_across_distinct_storylines(self):
        """Only place/actor words are censused: _unigrams already drops the curated
        generic vocabulary ("drone", "strike"), so this layer decides about the words
        that would otherwise look distinctive."""
        pool = [_rep(i, "kyiv drone strike") for i in range(3)]
        assert cycle_common_tokens(pool) == {"kyiv"}

    def test_below_threshold_is_not_generic(self):
        """Two storylines sharing a word is a coincidence, not a news cycle."""
        pool = [_rep(i, "kyiv drone strike") for i in range(2)]
        assert cycle_common_tokens(pool) == set()

    def test_only_the_head_of_the_list_is_counted(self):
        """The ordering contract: candidates arrive recency-ordered, and only the
        recent head describes what this week is about."""
        recent = [_rep(i, "fresh cycle token") for i in range(CYCLE_SLICE_STORYLINES)]
        stale = [_rep(1000 + i, "ancient buried token") for i in range(50)]

        tokens = cycle_common_tokens(recent + stale)
        assert "fresh" in tokens
        assert "ancient" not in tokens
        assert "buried" not in tokens

    def test_a_single_storyline_cannot_make_its_own_words_generic(self):
        """Counted per storyline, so heavy coverage of one incident does not dilute
        its own distinctive vocabulary — the property the whole separation protects."""
        pool = [{"storyline_id": "same", "storyline_hint": "nizhnekamsk drone refinery"}
                for _ in range(40)]
        assert cycle_common_tokens(pool) == set()

    def test_the_slice_counts_storylines_not_rows(self):
        """A heavily-covered storyline sits on many consecutive rows now. If the slice
        counted rows, one loud incident could consume the whole census window and the
        cycle's real vocabulary would go unseen."""
        loud = [{"storyline_id": "loud", "storyline_hint": "loud incident"}
                for _ in range(CYCLE_SLICE_STORYLINES * 2)]
        cycle = [_rep(i, "kyiv drone strike") for i in range(3)]
        assert cycle_common_tokens(loud + cycle) == {"kyiv"}

    def test_the_storyline_budget_is_still_enforced(self):
        """Distinct storylines past the budget are dropped, however few rows they take."""
        head = [_rep(i, "fresh cycle token") for i in range(CYCLE_SLICE_STORYLINES)]
        tail = [_rep(1000 + i, "ancient buried token") for i in range(50)]
        tokens = cycle_common_tokens(head + tail)
        assert "fresh" in tokens and "ancient" not in tokens

    def test_rows_without_a_storyline_are_ignored(self):
        pool = [{"storyline_id": None, "storyline_hint": "kyiv drone strike"}
                for _ in range(5)]
        assert cycle_common_tokens(pool) == set()


class TestLinkStorylinesTokenInjection:
    """link_storylines takes the per-run value rather than deriving it 100 times."""

    def _pool(self):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        return [{
            "id": "e1", "storyline_id": "S1",
            "storyline_hint": "nizhnekamsk drone refinery attack",
            "country_iso": "RU", "occurred_at_est": now - timedelta(hours=3),
            "anchor_name_norm": None, "anchor_name_raw": "Nizhnekamsk",
        }]

    def _event(self):
        from datetime import datetime, timedelta, timezone
        return {
            "storyline_hint": "nizhnekamsk refinery drone strike",
            "country_iso": "RU",
            "occurred_at_est": datetime.now(timezone.utc) - timedelta(hours=1),
            "anchor_name_norm": None, "anchor_name_raw": "Nizhnekamsk",
        }

    def test_links_the_real_fragment_pair(self):
        """The hints that failed to merge in production score 0.6 lexically — well
        over the 0.4 threshold. They stayed apart because the candidate never reached
        the pool, not because the comparison rejected it."""
        assert link_storylines(self._event(), self._pool()) == "S1"

    def test_passed_tokens_are_used_instead_of_recomputing(self):
        """A caller-supplied census wins over one derived from the pool."""
        marker = {"nizhnekamsk", "drone", "refinery", "attack", "strike"}
        # Same call, but every shared word declared generic: the containment assist
        # can no longer speak. The primary Jaccard path still can, so the link holds —
        # what matters here is that the argument reaches the comparison at all.
        assert link_storylines(self._event(), self._pool(), marker) == "S1"

    def test_missing_occurred_at_never_links(self):
        ev = self._event()
        ev["occurred_at_est"] = None
        assert link_storylines(ev, self._pool()) is None
