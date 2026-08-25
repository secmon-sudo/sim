"""Why the linking pool carries every member, added 2026-08-25.

link_storylines is correct: replayed over 3 days of real events it splits 1% of
repeated hints. Production split 73%. The whole difference was the candidate pool —
it held ONE representative per storyline, the most recent member, and a storyline's
wording drifts as it grows. The member a new report matches ends up hidden behind a
newer one that words the story differently.

    pool shape            storylines   split
    production (1 rep)          938      73%
    replay, 1 rep               616      14%
    replay, 3 reps              575       8%
    replay, 10 reps             543       3%
    replay, all members         550       1%

All members costs 45% more rows than one representative over 14 days (5962 vs 4097),
not the multiple the storyline-keyed design assumed: most storylines are singletons,
so the cap saved almost nothing and hid almost everything. Linking a full run against
a 5962-row pool measures 0.43 s.
"""

from datetime import datetime, timedelta

from src.core.storyline import tokenize_storyline_hint
from src.pipeline.pass_d_score import link_storylines

BASE = datetime(2026, 8, 25, 12, 0)


def _member(storyline: str, hint: str, hours_ago: float, iso: str = "UA") -> dict:
    return {
        "id": f"{storyline}-{hours_ago}",
        "storyline_id": storyline,
        "storyline_hint": hint,
        "country_iso": iso,
        "occurred_at_est": BASE - timedelta(hours=hours_ago),
        "anchor_name_norm": None,
        "anchor_name_raw": None,
    }


def _incoming(hint: str, iso: str = "UA") -> dict:
    return {
        "id": "incoming",
        "storyline_hint": hint,
        "country_iso": iso,
        "occurred_at_est": BASE,
        "anchor_name_norm": None,
        "anchor_name_raw": None,
    }


class TestDriftedStoryline:
    def test_matching_member_behind_a_drifted_newest_still_links(self):
        """The bug, in one case. Both rows belong to storyline S; the newest words the
        story differently. With only the newest as candidate the incoming report opens
        a second storyline for the same incident."""
        pool = [
            _member("S", "gaza hospital strike casualties", hours_ago=2),
            _member("S", "gaza israeli airstrike", hours_ago=30),
        ]
        assert link_storylines(_incoming("gaza israeli airstrike"), pool) == "S"

    def test_the_drifted_newest_alone_would_not_have_linked(self):
        """Pins WHY the case above matters: this is what the old pool offered."""
        pool = [_member("S", "gaza hospital strike casualties", hours_ago=2)]
        assert link_storylines(_incoming("gaza israeli airstrike"), pool) is None

    def test_identical_hints_never_open_a_second_storyline(self):
        pool = [_member("S", "kryvyi rih russian drone strike", hours_ago=18)]
        assert link_storylines(_incoming("kryvyi rih russian drone strike"), pool) == "S"


class TestStillDiscriminating:
    def test_a_different_country_does_not_link(self):
        """Breadth must not become promiscuity: the gates are unchanged, only what
        they get to see."""
        pool = [_member("S", "gaza israeli airstrike", hours_ago=5, iso="PS")]
        assert link_storylines(_incoming("gaza israeli airstrike", iso="IL"), pool) is None

    def test_a_generic_overlap_alone_does_not_link(self):
        """'shooting' is a category, not an incident — has_discriminating_overlap still
        refuses, however many members the pool now shows."""
        pool = [_member("S", "twin falls mass shooting", hours_ago=4, iso="US")]
        assert link_storylines(_incoming("dayton mass shooting", iso="US"), pool) is None

    def test_best_match_wins_across_members_of_different_storylines(self):
        pool = [
            _member("WRONG", "kherson russian drone attack", hours_ago=3),
            _member("RIGHT", "kryvyi rih russian drone strike", hours_ago=9),
        ]
        assert link_storylines(_incoming("kryvyi rih russian drone strike"), pool) == "RIGHT"


class TestTokenizerCache:
    def test_caller_cannot_poison_the_cache(self):
        """tokenize_storyline_hint promises a mutable set; the cache holds a frozenset
        and the wrapper copies it. Linking calls this thousands of times a run."""
        first = tokenize_storyline_hint("kryvyi rih russian drone strike")
        first.add("SENTINEL")
        assert "SENTINEL" not in tokenize_storyline_hint("kryvyi rih russian drone strike")

    def test_tokens_are_unchanged(self):
        assert tokenize_storyline_hint("runway incursion CAI") == {
            "runway", "incursion", "cai", "runway incursion", "incursion cai"}
