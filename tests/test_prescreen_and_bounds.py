"""
Tests for Faz 0.3 (occurred_at sanity bounds) and Faz 1.1 (deterministic
relevance pre-screen) added to Pass C.
"""

from datetime import datetime, timedelta, timezone

from src.pipeline.pass_c_classify import (
    PRESCREEN_SKIP_FLOOR,
    _parse_occurred_at,
    deterministic_relevance,
)


class TestOccurredAtBounds:
    def test_old_date_rejected(self):
        # Years-old anniversary/retrospective dates must be discarded.
        assert _parse_occurred_at("2019-06-01") is None

    def test_future_date_rejected(self):
        future = (datetime.now(timezone.utc) + timedelta(days=10)).strftime("%Y-%m-%d")
        assert _parse_occurred_at(future) is None

    def test_recent_date_kept(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        assert _parse_occurred_at(recent) is not None

    def test_garbage_returns_none(self):
        assert _parse_occurred_at(None) is None
        assert _parse_occurred_at("") is None
        assert _parse_occurred_at("not a date") is None


class TestStaleYearRepair:
    """A right day with the model's own year is a repair, not a retrospective.

    Measured 2026-08-13 over the classification corpus: 110 occurred_at estimates fell
    outside the sane window and 67 (61%) were correct to within two days once the year
    was replaced with the article's. The failure is the model anchoring absolute dates
    to its training cutoff — "2024-08-12" on a piece published 2026-08-12 — and it hit
    events as real as the Novorossiysk state-of-emergency CRITICAL.
    """

    def _pub(self, days_ago=0):
        return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)

    def test_stale_year_is_repaired_to_the_article_year(self):
        pub = self._pub()
        stale = pub.replace(year=pub.year - 2).strftime("%Y-%m-%d")
        got = _parse_occurred_at(stale, pub)
        assert got is not None and got.year == pub.year
        assert abs(got - pub) <= timedelta(days=2)

    def test_repair_needs_the_article_date(self):
        """Without published_at there is nothing to repair against — old behaviour."""
        pub = self._pub()
        stale = pub.replace(year=pub.year - 2).strftime("%Y-%m-%d")
        assert _parse_occurred_at(stale) is None

    def test_genuine_retrospective_stays_discarded(self):
        """A real anniversary piece lands far from publication once year-shifted.

        "25 years later, parents of Malki Roth sue mastermind" carried 2001-08-09 on a
        2026-08-12 article: shifting the year puts it 3 days out, past the tolerance.
        """
        pub = datetime(2026, 8, 12, 18, 23)
        assert _parse_occurred_at("2001-08-09", pub) is None
        assert _parse_occurred_at("2023-10-07", datetime(2026, 8, 11, 13, 7)) is None

    def test_repair_crosses_the_new_year(self):
        """December in a January article belongs to the previous year, not this one."""
        pub = datetime.now(timezone.utc).replace(tzinfo=None)
        prev_day = pub - timedelta(days=1)
        stale = prev_day.replace(year=prev_day.year - 3).strftime("%Y-%m-%d")
        got = _parse_occurred_at(stale, pub)
        assert got is not None
        assert abs(got - pub) <= timedelta(days=2)

    def test_repair_never_outruns_the_sane_bounds(self):
        """The repaired value must still satisfy the window it was rescued from."""
        pub = self._pub()
        future = (pub + timedelta(days=40)).replace(year=pub.year - 2).strftime("%Y-%m-%d")
        got = _parse_occurred_at(future, pub)
        assert got is None or abs(got - pub) <= timedelta(days=2)

    def test_tz_aware_publication_date_is_accepted(self):
        """published_at reaches Pass C tz-aware from psycopg; naive math must not crash."""
        pub = datetime.now(timezone.utc)
        stale = pub.replace(year=pub.year - 2).strftime("%Y-%m-%d")
        assert _parse_occurred_at(stale, pub) is not None

    def test_in_bounds_dates_are_untouched(self):
        recent = (self._pub(2)).strftime("%Y-%m-%d")
        pub = self._pub()
        assert _parse_occurred_at(recent, pub) == _parse_occurred_at(recent)


class TestDeterministicRelevance:
    def test_pure_junk_scores_below_floor(self):
        # No security vocabulary at all → skipped before an LLM call is spent.
        r = deterministic_relevance(
            "Airport unveils new luxury lounge", "Premium shopping opens next month."
        )
        assert r["score"] < PRESCREEN_SKIP_FLOOR
        assert r["has_security"] is False
        assert r["has_high_signal"] is False

    def test_sports_transfer_skipped(self):
        r = deterministic_relevance(
            "Premier League transfer news: striker signs deal", "The club confirmed."
        )
        assert r["score"] < PRESCREEN_SKIP_FLOOR

    def test_high_signal_passes(self):
        r = deterministic_relevance(
            "Explosion rocks Kabul airport", "A blast hit the terminal, casualties reported."
        )
        assert r["score"] >= PRESCREEN_SKIP_FLOOR
        assert r["has_high_signal"] is True

    def test_geopolitical_kept(self):
        # Broad coverage: geopolitical terms must NOT be skipped.
        r = deterministic_relevance(
            "Iran nuclear talks resume in Geneva", "Diplomats met to discuss the framework."
        )
        assert r["score"] >= PRESCREEN_SKIP_FLOOR

    def test_no_substring_false_positive(self):
        # "Warsaw" must not trigger via "war"; with no other signal it stays low.
        r = deterministic_relevance("Warsaw summit on trade", "Leaders met to discuss tariffs.")
        assert r["has_high_signal"] is False
        assert r["score"] < PRESCREEN_SKIP_FLOOR
