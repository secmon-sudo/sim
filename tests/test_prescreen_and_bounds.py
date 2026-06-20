"""
Tests for Faz 0.3 (occurred_at sanity bounds) and Faz 1.1 (deterministic
relevance pre-screen) added to Pass C.
"""

from datetime import datetime, timedelta

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
        future = (datetime.utcnow() + timedelta(days=10)).strftime("%Y-%m-%d")
        assert _parse_occurred_at(future) is None

    def test_recent_date_kept(self):
        recent = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
        assert _parse_occurred_at(recent) is not None

    def test_garbage_returns_none(self):
        assert _parse_occurred_at(None) is None
        assert _parse_occurred_at("") is None
        assert _parse_occurred_at("not a date") is None


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
