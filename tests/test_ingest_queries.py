"""Tests for the per-run search query set.

Two distinct jobs share one budget. Static tiers DISCOVER new incidents;
dynamic queries TRACK storylines that are already developing. Dynamic queries
run first, so every rule that limits them — the >=2 event floor, the
severity-scaled tracking window, the MAX_DYNAMIC_QUERIES cap — exists to stop
storyline tracking from eating the discovery budget and feeding itself
(search finds the same article, which re-scores the storyline, which keeps the
search alive).
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.pipeline.ingest_queries import MAX_DYNAMIC_QUERIES, build_search_queries


def _db(rows):
    """A connection whose storyline query returns `rows`.

    Row shape: (storyline_hint, last_update, max_severity, event_count)
    """
    db = MagicMock()
    db.transaction.return_value.__enter__ = lambda s: None
    db.transaction.return_value.__exit__ = lambda s, *a: False
    db.execute.return_value.fetchall.return_value = rows
    return db


def _ago(hours):
    return datetime.now(timezone.utc) - timedelta(hours=hours)


class TestStaticQueries:
    def test_returns_a_query_set_without_a_db(self):
        queries = build_search_queries(None)
        assert len(queries) > 50
        assert all(q.get("query") for q in queries)

    def test_no_duplicate_queries(self):
        # Duplicates waste a slot in the per-run cap on an identical fetch.
        queries = [q["query"].lower() for q in build_search_queries(None)]
        assert len(queries) == len(set(queries))

    def test_static_queries_are_not_marked_dynamic(self):
        assert all(not q.get("dynamic") for q in build_search_queries(None))

    def test_covers_the_core_beats(self):
        joined = " ".join(q["query"].lower() for q in build_search_queries(None))
        for beat in ("airport", "hotel", "airspace", "bomb"):
            assert beat in joined, beat


class TestDynamicQueries:
    def test_active_storyline_becomes_a_dynamic_query(self):
        queries = build_search_queries(_db([("Tehran missile strike", _ago(2), 85, 4)]))
        dynamic = [q for q in queries if q.get("dynamic")]
        assert "Tehran missile strike" in [q["query"] for q in dynamic]

    def test_dynamic_queries_run_before_static_ones(self):
        # pass_a takes the first MAX_QUERIES_PER_RUN entries, so ordering is
        # what actually guarantees tracked storylines get fetched.
        queries = build_search_queries(_db([("Tehran missile strike", _ago(2), 85, 4)]))
        assert queries[0].get("dynamic") is True

    def test_trailing_date_hint_is_stripped(self):
        # Hints carry a " Jun9"-style suffix that would pin the search to a
        # past date and stop matching today's coverage.
        queries = build_search_queries(_db([("Bandar Abbas port blast Jul22", _ago(1), 70, 3)]))
        assert "Bandar Abbas port blast" in [q["query"] for q in queries]

    def test_cap_is_enforced(self):
        rows = [(f"storyline {i}", _ago(1), 90, 5) for i in range(MAX_DYNAMIC_QUERIES + 10)]
        queries = build_search_queries(_db(rows))
        assert len([q for q in queries if q.get("dynamic")]) == MAX_DYNAMIC_QUERIES

    def test_highest_severity_wins_the_capped_slots(self):
        rows = [(f"low {i}", _ago(1), 30, 5) for i in range(MAX_DYNAMIC_QUERIES)]
        rows.append(("critical storyline", _ago(1), 95, 5))
        dynamic = [q["query"] for q in build_search_queries(_db(rows)) if q.get("dynamic")]
        assert "critical storyline" in dynamic

    def test_naive_timestamps_do_not_crash(self):
        # occurred_at_est comes back naive from some drivers; the age maths
        # would raise and wipe out every dynamic query for the run.
        queries = build_search_queries(
            _db([("Kyiv drone attack", datetime.utcnow() - timedelta(hours=2), 85, 4)]))
        assert "Kyiv drone attack" in [q["query"] for q in queries]

    def test_db_failure_degrades_to_static_only(self):
        db = MagicMock()
        db.transaction.side_effect = RuntimeError("connection lost")
        queries = build_search_queries(db)
        assert len(queries) > 50
        assert not any(q.get("dynamic") for q in queries)


class TestTrackingWindow:
    """Window scales with severity: loud storylines are tracked longer."""

    @pytest.mark.parametrize("severity,age_hours,tracked", [
        (85, 100, True),    # >=80 → 168h window
        (85, 200, False),
        (70, 48, True),     # >=60 → 72h window
        (70, 100, False),
        (30, 24, True),     # else → 36h window
        (30, 48, False),
    ])
    def test_window_by_severity(self, severity, age_hours, tracked):
        queries = build_search_queries(_db([("some storyline", _ago(age_hours), severity, 3)]))
        names = [q["query"] for q in queries]
        assert ("some storyline" in names) is tracked

    def test_stale_storyline_retires_itself(self):
        # No cron sweeps dynamic queries; ageing out of the window IS the
        # retirement mechanism.
        queries = build_search_queries(_db([("old news", _ago(500), 95, 9)]))
        assert not any(q.get("dynamic") for q in queries)


class TestRecencyOperatorApplies:
    """Today's freshness fix has to reach these queries too.

    The operator is appended centrally in fetch_rss_feed rather than baked into
    each query string, so assert on the composed URL, not on build_search_queries.
    """

    def test_dynamic_query_url_carries_the_operator(self):
        from urllib.parse import quote_plus

        from src.pipeline.ingest_sources import GOOGLE_NEWS_RSS, with_recency
        q = build_search_queries(_db([("Tehran missile strike", _ago(2), 85, 4)]))[0]
        url = GOOGLE_NEWS_RSS.format(query=quote_plus(with_recency(q["query"])))
        assert "when%3A2d" in url

    def test_every_built_query_gets_the_operator(self):
        from src.pipeline.ingest_sources import with_recency
        for q in build_search_queries(None):
            assert "when:" in with_recency(q["query"])
