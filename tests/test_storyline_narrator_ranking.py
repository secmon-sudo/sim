"""
Which storylines the narrator spends its per-run budget on — against a REAL Postgres.

The unit suite mocks db.execute() by substring-matching the SQL, so it can prove the
loop's caching but says nothing about the ORDER BY — and the ORDER BY was the bug.

Measured 2026-08-10: the ranking was `MAX(severity_score) DESC, COUNT(*) DESC`. Pass D
saturates severity at 100, so the first key tied across 267 of 353 qualifying storylines
and the order collapsed onto event_count. The ten biggest storylines of the whole 14-day
window took every slot permanently; all ten were already narrated at their current size,
so six consecutive production runs reported `generated: 0, skipped_cached: 10`. The
freshest of those ten had last seen an event three days earlier. 51 narratives had ever
been written, against 353 candidates.

Guarded like the smoke test: skipped unless SIM_SMOKE_DATABASE_URL points at localhost.
"""

import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.services.storyline_narrator import fetch_active_storylines

SMOKE_URL = os.environ.get("SIM_SMOKE_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not SMOKE_URL, reason="SIM_SMOKE_DATABASE_URL not set (CI-only, needs real Postgres)"
)

if SMOKE_URL and not re.search(r"@(localhost|127\.0\.0\.1)[:/]", SMOKE_URL):
    raise RuntimeError("SIM_SMOKE_DATABASE_URL must point at localhost — refusing to run")

NOW = datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture()
def db():
    import psycopg

    conn = psycopg.connect(SMOKE_URL, autocommit=True)
    schema = f"narrator_rank_{uuid.uuid4().hex[:8]}"
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}")
    # Only the columns the ranking query touches.
    conn.execute("""
        CREATE TABLE events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            storyline_id UUID,
            status TEXT,
            severity_score INT,
            occurred_at_est TIMESTAMP
        )""")
    conn.execute("""
        CREATE TABLE storyline_narratives (
            storyline_id UUID PRIMARY KEY,
            signature TEXT,
            event_count INT,
            updated_at TIMESTAMP DEFAULT NOW()
        )""")
    try:
        yield conn
    finally:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.close()


def _storyline(db, *, events: int, severity: int, last_activity, narrated_at: int = None):
    sid = uuid.uuid4()
    for i in range(events):
        db.execute(
            "INSERT INTO events (storyline_id, status, severity_score, occurred_at_est)"
            " VALUES (%s, 'scored', %s, %s)",
            (sid, severity, last_activity - timedelta(hours=i)),
        )
    if narrated_at is not None:
        db.execute(
            "INSERT INTO storyline_narratives (storyline_id, signature, event_count)"
            " VALUES (%s, 'sig', %s)", (sid, narrated_at))
    return str(sid)


def _ids(rows):
    return [r["storyline_id"] for r in rows]


class TestRankingPrefersActiveOverBiggest:
    def test_a_huge_stale_storyline_no_longer_blocks_a_fresh_one(self, db):
        """The exact production shape: both saturate at severity 100."""
        huge_and_done = _storyline(db, events=200, severity=100,
                                   last_activity=NOW - timedelta(days=3),
                                   narrated_at=200)
        small_and_live = _storyline(db, events=2, severity=100,
                                    last_activity=NOW - timedelta(hours=1))

        ranked = _ids(fetch_active_storylines(db))
        assert ranked.index(small_and_live) < ranked.index(huge_and_done)

    def test_never_narrated_outranks_narrated_at_the_same_size(self, db):
        narrated = _storyline(db, events=5, severity=100,
                              last_activity=NOW - timedelta(minutes=5), narrated_at=5)
        fresh = _storyline(db, events=5, severity=100,
                           last_activity=NOW - timedelta(hours=6))

        ranked = _ids(fetch_active_storylines(db))
        assert ranked.index(fresh) < ranked.index(narrated)

    def test_a_narrated_storyline_that_grew_is_reconsidered(self, db):
        """Cache hits must not be permanent — new events put it back in the queue."""
        grew = _storyline(db, events=9, severity=100,
                          last_activity=NOW - timedelta(minutes=10), narrated_at=4)
        unchanged = _storyline(db, events=9, severity=100,
                               last_activity=NOW - timedelta(minutes=5), narrated_at=9)

        ranked = _ids(fetch_active_storylines(db))
        assert ranked.index(grew) < ranked.index(unchanged)

    def test_recency_breaks_ties_within_the_stale_group(self, db):
        older = _storyline(db, events=3, severity=100, last_activity=NOW - timedelta(days=2))
        newer = _storyline(db, events=3, severity=100, last_activity=NOW - timedelta(hours=2))

        assert _ids(fetch_active_storylines(db)) == [newer, older]


class TestGatesStillApply:
    def test_below_min_events_is_excluded(self, db):
        _storyline(db, events=1, severity=100, last_activity=NOW)
        assert fetch_active_storylines(db) == []

    def test_below_min_severity_is_excluded(self, db):
        _storyline(db, events=5, severity=40, last_activity=NOW)
        assert fetch_active_storylines(db) == []

    def test_outside_the_lookback_is_excluded(self, db):
        _storyline(db, events=5, severity=100, last_activity=NOW - timedelta(days=30))
        assert fetch_active_storylines(db) == []

    def test_unscored_events_are_excluded(self, db):
        sid = uuid.uuid4()
        for _ in range(5):
            db.execute(
                "INSERT INTO events (storyline_id, status, severity_score, occurred_at_est)"
                " VALUES (%s, 'pending', 100, %s)", (sid, NOW))
        assert fetch_active_storylines(db) == []

    def test_budget_is_capped_per_run(self, db):
        from src.services.storyline_narrator import NARRATIVE_MAX_PER_RUN

        for i in range(NARRATIVE_MAX_PER_RUN + 5):
            _storyline(db, events=2, severity=100, last_activity=NOW - timedelta(hours=i))
        assert len(fetch_active_storylines(db)) == NARRATIVE_MAX_PER_RUN
