"""DB-backed tests for the storyline-linking candidate pool.

The pool used to be `ORDER BY occurred_at_est DESC LIMIT 200` over raw events, which
turned the configured 14-day window into roughly 21 hours at production volume. Measured
2026-08-11: one Ukrainian drone strike on the TANECO refinery became 14 storylines over
47 events in a day, because each run's pool no longer reached back far enough to see the
storylines the previous runs had created. That fragmentation is what printed three
contradictory casualty tolls in the RU SITREP as if they were three separate attacks.

The invariant this file pins: the pool is capped by STORYLINE, not by event count, so a
heavily-covered incident can never crowd older storylines out of its own candidate list.
Mocked db_conn cannot execute DISTINCT ON, which is precisely why these run against real
Postgres — the same gap that hid the alert_suppression upsert bug for a day.
"""

import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.pipeline.pass_d_score import _fetch_recent_events_for_linking

SMOKE_URL = os.environ.get("SIM_SMOKE_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not SMOKE_URL, reason="SIM_SMOKE_DATABASE_URL not set (CI-only DB test)"
)

if SMOKE_URL and not re.search(r"@(localhost|127\.0\.0\.1)[:/]", SMOKE_URL):
    raise RuntimeError("SIM_SMOKE_DATABASE_URL must point at localhost — refusing to run")


@pytest.fixture()
def conn():
    """A private, disposable schema holding a minimal `events` table.

    Built outside `public` so this never sees or touches the real migrated tables —
    the CI database is shared with the smoke test.
    """
    import psycopg

    schema = f"sim_link_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(SMOKE_URL) as c:
        c.execute(f"CREATE SCHEMA {schema}")
        c.execute(f"SET search_path TO {schema}")
        c.execute(
            """CREATE TABLE events (
                   id UUID PRIMARY KEY,
                   storyline_id UUID,
                   storyline_hint TEXT,
                   country_iso VARCHAR(2),
                   occurred_at_est TIMESTAMPTZ,
                   anchor_name_norm VARCHAR(16),
                   anchor_name_raw TEXT,
                   status VARCHAR(20)
               )"""
        )
        c.commit()
        try:
            yield c
        finally:
            c.rollback()
            c.execute(f"DROP SCHEMA {schema} CASCADE")
            c.commit()


def _add(conn, storyline, hint, hours_ago, status="scored", country="RU"):
    eid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO events (id, storyline_id, storyline_hint, country_iso,
                               occurred_at_est, status)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (eid, storyline, hint, country,
         datetime.now(timezone.utc) - timedelta(hours=hours_ago), status),
    )
    conn.commit()
    return eid


class TestLinkingPool:
    def test_one_representative_per_storyline(self, conn):
        s = str(uuid.uuid4())
        for i in range(5):
            _add(conn, s, f"nizhnekamsk drone attack {i}", hours_ago=i + 1)
        pool = _fetch_recent_events_for_linking(conn)
        assert len(pool) == 1

    def test_representative_is_the_most_recent_member(self, conn):
        s = str(uuid.uuid4())
        _add(conn, s, "oldest", hours_ago=40)
        _add(conn, s, "newest", hours_ago=2)
        _add(conn, s, "middle", hours_ago=20)
        pool = _fetch_recent_events_for_linking(conn)
        assert [p["storyline_hint"] for p in pool] == ["newest"]

    def test_busy_storyline_cannot_crowd_out_older_ones(self, conn):
        """The actual 2026-08-11 failure: an incident covered by many sources filled
        the pool with its own rows and hid the storylines it should have linked into."""
        loud = str(uuid.uuid4())
        for i in range(250):
            _add(conn, loud, "loud incident", hours_ago=1 + i / 1000)
        quiet = str(uuid.uuid4())
        _add(conn, quiet, "quiet older incident", hours_ago=30)

        pool = _fetch_recent_events_for_linking(conn)
        assert len(pool) == 2
        assert {p["storyline_hint"] for p in pool} == {"loud incident",
                                                       "quiet older incident"}

    def test_ordered_by_recency(self, conn):
        """cycle_common_tokens slices the head of this list, so the ordering is a
        contract, not a convenience."""
        for hours, name in ((50, "old"), (2, "new"), (25, "mid")):
            _add(conn, str(uuid.uuid4()), name, hours_ago=hours)
        pool = _fetch_recent_events_for_linking(conn)
        assert [p["storyline_hint"] for p in pool] == ["new", "mid", "old"]

    def test_window_is_honoured(self, conn):
        _add(conn, str(uuid.uuid4()), "inside window", hours_ago=13 * 24)
        _add(conn, str(uuid.uuid4()), "outside window", hours_ago=15 * 24)
        pool = _fetch_recent_events_for_linking(conn)
        assert [p["storyline_hint"] for p in pool] == ["inside window"]

    def test_unscored_and_hintless_rows_excluded(self, conn):
        _add(conn, str(uuid.uuid4()), "classified only", hours_ago=1, status="classified")
        _add(conn, str(uuid.uuid4()), None, hours_ago=1)
        _add(conn, str(uuid.uuid4()), "good", hours_ago=1)
        pool = _fetch_recent_events_for_linking(conn)
        assert [p["storyline_hint"] for p in pool] == ["good"]

    def test_reconciled_rows_included(self, conn):
        _add(conn, str(uuid.uuid4()), "reconciled", hours_ago=1, status="reconciled")
        assert len(_fetch_recent_events_for_linking(conn)) == 1

    def test_shape_matches_what_linking_reads(self, conn):
        _add(conn, str(uuid.uuid4()), "kyiv missile strike", hours_ago=1)
        row = _fetch_recent_events_for_linking(conn)[0]
        assert set(row) == {"id", "storyline_id", "storyline_hint", "country_iso",
                            "occurred_at_est", "anchor_name_norm", "anchor_name_raw"}
        assert row["country_iso"] == "RU"
