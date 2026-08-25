"""DB-backed tests for the storyline-linking candidate pool.

The pool used to be `ORDER BY occurred_at_est DESC LIMIT 200` over raw events, which
turned the configured 14-day window into roughly 21 hours at production volume. Measured
2026-08-11: one Ukrainian drone strike on the TANECO refinery became 14 storylines over
47 events in a day, because each run's pool no longer reached back far enough to see the
storylines the previous runs had created. That fragmentation is what printed three
contradictory casualty tolls in the RU SITREP as if they were three separate attacks.

The invariant this file pins: NOTHING is capped away. Every member of every storyline
inside the window is a candidate, so neither a heavily-covered incident nor a storyline's
own newest filing can hide the member a new report would have matched.

The pool was narrowed to one representative per storyline on 2026-08-11 and that turned
out to hide the very candidate that matters: a storyline's wording drifts as it grows, so
by the time it holds a dozen filings its newest member reads "gaza hospital strike
casualties" while the one a new report matches, "gaza israeli airstrike", sits behind it.
Measured 2026-08-25 over 3 days of real events, 73% of hints appearing on 2+ events were
split across storylines in production against 1% when the pool carries every member.

These run against real Postgres because a mocked db_conn cannot execute this query —
the same gap that hid the alert_suppression upsert bug for a day.
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
                   -- The pool query selects this so trusted_anchor can reject a
                   -- LOW-confidence anchor as a location key. Omitting it here made
                   -- the query fail against a real Postgres while every mocked test
                   -- still passed.
                   anchor_confidence VARCHAR(10),
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
    def test_every_member_of_a_storyline_is_a_candidate(self, conn):
        """One representative answered for the whole storyline and hid the rest."""
        s = str(uuid.uuid4())
        for i in range(5):
            _add(conn, s, f"nizhnekamsk drone attack {i}", hours_ago=i + 1)
        pool = _fetch_recent_events_for_linking(conn)
        assert len(pool) == 5

    def test_an_older_member_is_still_reachable_behind_a_drifted_newest(self, conn):
        """The regression in one case: a new report matching "oldest" must still find
        this storyline even though its most recent filing words the story differently."""
        s = str(uuid.uuid4())
        _add(conn, s, "oldest", hours_ago=40)
        _add(conn, s, "newest", hours_ago=2)
        _add(conn, s, "middle", hours_ago=20)
        pool = _fetch_recent_events_for_linking(conn)
        assert [p["storyline_hint"] for p in pool] == ["newest", "middle", "oldest"]
        assert len({p["storyline_id"] for p in pool}) == 1

    def test_busy_storyline_cannot_crowd_out_older_ones(self, conn):
        """The actual 2026-08-11 failure: an incident covered by many sources filled
        the pool with its own rows and hid the storylines it should have linked into."""
        loud = str(uuid.uuid4())
        for i in range(250):
            _add(conn, loud, "loud incident", hours_ago=1 + i / 1000)
        quiet = str(uuid.uuid4())
        _add(conn, quiet, "quiet older incident", hours_ago=30)

        pool = _fetch_recent_events_for_linking(conn)
        assert len(pool) == 251
        assert "quiet older incident" in {p["storyline_hint"] for p in pool}

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
        # anchor_confidence joined this set on 2026-08-16: should_link_storyline reaches
        # the anchor through trusted_anchor now, which cannot tell a resolved airport
        # from a bad fuzzy guess without it. A pool row missing the column reads as
        # trusted (fail-open), so leaving it out degraded linking silently.
        assert set(row) == {"id", "storyline_id", "storyline_hint", "country_iso",
                            "occurred_at_est", "anchor_name_norm", "anchor_name_raw",
                            "anchor_confidence"}
        assert row["country_iso"] == "RU"
