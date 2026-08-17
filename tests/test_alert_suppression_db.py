"""DB-backed tests for the alert_suppression upsert.

The rest of the suppression tests mock db_conn, which means the ON CONFLICT clause in
record_suppression is never executed — and that is exactly where the 11 Aug 2026 bug
lived: the clause refreshed expires_at but left alert_tier at the tier of the row's
FIRST claim, so suppression_blocks' one-off escalation allowance became unlimited and a
storyline re-paged at CRITICAL on every run.

Guarded the same way as the smoke test so it can never touch a production database:
skipped unless SIM_SMOKE_DATABASE_URL is set, and refused unless it points at localhost.

Run locally with e.g.:
  docker run --rm -d -p 5433:5432 -e POSTGRES_PASSWORD=pg postgres:16
  SIM_SMOKE_DATABASE_URL=postgresql://postgres:pg@localhost:5433/postgres \
      python -m pytest tests/test_alert_suppression_db.py -q
"""

import os
import re
import uuid

import pytest

from src.core.alerts import (
    active_suppression_tier,
    recent_paged_alerts,
    record_suppression,
    suppression_blocks,
)

SMOKE_URL = os.environ.get("SIM_SMOKE_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not SMOKE_URL, reason="SIM_SMOKE_DATABASE_URL not set (CI-only DB test)"
)

if SMOKE_URL and not re.search(r"@(localhost|127\.0\.0\.1)[:/]", SMOKE_URL):
    raise RuntimeError("SIM_SMOKE_DATABASE_URL must point at localhost — refusing to run")


@pytest.fixture()
def conn():
    """A connection whose search_path points at a private, disposable schema.

    Built in its own schema rather than in `public` so these tests neither see nor
    touch the real migrated tables: the CI database is shared with the smoke test, and
    an earlier version that dropped `public.events` failed outright against the real
    schema (alert_suppression's foreign key depends on it) and poisoned the whole
    session's transaction.

    Inside that schema, event_id carries no FK: these tests are about the upsert's tier
    arithmetic, and carrying the whole events schema in would couple them to it.
    """
    import psycopg

    schema = f"sim_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(SMOKE_URL) as c:
        c.execute(f"CREATE SCHEMA {schema}")
        c.execute(f"SET search_path TO {schema}")
        c.execute(
            """CREATE TABLE alert_suppression (
                   suppression_key VARCHAR(255) PRIMARY KEY,
                   first_fired_at  TIMESTAMP DEFAULT NOW(),
                   expires_at      TIMESTAMP NOT NULL,
                   alert_tier      VARCHAR(10),
                   event_id        UUID
               )"""
        )
        c.commit()
        try:
            yield c
        finally:
            c.rollback()
            c.execute(f"DROP SCHEMA {schema} CASCADE")
            c.commit()


def _evt():
    return str(uuid.uuid4())


class TestRecordSuppressionUpsert:
    def test_first_claim_stores_tier(self, conn):
        record_suppression(conn, "K", "ALERT", _evt())
        assert active_suppression_tier(conn, "K") == "ALERT"

    def test_escalation_raises_stored_tier(self, conn):
        """The 11 Aug 2026 bug: the row kept ALERT after paging at CRITICAL."""
        record_suppression(conn, "K", "ALERT", _evt())
        record_suppression(conn, "K", "CRITICAL", _evt())
        assert active_suppression_tier(conn, "K") == "CRITICAL"

    def test_escalation_is_a_one_off(self, conn):
        """The behaviour the raise exists to produce: WATCH pages, CRITICAL escalates
        once, and every later CRITICAL on the same key is muted."""
        record_suppression(conn, "K", "WATCH", _evt())
        assert suppression_blocks(conn, "K", "CRITICAL") is False

        record_suppression(conn, "K", "CRITICAL", _evt())
        assert suppression_blocks(conn, "K", "CRITICAL") is True
        assert suppression_blocks(conn, "K", "ALERT") is True

    def test_tier_is_never_lowered(self, conn):
        """Unreachable through dispatch_alert (suppression_blocks mutes the lower card
        first), asserted so a future caller cannot silently re-open the hole."""
        record_suppression(conn, "K", "CRITICAL", _evt())
        record_suppression(conn, "K", "WATCH", _evt())
        assert active_suppression_tier(conn, "K") == "CRITICAL"

    def test_event_id_follows_the_raise(self, conn):
        """The row should point at the event that set its current tier, not the first
        one to claim the key."""
        first, escalating, lower = _evt(), _evt(), _evt()
        record_suppression(conn, "K", "ALERT", first)
        record_suppression(conn, "K", "CRITICAL", escalating)
        assert str(_stored_event_id(conn, "K")) == escalating

        record_suppression(conn, "K", "WATCH", lower)
        assert str(_stored_event_id(conn, "K")) == escalating

    def test_ttl_is_refreshed_on_every_claim(self, conn):
        """Independent of the tier arithmetic — a re-claim extends the mute window."""
        record_suppression(conn, "K", "ALERT", _evt(), ttl_hours=1)
        short = _stored_expiry(conn, "K")
        record_suppression(conn, "K", "CRITICAL", _evt(), ttl_hours=8)
        assert _stored_expiry(conn, "K") > short

    def test_unrecognised_stored_tier_is_replaced(self, conn):
        """A legacy/garbled tier sorts below WATCH (as in tier_rank), so a real tier
        overwrites it rather than being held down by it."""
        conn.execute(
            """INSERT INTO alert_suppression (suppression_key, alert_tier, expires_at)
               VALUES ('K', 'bogus', NOW() + INTERVAL '4 hours')"""
        )
        conn.commit()
        record_suppression(conn, "K", "WATCH", _evt())
        assert active_suppression_tier(conn, "K") == "WATCH"


@pytest.fixture()
def conn_with_events(conn):
    """The same schema plus a minimal `events` stand-in for the candidate-fetch join.

    Dropped with the schema by the `conn` fixture, so nothing here can outlive the test.
    """
    conn.execute(
        """CREATE TABLE events (
               id UUID PRIMARY KEY,
               country_iso VARCHAR(2),
               source_title TEXT,
               storyline_hint TEXT,
               anchor_name_raw TEXT,
               anchor_name_norm VARCHAR(16),
               -- recent_paged_alerts selects and groups by this; see the note in
               -- test_linking_pool_db for why its absence only showed up in CI.
               anchor_confidence VARCHAR(10)
           )"""
    )
    conn.commit()
    return conn


def _add_event(conn, country="CO", title="quake", hint="h", loc="western Colombia"):
    eid = _evt()
    conn.execute(
        """INSERT INTO events (id, country_iso, source_title, storyline_hint,
                               anchor_name_raw)
           VALUES (%s, %s, %s, %s, %s)""",
        (eid, country, title, hint, loc),
    )
    conn.commit()
    return eid


class TestRecentPagedAlerts:
    def test_returns_live_claims_for_the_country(self, conn_with_events):
        eid = _add_event(conn_with_events, title="Colombia earthquake")
        record_suppression(conn_with_events, "K", "ALERT", eid)

        got = recent_paged_alerts(conn_with_events, "CO", None)
        assert [c["id"] for c in got] == [eid]
        assert got[0]["alert_tier"] == "ALERT"
        assert got[0]["source_title"] == "Colombia earthquake"

    def test_other_countries_excluded(self, conn_with_events):
        record_suppression(conn_with_events, "K", "ALERT",
                           _add_event(conn_with_events, country="UA"))
        assert recent_paged_alerts(conn_with_events, "CO", None) == []

    def test_expired_claims_excluded(self, conn_with_events):
        eid = _add_event(conn_with_events)
        conn_with_events.execute(
            """INSERT INTO alert_suppression (suppression_key, alert_tier, event_id,
                                              expires_at)
               VALUES ('K', 'ALERT', %s, NOW() - INTERVAL '1 hour')""",
            (eid,),
        )
        conn_with_events.commit()
        assert recent_paged_alerts(conn_with_events, "CO", None) == []

    def test_self_is_excluded(self, conn_with_events):
        eid = _add_event(conn_with_events)
        record_suppression(conn_with_events, "K", "ALERT", eid)
        assert recent_paged_alerts(conn_with_events, "CO", eid) == []

    def test_one_row_per_event_at_its_highest_tier(self, conn_with_events):
        """An event claims both a primary and a geo key, so it holds two rows. It must
        reach the model once, and carry the higher tier — 'CRITICAL' sorts below 'WATCH'
        alphabetically, which is why the query ranks tiers explicitly."""
        eid = _add_event(conn_with_events)
        record_suppression(conn_with_events, "primary", "WATCH", eid)
        record_suppression(conn_with_events, "geofp", "CRITICAL", eid)

        got = recent_paged_alerts(conn_with_events, "CO", None)
        assert len(got) == 1
        assert got[0]["alert_tier"] == "CRITICAL"

    def test_limit_is_honoured(self, conn_with_events):
        for i in range(5):
            record_suppression(conn_with_events, f"K{i}", "ALERT",
                               _add_event(conn_with_events))
        assert len(recent_paged_alerts(conn_with_events, "CO", None, limit=3)) == 3

    def test_countryless_event_sees_every_country(self, conn_with_events):
        """6% of tiered events carry no country_iso. Scoping their candidate list to a
        country they do not have skipped the duplicate check for them entirely; the
        model still has to affirm a match, so a wider list only costs prompt space."""
        record_suppression(conn_with_events, "K1", "ALERT",
                           _add_event(conn_with_events, country="UA"))
        record_suppression(conn_with_events, "K2", "ALERT",
                           _add_event(conn_with_events, country="CO"))
        assert len(recent_paged_alerts(conn_with_events, None, None)) == 2

    def test_country_scoping_still_applies_when_known(self, conn_with_events):
        record_suppression(conn_with_events, "K1", "ALERT",
                           _add_event(conn_with_events, country="UA"))
        record_suppression(conn_with_events, "K2", "ALERT",
                           _add_event(conn_with_events, country="CO"))
        got = recent_paged_alerts(conn_with_events, "CO", None)
        assert len(got) == 1


def _stored_event_id(conn, key):
    return conn.execute(
        "SELECT event_id FROM alert_suppression WHERE suppression_key = %s", (key,)
    ).fetchone()[0]


def _stored_expiry(conn, key):
    return conn.execute(
        "SELECT expires_at FROM alert_suppression WHERE suppression_key = %s", (key,)
    ).fetchone()[0]
