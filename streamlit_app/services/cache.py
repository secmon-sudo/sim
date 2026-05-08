"""
SIM — Streamlit Cache Service
Blueprint V20.1 §5.1

@st.cache_data wrappers with TTL for all dashboard queries.
Uses connection ID hash to enable proper cache hits.
"""

import json
from datetime import datetime

import streamlit as st


def _conn_id(conn) -> str:
    """Return a stable identifier for the connection to enable cache hits.
    Without this, passing the conn object directly causes cache misses
    every rerun since the object reference changes."""
    return str(id(conn))


def _safe_execute(conn, sql, params=None):
    """Execute SQL with automatic recovery from aborted transactions."""
    try:
        return conn.execute(sql, params)
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return conn.execute(sql, params)


@st.cache_data(ttl=60)
def get_recent_events(_conn_key: str, _db_conn, limit: int = 100) -> list[dict]:
    """Fetch recent events for the main table — excludes heavy text columns for performance."""
    from psycopg.rows import dict_row

    with _db_conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT id, source_title, event_type, alert_tier,
                      severity_score, system_confidence,
                      anchor_name_norm, anchor_confidence,
                      country_iso, latitude, longitude,
                      storyline_id, time_certainty,
                      occurred_at_est, ingested_at, status,
                      llm_provider, llm_model,
                      source_domain, source_url
               FROM events
               WHERE status IN ('classified', 'scored', 'reconciled')
               ORDER BY ingested_at DESC
               LIMIT %s""",
            (limit,),
        )
        return cur.fetchall()


@st.cache_data(ttl=60)
def get_event_detail(_conn_key: str, _db_conn, event_id: str) -> dict | None:
    """Fetch full event detail including text — only called on demand."""
    from psycopg.rows import dict_row

    with _db_conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT canonical_text, raw_text FROM events WHERE id = %s",
            (event_id,),
        )
        return cur.fetchone()


@st.cache_data(ttl=60)
def get_alert_events(_conn_key: str, _db_conn, hours: int = 24) -> list[dict]:
    """Fetch events with alert tiers OR high severity from the last N hours."""
    from psycopg.rows import dict_row

    with _db_conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT id, source_title, event_type, alert_tier,
                      severity_score, system_confidence,
                      anchor_name_norm, country_iso,
                      occurred_at_est, ingested_at,
                      source_url, source_domain,
                      canonical_text
               FROM events
               WHERE (alert_tier IS NOT NULL OR severity_score >= 65)
                 AND status IN ('classified', 'scored', 'reconciled')
                 AND ingested_at > NOW() - INTERVAL '%s hours'
               ORDER BY severity_score DESC, ingested_at DESC""",
            (hours,),
        )
        return cur.fetchall()


@st.cache_data(ttl=60)
def get_pipeline_stats(_conn_key: str, _db_conn) -> dict:
    """Get pipeline health metrics for telemetry dashboard — single optimized query."""
    # Combine multiple small queries into fewer round-trips
    event_counts = _safe_execute(
        _db_conn,
        """SELECT status, COUNT(*) FROM events GROUP BY status""",
    ).fetchall()

    alert_counts = _safe_execute(
        _db_conn,
        """SELECT alert_tier, COUNT(*)
           FROM events
           WHERE alert_tier IS NOT NULL
             AND ingested_at > NOW() - INTERVAL '24 hours'
           GROUP BY alert_tier""",
    ).fetchall()

    combined = _safe_execute(
        _db_conn,
        """SELECT
             (SELECT COUNT(*) FROM events WHERE ingested_at > NOW() - INTERVAL '24 hours') AS events_24h,
             (SELECT COUNT(*) FROM system_telemetry WHERE event_type = 'llm_call' AND timestamp > NOW() - INTERVAL '24 hours') AS llm_calls,
             (SELECT COALESCE(SUM((value_json->>'tokens_used')::int), 0) FROM system_telemetry WHERE event_type = 'llm_call' AND timestamp > NOW() - INTERVAL '24 hours') AS tokens,
             (SELECT COUNT(*) FROM system_telemetry WHERE event_type = 'stale_lock_cleared' AND timestamp > NOW() - INTERVAL '1 hour') AS stale_locks""",
    ).fetchone()

    last_run = _safe_execute(
        _db_conn,
        """SELECT value_json, timestamp
           FROM system_telemetry
           WHERE event_type = 'pipeline_run'
           ORDER BY timestamp DESC LIMIT 1""",
    ).fetchone()

    return {
        "llm_calls_24h": combined[0] if combined else 0,
        "tokens_used_24h": combined[2] if combined else 0,
        "stale_locks_1h": combined[3] if combined else 0,
        "event_counts": {row[0]: row[1] for row in event_counts} if event_counts else {},
        "last_run": last_run[0] if last_run and last_run[0] else None,
        "last_run_at": last_run[1].isoformat() if last_run and last_run[1] else None,
        "alert_counts": {row[0]: row[1] for row in alert_counts} if alert_counts else {},
        "events_24h": combined[0] if combined else 0,
    }


@st.cache_data(ttl=60)
def get_storyline_graph_data(_conn_key: str, _db_conn) -> list[dict]:
    """Fetch events with storyline links for graph visualization."""
    rows = _safe_execute(
        _db_conn,
        """SELECT id, source_title, event_type, storyline_id,
                  storyline_hint, anchor_name_norm, country_iso,
                  severity_score, occurred_at_est, alert_tier
           FROM events
           WHERE storyline_id IS NOT NULL
             AND status IN ('scored', 'reconciled')
           ORDER BY occurred_at_est DESC
           LIMIT 500""",
    ).fetchall()

    columns = [
        "id", "source_title", "event_type", "storyline_id",
        "storyline_hint", "anchor_name_norm", "country_iso",
        "severity_score", "occurred_at_est", "alert_tier",
    ]
    return [dict(zip(columns, row)) for row in rows]


@st.cache_data(ttl=60)
def get_geo_summary(_conn_key: str, _db_conn) -> list[dict]:
    """Get event counts by country for choropleth / summary."""
    rows = _safe_execute(
        _db_conn,
        """SELECT country_iso,
                  COUNT(*) AS total,
                  COUNT(*) FILTER (WHERE alert_tier = 'CRITICAL') AS critical,
                  COUNT(*) FILTER (WHERE alert_tier = 'ALERT') AS alert,
                  COUNT(*) FILTER (WHERE alert_tier = 'WATCH') AS watch
           FROM events
           WHERE country_iso IS NOT NULL
             AND status IN ('classified', 'scored', 'reconciled')
           GROUP BY country_iso
           ORDER BY total DESC
           LIMIT 20""",
    ).fetchall()
    columns = ["country_iso", "total", "critical", "alert", "watch"]
    return [dict(zip(columns, row)) for row in rows]


@st.cache_data(ttl=300)
def get_czib_zones(_conn_key: str, _db_conn, only_active: bool = True) -> list[dict]:
    """Fetch CZIB zones from database."""
    if only_active:
        sql = """SELECT czib_id, name, status, countries, country_names,
                        coordinates, issued_date, valid_until, valid_descr, updated_at
                 FROM czib_zones
                 WHERE status = 'Active'
                 ORDER BY updated_at DESC"""
    else:
        sql = """SELECT czib_id, name, status, countries, country_names,
                        coordinates, issued_date, valid_until, valid_descr, updated_at
                 FROM czib_zones
                 ORDER BY
                   CASE status
                     WHEN 'Active' THEN 1
                     WHEN 'Suspended' THEN 2
                     WHEN 'Withdrawn' THEN 3
                   END,
                   updated_at DESC"""
    rows = _safe_execute(_db_conn, sql).fetchall()
    columns = ["czib_id", "name", "status", "countries", "country_names",
               "coordinates", "issued_date", "valid_until", "valid_descr", "updated_at"]
    return [dict(zip(columns, row)) for row in rows]


@st.cache_data(ttl=300)
def get_czib_stats(_conn_key: str, _db_conn) -> dict:
    """Quick CZIB stats — single query instead of 3."""
    row = _safe_execute(
        _db_conn,
        """SELECT
             COUNT(*) FILTER (WHERE status = 'Active') AS active,
             COUNT(*) FILTER (WHERE status = 'Suspended') AS suspended,
             (SELECT COUNT(DISTINCT c) FROM czib_zones, unnest(countries) AS c WHERE status = 'Active') AS country_count
           FROM czib_zones""",
    ).fetchone()
    return {
        "active": row[0] if row else 0,
        "suspended": row[1] if row else 0,
        "countries": row[2] if row else 0,
    }
