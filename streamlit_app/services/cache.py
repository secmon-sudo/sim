"""
SIM — Streamlit Cache Service
Blueprint V20.1 §5.1

@st.cache_data wrappers with 60-second TTL for all dashboard queries.
"""

import json
from datetime import datetime

import streamlit as st


def _safe_execute(conn, sql, params=None):
    """Execute SQL with automatic rollback on aborted transaction."""
    try:
        return conn.execute(sql, params)
    except Exception:
        # Transaction may be aborted from a previous error — rollback and retry once
        try:
            conn.rollback()
        except Exception:
            pass
        return conn.execute(sql, params)


@st.cache_data(ttl=60)
def get_recent_events(_db_conn, limit: int = 200) -> list[dict]:
    """Fetch recent events for the main table and map."""
    rows = _safe_execute(
        _db_conn,
        """SELECT id, source_title, event_type, alert_tier,
                  severity_score, system_confidence,
                  anchor_name_norm, anchor_confidence,
                  country_iso, latitude, longitude,
                  storyline_id, time_certainty,
                  occurred_at_est, ingested_at, status,
                  llm_provider, llm_model,
                  source_domain, source_url,
                  canonical_text, raw_text
           FROM events
           WHERE status IN ('classified', 'scored', 'reconciled')
           ORDER BY ingested_at DESC
           LIMIT %s""",
        (limit,),
    ).fetchall()

    columns = [
        "id", "source_title", "event_type", "alert_tier",
        "severity_score", "system_confidence",
        "anchor_name_norm", "anchor_confidence",
        "country_iso", "latitude", "longitude",
        "storyline_id", "time_certainty",
        "occurred_at_est", "ingested_at", "status",
        "llm_provider", "llm_model",
        "source_domain", "source_url",
        "canonical_text", "raw_text",
    ]
    return [dict(zip(columns, row)) for row in rows]


@st.cache_data(ttl=60)
def get_alert_events(_db_conn, hours: int = 24) -> list[dict]:
    """Fetch events with alert tiers from the last N hours."""
    rows = _safe_execute(
        _db_conn,
        """SELECT id, source_title, event_type, alert_tier,
                  severity_score, system_confidence,
                  anchor_name_norm, country_iso,
                  occurred_at_est, ingested_at,
                  source_url, source_domain, canonical_text
           FROM events
           WHERE alert_tier IS NOT NULL
             AND ingested_at > NOW() - INTERVAL '%s hours'
           ORDER BY
             CASE alert_tier
               WHEN 'CRITICAL' THEN 1
               WHEN 'ALERT' THEN 2
               WHEN 'WATCH' THEN 3
             END,
             ingested_at DESC""",
        (hours,),
    ).fetchall()

    columns = [
        "id", "source_title", "event_type", "alert_tier",
        "severity_score", "system_confidence",
        "anchor_name_norm", "country_iso",
        "occurred_at_est", "ingested_at",
        "source_url", "source_domain", "canonical_text",
    ]
    return [dict(zip(columns, row)) for row in rows]


@st.cache_data(ttl=60)
def get_pipeline_stats(_db_conn) -> dict:
    """Get pipeline health metrics for telemetry dashboard."""
    llm = _safe_execute(
        _db_conn,
        """SELECT COUNT(*) AS calls,
                  COALESCE(SUM((value_json->>'tokens_used')::int), 0) AS tokens
           FROM system_telemetry
           WHERE event_type = 'llm_call'
             AND timestamp > NOW() - INTERVAL '24h'""",
    ).fetchone()

    stale = _safe_execute(
        _db_conn,
        """SELECT COUNT(*) AS n FROM system_telemetry
           WHERE event_type = 'stale_lock_cleared'
             AND timestamp > NOW() - INTERVAL '1h'""",
    ).fetchone()

    event_counts = _safe_execute(
        _db_conn,
        """SELECT status, COUNT(*) FROM events GROUP BY status""",
    ).fetchall()

    last_run = _safe_execute(
        _db_conn,
        """SELECT value_json, timestamp
           FROM system_telemetry
           WHERE event_type = 'pipeline_run'
           ORDER BY timestamp DESC LIMIT 1""",
    ).fetchone()

    quota = _safe_execute(
        _db_conn,
        """SELECT value_json->>'daily_quota' AS dq,
                  value_json->>'daily_used' AS du
           FROM system_telemetry
           WHERE event_type = 'llm_call'
           ORDER BY timestamp DESC LIMIT 1""",
    ).fetchone()

    alert_counts = _safe_execute(
        _db_conn,
        """SELECT alert_tier, COUNT(*)
           FROM events
           WHERE alert_tier IS NOT NULL
             AND ingested_at > NOW() - INTERVAL '24h'
           GROUP BY alert_tier""",
    ).fetchall()

    events_24h = _safe_execute(
        _db_conn,
        """SELECT COUNT(*) FROM events
           WHERE ingested_at > NOW() - INTERVAL '24h'""",
    ).fetchone()

    return {
        "llm_calls_24h": llm[0] if llm else 0,
        "tokens_used_24h": llm[1] if llm else 0,
        "stale_locks_1h": stale[0] if stale else 0,
        "event_counts": {row[0]: row[1] for row in event_counts} if event_counts else {},
        "last_run": last_run[0] if last_run and last_run[0] else None,
        "last_run_at": last_run[1].isoformat() if last_run and last_run[1] else None,
        "daily_quota": int(quota[0]) if quota and quota[0] else 1000,
        "daily_used": int(quota[1]) if quota and quota[1] else 0,
        "alert_counts": {row[0]: row[1] for row in alert_counts} if alert_counts else {},
        "events_24h": events_24h[0] if events_24h else 0,
    }


@st.cache_data(ttl=60)
def get_storyline_graph_data(_db_conn) -> list[dict]:
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
def get_geo_summary(_db_conn) -> list[dict]:
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
def get_czib_zones(_db_conn, only_active: bool = True) -> list[dict]:
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
def get_czib_stats(_db_conn) -> dict:
    """Quick CZIB stats for sidebar."""
    active = _safe_execute(
        _db_conn, "SELECT COUNT(*) FROM czib_zones WHERE status = 'Active'"
    ).fetchone()
    suspended = _safe_execute(
        _db_conn, "SELECT COUNT(*) FROM czib_zones WHERE status = 'Suspended'"
    ).fetchone()
    total_countries = _safe_execute(
        _db_conn,
        "SELECT COUNT(DISTINCT unnest(countries)) FROM czib_zones WHERE status = 'Active'"
    ).fetchone()
    return {
        "active": active[0] if active else 0,
        "suspended": suspended[0] if suspended else 0,
        "countries": total_countries[0] if total_countries else 0,
    }
