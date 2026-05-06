"""
SIM — Streamlit Cache Service
Blueprint V20.1 §5.1

@st.cache_data wrappers with 60-second TTL for all dashboard queries.
"""

import json
from datetime import datetime

import streamlit as st


@st.cache_data(ttl=60)
def get_recent_events(_db_conn, limit: int = 200) -> list[dict]:
    """Fetch recent events for the main table and map."""
    rows = _db_conn.execute(
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
    ]
    return [dict(zip(columns, row)) for row in rows]


@st.cache_data(ttl=60)
def get_alert_events(_db_conn, hours: int = 24) -> list[dict]:
    """Fetch events with alert tiers from the last N hours."""
    rows = _db_conn.execute(
        """SELECT id, source_title, event_type, alert_tier,
                  severity_score, system_confidence,
                  anchor_name_norm, country_iso,
                  occurred_at_est, ingested_at
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
    ]
    return [dict(zip(columns, row)) for row in rows]


@st.cache_data(ttl=60)
def get_pipeline_stats(_db_conn) -> dict:
    """Get pipeline health metrics for telemetry dashboard."""
    llm = _db_conn.execute(
        """SELECT COUNT(*) AS calls,
                  COALESCE(SUM((value_json->>'tokens_used')::int), 0) AS tokens
           FROM system_telemetry
           WHERE event_type = 'llm_call'
             AND timestamp > NOW() - INTERVAL '24h'""",
    ).fetchone()

    stale = _db_conn.execute(
        """SELECT COUNT(*) AS n FROM system_telemetry
           WHERE event_type = 'stale_lock_cleared'
             AND timestamp > NOW() - INTERVAL '1h'""",
    ).fetchone()

    event_counts = _db_conn.execute(
        """SELECT status, COUNT(*) FROM events GROUP BY status""",
    ).fetchall()

    last_run = _db_conn.execute(
        """SELECT value_json, timestamp
           FROM system_telemetry
           WHERE event_type = 'pipeline_run'
           ORDER BY timestamp DESC LIMIT 1""",
    ).fetchone()

    # Extract quota info from most recent llm_call telemetry
    quota = _db_conn.execute(
        """SELECT value_json->>'daily_quota' AS dq,
                  value_json->>'daily_used' AS du
           FROM system_telemetry
           WHERE event_type = 'llm_call'
           ORDER BY timestamp DESC LIMIT 1""",
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
    }


@st.cache_data(ttl=60)
def get_storyline_graph_data(_db_conn) -> list[dict]:
    """Fetch events with storyline links for graph visualization."""
    rows = _db_conn.execute(
        """SELECT id, source_title, event_type, storyline_id,
                  storyline_hint, anchor_name_norm, country_iso,
                  severity_score, occurred_at_est
           FROM events
           WHERE storyline_id IS NOT NULL
             AND status IN ('scored', 'reconciled')
           ORDER BY occurred_at_est DESC
           LIMIT 500""",
    ).fetchall()

    columns = [
        "id", "source_title", "event_type", "storyline_id",
        "storyline_hint", "anchor_name_norm", "country_iso",
        "severity_score", "occurred_at_est",
    ]
    return [dict(zip(columns, row)) for row in rows]
