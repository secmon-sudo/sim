"""
SIM — Alert Engine
Blueprint V20.1 §5.2 + §5.3

3-tier alert evaluation (WATCH / ALERT / CRITICAL) with
composite suppression key to prevent duplicate notifications.
"""

import logging
from dataclasses import dataclass

from src.core.geo import geo_key

logger = logging.getLogger(__name__)


@dataclass
class AlertTier:
    name: str
    color: str
    notify_channels: list[str]


TIERS = {
    "CRITICAL": AlertTier("CRITICAL", "#DC2626", ["telegram", "email", "sms"]),
    "ALERT":    AlertTier("ALERT",    "#EA580C", ["telegram", "email"]),
    "WATCH":    AlertTier("WATCH",    "#CA8A04", ["telegram"]),
}


def evaluate_alert_tier(event: dict) -> str | None:
    """
    Evaluate which alert tier an event qualifies for.

    Returns: 'CRITICAL', 'ALERT', 'WATCH', or None
    """
    sev = event.get("severity_score", 0)
    conf = event.get("system_confidence", 0.0)
    anc = event.get("anchor_confidence", "LOW")
    time_ = event.get("time_certainty", "unknown")

    # CRITICAL — original V19 gate, unchanged
    if sev >= 80 and conf >= 0.8 and anc == "HIGH" and time_ != "unknown":
        return "CRITICAL"

    # ALERT — mid tier, actionable but not highest urgency
    if sev >= 65 and conf >= 0.65 and anc in ("HIGH", "MEDIUM") and time_ != "unknown":
        return "ALERT"

    # WATCH — early signal, situational awareness only
    if sev >= 45 and conf >= 0.5 and time_ in ("same_day", "previous_day"):
        return "WATCH"

    return None


def build_suppression_key(event: dict) -> str:
    """
    Build composite key to prevent duplicate alerts.
    Uses IATA code (not raw text) to avoid key fragmentation.
    """
    return "|".join([
        str(event.get("storyline_id") or "no_storyline"),
        event.get("anchor_name_norm") or "UNKNOWN",
        str(int(event.get("severity_score", 0) // 10) * 10),
    ])


def build_geo_suppression_key(event: dict) -> str | None:
    """Storyline-independent suppression fingerprint (safety net).

    The primary key keys off storyline_id, so when the same real-world event is split
    across several storyline_ids (paraphrased hints that fall below the Jaccard
    threshold), each fragment produces a different primary key and every source pages
    separately. This fingerprint drops storyline_id and keys off coarse geography
    instead — country + resolved location + severity bucket — so near-identical alerts
    within the suppression TTL collapse regardless of storyline fragmentation.

    Location resolution prefers the precise IATA anchor, then a coarse geo_key derived
    from the raw location text. Returns None when no usable location is known (so the
    net is never so broad that it mutes unrelated same-country alerts).
    """
    loc = event.get("anchor_name_norm") or geo_key(
        event.get("anchor_name_raw"), event.get("country_iso")
    )
    if not loc or loc == "UNKNOWN":
        return None
    return "|".join([
        "geofp",
        event.get("country_iso") or "??",
        loc,
        str(int(event.get("severity_score", 0) // 10) * 10),
    ])


def is_suppressed(db_conn, suppression_key: str) -> bool:
    """Check if an alert with this key was already fired within TTL."""
    row = db_conn.execute(
        """SELECT 1 FROM alert_suppression
           WHERE suppression_key = %s AND expires_at > NOW()""",
        (suppression_key,),
    ).fetchone()
    return row is not None


def record_suppression(db_conn, suppression_key: str, tier: str, event_id: str, ttl_hours: int = 4):
    """Record a suppression entry so future duplicates are muted."""
    db_conn.execute(
        """INSERT INTO alert_suppression (suppression_key, alert_tier, event_id, expires_at)
           VALUES (%s, %s, %s, NOW() + (%s * INTERVAL '1 hour'))
           ON CONFLICT (suppression_key) DO UPDATE SET expires_at = EXCLUDED.expires_at""",
        (suppression_key, tier, event_id, ttl_hours),
    )
    db_conn.commit()
