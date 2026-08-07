"""
SIM — Alert Engine
Blueprint V20.1 §5.2 + §5.3

3-tier alert evaluation (WATCH / ALERT / CRITICAL) with
composite suppression key to prevent duplicate notifications.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.core.geo import geo_key

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as _f:
    _SETTINGS = json.load(_f)


@dataclass
class AlertTier:
    name: str
    color: str
    notify_channels: list[str]


TIERS = {
    "CRITICAL": AlertTier("CRITICAL", "#DC2626", ["telegram"]),
    "ALERT":    AlertTier("ALERT",    "#EA580C", ["telegram"]),
    "WATCH":    AlertTier("WATCH",    "#CA8A04", ["telegram"]),
}

# Official government travel advisories are actionable at the COUNTRY level, so they
# never carry an airport anchor and the standard anchor/time gates would drop them.
# They are pre-filtered upstream (Level 3-4 / "do not travel", curated high-risk
# countries) so gate them on severity alone.
ADVISORY_EVENT_TYPES = {"travel_advisory", "travel_ban"}

# A standing country-level advisory is not a breaking incident, so it must not outrank
# one. At the old floor of 55 every single advisory reached ALERT (88 of 88 over 7 days)
# while 108 of 252 confirmed mass-casualty-grade fresh events sat at WATCH — a Level-2
# Belgium advisory literally outranking a suicide bombing that killed 14.
#
# Advisory severity is effectively three discrete values (60 / 75 / 90), so 75 is the
# natural seam: it keeps the Level-3-4 "do not travel" advisories the docstring above
# describes at ALERT and drops the routine Level-2 bulk to WATCH.
ADVISORY_ALERT_SEVERITY_MIN = _SETTINGS.get("alert", {}).get(
    "advisory_alert_severity_min", 75
)

# Severity floor: a confirmed high-severity event that is happening NOW cannot rank
# below WATCH, whatever the corroboration signals say. system_confidence measures how
# well-sourced a report is, which is a fair reason to rank one incident above another
# but not a fair reason to rank a mass-casualty attack below a travel advisory.
# Deliberately does NOT reach CRITICAL — that tier still has to be earned on location
# and confidence.
SEVERITY_ALERT_FLOOR = _SETTINGS.get("alert", {}).get("severity_alert_floor", 90)

# A time_certainty that pins the event to roughly "now" — the only values that make
# an event newsworthy as a page rather than as background.
FRESH_TIME_CERTAINTY = ["same_day", "previous_day"]

# Tier thresholds live in config/settings.json -> alert.tiers. They used to be
# duplicated here as literals, which meant editing the config changed nothing —
# a silent trap for anyone tuning alert volume.
#
# Each tier is evaluated in order and takes the first match:
#   severity_min / confidence_min   — inclusive floors
#   require_location                — event must resolve to a place (IATA anchor or coords)
#   require_location_or_fresh       — ...or, failing that, carry a fresh time_certainty
#   time_certainty_include          — require time_certainty to be in this list
#   time_certainty_exclude          — reject when time_certainty is in this list
#   anchor_confidence               — allowed values (omit to accept any)
#
# Calibration note (7 Aug 2026). The original V19 gates keyed ALERT/CRITICAL on
# anchor_confidence and on time_certainty != unknown. Measured against a week of
# real corpus, both signals turned out to be near-constant and the conjunction was
# unsatisfiable — 1455 of 1469 alerted events were anchor LOW (the anchor level
# really means "an IATA airport was resolved", which conflict reporting rarely
# offers) and 86% of all scored events carry time_certainty='unknown'. So the
# gates never fired and a fallback in dispatch_alert force-labelled everything
# ALERT, which is why CRITICAL had never once fired. The gates below are set from
# the observed distributions (confidence p50=0.41, p90=0.62, p95=0.66, max=0.78)
# and treat "we know WHERE it happened" as a first-class signal alongside "we know
# WHEN": ALERT needs one of the two, CRITICAL needs both.
_DEFAULT_TIER_RULES = {
    "CRITICAL": {"severity_min": 80, "confidence_min": 0.62,
                 "require_location": True,
                 "time_certainty_include": FRESH_TIME_CERTAINTY},
    "ALERT": {"severity_min": 65, "confidence_min": 0.50,
              "require_location_or_fresh": True},
    "WATCH": {"severity_min": 45, "confidence_min": 0.40,
              "time_certainty_include": FRESH_TIME_CERTAINTY},
}

# Severity descends across tiers, so evaluation order is fixed rather than taken
# from dict order in the config — a reordered config file must not silently make
# every CRITICAL event fire as WATCH.
TIER_ORDER = ("CRITICAL", "ALERT", "WATCH")

_TIER_RANK = {name: i for i, name in enumerate(reversed(TIER_ORDER), start=1)}


def tier_rank(tier: str | None) -> int:
    """Order tiers so escalations are comparable. None/unrecognised sorts below WATCH."""
    return _TIER_RANK.get(tier or "", 0)


def _tier_rules() -> dict:
    """Merge configured tier rules over the built-in defaults, per tier."""
    configured = _SETTINGS.get("alert", {}).get("tiers", {})
    rules = {}
    for name in TIER_ORDER:
        merged = dict(_DEFAULT_TIER_RULES[name])
        merged.update(configured.get(name) or {})
        rules[name] = merged
    return rules


TIER_RULES = _tier_rules()


def is_located(event: dict) -> bool:
    """True when the event resolved to a real place.

    Either an IATA anchor (aviation events) or coordinates from the city gazetteer
    (everything else). Pass D resolves both, so this is the honest "we know where
    this happened" test — unlike anchor_confidence, which only ever means "an IATA
    airport matched" and is LOW for ~99% of the corpus.
    """
    return bool(event.get("anchor_name_norm")) or event.get("latitude") is not None


def _matches_tier(rule: dict, sev, conf, anc: str, time_: str, located: bool) -> bool:
    if sev < rule["severity_min"] or conf < rule["confidence_min"]:
        return False
    fresh = time_ in FRESH_TIME_CERTAINTY
    if rule.get("require_location") and not located:
        return False
    if rule.get("require_location_or_fresh") and not (located or fresh):
        return False
    allowed = rule.get("anchor_confidence")
    if allowed is not None and anc not in allowed:
        return False
    excluded = rule.get("time_certainty_exclude")
    if excluded and time_ in excluded:
        return False
    required = rule.get("time_certainty_include")
    if required and time_ not in required:
        return False
    return True


def evaluate_alert_tier(event: dict) -> str | None:
    """
    Evaluate which alert tier an event qualifies for.

    Returns: 'CRITICAL', 'ALERT', 'WATCH', or None
    """
    sev = event.get("severity_score", 0)
    conf = event.get("system_confidence", 0.0)
    anc = event.get("anchor_confidence", "LOW")
    time_ = event.get("time_certainty") or "unknown"
    located = is_located(event)

    # Travel advisory path — country-level official warning, no airport anchor and its
    # "time" is the standing advisory date, so bypass the anchor/time gates and key on
    # severity only (already pre-filtered to Level 3-4 / "do not travel" upstream).
    if event.get("event_type") in ADVISORY_EVENT_TYPES:
        return "ALERT" if sev >= ADVISORY_ALERT_SEVERITY_MIN else "WATCH"

    tier = None
    for name in TIER_ORDER:
        if _matches_tier(TIER_RULES[name], sev, conf, anc, time_, located):
            tier = name
            break

    # Severity floor — see SEVERITY_ALERT_FLOOR. Applied after the ladder so it can
    # only ever raise a tier to ALERT, never lower one or manufacture a CRITICAL.
    if (sev >= SEVERITY_ALERT_FLOOR
            and time_ in FRESH_TIME_CERTAINTY
            and tier_rank(tier) < tier_rank("ALERT")):
        return "ALERT"

    return tier


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
