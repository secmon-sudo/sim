"""
SIM — Pass D: Resolution, Storyline, Spatial & Scoring
Blueprint V20.1 §4 PASS D

Resolves anchors, links storylines, and computes severity/confidence scores.
Optimized to fetch recent events once (batch) instead of N+1 per event.
"""

import json
import logging
import uuid
from pathlib import Path

from src.core.alerts import build_suppression_key, evaluate_alert_tier, is_suppressed, record_suppression
from src.core.anchor import get_anchor_confidence_level, normalize_anchor
from src.core.storyline import should_link_storyline
from src.services.telegram_notifier import send_telegram_alert

logger = logging.getLogger(__name__)

# Load settings
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as f:
    _SETTINGS = json.load(f)

_SCORING = _SETTINGS.get("scoring", {})
PROXIMITY_BONUS = _SCORING.get("proximity_bonus", 30)
CZIB_BONUS = _SCORING.get("czib_bonus", 20)
MAX_SEVERITY = _SCORING.get("max_severity", 100)
LLM_CONF_WEIGHT = _SCORING.get("llm_confidence_weight", 0.4)
ANCHOR_CONF_WEIGHT = _SCORING.get("anchor_confidence_weight", 0.3)
DIVERSITY_WEIGHT = _SCORING.get("diversity_weight", 0.3)

# Casualty bonus config
_CASUALTY = _SCORING.get("casualty_bonus", {})
CASUALTY_DEATHS_THRESHOLD = _CASUALTY.get("deaths_threshold", 3)
CASUALTY_INJURIES_THRESHOLD = _CASUALTY.get("injuries_threshold", 10)
CASUALTY_BONUS_POINTS = _CASUALTY.get("bonus_points", 20)


def _safe_int(value) -> int:
    """Convert a value to int, returning 0 for non-numeric strings like 'multiple', 'several'."""
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def _extract_casualties(llm_parsed: dict) -> dict:
    """Extract casualty counts from LLM parsed output."""
    casualties = llm_parsed.get("casualties") if isinstance(llm_parsed.get("casualties"), dict) else {}
    if casualties is None:
        casualties = {}
    return {
        "deaths": _safe_int(casualties.get("deaths")),
        "injuries": _safe_int(casualties.get("injuries")),
        "missing": _safe_int(casualties.get("missing")),
    }


def compute_casualty_bonus(llm_parsed: dict) -> int:
    """Compute severity bonus based on casualty counts."""
    counts = _extract_casualties(llm_parsed)
    deaths = counts.get("deaths", 0)
    injuries = counts.get("injuries", 0)

    if deaths >= CASUALTY_DEATHS_THRESHOLD or injuries >= CASUALTY_INJURIES_THRESHOLD:
        return CASUALTY_BONUS_POINTS
    return 0


def compute_severity(event_type: str, anchor_data: dict | None, db_conn, llm_parsed: dict | None = None) -> int:
    """
    Severity = Base_Type_Weight + Proximity_Bonus (+30) + CZIB_Bonus (+20) + Casualty_Bonus (+20). Max 100.
    """
    base = 20  # default
    try:
        row = db_conn.execute(
            "SELECT severity_base FROM event_type_catalog WHERE code = %s",
            (event_type,),
        ).fetchone()
        if row:
            base = row[0]
    except Exception:
        pass

    score = base

    if anchor_data:
        if anchor_data.get("confidence", 0) >= 0.6:
            score += PROXIMITY_BONUS
        if anchor_data.get("czib_flag"):
            score += CZIB_BONUS

    # Mass casualty bonus
    if llm_parsed:
        score += compute_casualty_bonus(llm_parsed)

    return min(score, MAX_SEVERITY)


def compute_confidence(llm_confidence: float, anchor_confidence: float, diversity_score: float = 0.5) -> float:
    """
    Confidence = Max(0.0, Min(1.0, (llm_conf * 0.4) + (anchor_score * 0.3) + (diversity * 0.3)))
    """
    raw = (llm_confidence * LLM_CONF_WEIGHT
           + anchor_confidence * ANCHOR_CONF_WEIGHT
           + diversity_score * DIVERSITY_WEIGHT)
    return max(0.0, min(1.0, round(raw, 3)))


def resolve_anchor_for_event(db_conn, event: dict) -> dict:
    """Resolve anchor for a single event and return anchor data."""
    raw_anchor = event.get("anchor_name_raw")
    if not raw_anchor:
        return {"norm": None, "confidence": 0.0, "level": "LOW", "czib_flag": False}

    norm, conf = normalize_anchor(raw_anchor, db_conn)

    czib = False
    lat = None
    lon = None
    country = event.get("country_iso")

    if norm:
        try:
            row = db_conn.execute(
                "SELECT czib_flag, latitude, longitude, country_iso FROM anchor_master WHERE iata_code = %s",
                (norm,),
            ).fetchone()
            if row:
                czib = row[0] or False
                lat = row[1]
                lon = row[2]
                country = row[3] or country
        except Exception:
            pass

    return {
        "norm": norm,
        "confidence": conf,
        "level": get_anchor_confidence_level(conf),
        "czib_flag": czib,
        "latitude": lat,
        "longitude": lon,
        "country_iso": country,
    }


def link_storylines(event: dict, recent_events: list[dict]) -> str | None:
    """Try to link this event to an existing storyline."""
    # Guard: skip storyline linking if occurred_at_est is missing
    if event.get("occurred_at_est") is None:
        return None

    for existing in recent_events:
        if should_link_storyline(event, existing):
            return existing.get("storyline_id")
    return None


def _fetch_recent_events_for_linking(db_conn) -> list[dict]:
    """Fetch recent scored/reconciled events once for storyline linking."""
    try:
        rows = db_conn.execute(
            """SELECT id, storyline_id, storyline_hint, country_iso, occurred_at_est
               FROM events
               WHERE status IN ('scored', 'reconciled')
                 AND storyline_hint IS NOT NULL
                 AND occurred_at_est > NOW() - INTERVAL '14 days'
               ORDER BY occurred_at_est DESC LIMIT 200""",
        ).fetchall()

        return [
            {
                "id": str(r[0]),
                "storyline_id": str(r[1]) if r[1] else None,
                "storyline_hint": r[2],
                "country_iso": r[3],
                "occurred_at_est": r[4],
            }
            for r in rows
        ]
    except Exception:
        logger.exception("Error fetching recent events for storyline linking")
        return []


def score_single_event(db_conn, event_id: str, recent_events: list[dict]) -> dict | None:
    """Score a single classified event: resolve anchor, compute severity/confidence, assign alert tier."""
    try:
        row = db_conn.execute(
            """SELECT id, event_type, anchor_name_raw, country_iso,
                      llm_parsed_output, storyline_hint, occurred_at_est,
                      source_title, source_url
               FROM events WHERE id = %s AND status = 'classified'""",
            (event_id,),
        ).fetchone()

        if not row:
            return None

        # llm_parsed_output is now stored as native JSONB (dict) in psycopg 3
        llm_parsed = row[4]
        if isinstance(llm_parsed, str):
            llm_parsed = json.loads(llm_parsed or "{}")
        elif llm_parsed is None:
            llm_parsed = {}

        event = {
            "id": str(row[0]),
            "event_type": row[1],
            "anchor_name_raw": row[2],
            "country_iso": row[3],
            "llm_parsed": llm_parsed,
            "storyline_hint": row[5],
            "occurred_at_est": row[6],
            "source_title": row[7],
            "source_url": row[8],
        }

        # 1. Resolve anchor
        anchor = resolve_anchor_for_event(db_conn, event)

        # 2. Compute severity (with casualty bonus)
        severity = compute_severity(event["event_type"], anchor, db_conn, llm_parsed)

        # 3. Compute confidence
        llm_conf = event["llm_parsed"].get("confidence", 0.5)
        system_conf = compute_confidence(llm_conf, anchor["confidence"])

        # 4. Evaluate alert tier
        alert_data = {
            "severity_score": severity,
            "system_confidence": system_conf,
            "anchor_confidence": anchor["level"],
            "time_certainty": event["llm_parsed"].get("time_certainty", "unknown"),
        }
        alert_tier = evaluate_alert_tier(alert_data)

        # 5. Try storyline linking (using pre-fetched recent events)
        storyline_id = link_storylines(event, recent_events)
        if not storyline_id:
            storyline_id = str(uuid.uuid4())

        # Prepare event dict for suppression & notification
        event["storyline_id"] = storyline_id
        event["anchor_name_norm"] = anchor["norm"]
        event["severity_score"] = severity
        event["system_confidence"] = system_conf
        event["alert_tier"] = alert_tier

        # Send Telegram alert for high-severity events (severity >= 80)
        # Suppression key prevents duplicate notifications for the same storyline
        if severity >= 80:
            # Ensure alert_tier is set for the message formatting
            if not alert_tier:
                event["alert_tier"] = "ALERT"
            supp_key = build_suppression_key(event)
            if not is_suppressed(db_conn, supp_key):
                if send_telegram_alert(event):
                    record_suppression(db_conn, supp_key, event["alert_tier"], event_id, ttl_hours=4)

        # 6. Update event — wrapped in savepoint for isolation
        with db_conn.transaction():
            db_conn.execute(
                """UPDATE events
                   SET anchor_name_norm = %s,
                       anchor_confidence = %s,
                       latitude = %s,
                       longitude = %s,
                       country_iso = COALESCE(%s, country_iso),
                       severity_score = %s,
                       system_confidence = %s,
                       alert_tier = %s,
                       storyline_id = %s,
                       status = 'scored',
                       updated_at = NOW()
                   WHERE id = %s""",
                (
                    anchor["norm"],
                    anchor["level"],
                    anchor.get("latitude"),
                    anchor.get("longitude"),
                    anchor.get("country_iso"),
                    severity,
                    system_conf,
                    alert_tier,
                    storyline_id,
                    event_id,
                ),
            )
        db_conn.commit()

        logger.info(
            "Scored event %s: severity=%d, confidence=%.2f, tier=%s, anchor=%s",
            event_id[:8], severity, system_conf, alert_tier, anchor["norm"],
        )
        return {
            "event_id": event_id,
            "severity": severity,
            "confidence": system_conf,
            "alert_tier": alert_tier,
            "anchor_norm": anchor["norm"],
            "storyline_id": storyline_id,
        }

    except Exception:
        try:
            db_conn.rollback()
        except Exception:
            pass
        logger.exception("Error scoring event %s", event_id)
        return None


def run_pass_d(db_conn) -> dict:
    """
    Execute Pass D: Resolution, Storyline, Spatial & Scoring.
    Fetches recent events once for storyline linking (eliminates N+1).

    Returns: stats dict
    """
    stats = {
        "events_scored": 0,
        "events_failed": 0,
        "alerts_generated": {"CRITICAL": 0, "ALERT": 0, "WATCH": 0},
    }

    try:
        # Fetch recent events once for storyline linking
        recent_events = _fetch_recent_events_for_linking(db_conn)

        rows = db_conn.execute(
            "SELECT id FROM events WHERE status = 'classified' ORDER BY ingested_at ASC",
        ).fetchall()

        for row in rows:
            result = score_single_event(db_conn, str(row[0]), recent_events)
            if result:
                stats["events_scored"] += 1
                tier = result.get("alert_tier")
                if tier:
                    stats["alerts_generated"][tier] = stats["alerts_generated"].get(tier, 0) + 1
            else:
                stats["events_failed"] += 1

    except Exception:
        logger.exception("Error in Pass D")

    # Log telemetry
    try:
        db_conn.execute(
            "INSERT INTO system_telemetry(event_type, value_json) VALUES ('pass_d', %s)",
            (json.dumps(stats),),
        )
        db_conn.commit()
    except Exception:
        logger.exception("Failed to log Pass D telemetry")

    logger.info("Pass D complete: %s", stats)
    return stats
