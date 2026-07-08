"""
SIM — Pass D: Resolution, Storyline, Spatial & Scoring
Blueprint V20.1 §4 PASS D

Resolves anchors, links storylines, and computes severity/confidence scores.
Optimized to fetch recent events once (batch) instead of N+1 per event.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.core.alerts import (
    build_geo_suppression_key,
    build_suppression_key,
    evaluate_alert_tier,
    is_suppressed,
    record_suppression,
)
from src.core.anchor import get_anchor_confidence_level, normalize_anchor
from src.core.geo import geo_coords
from src.core.storyline import jaccard_similarity, should_link_storyline
from src.core.storyline_alert_state import get_peak_tier, is_escalation, register_alert
from src.services.telegram_notifier import send_telegram_alert

logger = logging.getLogger(__name__)

# Load settings
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as f:
    _SETTINGS = json.load(f)

SOURCE_CREDIBILITY = {
    "reuters.com": 1.0,
    "bbc.co.uk": 0.95,
    "defense.gov": 0.95,
    "timesofisrael.com": 0.95,
    "jpost.com": 0.95,
    "haaretz.com": 0.95,
    "ynetnews.com": 0.95,
    "breakingdefense.com": 0.90,
    "militarytimes.com": 0.90,
    "warontherocks.com": 0.90,
    "longwarjournal.org": 0.90,
    "centcom.mil": 0.95,
    "cnn.com": 0.90,
    "foxnews.com": 0.90,
    "wsj.com": 0.95,
    "nytimes.com": 0.95,
    "dropsitenews.com": 0.85,
    "presstv.ir": 0.85,
    "nitter.net": 0.80,
    "nitter.privacydev.net": 0.80,
    "nitter.poast.org": 0.80,
    "reddit.com": 0.50,
    "aljazeera.com": 0.90,
    "crisisgroup.org": 0.92,
    "bellingcat.com": 0.90,
    "thecipherbrief.com": 0.88,
    "foreignpolicy.com": 0.90,
    "defenseone.com": 0.90,
    "twz.com": 0.85,
    "defensenews.com": 0.90,
    "al-monitor.com": 0.85,
    "themoscowtimes.com": 0.85,
    "meduza.io": 0.85,
    "warsawinstitute.org": 0.82,
    "un.org": 0.95,
    "bbc.com": 0.95,
    "jamestown.org": 0.88,
    "thesoufancenter.org": 0.88,
    "ctc.westpoint.edu": 0.92,
    "counterextremism.com": 0.85,
}

_SCORING = _SETTINGS.get("scoring", {})
PROXIMITY_BONUS = _SCORING.get("proximity_bonus", 30)
CZIB_BONUS = _SCORING.get("czib_bonus", 20)
MAX_SEVERITY = _SCORING.get("max_severity", 100)
AVIATION_NEXUS_BONUS = _SCORING.get("aviation_nexus_bonus", 15)
ALERT_SUPPRESSION_TTL_HOURS = _SETTINGS.get("alert", {}).get("suppression_ttl_hours", 4)
ALERT_SEVERITY_MIN = 80
NEW_ACTIVITY_WINDOW_HOURS = _SETTINGS.get("alert", {}).get("new_activity_window_hours", 24)
# Events ingested longer ago than this never notify — they are old news being
# (re)processed late (e.g. orphan recovery), not breaking incidents.
ALERT_MAX_AGE_DAYS = _SETTINGS.get("alert", {}).get("alert_max_age_days", 2)

# After this many hours with no new page, an alerted storyline is considered quiet and
# gets a single closure note (the counterpart to the escalation cue).
STORYLINE_QUIET_HOURS = _SETTINGS.get("alert", {}).get("storyline_quiet_hours", 12)

# Accidental (safety/emniyet) event types — kept for coverage but de-prioritized,
# because the platform's mission is SECURITY (hostile/intentional) events. A
# genuinely intentional incident would be classified as a security type by the LLM.
SAFETY_EVENT_TYPES = {
    "bird_strike", "engine_failure", "emergency_landing", "depressurization",
}
SAFETY_SEVERITY_CAP = _SCORING.get("safety_severity_cap", 40)

# Generic "umbrella" event types the LLM reaches for on any geopolitics/policy-flavoured
# story. Used by the incident gate: an umbrella label with no located anchor and no
# casualties is commentary/analysis, not an actionable incident, and is capped so a
# single LLM mislabel can never become a near-critical alert.
# NB: travel_advisory is deliberately NOT here — official advisories are country-level
# (no airport anchor, no casualties) but ARE actionable, and have their own alert path.
GENERIC_UMBRELLA_TYPES = {
    "geopolitical_conflict", "political_event", "civil_unrest",
    "humanitarian_crisis",
}
INCIDENT_GATE_CAP = _SCORING.get("incident_gate_cap", 50)
SAFETY_LIFT_ON_MASS_CASUALTY = _SCORING.get("safety_lift_cap_on_mass_casualty", True)
LLM_CONF_WEIGHT = _SCORING.get("llm_confidence_weight", 0.4)
ANCHOR_CONF_WEIGHT = _SCORING.get("anchor_confidence_weight", 0.3)
DIVERSITY_WEIGHT = _SCORING.get("diversity_weight", 0.3)

# Casualty bonus config
_CASUALTY = _SCORING.get("casualty_bonus", {})
CASUALTY_DEATHS_THRESHOLD = _CASUALTY.get("deaths_threshold", 3)
CASUALTY_INJURIES_THRESHOLD = _CASUALTY.get("injuries_threshold", 10)
CASUALTY_BONUS_POINTS = _CASUALTY.get("bonus_points", 20)

# Storyline linking config (previously hardcoded 0.35/14 and ignored these keys)
_STORYLINE = _SETTINGS.get("storyline", {})
STORYLINE_JACCARD_THRESHOLD = _STORYLINE.get("jaccard_threshold", 0.4)
STORYLINE_TIME_WINDOW_DAYS = _STORYLINE.get("time_window_days", 14)
STORYLINE_COUNTRY_MATCH_REQUIRED = _STORYLINE.get("country_match_required", True)
STORYLINE_ANCHOR_ASSIST_THRESHOLD = _STORYLINE.get("anchor_assist_threshold", 0.2)
STORYLINE_ANCHOR_ASSIST_MAX_HOURS = _STORYLINE.get("anchor_assist_max_hours", 72)
# Layer 2 — LLM adjudication of same-place/near-time candidates the deterministic
# linker left unlinked (paraphrases with near-zero lexical overlap). Bounded to the
# ambiguous residue and run on the bulk router so it never touches classification quota.
STORYLINE_LLM_ADJUDICATION = _STORYLINE.get("llm_adjudication_enabled", True)
STORYLINE_ADJUDICATION_WINDOW_HOURS = _STORYLINE.get("adjudication_window_hours", 48)
STORYLINE_ADJUDICATION_MAX_CANDIDATES = _STORYLINE.get("adjudication_max_candidates", 6)


def _safe_float(value, default: float = 0.5, lo: float = 0.0, hi: float = 1.0) -> float:
    """Coerce an LLM-supplied numeric field (e.g. confidence) to a bounded float.

    Guards against null / "high" / other non-numeric values that would otherwise
    raise mid-scoring and leave the event stuck in 'classified' forever.
    """
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


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


# Event types that are inherently aviation-related (security + safety).
AVIATION_EVENT_TYPES = {
    "bomb_threat", "active_shooter", "hijacking", "runway_incursion",
    "emergency_landing", "bird_strike", "engine_failure", "fire_on_board",
    "depressurization", "unruly_passenger", "drone_incursion",
    "drone_attack_critical_infra", "drone_airport_attack", "laser_attack",
    "suspicious_package", "evacuation", "aviation_personnel_attack",
    "pilot_attacked", "cabin_crew_attacked", "ground_staff_attacked",
    "air_traffic_controller_threat",
}

# Word-boundary aviation vocabulary — used to detect an aviation nexus on events
# whose type is generic (e.g. a bombing/protest/military action that nonetheless
# threatens an airport, aircraft, or airspace).
_AVIATION_TERMS = re.compile(
    r"\b(airport|airfield|aerodrome|aircraft|airplane|airliner|airline|aviation|"
    r"runway|taxiway|tarmac|airspace|terminal|jetway|cockpit|cabin crew|"
    r"air traffic|flight|jet|notam|departures|arrivals|boarding gate)\b",
    re.IGNORECASE,
)


def compute_aviation_bonus(event: dict, anchor_data: dict | None) -> int:
    """Bonus for events with a genuine aviation nexus (keeps aviation 'front of mind'
    without narrowing coverage — geopolitical events simply don't earn it).

    Triggers on: an aviation event_type, an aviation/airport keyword in the title or
    anchor text, or an LLM-flagged direct aviation impact. Returns AVIATION_NEXUS_BONUS
    or 0.
    """
    if not AVIATION_NEXUS_BONUS:
        return 0
    if event.get("event_type") in AVIATION_EVENT_TYPES:
        return AVIATION_NEXUS_BONUS

    llm_parsed = event.get("llm_parsed") or {}
    if str(llm_parsed.get("aviation_impact", "")).lower() == "direct":
        return AVIATION_NEXUS_BONUS

    blob = " ".join(str(x) for x in (
        event.get("source_title") or "",
        event.get("anchor_name_raw") or "",
        llm_parsed.get("anchor_name") or "",
        event.get("storyline_hint") or "",
    ))
    if _AVIATION_TERMS.search(blob):
        return AVIATION_NEXUS_BONUS

    return 0


def apply_safety_downrank(event_type: str, severity: int, llm_parsed: dict | None) -> tuple[int, bool]:
    """De-prioritize accidental safety events (returns (severity, is_safety)).

    Routine safety events are capped below the alert threshold so they stay in the
    feed without raising security alerts. Mass-casualty safety events (e.g. a fatal
    crash) are tagged safety but NOT capped, so genuinely major incidents still
    surface — balancing "security focus" with "don't miss big events".
    """
    if event_type not in SAFETY_EVENT_TYPES:
        return severity, False
    if SAFETY_LIFT_ON_MASS_CASUALTY and compute_casualty_bonus(llm_parsed or {}) > 0:
        return severity, True
    return min(severity, SAFETY_SEVERITY_CAP), True


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

    # Incident gate (defense-in-depth): a generic umbrella type with NO located anchor
    # and NO reported casualties is analysis/commentary, not an actionable incident.
    # Cap it so an LLM mislabel (e.g. an inflation survey tagged geopolitical_conflict)
    # can never be near-critical, regardless of the type's catalog base.
    has_proximity = bool(anchor_data and anchor_data.get("confidence", 0) >= 0.6)
    if (event_type in GENERIC_UMBRELLA_TYPES
            and not has_proximity
            and not _has_casualties(llm_parsed)):
        score = min(score, INCIDENT_GATE_CAP)

    return min(score, MAX_SEVERITY)


def _has_casualties(llm_parsed: dict | None) -> bool:
    """True if the LLM extracted any deaths/injuries/missing (a real-incident signal)."""
    if not llm_parsed:
        return False
    casualties = llm_parsed.get("casualties") or {}
    if not isinstance(casualties, dict):
        return False
    try:
        return any(int(casualties.get(k) or 0) > 0 for k in ("deaths", "injuries", "missing"))
    except (TypeError, ValueError):
        return False


def compute_confidence(llm_confidence: float, anchor_confidence: float, diversity_score: float = 0.5) -> float:
    """
    Confidence = Max(0.0, Min(1.0, (llm_conf * 0.4) + (anchor_score * 0.3) + (diversity * 0.3)))
    """
    raw = (llm_confidence * LLM_CONF_WEIGHT
           + anchor_confidence * ANCHOR_CONF_WEIGHT
           + diversity_score * DIVERSITY_WEIGHT)
    return max(0.0, min(1.0, round(raw, 3)))


def compute_diversity_score(db_conn, storyline_id: str | None) -> float:
    """Compute source diversity score based on unique domains covering this storyline.

    Returns a value between 0.0 and 1.0:
      - 1 source  → 0.3 (single report, low corroboration)
      - 2 sources → 0.5 (baseline corroboration)
      - 3 sources → 0.7 (good corroboration)
      - 4+ sources → 0.9+ (strong multi-source confirmation)
    """
    if not storyline_id:
        return 0.3  # Single event with no storyline peers
    try:
        row = db_conn.execute(
            """SELECT COUNT(DISTINCT source_domain)
               FROM events
               WHERE storyline_id = %s AND source_domain IS NOT NULL""",
            (storyline_id,),
        ).fetchone()
        unique_domains = row[0] if row else 1
    except Exception:
        return 0.5
    # Map domain count to 0.0–1.0 range with diminishing returns
    if unique_domains <= 1:
        return 0.3
    elif unique_domains == 2:
        return 0.5
    elif unique_domains == 3:
        return 0.7
    elif unique_domains == 4:
        return 0.85
    else:
        return min(1.0, 0.85 + (unique_domains - 4) * 0.03)


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

    # City-level fallback: most conflict events never resolve to an IATA airport, so the
    # anchor_master lookup above leaves lat/lon empty. Resolve the raw location text
    # against the curated city gazetteer so these events still carry coordinates.
    if lat is None or lon is None:
        coords = geo_coords(raw_anchor, country)
        if coords:
            lat, lon, coord_iso = coords
            country = country or coord_iso

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
    """Link this event to the BEST-matching existing storyline.

    Uses config-driven thresholds (settings.json -> storyline.*) and picks the
    candidate with the highest Jaccard similarity among those that pass
    should_link_storyline — not merely the first match (which was order-dependent).
    """
    # Guard: skip storyline linking if occurred_at_est is missing
    if event.get("occurred_at_est") is None:
        return None

    event_hint = event.get("storyline_hint") or ""
    best_id: str | None = None
    best_sim = -1.0

    for existing in recent_events:
        if not should_link_storyline(
            event,
            existing,
            threshold=STORYLINE_JACCARD_THRESHOLD,
            max_days=STORYLINE_TIME_WINDOW_DAYS,
            country_match_required=STORYLINE_COUNTRY_MATCH_REQUIRED,
            anchor_assist_threshold=STORYLINE_ANCHOR_ASSIST_THRESHOLD,
            anchor_assist_max_hours=STORYLINE_ANCHOR_ASSIST_MAX_HOURS,
        ):
            continue
        sim = jaccard_similarity(event_hint, existing.get("storyline_hint") or "")
        if sim > best_sim:
            best_sim = sim
            best_id = existing.get("storyline_id")

    return best_id


def run_storyline_closures(db_conn) -> int:
    """Emit a 'storyline quiet' note for each alerted storyline that has gone silent.

    Best-effort and idempotent: `find_and_close_quiet` atomically flips each storyline to
    closed as it returns it, so re-running (or a concurrent pass) never double-closes.
    """
    from src.core.storyline_alert_state import find_and_close_quiet
    from src.services.telegram_notifier import send_storyline_closure

    closed = find_and_close_quiet(db_conn, STORYLINE_QUIET_HOURS)
    for c in closed:
        send_storyline_closure(c["peak_tier"], c["label"], STORYLINE_QUIET_HOURS)
    if closed:
        logger.info("Closed %d quiet storyline(s)", len(closed))
    return len(closed)


def _alert_label(event: dict) -> str:
    """Short human context stored with alert state, reused in the quiet-closure note."""
    title = str(event.get("source_title") or "").strip()[:80]
    anchor = str(event.get("anchor_name_norm") or "Unknown")
    country = str(event.get("country_iso") or "")
    loc = f"{anchor} {country}".strip()
    return f"{title} · {loc}" if title else loc


def dispatch_alert(db_conn, event: dict, event_id: str) -> str:
    """Outbox-ordered Telegram alert dispatch.

    Correctness fix: the suppression record is committed BEFORE the alert is sent,
    so a crash between sending and recording can never produce a duplicate alert.
    If the send fails outright, the suppression claim is released so a sibling event
    in the same storyline can retry.

    Returns one of: 'skipped' (below threshold), 'suppressed' (already alerted),
    'sent', or 'failed'.
    """
    if event.get("severity_score", 0) < ALERT_SEVERITY_MIN:
        return "skipped"

    # Freshness gate: an event ingested weeks ago that is only being scored now
    # (orphan recovery, backlog replay) is stale news — score it, but never page.
    ingested_at = event.get("ingested_at")
    if ingested_at:
        if ingested_at.tzinfo is None:
            ingested_at = ingested_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ingested_at
        if age > timedelta(days=ALERT_MAX_AGE_DAYS):
            logger.info(
                "Alert skipped for stale event %s (ingested %.1f days ago)",
                event_id[:8], age.total_seconds() / 86400,
            )
            return "skipped"

    if not event.get("alert_tier"):
        event["alert_tier"] = "ALERT"

    supp_key = build_suppression_key(event)
    # Storyline-independent safety net: mutes same-place/same-severity duplicates even
    # when the storyline_id fragments across paraphrased sources (None if no location).
    geo_supp_key = build_geo_suppression_key(event)
    supp_keys = [k for k in (supp_key, geo_supp_key) if k]

    if any(is_suppressed(db_conn, k) for k in supp_keys):
        return "suppressed"

    # Durably claim the alert slot(s) first (record_suppression commits internally).
    for k in supp_keys:
        record_suppression(db_conn, k, event["alert_tier"], event_id,
                           ttl_hours=ALERT_SUPPRESSION_TTL_HOURS)

    # Escalation context: if this storyline already paged at a lower tier, mark the card
    # so the higher-tier alert reads as "this got worse", not an unrelated fresh event.
    storyline_id = event.get("storyline_id")
    tier = event["alert_tier"]
    try:
        prev_peak = get_peak_tier(db_conn, storyline_id)
        if is_escalation(prev_peak, tier):
            event["escalation_from"] = prev_peak
    except Exception:
        logger.exception("Escalation check failed for %s", event_id[:8])

    if send_telegram_alert(event):
        # Persist paging history (peak tier / recency) for escalation + quiet-closure.
        register_alert(db_conn, storyline_id, tier,
                       int(event.get("severity_score", 0)), _alert_label(event))
        return "sent"

    # Send failed — release the claim(s) so a sibling event can retry next time.
    logger.warning("Telegram alert send failed for %s; releasing suppression", event_id[:8])
    try:
        db_conn.execute(
            "DELETE FROM alert_suppression WHERE suppression_key = ANY(%s)", (supp_keys,)
        )
        db_conn.commit()
    except Exception:
        db_conn.rollback()
        logger.exception("Failed to release suppression for %s", event_id[:8])
    return "failed"


def _fetch_recent_events_for_linking(db_conn) -> list[dict]:
    """Fetch recent scored/reconciled events once for storyline linking."""
    try:
        rows = db_conn.execute(
            """SELECT id, storyline_id, storyline_hint, country_iso, occurred_at_est,
                      anchor_name_norm, anchor_name_raw
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
                "anchor_name_norm": r[5],
                "anchor_name_raw": r[6],
            }
            for r in rows
        ]
    except Exception:
        logger.exception("Error fetching recent events for storyline linking")
        return []


def score_single_event(db_conn, event_id: str, recent_events: list[dict],
                       adjudicator=None) -> dict | None:
    """Score a single classified event: resolve anchor, compute severity/confidence, assign alert tier.

    adjudicator: optional callable(event, recent_events) -> storyline_id | None, invoked
    ONLY when deterministic linking finds no match, to resolve same-place paraphrases.
    """
    try:
        row = db_conn.execute(
            """SELECT id, event_type, anchor_name_raw, country_iso,
                      llm_parsed_output, storyline_hint, occurred_at_est,
                      source_title, source_url, ingested_at, source_domain
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

        # Fallback occurred_at_est to ingested_at if LLM couldn't determine a date
        occurred_at_est = row[6] if row[6] else row[9]

        event = {
            "id": str(row[0]),
            "event_type": row[1],
            "anchor_name_raw": row[2],
            "country_iso": row[3],
            "llm_parsed": llm_parsed,
            "storyline_hint": row[5],
            "occurred_at_est": occurred_at_est,
            # True when occurred_at_est is really the ingestion time, not an
            # incident time — notifier renders it as such instead of lying.
            "occurred_at_is_fallback": row[6] is None,
            "ingested_at": row[9],
            "source_title": row[7],
            "source_url": row[8],
            "source_domain": row[10],
        }

        # 1. Resolve anchor
        anchor = resolve_anchor_for_event(db_conn, event)
        # Expose normalized anchor for the hybrid storyline linker (anchor-assist)
        event["anchor_name_norm"] = anchor.get("norm")

        # 2. Compute severity (with casualty bonus), then apply aviation-nexus bonus
        #    so aviation-threatening events rank ahead of equivalent generic ones.
        severity = compute_severity(event["event_type"], anchor, db_conn, llm_parsed)
        severity = min(severity + compute_aviation_bonus(event, anchor), MAX_SEVERITY)
        # De-prioritize accidental safety events (kept for coverage, tagged is_safety).
        severity, is_safety = apply_safety_downrank(event["event_type"], severity, llm_parsed)

        # 3. Try storyline linking first (needed for diversity score)
        storyline_id = link_storylines(event, recent_events)
        if not storyline_id and adjudicator is not None:
            # Deterministic linking failed — let the LLM adjudicator judge same-place,
            # near-time candidates (paraphrases the lexical path could not confirm).
            try:
                storyline_id = adjudicator(event, recent_events)
            except Exception:
                logger.exception("Storyline adjudicator failed for %s", event_id[:8])
        if not storyline_id:
            storyline_id = str(uuid.uuid4())

        # 4. Compute confidence (with real source diversity and credibility multiplier)
        llm_conf = _safe_float(event["llm_parsed"].get("confidence", 0.5))
        diversity = compute_diversity_score(db_conn, storyline_id)
        system_conf = compute_confidence(llm_conf, anchor["confidence"], diversity)

        # Apply source credibility weighting
        credibility_multiplier = 1.0
        domain = event.get("source_domain")
        if domain:
            domain = domain.lower()
            if domain in SOURCE_CREDIBILITY:
                credibility_multiplier = SOURCE_CREDIBILITY[domain]
            else:
                for parent_domain, score in SOURCE_CREDIBILITY.items():
                    if domain.endswith("." + parent_domain):
                        credibility_multiplier = score
                        break
        system_conf = float(system_conf * credibility_multiplier)

        # 5. Evaluate alert tier
        alert_data = {
            "severity_score": severity,
            "system_confidence": system_conf,
            "anchor_confidence": anchor["level"],
            "time_certainty": event["llm_parsed"].get("time_certainty", "unknown"),
            "event_type": event["event_type"],
        }
        alert_tier = evaluate_alert_tier(alert_data)

        # Prepare event dict for suppression & notification
        event["storyline_id"] = storyline_id
        event["anchor_name_norm"] = anchor["norm"]
        event["country_iso"] = anchor.get("country_iso") or event.get("country_iso")
        event["severity_score"] = severity
        event["system_confidence"] = system_conf
        event["alert_tier"] = alert_tier

        # Detect "new activity" — the first GENUINE SECURITY event for this country/
        # location within the lookback window (drives the NEW LOCATION/COUNTRY labels).
        # Excludes archived (noise/aged-out) and is_safety events so that prior noise or
        # accidental safety incidents don't mask a real new threat. Single scan (no N+1).
        country_quiet = False
        location_quiet = False
        resolved_country = event["country_iso"]
        resolved_anchor_norm = event["anchor_name_norm"]
        if occurred_at_est and (resolved_country or resolved_anchor_norm):
            try:
                row = db_conn.execute(
                    """SELECT
                           COUNT(*) FILTER (WHERE country_iso = %s)        AS country_cnt,
                           COUNT(*) FILTER (WHERE anchor_name_norm = %s)   AS location_cnt
                       FROM events
                       WHERE status IN ('scored', 'reconciled')
                         AND COALESCE(is_safety, FALSE) = FALSE
                         AND occurred_at_est >= %s - make_interval(hours => %s)
                         AND occurred_at_est <= %s
                         AND id != %s""",
                    (resolved_country, resolved_anchor_norm, occurred_at_est,
                     NEW_ACTIVITY_WINDOW_HOURS, occurred_at_est, event["id"]),
                ).fetchone()
                if row:
                    if resolved_country and row[0] == 0:
                        country_quiet = True
                    if resolved_anchor_norm and row[1] == 0:
                        location_quiet = True
            except Exception:
                logger.exception("Failed to query new-activity window for event %s", event["id"])

        event["country_quiet_24h"] = country_quiet
        event["location_quiet_24h"] = location_quiet

        # Send Telegram alert for high-severity events (suppression-claim BEFORE
        # send to prevent duplicate notifications; see dispatch_alert).
        dispatch_alert(db_conn, event, event_id)

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
                       is_safety = %s,
                       occurred_at_est = COALESCE(occurred_at_est, %s),
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
                    is_safety,
                    occurred_at_est,
                    event_id,
                ),
            )
        db_conn.commit()

        # Make this freshly-scored event visible to the REST of the same Pass D batch.
        # recent_events is fetched once per pass and only contains already
        # scored/reconciled rows, so sibling reports of one incident that arrive
        # together were blind to each other and each spawned a new storyline_id.
        # Advertising the just-committed event lets later siblings link into it,
        # clustering multi-source reports into a single storyline.
        recent_events.append({
            "id": event["id"],
            "storyline_id": storyline_id,
            "storyline_hint": event.get("storyline_hint"),
            "country_iso": event.get("country_iso"),
            "occurred_at_est": occurred_at_est,
            "anchor_name_norm": event.get("anchor_name_norm"),
            "anchor_name_raw": event.get("anchor_name_raw"),
        })

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

        # Build the Layer 2 LLM adjudicator once per pass (bulk router, isolated quota).
        # Any init failure degrades gracefully to deterministic-only linking.
        adjudicator = None
        if STORYLINE_LLM_ADJUDICATION:
            try:
                from src.core.llm_router import build_bulk_router
                from src.core.storyline_adjudicator import adjudicate_storyline
                _adj_router = build_bulk_router()

                def adjudicator(event, recent, _router=_adj_router):
                    return adjudicate_storyline(
                        event, recent, _router,
                        window_hours=STORYLINE_ADJUDICATION_WINDOW_HOURS,
                        max_candidates=STORYLINE_ADJUDICATION_MAX_CANDIDATES,
                    )
            except Exception:
                logger.exception("Failed to init storyline adjudicator; deterministic only")
                adjudicator = None

        rows = db_conn.execute(
            "SELECT id FROM events WHERE status = 'classified' ORDER BY ingested_at ASC",
        ).fetchall()

        for row in rows:
            result = score_single_event(db_conn, str(row[0]), recent_events, adjudicator)
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
