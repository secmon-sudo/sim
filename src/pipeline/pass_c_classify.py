"""
SIM — Pass C: LLM Classification
Blueprint V20.1 §4 PASS C

Classifies deduped events using multi-provider LLM router.
Uses HeartbeatWorker to keep locks alive during long calls.
"""

import json
import logging
import uuid
from datetime import datetime as dt, timezone

from src.core.heartbeat import HeartbeatWorker
from src.core.llm_client import call_llm, log_llm_telemetry
from src.core.llm_router import LLMRouter
from src.pipeline.pass_b_dedup import acquire_lock, get_events_for_classification, release_lock

logger = logging.getLogger(__name__)

CLASSIFICATION_SYSTEM_PROMPT = """You are a global security and geopolitical incident classifier.
Your job is to analyze news reports and determine if they describe REAL security incidents, conflicts, or threats.

STEP 1 — RELEVANCE CHECK:
Score the relevance of this text to security monitoring (0-100):
- 90-100: Active security incident, attack, or military conflict with confirmed details
- 70-89: Credible threat, escalation, or developing security situation
- 50-69: Related security event but limited details, or indirect impact
- 30-49: Tangentially related — mentions security topics but is NOT an incident (policy, opinion, analysis)
- 10-29: Mostly irrelevant — hobby content, entertainment, historical, reviews
- 0-9: Completely irrelevant — no security connection whatsoever

STEP 2 — CLASSIFICATION (if relevance >= 30):
Extract the following fields:

1. event_type: One of:
   bomb_threat, active_shooter, hijacking, runway_incursion,
   emergency_landing, bird_strike, engine_failure, fire_on_board, depressurization,
   unruly_passenger, drone_incursion, drone_attack_critical_infra, drone_airport_attack,
   laser_attack, suspicious_package, evacuation,
   security_incident, aviation_personnel_attack, pilot_attacked, cabin_crew_attacked, ground_staff_attacked,
   geopolitical_conflict, military_action, missile_strike, war_escalation, ceasefire_violation, civilian_casualties,
   political_event, civil_unrest, terrorism, african_terrorism, insurgency_attack, extremist_violence, jihadist_attack,
   mass_casualty_event, mass_shooting, mass_stabbing, suicide_bombing, vehicle_ramming,
   resort_attack, beach_attack, tourist_bus_attack, cruise_ship_attack,
   other_aviation_related,
   noise

2. sub_type: More specific classification if applicable, or null
3. anchor_name: Airport, military base, port, hotel, resort, or location name mentioned (raw text)
4. country_iso: 2-letter ISO country code (e.g. "US", "EG", "GB", "NG", "ML", "SO")
5. occurred_at: Best estimate of when the event occurred (ISO 8601 format), or null
6. time_certainty: One of: same_day, previous_day, this_week, approximate, unknown
7. storyline_hint: A short phrase describing the core event for grouping related articles
8. confidence: Your confidence in the classification (0.0 to 1.0)
9. casualties: If mentioned, extract {"deaths": int, "injuries": int, "missing": int}. If unknown, null.
10. relevance_score: Integer 0-100 from Step 1
11. relevance_reasoning: One sentence explaining why this relevance score was given

WHEN TO USE event_type "noise" (relevance < 30):
- Flight simulators, plane spotting, aviation photography, model aircraft
- Historical articles, documentaries, anniversaries, museum exhibits
- Airline/hotel/seat reviews, trip reports, lounge reviews
- Movies, TV shows, video games, books
- Delivery flights, new liveries, route announcements, frequent flyer programs
- Reddit hobby discussions: "what is this plane", "spotted this", personal travel
- Opinion editorials, policy analysis with NO actual incident
- Generic street crime with NO link to aviation/infrastructure/military

WHEN TO CLASSIFY (relevance >= 30, even if borderline):
- Any mention of an actual attack, shooting, bombing, stabbing at a specific location
- Military operations, airstrikes, troop movements, escalations
- Drone attacks on infrastructure, airports, bases
- Mass casualty events regardless of location
- Terrorism or insurgency attacks anywhere
- Personnel attacks at airports, airlines, hotels
- Active threats, bomb scares, evacuations
- Civil unrest that threatens critical infrastructure

PRIORITY RULES:
- Aviation personnel attacked → event_type: aviation_personnel_attack, HIGH priority
- Drone attack on critical infrastructure → event_type: drone_attack_critical_infra
- Mass casualty (3+ deaths OR 10+ injuries) → event_type: mass_casualty_event
- African terrorism (Sahel, Horn of Africa) → event_type: african_terrorism
- War escalation, ceasefire violations → event_type: war_escalation or ceasefire_violation
- Resort/hotel/beach attacks → event_type: resort_attack

IMPORTANT: When in doubt, classify the event rather than marking as noise.
It is better to let a borderline event through than to miss a real incident.

Respond ONLY with valid JSON. No markdown, no explanation."""




def _parse_occurred_at(raw: str | None):
    """Safely parse LLM's occurred_at ISO 8601 string into a naive datetime.
    Returns None if the value is missing, empty, or unparseable.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    # Try common ISO formats LLMs produce
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            parsed = dt.strptime(raw, fmt)
            # Strip timezone info → naive timestamp (DB column is TIMESTAMP without tz)
            return parsed.replace(tzinfo=None)
        except ValueError:
            continue
    # Last resort: dateutil-style fallback
    try:
        # Handle "Z" suffix
        cleaned = raw.replace("Z", "+00:00")
        parsed = dt.fromisoformat(cleaned)
        return parsed.replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


class LLMParseError(Exception):
    """Raised when LLM output cannot be parsed as valid classification JSON."""
    pass


def validate_and_parse(content: str) -> dict:
    """
    Parse and validate LLM classification output.
    Handles common LLM JSON issues: markdown wrapping, trailing commas,
    single quotes, text before/after JSON.
    """
    import re

    if not content:
        raise LLMParseError("Empty LLM response")

    text = content.strip()

    # Strip markdown code block if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Extract JSON object if there's text before/after it
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)

    # Fix trailing commas before closing braces/brackets (most common LLM issue)
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # Remove control characters that break JSON
    text = re.sub(r'[\x00-\x1f]', lambda m: ' ' if m.group() not in '\n\r\t' else m.group(), text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMParseError(f"Invalid JSON: {e}") from e

    # Validate required fields
    if not isinstance(parsed, dict):
        raise LLMParseError(f"Expected dict, got {type(parsed).__name__}")

    # Ensure event_type is present
    if "event_type" not in parsed:
        parsed["event_type"] = "other_aviation_related"

    return parsed



def classify_single_event(db_conn, router: LLMRouter, event: dict, worker_id: uuid.UUID) -> dict | None:
    """
    Classify a single event using LLM with heartbeat protection.

    Returns parsed classification dict, or None on failure.
    """
    event_id = event["id"]

    # Acquire lock
    if not acquire_lock(db_conn, event_id, worker_id):
        logger.debug("Could not acquire lock for event %s", event_id)
        return None

    try:
        with HeartbeatWorker(event_id, str(worker_id), interval=60):
            # Build prompt — include title for better relevance judgment
            source_title = event.get('source_title', '') or ''
            source_domain = event.get('source_domain', 'unknown') or 'unknown'
            canonical_text = event.get('canonical_text', '') or ''
            
            prompt = f"""Classify this news report:

Headline: {source_title[:500]}
Source: {source_domain}
Text: {canonical_text[:3000]}"""

            # Call LLM through multi-provider router
            result = call_llm(
                router,
                prompt=prompt,
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
                max_tokens=1024,
            )

            # Parse response
            parsed = validate_and_parse(result.get("content", ""))

            # Graduated relevance handling using LLM's relevance_score
            event_type = parsed.get("event_type", "other_aviation_related")
            relevance = int(parsed.get("relevance_score", 50))

            # Tier 1: Clear noise (relevance < 20) → archive immediately
            # Use 'other_aviation_related' as FK-safe type; the real signal is status='archived'
            # The original LLM classification is preserved in llm_parsed_output for auditing
            if relevance < 20 or (event_type == "noise" and relevance < 30):
                archive_type = "other_aviation_related"  # FK-safe fallback
                db_conn.execute(
                    """UPDATE events
                       SET event_type = %s,
                           llm_raw_output = %s,
                           llm_parsed_output = %s,
                           llm_provider = %s,
                           llm_model = %s,
                           status = 'archived',
                           updated_at = NOW()
                       WHERE id = %s""",
                    (
                        archive_type,
                        json.dumps(result.get("response", {})),
                        json.dumps(parsed),
                        result.get("provider"),
                        result.get("model"),
                        event_id,
                    ),
                )
                db_conn.commit()
                log_llm_telemetry(db_conn, result, router, success=True)
                logger.info("Event %s archived — relevance=%d, llm_type=%s, reason=%s",
                            event_id[:8], relevance, event_type,
                            parsed.get("relevance_reasoning", "")[:80])
                release_lock(db_conn, event_id, worker_id)
                return parsed


            # Tier 2: Low relevance (20-40) or noise with some relevance → classify but flag
            # These events proceed through the pipeline but with reduced priority
            if relevance < 40 or event_type == "noise":
                # Re-classify noise with some relevance as other_aviation_related
                # so it still flows through scoring but won't get high priority
                if event_type == "noise":
                    event_type = "other_aviation_related"
                    parsed["event_type"] = event_type
                logger.info("Event %s low-relevance (%d) — classifying as %s",
                            event_id[:8], relevance, event_type)

            # Tier 3: Relevant (40+) → proceed normally with classification

            # Validate event_type against active catalog
            active_check = db_conn.execute(
                "SELECT code FROM event_type_catalog WHERE code = %s AND active = TRUE",
                (event_type,),
            ).fetchone()
            if not active_check:
                event_type = "other_aviation_related"

            # Validate sub_type against active catalog
            sub_type = parsed.get("sub_type")
            if sub_type:
                sub_check = db_conn.execute(
                    "SELECT code FROM event_type_catalog WHERE code = %s AND active = TRUE",
                    (sub_type,),
                ).fetchone()
                if not sub_check:
                    sub_type = None

            # Sanitize country_iso: must be exactly 2 uppercase ASCII letters
            raw_iso = parsed.get("country_iso") or ""
            country_iso = raw_iso.strip().upper()[:2] if raw_iso else None
            if country_iso and (len(country_iso) != 2 or not country_iso.isalpha()):
                country_iso = None

            # Parse occurred_at from LLM output into a timestamp
            occurred_at_est = _parse_occurred_at(parsed.get("occurred_at"))

            # Update event with classification — use json.dumps for JSONB columns
            db_conn.execute(
                """UPDATE events
                   SET llm_raw_output    = %s,
                       llm_parsed_output = %s,
                       event_type        = %s,
                       sub_type          = %s,
                       anchor_name_raw   = %s,
                       country_iso       = %s,
                       storyline_hint    = %s,
                       time_certainty    = %s,
                       occurred_at_est   = COALESCE(%s, occurred_at_est),
                       llm_provider      = %s,
                       llm_model         = %s,
                       status            = 'classified',
                       updated_at        = NOW()
                   WHERE id = %s AND lock_owner = %s""",
                (
                    json.dumps(result.get("response", {})),
                    json.dumps(parsed),
                    event_type,
                    sub_type,
                    parsed.get("anchor_name"),
                    country_iso,
                    parsed.get("storyline_hint"),
                    parsed.get("time_certainty", "unknown"),
                    occurred_at_est,
                    result.get("provider"),
                    result.get("model"),
                    event_id,
                    str(worker_id),
                ),
            )
            db_conn.commit()

            # Log telemetry
            log_llm_telemetry(db_conn, result, router, success=True)

            logger.info(
                "Classified event %s as %s via %s/%s (%.0fms)",
                event_id[:8],
                event_type,
                result.get("provider"),
                result.get("model", "")[:30],
                result.get("latency_ms", 0),
            )
            return parsed

    except LLMParseError as e:
        logger.warning("LLM parse error for event %s: %s", event_id[:8], e)
        try:
            db_conn.execute(
                """UPDATE events
                   SET llm_parse_error = %s,
                       event_type = 'other_aviation_related',
                       status = 'classified',
                       updated_at = NOW()
                   WHERE id = %s""",
                (str(e), event_id),
            )
            db_conn.commit()
        except Exception:
            db_conn.rollback()
        return None

    except RuntimeError as e:
        # All LLM accounts exhausted
        logger.error("All LLM accounts exhausted: %s", e)
        return None

    except Exception:
        db_conn.rollback()
        logger.exception("Unexpected error classifying event %s", event_id[:8])
        return None

    finally:
        # Idempotent lock release with explicit commit/rollback
        release_lock(db_conn, event_id, worker_id)


def run_pass_c(db_conn, router: LLMRouter, limit: int = 50) -> dict:
    """
    Execute Pass C: LLM Classification.

    1. Get deduped events ready for classification
    2. Classify each with LLM (heartbeat-protected)
    3. Return stats

    Returns: stats dict
    """
    worker_id = uuid.uuid4()

    stats = {
        "worker_id": str(worker_id),
        "events_available": 0,
        "events_classified": 0,
        "events_failed": 0,
        "llm_exhausted": False,
    }

    events = get_events_for_classification(db_conn, limit=limit)
    stats["events_available"] = len(events)

    if not events:
        logger.info("Pass C: No events to classify")
        return stats

    for event in events:
        try:
            result = classify_single_event(db_conn, router, event, worker_id)
            if result:
                stats["events_classified"] += 1
            else:
                stats["events_failed"] += 1
        except RuntimeError:
            stats["llm_exhausted"] = True
            logger.error("LLM accounts exhausted, stopping Pass C")
            break

    # Log telemetry
    try:
        db_conn.execute(
            "INSERT INTO system_telemetry(event_type, value_json) VALUES ('pass_c', %s)",
            (json.dumps(stats),),
        )
        db_conn.commit()
    except Exception:
        logger.exception("Failed to log Pass C telemetry")

    logger.info("Pass C complete: %s", stats)
    return stats
