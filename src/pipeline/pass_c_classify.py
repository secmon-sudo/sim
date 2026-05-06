"""
SIM — Pass C: LLM Classification
Blueprint V20.1 §4 PASS C

Classifies deduped events using multi-provider LLM router.
Uses HeartbeatWorker to keep locks alive during long calls.
"""

import json
import logging
import uuid

from src.core.heartbeat import HeartbeatWorker
from src.core.llm_client import call_llm, log_llm_telemetry
from src.core.llm_router import LLMRouter
from src.pipeline.pass_b_dedup import acquire_lock, get_events_for_classification, release_lock

logger = logging.getLogger(__name__)

CLASSIFICATION_SYSTEM_PROMPT = """You are a global security incident classifier.
Analyze the following news text and extract:
1. event_type: One of: bomb_threat, active_shooter, hijacking, runway_incursion,
   emergency_landing, bird_strike, engine_failure, fire_on_board, depressurization,
   unruly_passenger, drone_incursion, laser_attack, suspicious_package, evacuation,
   security_incident, geopolitical_conflict, military_action, missile_strike,
   political_event, civil_unrest, terrorism, other_aviation_related
2. sub_type: More specific classification if applicable (same codes), or null
3. anchor_name: Airport or location name mentioned (raw text)
4. country_iso: 2-letter ISO country code (e.g. "US", "EG", "GB")
5. occurred_at: Best estimate of when the event occurred (ISO 8601 format), or null
6. time_certainty: One of: same_day, previous_day, this_week, approximate, unknown
7. storyline_hint: A short phrase describing the core event for grouping related articles
8. confidence: Your confidence in the classification (0.0 to 1.0)

Respond ONLY with valid JSON. No markdown, no explanation."""


class LLMParseError(Exception):
    """Raised when LLM output cannot be parsed as valid classification JSON."""
    pass


def validate_and_parse(content: str) -> dict:
    """
    Parse and validate LLM classification output.
    Extracts JSON from response, handling potential markdown wrapping.
    """
    if not content:
        raise LLMParseError("Empty LLM response")

    # Strip markdown code block if present
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

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
        with HeartbeatWorker(db_conn, event_id, str(worker_id), interval=60):
            # Build prompt
            prompt = f"""Classify this aviation security report:

Title/Source: {event.get('source_domain', 'unknown')}
Text: {event.get('canonical_text', '')[:3000]}"""

            # Call LLM through multi-provider router
            result = call_llm(
                router,
                prompt=prompt,
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
                max_tokens=1024,
            )

            # Parse response
            parsed = validate_and_parse(result.get("content", ""))

            # Validate event_type against active catalog
            event_type = parsed.get("event_type", "other_aviation_related")
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

            # Update event with classification
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
                    parsed.get("country_iso"),
                    parsed.get("storyline_hint"),
                    parsed.get("time_certainty", "unknown"),
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
