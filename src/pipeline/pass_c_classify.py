"""
SIM — Pass C: LLM Classification
Blueprint V20.1 §4 PASS C

Classifies deduped events using multi-provider LLM router.
Uses HeartbeatWorker to keep locks alive during long calls.
"""

import json
import logging
import re
import uuid
from datetime import datetime as dt, timedelta, timezone
from pathlib import Path

from src.core.heartbeat import HeartbeatWorker
from src.core.llm_client import call_llm, log_llm_telemetry
from src.core.llm_router import LLMRouter
from src.pipeline.pass_a_ingest import (
    _HIGH_SIGNAL_TERMS,
    _SECURITY_KEYWORD_PATTERN,
    is_noise,
)
from src.pipeline.pass_b_dedup import acquire_lock, get_events_for_classification, release_lock

logger = logging.getLogger(__name__)

# Config: sanity bounds for occurred_at + deterministic pre-screen
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
try:
    with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as _f:
        _SETTINGS = json.load(_f)
except (OSError, json.JSONDecodeError):
    _SETTINGS = {}
_INGESTION = _SETTINGS.get("ingestion", {})
MAX_EVENT_AGE_DAYS = _INGESTION.get("max_event_age_days", 30)
MAX_EVENT_FUTURE_DAYS = _INGESTION.get("max_event_future_days", 1)

_CLASSIFICATION = _SETTINGS.get("classification", {})
PRESCREEN_ENABLED = _CLASSIFICATION.get("deterministic_prescreen_enabled", True)
PRESCREEN_SKIP_FLOOR = _CLASSIFICATION.get("deterministic_skip_floor", 15)

# Word-boundary pattern for high-signal terms only (subset of the full security
# pattern). Used to (a) score relevance and (b) override LLM false-negatives —
# if a hard signal like "explosion"/"airstrike"/"killed" is present we never let
# the LLM silently archive the event as noise.
_HIGH_SIGNAL_PATTERN = re.compile(
    "|".join(rf"\b{re.escape(t)}\b" for t in sorted(_HIGH_SIGNAL_TERMS)),
    re.IGNORECASE,
)
_CASUALTY_NUM_PATTERN = re.compile(
    r"\b\d+\s+(killed|dead|deaths?|injured|wounded|casualties|fatalities|missing)\b",
    re.IGNORECASE,
)


def deterministic_relevance(title: str, text: str, trusted_domain: bool = False) -> dict:
    """Zero-LLM relevance estimate used to skip clearly off-topic articles before
    spending an LLM call (token-positive) and to guard against LLM false-negatives.

    Returns a dict with an integer ``score`` (0-100) and boolean signals. The score
    is intentionally conservative: an article only scores low when it contains NO
    security vocabulary at all (none of the ~400 emergency/geopolitical keywords or
    high-signal terms), which for a real incident is extremely unlikely.
    """
    blob = f"{title} {text}"
    has_high_signal = bool(_HIGH_SIGNAL_PATTERN.search(blob))
    has_security = has_high_signal or bool(_SECURITY_KEYWORD_PATTERN.search(blob))
    has_casualty = bool(_CASUALTY_NUM_PATTERN.search(blob))
    noisy = is_noise(f"{title} {text[:500]}")

    score = 0
    if has_high_signal:
        score += 45
    elif has_security:
        score += 25
    if has_casualty:
        score += 15
    if trusted_domain:
        score += 10
    if noisy and not has_high_signal:
        score -= 30
    score = max(0, min(100, score))

    return {
        "score": score,
        "has_security": has_security,
        "has_high_signal": has_high_signal,
        "has_casualty": has_casualty,
        "noisy": noisy,
    }

CLASSIFICATION_SYSTEM_PROMPT = """You are a global security and geopolitical incident classifier.
Your job is to analyze news reports and determine if they describe REAL security incidents, conflicts, or threats.

STEP 1 — RELEVANCE CHECK:
Score the relevance of this text to security monitoring (0-100).
IMPORTANT: Score ONLY based on DIRECT, ACTIONABLE security threats.
Generic geopolitical analysis, opinion pieces, commentary, or distant regional news
without a specific incident should score below 30.
- 90-100: Active security incident, attack, or military conflict with confirmed details
- 70-89: Credible threat, escalation, or developing security situation
- 50-69: Related security event but limited details, or indirect impact
- 30-49: Tangentially related — mentions security topics but is NOT an incident (policy, opinion, analysis)
- 10-29: Mostly irrelevant — hobby content, entertainment, historical, reviews, generic commentary
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
   political_event, civil_unrest, protest, mass_demonstration, riot, general_strike, coup_attempt,
   terrorism, african_terrorism, insurgency_attack, extremist_violence, jihadist_attack,
   mass_casualty_event, mass_shooting, mass_stabbing, suicide_bombing, vehicle_ramming,
   resort_attack, beach_attack, tourist_bus_attack, cruise_ship_attack,
   travel_advisory, travel_ban, embassy_closure,
   other_aviation_related,
   noise

2. sub_type: More specific classification if applicable, or null
3. anchor_name: Airport, military base, port, hotel, resort, or location name mentioned (raw text). If none, null.
4. country_iso: 2-letter ISO country code (e.g. "US", "EG", "GB", "NG", "ML", "SO"). If unknown, null.
5. occurred_at: Best estimate of when the event occurred (ISO 8601 format), or null
6. time_certainty: One of: same_day, previous_day, this_week, approximate, unknown
7. storyline_hint: A STRICTLY ENGLISH, structured 4-6 word identifier for grouping related articles about the EXACT same event.
   Format: "[LOCATION] [ACTOR/ENTITY] [ACTION] [DATE-HINT]"
   Examples:
   - "Istanbul Ataturk bomb threat Jun8"
   - "Delta DL54 emergency Atlanta Jun7"
   - "Sahel JNIM convoy ambush Jun6"
   - "Tehran drone strike refinery Jun8"
   - "Somalia Shabaab base attack Jun5"
   Rules:
   - ALWAYS in English regardless of source language
   - MUST include location name (city/airport/base)
   - MUST include the specific actor, flight number, or entity if known
   - MUST include a short date hint (MonDD format, e.g. Jun8, May15)
   - NEVER use generic phrases like "emergency landing" or "bomb threat" alone
   - Two articles about the SAME event MUST produce the SAME hint
8. confidence: Your confidence in the classification (0.0 to 1.0)
9. casualties: If mentioned, extract {"deaths": int, "injuries": int, "missing": int}. If unknown, null.
10. relevance_score: Integer 0-100 from Step 1
11. relevance_reasoning: One sentence explaining why this relevance score was given
12. aviation_impact: How this event threatens civil aviation operations. One of:
    - "direct": targets/disrupts an airport, aircraft, airline, airspace, or aviation personnel
      (e.g. airport attack, drone near runway, airspace closure, crew assault, GPS jamming of flights)
    - "indirect": nearby or regional event that could spill over to aviation
      (e.g. conflict/airstrikes near a city with an airport, unrest affecting airport access)
    - "none": no plausible connection to aviation operations
    Aviation is the PRIORITY domain — assess this field carefully for every event.

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
- Mass protests, demonstrations, or riots that threaten stability or cause casualties
- Government crackdowns on protesters with violence
- General strikes affecting transportation, airports, or critical infrastructure
- Coup attempts or martial law declarations
- Country travel advisories (Level 3/4), travel bans, or "do not travel" warnings
- Embassy or consulate closures due to security threats
- State of emergency declarations related to security

PRIORITY RULES:
- Aviation personnel attacked → event_type: aviation_personnel_attack, HIGH priority
- Drone attack on critical infrastructure → event_type: drone_attack_critical_infra
- Mass casualty (3+ deaths OR 10+ injuries) → event_type: mass_casualty_event
- African terrorism (Sahel, Horn of Africa) → event_type: african_terrorism
- War escalation, ceasefire violations → event_type: war_escalation or ceasefire_violation
- Resort/hotel/beach attacks → event_type: resort_attack
- Protest with violence or casualties → event_type: riot, HIGH priority
- Mass demonstration (10K+ participants or nationwide) → event_type: mass_demonstration
- Peaceful protest (significant, large-scale) → event_type: protest
- General/nationwide strike → event_type: general_strike
- Coup attempt or martial law → event_type: coup_attempt, CRITICAL priority
- Country travel advisory Level 3-4 or "do not travel" → event_type: travel_advisory or travel_ban
- Embassy/consulate closure due to security → event_type: embassy_closure

IMPORTANT: When in doubt, classify the event rather than marking as noise.
It is better to let a borderline event through than to miss a real incident.

Respond ONLY with valid JSON. No markdown, no explanation."""




def _normalize_storyline_hint(hint: str | None) -> str | None:
    """Normalize storyline hint for consistent Jaccard matching.

    - Lowercases
    - Strips punctuation (except hyphens)
    - Collapses whitespace
    """
    if not hint or not isinstance(hint, str):
        return None
    import re as _re
    hint = _re.sub(r'\s+', ' ', hint.strip().lower())
    hint = _re.sub(r'[^\w\s-]', '', hint)
    # Drop a malformed date-hint when the LLM had no day (it is required to append a
    # "MonDD" token, e.g. "Jun8", but emits "JunUnknown"/"JunTBD" when the day is
    # unknown). Left in place it both uglifies the title and pollutes Jaccard with a
    # "jununknown" token. Well-formed "jun8" hints are untouched here and are handled
    # by the tokenizer's date-token filter.
    _months = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
    hint = _re.sub(rf'\b(?:{_months})(?:unknown|tbd)\b', '', hint)
    hint = _re.sub(r'\bunknown\b', '', hint)
    hint = _re.sub(r'\s+', ' ', hint).strip()
    return hint if hint else None


def _within_sane_bounds(parsed) -> bool:
    """Reject LLM-estimated timestamps that are implausibly old or in the future.

    LLMs sometimes hallucinate dates years in the past (anniversary/retrospective
    articles) or in the future. Such values pollute storyline time windows and the
    weekly forecast, so they are discarded (caller falls back to None/'unknown').
    """
    now = dt.now(timezone.utc).replace(tzinfo=None)
    if parsed > now + timedelta(days=MAX_EVENT_FUTURE_DAYS):
        return False
    if parsed < now - timedelta(days=MAX_EVENT_AGE_DAYS):
        return False
    return True


def _parse_occurred_at(raw: str | None):
    """Safely parse LLM's occurred_at ISO 8601 string into a naive datetime.
    Returns None if the value is missing, empty, unparseable, or outside sane bounds.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()

    parsed = None
    # Try common ISO formats LLMs produce
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            # Strip timezone info → naive timestamp (DB column is TIMESTAMP without tz)
            parsed = dt.strptime(raw, fmt).replace(tzinfo=None)
            break
        except ValueError:
            continue

    # Last resort: dateutil-style fallback
    if parsed is None:
        try:
            cleaned = raw.replace("Z", "+00:00")  # Handle "Z" suffix
            parsed = dt.fromisoformat(cleaned).replace(tzinfo=None)
        except (ValueError, TypeError):
            return None

    if not _within_sane_bounds(parsed):
        logger.info("Discarded out-of-bounds occurred_at estimate: %s", raw[:40])
        return None
    return parsed


def _safe_relevance(value, default: int = 50) -> int:
    """Coerce the LLM's relevance_score to an int in [0, 100].

    LLMs occasionally emit null or non-numeric values ("high", "N/A"); a bare
    int() would raise and permanently fail the event, so fall back to default.
    """
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return default


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

def update_domain_penalty(db_conn, domain: str, is_noise: int):
    """Update penalty stats for a domain in the database."""
    if not domain or domain == "unknown":
        return
    try:
        db_conn.execute(
            """INSERT INTO domain_penalties (domain, total_events, false_positives, penalty_score, last_seen)
               VALUES (%s, 1, %s, %s, NOW())
               ON CONFLICT (domain) DO UPDATE SET
                   total_events = domain_penalties.total_events + 1,
                   false_positives = domain_penalties.false_positives + EXCLUDED.false_positives,
                   last_seen = NOW(),
                   penalty_score = (domain_penalties.false_positives + EXCLUDED.false_positives)::float / (domain_penalties.total_events + 1)
            """,
            (domain, is_noise, float(is_noise))
        )
    except Exception:
        logger.exception("Error updating domain penalty for: %s", domain)


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
            
            # Detect travel advisory sources for special handling
            is_travel_advisory = source_domain in ('travel.state.gov', 'gov.uk', 'smartraveller.gov.au')

            # ── Deterministic pre-screen (zero-LLM, token-positive) ──
            # Skip clearly off-topic articles (no security vocabulary at all) before
            # spending an LLM call. Travel advisories always go to the LLM. The same
            # signals also guard against LLM false-negatives later (see Tier 1 below).
            det = deterministic_relevance(source_title, canonical_text)
            if PRESCREEN_ENABLED and not is_travel_advisory and det["score"] < PRESCREEN_SKIP_FLOOR:
                update_domain_penalty(db_conn, source_domain, 1)
                db_conn.execute(
                    """UPDATE events
                       SET event_type = 'other_aviation_related',
                           llm_parsed_output = %s,
                           status = 'archived',
                           updated_at = NOW()
                       WHERE id = %s""",
                    (json.dumps({"prescreen": det, "archived_reason": "deterministic_prescreen"}), event_id),
                )
                db_conn.commit()
                logger.info(
                    "Event %s prescreen-archived (score=%d, no security signal) — saved 1 LLM call",
                    event_id[:8], det["score"],
                )
                release_lock(db_conn, event_id, worker_id)
                return {"event_type": "other_aviation_related", "_prescreen_skipped": True}

            prompt = f"""Classify this news report:

Headline: {source_title[:500]}
Source: {source_domain}
Text: {canonical_text[:3000]}"""

            if is_travel_advisory:
                prompt += "\n\nIMPORTANT: This is an official government Travel Advisory. Classify as travel_advisory, travel_ban, or embassy_closure as appropriate."

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
            relevance = _safe_relevance(parsed.get("relevance_score", 50))

            # LLM false-negative guard: if a hard deterministic signal is present
            # (explosion/airstrike/killed/etc.) but the LLM scored this as noise,
            # keep it in the pipeline rather than silently archiving. Better a
            # low-priority event than a missed real incident.
            if det["has_high_signal"] and relevance < 30:
                logger.warning(
                    "Event %s: LLM relevance=%d but high-signal term present — overriding archive, keeping event",
                    event_id[:8], relevance,
                )
                relevance = max(relevance, 30)
                if event_type == "noise":
                    event_type = "other_aviation_related"
                    parsed["event_type"] = event_type

            # Tier 1: Clear noise (relevance < 20) → archive immediately
            # Use 'other_aviation_related' as FK-safe type; the real signal is status='archived'
            # The original LLM classification is preserved in llm_parsed_output for auditing
            if relevance < 30 or (event_type == "noise" and relevance < 40):
                archive_type = "other_aviation_related"  # FK-safe fallback
                update_domain_penalty(db_conn, source_domain, 1)
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
            if relevance < 50 or event_type == "noise":
                # Re-classify noise with some relevance as other_aviation_related
                # so it still flows through scoring but won't get high priority
                if event_type == "noise":
                    event_type = "other_aviation_related"
                    parsed["event_type"] = event_type
                logger.info("Event %s low-relevance (%d) — classifying as %s",
                            event_id[:8], relevance, event_type)

            # Tier 3: Relevant (40+) → proceed normally with classification
            update_domain_penalty(db_conn, source_domain, 0)

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

            # Update event with classification — psycopg 3 writes dicts to JSONB natively
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
                    _normalize_storyline_hint(parsed.get("storyline_hint")),
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
