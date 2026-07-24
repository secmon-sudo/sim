"""
SIM — Pass C: LLM Classification
Blueprint V20.1 §4 PASS C

Classifies deduped events using multi-provider LLM router.
Uses HeartbeatWorker to keep locks alive during long calls.
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime as dt, timedelta, timezone
from pathlib import Path

from src.core.heartbeat import HeartbeatWorker
from src.core.llm_client import LLMAllThrottled, LLMRequestTooLarge, call_llm, log_llm_telemetry
from src.core.llm_router import LLMRouter
from src.core.storyline import strip_date_hint
from src.pipeline.ingest_filters import (
    _HIGH_SIGNAL_TERMS,
    _SECURITY_KEYWORD_PATTERN,
    is_noise,
)
from src.pipeline.pass_b_dedup import acquire_lock, get_events_for_classification, release_lock

# Pending 'deduped' events above this logs a WARNING: at ~40 ingested/run it means
# the queue is more than two full runs behind even at the raised per-run limit.
QUEUE_DEPTH_ALERT_THRESHOLD = 400

# FK-safe fallback for events we could not — or need not — classify: parse
# failures, 'noise' verdicts, missing types, and the sub-relevance tail. Kept
# DISTINCT from the genuine 'other_aviation_related' aviation category so these
# never surface in SITREP daily records mislabeled as aviation. Requires the
# 'unclassified' catalog row (migration 019), which the workflow applies before
# this pass runs.
FALLBACK_EVENT_TYPE = "unclassified"

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

# Batch classification: how many reports to classify per LLM call. The ~2300-token
# system prompt is paid ONCE per call instead of once per event, and each call burns
# one RPM slot for N events — the free tier's two scarcest currencies. Sized so a
# full batch (system + N truncated reports + JSON array output) stays inside Groq's
# 8K TPM window. 1 disables batching (classic per-event path).
BATCH_CLASSIFY_SIZE = int(_CLASSIFICATION.get("llm_batch_size", 6))
# Per-report truncation inside a batch prompt (chars). Tighter than the single-event
# path's 3000 so the whole batch fits the TPM window; headlines carry most signal.
BATCH_TEXT_CHARS = 1200
BATCH_TITLE_CHARS = 300

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
7. storyline_hint: A STRICTLY ENGLISH, structured 3-5 word identifier for grouping related articles about the EXACT same event.
   Format: "[LOCATION] [ACTOR/ENTITY] [ACTION]"
   Examples:
   - "Istanbul Ataturk bomb threat"
   - "Delta DL54 emergency Atlanta"
   - "Sahel JNIM convoy ambush"
   - "Tehran drone strike refinery"
   - "Somalia Shabaab base attack"
   Rules:
   - ALWAYS in English regardless of source language
   - MUST include location name (city/airport/base)
   - MUST include the specific actor, flight number, or entity if known
   - Do NOT include any date or time token — event timing is captured separately
     in occurred_at, never in the hint
   - NEVER use generic phrases like "emergency landing" or "bomb threat" alone
   - Two articles about the SAME event MUST produce the SAME hint
   Consistency rules (critical — the hint is used to group multi-source reports):
   - Use the most specific COMMON place NAME (the city/airport), NEVER a descriptor
     like "capital", "the north", "border area", or "the region". Write "Kyiv", not
     "Ukrainian capital"; "Gaza", not "the enclave".
   - Use the canonical English spelling: "Kyiv" (not "Kiev"), "Kharkiv" (not "Kharkov"),
     "Odesa" (not "Odessa"), "Aleppo", "Sanaa".
   - Order the tokens LOCATION → ACTOR → ACTION every time, so paraphrases converge.
   - If several places are named, use the PRIMARY target/impact location only.
   - NEVER merge a multi-word proper noun into one token. Write "China Coast Guard"
     as three words, "Al Shabaab" as two — never "chinacoastguard"/"alshabaab". The
     hint doubles as a live news-search query, and glued tokens match nothing.
   - Use plain, searchable words: real place/actor/action terms only. Do NOT invent
     compounds, hashtags, or codes, and do not pad with filler like "situation",
     "update", "news", or "crisis".
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
- Economics/markets/finance: inflation, CPI, interest rates, central-bank surveys,
  stock/currency/oil-price moves, trade or tariff figures — even when they mention a
  country, sanctions, or a "deal" (e.g. "CPI expectations before a U.S.-Iran deal")
- Corporate / ESG / activism: boycotts, divestment, companies "remaining in" or exiting
  a country, brand statements, shareholder pressure, sustainability pledges
- Diplomatic/policy commentary with NO physical incident: negotiations, statements,
  sanctions announcements, treaty debate, election punditry

CRITICAL — do NOT use `geopolitical_conflict` (or any conflict/military type) for the
above. Those types are ONLY for an actual armed event (a strike, attack, clash, troop
movement, escalation on the ground). Economic, corporate, or diplomatic stories with no
physical incident are `noise` (or `political_event` if a concrete government action).

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
    # Drop date-hint tokens entirely ("jun8", "nov20", "jununknown", "juntbd").
    # The prompt no longer asks for them (since 2026-07-09), but when the article
    # stated no date the LLM used to FABRICATE one from training memory — which then
    # showed up verbatim in Telegram cards. The token was never used for matching
    # anyway (the Jaccard tokenizer filters date tokens); time lives in occurred_at.
    hint = strip_date_hint(hint)
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

    text = content.strip() if content else ""
    # Whitespace-only content (common when a reasoning model spends its whole
    # budget "thinking" and returns an empty message) reaches json.loads("") as
    # the misleading "Expecting value: line 1 column 1 (char 0)". Catch it here.
    if not text:
        raise LLMParseError("Empty LLM response")

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
        # strict=False tolerates raw control characters INSIDE string values
        # (e.g. a literal newline in a quoted summary) — the \x00-\x1f regex above
        # deliberately preserves \n\r\t as inter-token whitespace, so one inside a
        # string would otherwise fail with "Invalid control character".
        parsed = json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        raise LLMParseError(f"Invalid JSON: {e}") from e

    # Validate required fields
    if not isinstance(parsed, dict):
        raise LLMParseError(f"Expected dict, got {type(parsed).__name__}")

    # Ensure event_type is present
    if "event_type" not in parsed:
        parsed["event_type"] = FALLBACK_EVENT_TYPE

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


def _is_travel_advisory(event: dict) -> bool:
    source_domain = event.get('source_domain', 'unknown') or 'unknown'
    return source_domain in ('travel.state.gov', 'gov.uk', 'smartraveller.gov.au')


def _try_prescreen_archive(db_conn, event: dict, det: dict) -> bool:
    """Deterministic pre-screen (zero-LLM, token-positive).

    Archives clearly off-topic articles (no security vocabulary at all) before
    spending an LLM call; travel advisories always go to the LLM. Returns True
    if the event was archived. Caller holds (and releases) the lock.
    """
    if not PRESCREEN_ENABLED or _is_travel_advisory(event) or det["score"] >= PRESCREEN_SKIP_FLOOR:
        return False
    event_id = event["id"]
    source_domain = event.get('source_domain', 'unknown') or 'unknown'
    with db_conn.transaction():  # penalty + archive land together (conn is autocommit)
        update_domain_penalty(db_conn, source_domain, 1)
        db_conn.execute(
            """UPDATE events
               SET event_type = 'unclassified',
                   llm_parsed_output = %s,
                   status = 'archived',
                   updated_at = NOW()
               WHERE id = %s""",
            (json.dumps({"prescreen": det, "archived_reason": "deterministic_prescreen"}), event_id),
        )
    logger.info(
        "Event %s prescreen-archived (score=%d, no security signal) — saved 1 LLM call",
        event_id[:8], det["score"],
    )
    return True


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

            det = deterministic_relevance(source_title, canonical_text)
            if _try_prescreen_archive(db_conn, event, det):
                release_lock(db_conn, event_id, worker_id)
                return {"event_type": FALLBACK_EVENT_TYPE, "_prescreen_skipped": True}

            prompt = f"""Classify this news report:

Headline: {source_title[:500]}
Source: {source_domain}
Text: {canonical_text[:3000]}"""

            if _is_travel_advisory(event):
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
            return _apply_llm_classification(db_conn, router, event, det, parsed, result, worker_id)

    except LLMParseError as e:
        logger.warning("LLM parse error for event %s: %s", event_id[:8], e)
        try:
            db_conn.execute(
                """UPDATE events
                   SET llm_parse_error = %s,
                       event_type = 'unclassified',
                       status = 'classified',
                       updated_at = NOW()
                   WHERE id = %s""",
                (str(e), event_id),
            )
            db_conn.commit()
        except Exception:
            db_conn.rollback()
        return None

    except LLMAllThrottled as e:
        # Every slot is momentarily on a TPM/cooldown window — expected under
        # free-tier pacing: run_pass_c waits for the refill and retries this event.
        # INFO, not ERROR: nothing failed, no request was even sent.
        logger.info("All LLM slots throttled, deferring to pacing: %s", e)
        raise

    except RuntimeError as e:
        # All LLM accounts exhausted after real attempts (requests sent and failed).
        # Propagate so run_pass_c breaks the loop instead of hammering call_llm for
        # every remaining event and spamming the log while nothing can succeed.
        logger.error("All LLM accounts exhausted: %s", e)
        raise

    except Exception:
        db_conn.rollback()
        logger.exception("Unexpected error classifying event %s", event_id[:8])
        return None

    finally:
        # Idempotent lock release with explicit commit/rollback. requeue=True flips a
        # still-'locked' event back to 'deduped' so the pacing retry (or at worst the
        # next run) can pick it up without waiting for the orphan sweep.
        release_lock(db_conn, event_id, worker_id, requeue=True)


def _apply_llm_classification(db_conn, router: LLMRouter, event: dict, det: dict,
                              parsed: dict, result: dict, worker_id: uuid.UUID) -> dict | None:
    """Apply a parsed LLM classification to an event (tiering, validation, DB update).

    Shared by the single-event and batched paths. Caller holds the lock.
    """
    event_id = event["id"]
    source_domain = event.get('source_domain', 'unknown') or 'unknown'

    # Graduated relevance handling using LLM's relevance_score
    event_type = parsed.get("event_type", FALLBACK_EVENT_TYPE)
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
            event_type = FALLBACK_EVENT_TYPE
            parsed["event_type"] = event_type

    # Tier 1: Clear noise (relevance < 20) → archive immediately
    # Use 'unclassified' as FK-safe type; the real signal is status='archived'
    # The original LLM classification is preserved in llm_parsed_output for auditing
    if relevance < 30 or (event_type == "noise" and relevance < 40):
        archive_type = FALLBACK_EVENT_TYPE  # FK-safe fallback
        with db_conn.transaction():  # penalty + archive land together (conn is autocommit)
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
        log_llm_telemetry(db_conn, result, router, success=True)
        logger.info("Event %s archived — relevance=%d, llm_type=%s, reason=%s",
                    event_id[:8], relevance, event_type,
                    parsed.get("relevance_reasoning", "")[:80])
        return parsed


    # Tier 2: Low relevance (20-40) or noise with some relevance → classify but flag
    # These events proceed through the pipeline but with reduced priority
    if relevance < 50 or event_type == "noise":
        # Re-classify noise with some relevance as the neutral fallback type
        # so it still flows through scoring but won't get high priority
        if event_type == "noise":
            event_type = FALLBACK_EVENT_TYPE
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
        event_type = FALLBACK_EVENT_TYPE

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


# Appended to the system prompt for batched calls. json_object mode requires an
# object at the top level, so the per-report results ride in a "results" array.
BATCH_SYSTEM_SUFFIX = """

BATCH MODE: You will receive several numbered news reports in one message.
Classify EACH report INDEPENDENTLY using the schema above.
Respond ONLY with valid JSON of the form:
{"results": [{"report": 1, ...all fields...}, {"report": 2, ...}, ...]}
Include exactly one object per report, carrying its "report" number."""


def _batch_prompt(llm_events: list[dict]) -> str:
    blocks = [f"Classify each of these {len(llm_events)} news reports:"]
    for i, event in enumerate(llm_events, 1):
        title = (event.get('source_title', '') or '')[:BATCH_TITLE_CHARS]
        domain = event.get('source_domain', 'unknown') or 'unknown'
        text = (event.get('canonical_text', '') or '')[:BATCH_TEXT_CHARS]
        block = f"REPORT {i}:\nHeadline: {title}\nSource: {domain}\nText: {text}"
        if _is_travel_advisory(event):
            block += ("\nIMPORTANT: This is an official government Travel Advisory. "
                      "Classify as travel_advisory, travel_ban, or embassy_closure as appropriate.")
        blocks.append(block)
    return "\n\n".join(blocks)


def _parse_batch_response(content: str, expected: int) -> dict[int, dict]:
    """Parse a batch response into {report_number: parsed_item}.

    Outer-JSON failures raise LLMParseError (whole batch stays queued);
    per-item defects just drop that item — its event stays queued.
    """
    parsed = validate_and_parse(content)  # reuses markdown/trailing-comma repair
    results = parsed.get("results")
    if not isinstance(results, list):
        raise LLMParseError("Batch response missing 'results' array")
    items: dict[int, dict] = {}
    for pos, item in enumerate(results, 1):
        if not isinstance(item, dict):
            continue
        try:
            report_no = int(item.get("report", pos))
        except (TypeError, ValueError):
            report_no = pos
        if 1 <= report_no <= expected and report_no not in items:
            item.setdefault("event_type", FALLBACK_EVENT_TYPE)
            items[report_no] = item
    return items


def classify_event_batch(db_conn, router: LLMRouter, events: list[dict], worker_id: uuid.UUID) -> dict:
    """Classify a chunk of events with ONE LLM call (plus zero-cost prescreens).

    Returns {"classified": int, "failed": int}. Events whose lock can't be
    acquired (already handled by an earlier attempt of this chunk) are skipped
    without counting. On throttle/exhaustion the LLM-pending locks are released
    with requeue so run_pass_c's pacing retry can re-acquire them, then the
    exception propagates — mirroring the single-event contract.
    """
    stats = {"classified": 0, "failed": 0}
    llm_events: list[dict] = []

    for event in events:
        event_id = event["id"]
        if not acquire_lock(db_conn, event_id, worker_id):
            continue  # already archived/classified (e.g. pre-retry) or raced
        try:
            det = deterministic_relevance(
                event.get('source_title', '') or '',
                event.get('canonical_text', '') or '',
            )
            event["_det"] = det
            if _try_prescreen_archive(db_conn, event, det):
                stats["classified"] += 1
                release_lock(db_conn, event_id, worker_id)
            else:
                llm_events.append(event)  # lock intentionally kept for the LLM leg
        except Exception:
            db_conn.rollback()
            logger.exception("Batch prescreen failed for event %s", event_id[:8])
            stats["failed"] += 1
            release_lock(db_conn, event_id, worker_id, requeue=True)

    if not llm_events:
        return stats

    def _release_pending(requeue: bool):
        for ev in llm_events:
            release_lock(db_conn, ev["id"], worker_id, requeue=requeue)

    try:
        result = call_llm(
            router,
            prompt=_batch_prompt(llm_events),
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT + BATCH_SYSTEM_SUFFIX,
            # 450/event + 512 headroom: a low-effort reasoning preamble plus six full
            # classification objects overflowed the old 280/event budget, truncating
            # the JSON mid-string (2026-07-10 run: 21/21 batches unparseable).
            max_tokens=450 * len(llm_events) + 512,
        )
        items = _parse_batch_response(result.get("content", ""), expected=len(llm_events))
    except LLMAllThrottled:
        _release_pending(requeue=True)
        raise
    except RuntimeError as e:
        logger.error("All LLM accounts exhausted: %s", e)
        _release_pending(requeue=True)
        raise
    except LLMParseError as e:
        # Whole-batch parse failure: leave the events queued for the pacing retry /
        # next run rather than mislabeling all of them.
        # result is always bound here: LLMParseError is only raised by
        # _parse_batch_response, after call_llm has returned.
        logger.warning(
            "Batch parse error (%d events left queued): %s [model=%s finish_reason=%s head=%r]",
            len(llm_events), e, result.get("model", "?"),
            result.get("finish_reason", "?"), (result.get("content") or "")[:160],
        )
        # Garbage JSON is a slot-quality signal (degraded :free upstream), not a
        # prompt problem: sideline the slot so the next chunk rotates to the next
        # cascade slot instead of feeding the same broken upstream until fail-fast.
        router.penalize_model_slot(
            result.get("provider", ""), result.get("account", ""), result.get("model", ""),
        )
        _release_pending(requeue=True)
        stats["failed"] += len(llm_events)
        stats["parse_error"] = True
        return stats
    except Exception:
        db_conn.rollback()
        logger.exception("Unexpected batch classification error (%d events)", len(llm_events))
        _release_pending(requeue=True)
        stats["failed"] += len(llm_events)
        return stats

    for i, event in enumerate(llm_events, 1):
        item = items.get(i)
        try:
            if item is None:
                logger.warning("Batch response missing report %d (event %s) — left queued",
                               i, event["id"][:8])
                stats["failed"] += 1
                continue
            if _apply_llm_classification(db_conn, router, event, event["_det"], item, result, worker_id):
                stats["classified"] += 1
            else:
                stats["failed"] += 1
        except Exception:
            db_conn.rollback()
            logger.exception("Failed applying batch classification to event %s", event["id"][:8])
            stats["failed"] += 1
        finally:
            release_lock(db_conn, event["id"], worker_id, requeue=(item is None))

    return stats


# Pacing bounds — cap a single wait for a token-window refill, and the cumulative pacing
# time per run, so a real provider outage still aborts the pass promptly.
PASS_C_PACING_MAX_WAIT = 30.0
PASS_C_PACING_TOTAL_BUDGET = 180.0

# Abort the pass after this many whole-batch parse failures in a row: a model that
# systematically returns unparseable output would otherwise burn ~90s per chunk until
# the workflow's 30-minute timeout kills the run before Pass D/E and alerting.
PASS_C_MAX_CONSECUTIVE_PARSE_ERRORS = 3


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

    # Queue-depth telemetry: a saturated batch (available == limit) means events
    # are waiting more than one run for classification. Log-only by user request
    # (2026-07-09): internal pipeline chatter must not reach Telegram — the channel
    # is for incident alerts. The backlog is still visible in stats/telemetry and
    # this WARNING line.
    try:
        row = db_conn.execute(
            "SELECT COUNT(*) FROM events WHERE status = 'deduped' AND classification_lock = FALSE"
        ).fetchone()
        queue_depth = int(row[0]) if row else 0
        stats["queue_depth"] = queue_depth
        if queue_depth > QUEUE_DEPTH_ALERT_THRESHOLD:
            logger.warning(
                "Pass C classification queue depth is %d (threshold %d, per-run limit %d) "
                "— ingest is outpacing LLM classification capacity",
                queue_depth, QUEUE_DEPTH_ALERT_THRESHOLD, limit,
            )
    except Exception:
        logger.exception("Pass C: queue-depth check failed (non-fatal)")

    if not events:
        logger.info("Pass C: No events to classify")
        return stats

    # Pacing: when every slot is momentarily throttled (per-minute token windows drained),
    # wait for the soonest refill and retry rather than aborting the whole pass — TPM is
    # far tighter than RPM on the free tier, so a backlog otherwise stops after a handful
    # of events. Bounded per-wait and per-run so a genuine outage still fails fast.
    #
    # Batching: BATCH_CLASSIFY_SIZE > 1 classifies whole chunks per LLM call. A paced
    # retry re-runs the same chunk; events its first attempt already completed fail
    # acquire_lock and are skipped, so nothing is double-counted or re-billed.
    chunk_size = max(1, BATCH_CLASSIFY_SIZE)
    paced_total = 0.0
    exhausted = False
    consecutive_parse_errors = 0
    for start in range(0, len(events), chunk_size):
        chunk = events[start:start + chunk_size]
        while True:
            try:
                if chunk_size > 1:
                    batch = classify_event_batch(db_conn, router, chunk, worker_id)
                    stats["events_classified"] += batch["classified"]
                    stats["events_failed"] += batch["failed"]
                    if batch.get("parse_error"):
                        consecutive_parse_errors += 1
                    else:
                        consecutive_parse_errors = 0
                else:
                    result = classify_single_event(db_conn, router, chunk[0], worker_id)
                    if result:
                        stats["events_classified"] += 1
                    else:
                        stats["events_failed"] += 1
                break  # this chunk is done → move on
            except LLMRequestTooLarge as e:
                # This chunk's payload is the problem, not the accounts — a paced
                # retry of the identical prompt can never succeed. Leave the events
                # queued (locks were released with requeue) and move to the next chunk.
                logger.error("Pass C chunk of %d skipped, request too large: %s",
                             len(chunk), e)
                stats["events_failed"] += len(chunk)
                break
            except RuntimeError:
                wait = router.seconds_until_available()
                if (wait is None
                        or wait > PASS_C_PACING_MAX_WAIT
                        or paced_total + wait > PASS_C_PACING_TOTAL_BUDGET):
                    exhausted = True
                    break
                logger.info(
                    "Pass C paced: all slots throttled, waiting %.1fs for token refill",
                    wait,
                )
                time.sleep(wait + 0.5)
                paced_total += wait + 0.5
        if exhausted:
            stats["llm_exhausted"] = True
            logger.error("LLM accounts exhausted, stopping Pass C")
            break
        if consecutive_parse_errors >= PASS_C_MAX_CONSECUTIVE_PARSE_ERRORS:
            stats["aborted_on_parse_errors"] = True
            logger.error(
                "Pass C aborted: %d consecutive batch parse failures — LLM output is "
                "systematically unparseable, leaving remaining events queued",
                consecutive_parse_errors,
            )
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
