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

from src.core.geo import is_african
from src.core.heartbeat import HeartbeatWorker
from src.core.llm_client import LLMAllThrottled, LLMRequestTooLarge, call_llm, log_llm_telemetry
from src.core.llm_router import LLMRouter
from src.core.storyline import strip_date_hint
from src.pipeline.ingest_filters import (
    _HIGH_SIGNAL_TERMS,
    _SECURITY_KEYWORD_PATTERN,
    _is_airport_intrusion,
    _is_aviation_security_incident,
    _is_bare_security_incident,
    _is_flight_disruption,
    _is_screening_breach,
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

# `african_terrorism` is the one event type whose definition is geographic. The catalog
# makes `terrorism` its parent with the same severity_base (95), so demoting an
# out-of-region classification to the parent is label-only — it never changes scoring.
GEO_SCOPED_EVENT_TYPE = "african_terrorism"
GEO_SCOPED_FALLBACK = "terrorism"

# Aviation is the priority domain, and "which carrier stopped flying where" is
# the single highest-value line in these reports — but its vocabulary is low on
# the generic security keywords the relevance heuristic and LLM both key on, so
# genuine flight-disruption headlines ("Qatar Airways suspends flights to
# Bahrain") were being prescreen-dropped (det score 0) or LLM-archived (sev 0).
# A flight-disruption headline that is NOT weather noise is floored to this
# relevance so it always survives to scoring, where the aviation nexus bonus
# ranks it. Weather cancellations still fall through (is_noise catches them).
AVIATION_RELEVANCE_FLOOR = 40

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

# Batch classification: how many reports to classify per LLM call. The ~2650-token
# system prompt is paid ONCE per call instead of once per event, and each call burns
# one RPM slot for N events — the free tier's two scarcest currencies. Sized so a
# full batch (system + N truncated reports + JSON array output) stays inside Groq's
# 8K TPM window. 1 disables batching (classic per-event path).
#
# The prompt grew ~320 tokens on 2026-08-11 with the report_kind field, which is the
# largest single item of headroom spent so far. Anything else added here should come
# with the same arithmetic.
#
# Lowered 6 → 4 on 2026-08-12. Real batches of 6 peaked at ~7400 tokens (telemetry,
# 7 days) but a batch of 6 whose reports all hit the truncation limit estimates ~8270 —
# past Groq's 8K ceiling, where llm_client's size guard silently drops the slot. Those
# slots are exactly where the wall-clock ceiling (model_profiles item 5) sends work when
# the OpenRouter primary goes slow, and a failover target the payload might not fit is
# not a failover target. A batch of 4 tops out at ~6595, at the cost of ~50% more calls —
# which the free tier's RPM/RPD can absorb but its per-request window cannot.
BATCH_CLASSIFY_SIZE = int(_CLASSIFICATION.get("llm_batch_size", 4))
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

# Unambiguous hostile ACTS — a deliberate subset of _HIGH_SIGNAL_TERMS with every
# outcome word ("killed", "dead", "casualties"), every ambient-politics word ("war",
# "conflict", "sanctions", "nuclear") and every humanitarian word ("refugee",
# "famine") removed. Those belong in a recall-tuned relevance score; they are useless
# for deciding that a specific incident occurred, because a flood report and an
# opinion column both carry them.
#
# NOT used to override the LLM (see the note in _apply_llm_classification for why
# that failed). This measures how often the classifier archives something whose
# HEADLINE claims a hostile act — the honest false-negative signal. Watch
# pass_c.high_signal_archived: a rising count is the evidence that would justify
# building a real guard, and the sample to build it from.
#
# Calibration note (10 Aug 2026). The first version listed only NOUN phrases for each
# act ("drone attack", "missile strike"), so every headline that used the verb slipped
# past and the counter read as reassuring when it was not: over 700 archived events it
# matched 1, while a loose hostile-act scan matched 16. The verb forms below were added
# and re-measured on those same 700 — 1 → 22 matches, of which 20 are genuine hostile
# acts. The recovered sample is dominated by exactly the class this system has never
# surfaced: the Leipzig airport explosive-drone incident, archived across five separate
# headlines, plus "A drone carrying explosives attacked a Ukrainian An-124 in Germany".
HOSTILE_ACT_PATTERN = re.compile(
    # The plural `s?` on the three attack phrases is load-bearing. Without it
    # "drone attacks", "missile attacks" and "terrorist attacks" — three of the most
    # common constructions in the corpus — did not match, while the neighbouring
    # "drone strikes?" did. Measured 2026-08-17.
    r"\b(explosions?|bombings?|shelling|airstrikes?|air strikes?|missile strikes?|"
    r"missile attacks?|drone attacks?|drone strikes?|gunfire|assassinat(ion|ed)|"
    r"massacred?|ambush|suicide bomb(er)?|car bomb|truck bomb|improvised explosive|"
    r"terror(ist)?s? attacks?|artillery|mortar|kidnapped|abducted"
    # Verb forms of the same acts — "the enemy ATTACKED Kharkiv", "vessels ATTACKED".
    r"|attacked|bombed|shelled|stormed|detonated|hijacked|opened fire|shot dead|shot down"
    # Weapon + verb — "Russian drones TARGET Naftogaz", "drones HIT Erbil".
    r"|(drones?|missiles?|rockets?|uavs?)\s+(target(ed|s)?|hit|struck|strike[sd]?)"
    # ── Casualty verbs ────────────────────────────────────────────────────────
    # "kill" is the single most common verb in conflict reporting and was absent from
    # this vocabulary in every form, as were injure/wound/down. The whole list was
    # written around past-tense passive constructions while wire copy writes present
    # active. Measured 2026-08-17 over 14 days: 4131 events were prescreen-archived
    # at score 0 — "no security vocabulary at all" — and 89 of them were unambiguous
    # mass-casualty attacks, ~6 a day, none of which an LLM ever saw:
    # "Ukrainian drone kills 12, injured 39 in Russia's Tatarstan", "Ukrainian drone
    # kills seven on beach of Russian Black Sea resort" (The Times), "Terrorists
    # attack Benue communities, kill, injure residents" (Punch).
    #
    # Both frames are anchored — a weapon/actor subject, or an explicitly civilian
    # object — because a bare "kills" is where the metaphors live ("the deal kills
    # jobs"). Checked against the control set that motivated the anchoring: "the film
    # bombed at the box office", "workers prepare for strike", "stocks attack record
    # highs" and "transporters end strike" all stay unmatched.
    # The ['’"]? after the subject is not decoration. Wire copy scare-quotes the
    # attributed actor — "Moment 'Ukrainian drone' kills kids on beach" (News.com.au)
    # — and the closing quote sits between the noun and the verb, so a bare \s+ misses
    # it. That was the 1 of 12 genuinely-archived events this frame failed to recover.
    r"|(drones?|missiles?|rockets?|uavs?|strikes?|shelling|bombardment|troops|forces|"
    r"militants?|gunmen|rebels?|insurgents?|terrorists?|jets?|warplanes?|raids?)"
    r"['’\"]?\s+([\w'’-]+\s+){0,3}(kills?|killed|injur(e|es|ed)|wounds?|wounded)"
    r"|(kills?|killed|injur(e|es|ed)|wounded)\s+([\w'’-]+\s+){0,2}"
    r"(civilians?|people|residents?|children|women|worshippers|passengers|pilgrims)"
    # "NATO jet DOWNS drone", "air defences DOWNED 12 drones".
    r"|(down(s|ed)?|intercept(s|ed)?|shoot(s)? down)\s+([\w'’-]+\s+){0,3}(drones?|missiles?|aircraft|jets?|uavs?)"
    # ── Report frames ─────────────────────────────────────────────────────────
    # The verb forms above cover "drones ATTACKED X". They miss the shapes wire copy
    # uses just as often, because there the act is a bare NOUN carrying a preposition
    # or a delivery verb. Measured 2026-08-13 over 7 days: 168 of 2011 prescreen-archived
    # events match one of the four frames below (~24 extra classification calls a day).
    #
    # The sample that motivated them: the Novorossiysk naval-base strike was reported by
    # five separate outlets and ALL FIVE were archived unseen ("Ukraine launched a
    # coordinated attack on the Russian naval base…", "Ukraine Carries Out Major Strike
    # on…", "…Hits Russia's Novorossiysk Port"). The one report that happened to use a
    # covered phrasing paged at confidence 0.51 — and confidence is built from
    # corroborating sources, so the four gates that read it were reading a number those
    # five drops had suppressed. A prescreen miss is never one lost article.
    #
    # 1. Bare act noun + preposition: "attack ON the naval base", "attacks ON civilians".
    #    The noun is deliberately not listed alone — "under attack from critics" is a
    #    metaphor, and the preposition frame is what separates it from an incident.
    r"|(attacks?|strikes?|assaults?|offensives?|raids?|bombardments?)\s+(on|against)\b"
    # 2. Delivery verb + act: "LAUNCHED a coordinated attack", "CARRIED OUT a strike".
    r"|(launch(ed|es|ing)?|carr(y|ies|ied)\s+out|conduct(ed|s|ing)?|mount(ed|s|ing)?|"
    r"unleash(ed|es|ing)?)\s+(the\s+|a\s+|an\s+|its\s+|their\s+|[\w'’-]+\s+){0,3}"
    r"(attacks?|strikes?|raids?|offensives?|assaults?|bombardments?)"
    # 3. Armed-actor subject + kinetic verb. Subject-side rather than object-side because
    #    the object is usually a place name no dictionary holds ("Russian Forces SEIZE
    #    Vodyanoe"), and the armed subject is what keeps these bare verbs safe: "Boys,
    #    ages 4 and 7 … HIT woman walking her dog" and "Trucker CAPTURES pilot's
    #    maneuver" both carry the verb, neither has one.
    r"|(forces|troops|army|navy|air force|militants?|gunmen|rebels?|insurgents?|"
    r"fighters?|jets?|warplanes?|artillery|militia|units?)\s+([\w'’-]+\s+){0,2}"
    r"(hits?|struck|strikes?|target(ed|s)?|seiz(e|ed|es)|captur(e|ed|es)|overran|"
    r"overrun|shell(ed|s)?|storm(ed|s)?|raid(ed|s)?)"
    # 4. Kinetic verb + military/energy asset, for the headlines whose subject is a bare
    #    country name no subject list can hold ("Ukraine HITS Russia's Novorossiysk
    #    PORT"). The asset object plays the role the armed subject plays in frame 3.
    r"|(hits?|struck|targeted)\s+([\w'’-]+\s+){0,3}"
    r"(naval base|air ?base|military base|port|refinery|depot|terminal|substation|"
    r"power (plant|station|grid)|pipeline|airport|barracks|checkpoint|convoy|warehouse)"
    # Ordnance found rather than delivered — the Leipzig airport class.
    r"|explosive (device|drone|belt)"
    # Airspace incursion, the aviation-adjacent signal this pipeline exists to catch.
    r"|drones? (spotted|sighted|flew) over"
    r")\b",
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
    # The verb construction most wire copy actually uses. _HIGH_SIGNAL_TERMS and the
    # security keywords are built from NOUN PHRASES ("drone attack", "air strike", "car
    # bomb") — the right shape for precision, but they match nothing in "Drones ATTACKED
    # the petrochemical center" or "The enemy ATTACKED Kharkiv". HOSTILE_ACT_PATTERN
    # already carries those forms; it was written for the high_signal_archived counter,
    # so the two vocabularies were measuring different things while only one of them
    # decided anything.
    #
    # Measured 2026-08-11 over 7 days: 64 of 1779 prescreen-archived events matched the
    # verb forms and scored 0 — archived without an LLM ever seeing them, among them "A
    # drone carrying explosives attacked a Ukrainian An-124 in Germany". Costs ~9 extra
    # classification calls a day.
    #
    # Deliberately NOT folded into has_high_signal. That flag suppresses the is_noise()
    # penalty below, and a bare verb is exactly where the metaphors live: "the film
    # BOMBED at the box office", "stock market ATTACKED by inflation fears". is_noise()
    # catches both, so a verb-only match is scored as ordinary security vocabulary and
    # left subject to that veto, while a noun-phrase hit still overrides it.
    has_hostile_act = bool(HOSTILE_ACT_PATTERN.search(blob))
    has_casualty = bool(_CASUALTY_NUM_PATTERN.search(blob))
    noisy = is_noise(f"{title} {text[:500]}")
    # Aviation stopped flying, and not because of weather — the security scope.
    # is_noise() catches snowstorm/maintenance cancellations, so excluding noisy
    # here leaves only disruptions worth keeping (mirrors the ingest gate).
    has_flight_disruption = _is_flight_disruption(blob, title) and not noisy
    # A prohibited item carried through passenger screening. Measured 2026-08-27, the
    # class was invisible to every vocabulary above: "Businessman flies to Delhi with
    # 31 live rounds after passing through Dhaka airport security" scored 0 here with
    # has_security False, so the prescreen would have archived it unread even if the
    # ingest budget had let it through. Six such headlines in 14 days were archived
    # without an LLM ever seeing them, among them "Live bullet found aboard United
    # flight". Unlike the disruption flag this one is NOT suppressed by is_noise():
    # the conjunction already requires three vocabularies to meet in one headline,
    # which is where the metaphors do not survive.
    has_screening_breach = _is_screening_breach(title)
    # Bomb threat, runway incursion, drone sighting, GNSS jamming, laser, stowaway.
    # Measured 2026-08-27: ten such headlines in 14 days were archived here at score 0,
    # including a stowaway found dead in a wheel well at Gatwick and three filings of
    # the Sydney runway-incursion investigation whose siblings were scored normally.
    has_aviation_incident = _is_aviation_security_incident(title)
    # A bare security noun reporting harm. The lexicon carries phrases, not the bare
    # words, so "Russian attack damages Nova Poshta warehouses in Kyiv Oblast" matched
    # nothing at all. Measured 2026-08-31 by the weekly vocabulary audit: every one of
    # the 2273 events the prescreen archived in seven days scored exactly 0, and 102 of
    # them name a security noun alongside something killed, wounded or destroyed.
    #
    # Counted as ordinary security vocabulary, NOT as high signal: it stays subject to
    # the is_noise() penalty below, which is the existing guard against the metaphors
    # this vocabulary is full of. That is deliberate and different from the
    # screening-breach flag, whose three-way conjunction leaves no room for metaphor.
    has_bare_incident = _is_bare_security_incident(title)
    # A person through the fence, onto the airfield, into the wheel well. Measured
    # 2026-08-31: 8 such headlines in seven days, all prescreen-archived, including a
    # Manchester breach that diverted 20+ flights and a stowaway found dead at Gatwick.
    has_airport_intrusion = _is_airport_intrusion(title)
    has_security = (has_high_signal or has_flight_disruption or has_hostile_act
                    or has_screening_breach or has_aviation_incident
                    or has_bare_incident or has_airport_intrusion
                    or bool(_SECURITY_KEYWORD_PATTERN.search(blob)))

    score = 0
    if has_high_signal:
        score += 45
    elif has_security:
        score += 25
    if has_flight_disruption:
        score += 20  # ensure it clears the prescreen floor even with no other keyword
    if has_screening_breach:
        score += 20  # same reason: clears the floor (15) even against the noise penalty
    if has_aviation_incident:
        score += 20
    if has_airport_intrusion:
        score += 20
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
        "has_hostile_act": has_hostile_act,
        "has_flight_disruption": has_flight_disruption,
        "has_screening_breach": has_screening_breach,
        "has_aviation_incident": has_aviation_incident,
        "has_bare_incident": has_bare_incident,
        "has_airport_intrusion": has_airport_intrusion,
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
   laser_attack, suspicious_package, evacuation, airspace_closure, gnss_interference,
   security_incident, aviation_personnel_attack, pilot_attacked, cabin_crew_attacked, ground_staff_attacked,
   geopolitical_conflict, military_action, missile_strike, war_escalation, ceasefire_violation, civilian_casualties,
   political_event, civil_unrest, protest, mass_demonstration, riot, general_strike, coup_attempt,
   terrorism, african_terrorism, insurgency_attack, extremist_violence, jihadist_attack,
   mass_casualty_event, mass_shooting, mass_stabbing, suicide_bombing, vehicle_ramming,
   resort_attack, beach_attack, tourist_bus_attack, cruise_ship_attack,
   travel_advisory, travel_ban, embassy_closure,
   other_aviation_related,
   noise

   USE gnss_interference when GPS/GNSS signals are JAMMED or SPOOFED in a way that
   affects aircraft or shipping — including reports of navigation loss, false position
   or false terrain warnings in a named airspace or corridor. The hazard travels with
   the aircraft rather than sitting at a place, so a route or region is a valid
   location for it; do not downgrade it to security_incident for lacking a point.

   USE airspace_closure when airspace or an airport is CLOSED, SUSPENDED, RESTRICTED
   or flights are HALTED/DIVERTED/REROUTED — this is the single most operationally
   actionable aviation event, so do not fall back to other_aviation_related for it.
   Set sub_type to "planned" when the closure is scheduled and benign (air show,
   flypast, national day parade, military exercise, VIP movement, maintenance) and to
   "incident" when it follows a drone sighting, strike, attack, accident, volcanic ash,
   security threat or conflict. If the article does not say which, use "incident" —
   an unexplained closure is the case worth looking at.

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

13. report_kind: What this ARTICLE is, independent of how serious its subject is.
    An alert claims something is happening NOW, so a report ABOUT an earlier incident
    must be recognisable even when the incident itself was severe. One of:
    - "new_incident": reports an incident, attack, strike or operational disruption as
      NEWLY happening. Includes a closure, suspension or evacuation being announced now
      (e.g. "Airport suspends operations after drone sighting"), and includes a first
      report that is still fragmentary.
    - "followup": further developments in an incident already reported — arrests,
      charges, trials, sentences, funerals, compensation, demolitions, repairs,
      reopenings, released footage or police briefings, revised death tolls, "one week
      later" / anniversary pieces, recovery and rescue-completed stories.
      Forensic and mortuary procedure belongs here too: autopsies and post-mortems,
      bodies recovered from rubble, remains identified or repatriated, burials. The
      procedure is evidence that the incident is already over and already reported.
    - "roundup": covers several separate incidents at once, or is a running live blog /
      daily war summary / timeline ("Day 1,625", "live updates", "key moments").
    - "commentary": analysis, opinion, explainer, or pure reaction — condemnations,
      statements, warnings, diplomatic responses — with no new physical event.
    Judge the ARTICLE, not the subject: "Police release video of the mass shooting" is
    followup even though a mass shooting is severe, and "Autopsy conducted on the
    shooting suspect" is followup because the news is the autopsy, not the shooting.
    Asking "did something new physically happen today?" is the test — an autopsy, an
    arrest or a funeral is not that. If a report describes a fresh incident AND adds
    later detail, it is new_incident. When genuinely unsure, answer new_incident.

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
- African terrorism → event_type: african_terrorism. ONLY for events physically
  located in Africa (Sahel, Horn of Africa, Lake Chad, Mozambique). NEVER use it for
  Asia or the Middle East — Pakistani, Indian or Afghan insurgency is `terrorism` or
  `insurgency_attack`.
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


# How far a year-repaired timestamp may sit from the article's own publication date
# before the repair is refused. Deliberately tight: at this distance the model is
# echoing the date it read in the article, which is the only case the repair is for.
STALE_YEAR_TOLERANCE_DAYS = 2


def _repair_stale_year(parsed, published_at):
    """Rescue a timestamp whose day is right and whose YEAR is the model's own.

    Models anchor absolute dates to their training cutoff: measured 2026-08-13 over the
    current classification corpus, 110 occurred_at estimates fell outside the sane
    window and 67 of them (61%) were correct to within two days once the year was
    replaced with the article's — "2024-08-12" for a piece published 2026-08-12, on
    events as real as the Novorossiysk state-of-emergency CRITICAL. Discarding those
    threw away a usable incident time and fell back to publication time, which is a
    different quantity and the one Pass D's fallback exists to avoid.

    Only the article's own date can separate this from a genuine retrospective, so a
    repair is attempted solely against `published_at` and only within
    STALE_YEAR_TOLERANCE_DAYS of it. A real anniversary piece ("2023-08-06" in a story
    published 2026-08-13) lands a week out and stays discarded.

    Returns the repaired datetime, or None when no repair is warranted.
    """
    if published_at is None:
        return None
    if published_at.tzinfo is not None:
        published_at = published_at.astimezone(timezone.utc).replace(tzinfo=None)

    best = None
    # Both neighbouring years, so a repair still works across a New Year boundary —
    # "2024-12-31" in a piece published 2026-01-02 belongs to 2025, not 2026.
    for year in (published_at.year, published_at.year - 1):
        try:
            candidate = parsed.replace(year=year)
        except ValueError:
            continue  # 29 Feb into a common year
        if abs(candidate - published_at) > timedelta(days=STALE_YEAR_TOLERANCE_DAYS):
            continue
        if not _within_sane_bounds(candidate):
            continue
        if best is None or abs(candidate - published_at) < abs(best - published_at):
            best = candidate
    return best


def _parse_occurred_at(raw: str | None, published_at=None):
    """Safely parse LLM's occurred_at ISO 8601 string into a naive datetime.
    Returns None if the value is missing, empty, unparseable, or outside sane bounds.

    `published_at` is optional and only enables the stale-year repair below; callers
    that omit it keep the original discard-on-out-of-bounds behaviour.
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
        repaired = _repair_stale_year(parsed, published_at)
        if repaired is not None:
            logger.info("Repaired stale-year occurred_at estimate: %s → %s",
                        raw[:40], repaired.isoformat())
            return repaired
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


# The article-shape classes the alert gate reads (see REPORT_KIND_NOT_NEWS in
# core.alerts). NEW_INCIDENT is the fail-safe: an absent, misspelled or invented value
# resolves to it, so a degraded model reply can only ever let alerts through, never
# silence them.
REPORT_KIND_NEW = "new_incident"
REPORT_KINDS = {REPORT_KIND_NEW, "followup", "roundup", "commentary"}


# Vocabulary keyed on letters alone, so "follow-up", "follow up" and "followup" are one
# value. Models are consistent about the WORD and careless about the separator.
_REPORT_KIND_BY_LETTERS = {k.replace("_", ""): k for k in REPORT_KINDS}


def _safe_report_kind(value) -> str:
    """Coerce the LLM's report_kind to a known class, defaulting to new_incident.

    Deliberately strict about the vocabulary and lenient about everything else: this
    field can only ever WITHHOLD a page, so an unrecognised value must not be trusted
    to mean "not news". Models paraphrase enum values ("follow-up", "Roundup "), which
    is worth normalising; they also invent them, which is not worth guessing at.
    """
    if not isinstance(value, str):
        return REPORT_KIND_NEW
    letters = re.sub(r"[^a-z]", "", value.lower())
    return _REPORT_KIND_BY_LETTERS.get(letters, REPORT_KIND_NEW)


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
    """Record one CLAIM this domain made, and whether the claim held up.

    penalty_score is read as credibility — fetch_penalized_domains() bars a domain from
    label_cluster()'s independence count and the official-source check, and
    check_domain_penalty() drops its items at ingest. So the denominator has to be the
    claims a domain made, not everything it published.

    It used to be everything it published, and the two are completely different
    measurements. Bloomberg publishes finance, CNBC publishes markets: most of what they
    give us is correctly archived as off-topic, which drove them to penalty_score >= 0.5
    and disqualified them as corroborating sources. Measured 2026-08-13: of the 1679
    domains at >= 0.5, the list included reuters.com, bloomberg.com, cnbc.com, ft.com,
    thetimes.com, washingtonpost.com, economist.com, politico.eu, nhk.or.jp and
    aa.com.tr — i.e. the signal was ranking outlets by how much non-security news they
    write, and the SITREP was treating that as unreliability.

    So callers must invoke this ONLY for items that claimed a security event:
      is_noise=1  the headline claimed a hostile act and the classifier found no
                  incident behind it — the clickbait class, the one thing this score
                  should be measuring;
      is_noise=0  the claim held up and the event was classified.
    An article that never claimed anything is not an observation about the domain's
    credibility and must not touch either counter — incrementing total_events alone
    would be just as wrong in the other direction, diluting real offenders toward 0.
    """
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
    # Deliberately NO domain penalty here. Reaching this line means the article carried
    # no security vocabulary at all — an off-topic subject, not a false claim — so it
    # says nothing about whether this outlet can be believed (see update_domain_penalty).
    # Charging it was also self-reinforcing: every headline the prescreen's vocabulary
    # could not read punished the outlet that reported it, and the penalty then withheld
    # that outlet's corroboration. Measured 2026-08-13, that mis-charged 1538 archives
    # across 426 domains, refunded in db/maintenance/2026-08-13_refund_*.sql.
    #
    # The clickbait shape the score DOES want ("Iran Launches Missile Attack On Bahrain"
    # with nothing in the body) cannot land here: a hostile-act headline scores at or
    # above the floor, so it is judged on the LLM path where the charge is made.
    with db_conn.transaction():
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


# Active event-type codes, read once per process instead of twice per event.
#
# The catalog is 59 rows and changes when someone edits it, not while a run is in
# flight, but it was being probed with two single-row SELECTs for every classified
# event — 200 round trips across the Supabase WAN on a 100-event run, to answer a
# question whose whole answer set fits in a few hundred bytes.
_ACTIVE_CODES: set[str] | None = None


def _active_event_codes(db_conn) -> set[str]:
    """The active catalog as a set, cached for the life of the process.

    On a query failure it returns an empty set rather than caching it, so the next
    event retries instead of silently rejecting every type for the rest of the run.
    """
    global _ACTIVE_CODES
    if _ACTIVE_CODES is not None:
        return _ACTIVE_CODES
    try:
        rows = db_conn.execute(
            "SELECT code FROM event_type_catalog WHERE active = TRUE"
        ).fetchall()
    except Exception:
        logger.exception("Active event-type catalog lookup failed")
        return set()
    _ACTIVE_CODES = {r[0] for r in rows}
    return _ACTIVE_CODES


def _apply_llm_classification(db_conn, router: LLMRouter, event: dict, det: dict,
                              parsed: dict, result: dict, worker_id: uuid.UUID,
                              log_telemetry: bool = True) -> dict | None:
    """Apply a parsed LLM classification to an event (tiering, validation, DB update).

    Shared by the single-event and batched paths. Caller holds the lock.

    log_telemetry=False for the batched path: one LLM call covers the whole chunk, so
    the caller logs it once. Logging here per event wrote the SAME call N times (same
    latency_ms, same token counts), inflating system_telemetry ~4.7x and making every
    per-call metric derived from it — call counts, average latency, token totals —
    wrong by that factor.
    """
    event_id = event["id"]
    source_domain = event.get('source_domain', 'unknown') or 'unknown'

    # Graduated relevance handling using LLM's relevance_score
    event_type = parsed.get("event_type", FALLBACK_EVENT_TYPE)
    relevance = _safe_relevance(parsed.get("relevance_score", 50))

    # There used to be an "LLM false-negative guard" here: any _HIGH_SIGNAL_TERMS hit
    # with relevance < 30 floored relevance to 30 and kept the event, on the reasoning
    # that a low-priority event beats a missed incident. Measured over the 7 days to
    # 2026-08-10 it rescued 642 events and produced ZERO alerts — and it could not have
    # produced one, because rescuing a "noise" verdict also stamps FALLBACK_EVENT_TYPE,
    # whose catalog severity_base is 20 against an ALERT floor of 65 and a severity
    # floor of 90. The rescue and the cap were the same line of code: a safety net woven
    # from the hole it was meant to cover.
    #
    # The trigger was hopeless too, for a reason worth remembering: _HIGH_SIGNAL_TERMS
    # exists to SCORE relevance, so it is deliberately broad ("war", "conflict",
    # "killed", "sanctions", "refugee", "nuclear"), and it matched title+body. A
    # recall-tuned vocabulary makes a terrible precision-tuned veto. Narrowing it to
    # unambiguous hostile acts in the title alone still left 38 rescues of which 2 were
    # real incidents — the rest were a tech blog on the domain explosion.com, a metal
    # band called Car Bomb, the 1933 Simele massacre and a kidnapped Serbian eagle.
    #
    # It also laundered domain reputations: the keep path credits the domain
    # (update_domain_penalty(..., 0)), so explosion.com sat at penalty_score 0.000 on
    # 3 rescued events. Archiving them scores those domains honestly.
    #
    # Replaced by measurement rather than another guess — see HOSTILE_ACT_PATTERN and
    # the high_signal_archived counter, which make the real false-negative rate visible
    # so any future guard can be built on evidence.

    # Aviation-priority guard: a genuine flight disruption (not weather) is never
    # archived. Floored above the noise tiers so it survives to scoring, where
    # the aviation nexus bonus ranks it. Mirrors the high-signal guard above.
    if det.get("has_flight_disruption") and relevance < AVIATION_RELEVANCE_FLOOR:
        logger.warning(
            "Event %s: LLM relevance=%d but flight-disruption present — flooring to %d, keeping event",
            event_id[:8], relevance, AVIATION_RELEVANCE_FLOOR,
        )
        relevance = AVIATION_RELEVANCE_FLOOR
        if event_type == "noise":
            event_type = FALLBACK_EVENT_TYPE
            parsed["event_type"] = event_type

    # Tier 1: Clear noise (relevance < 20) → archive immediately
    # Use 'unclassified' as FK-safe type; the real signal is status='archived'
    # The original LLM classification is preserved in llm_parsed_output for auditing
    if relevance < 30 or (event_type == "noise" and relevance < 40):
        archive_type = FALLBACK_EVENT_TYPE  # FK-safe fallback
        # The headline asserted a hostile act and the classifier, having read the body,
        # found no incident behind it. That gap IS the clickbait class, and it is the only
        # archive that says anything about whether this outlet can be believed — an
        # off-topic subject (a finance column, a film release) is charged nothing, because
        # penalty_score is read as credibility, not as topicality. See
        # update_domain_penalty. Computed here rather than after the write below because
        # the charge belongs inside the same transaction as the archive.
        claimed_hostile_act = bool(HOSTILE_ACT_PATTERN.search(event.get("source_title") or ""))
        with db_conn.transaction():  # penalty + archive land together (conn is autocommit)
            if claimed_hostile_act:
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
        if log_telemetry:
            log_llm_telemetry(db_conn, result, router, success=True,
                              purpose="classify_single")
        logger.info("Event %s archived — relevance=%d, llm_type=%s, reason=%s",
                    event_id[:8], relevance, event_type,
                    parsed.get("relevance_reasoning", "")[:80])
        # Flagged AFTER the row is written so this private marker never lands in the
        # stored llm_parsed_output. Counted by the callers into
        # pass_c.high_signal_archived — the measurement that replaced the old
        # false-negative override (see the note above). Same condition the penalty above
        # charges on: this counter and that charge are now two readings of one signal, so
        # a rise in high_signal_archived is also the domain-penalty inflow.
        if claimed_hostile_act:
            logger.info("Event %s archived despite a hostile-act headline: %.90s",
                        event_id[:8], event.get("source_title") or "")
            parsed["_high_signal_archived"] = True
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

    # Validate event_type and sub_type against the active catalog. An empty set means
    # the lookup failed, and rejecting every type on a transient DB error would rewrite
    # a whole run as unclassified — so treat it as "cannot check" and keep the label.
    active_codes = _active_event_codes(db_conn)
    if active_codes and event_type not in active_codes:
        event_type = FALLBACK_EVENT_TYPE

    sub_type = parsed.get("sub_type")
    if sub_type and active_codes and sub_type not in active_codes:
        sub_type = None

    # Sanitize country_iso: must be exactly 2 uppercase ASCII letters
    raw_iso = parsed.get("country_iso") or ""
    country_iso = raw_iso.strip().upper()[:2] if raw_iso else None
    if country_iso and (len(country_iso) != 2 or not country_iso.isalpha()):
        country_iso = None

    # Normalize report_kind into `parsed` (the dict stored as llm_parsed_output) so the
    # alert gate reads a known value on every event, including ones classified before
    # the field existed. Both classification paths land here, so this is the single
    # place the vocabulary is enforced.
    parsed["report_kind"] = _safe_report_kind(parsed.get("report_kind"))

    # Keep the geographically-scoped type inside its geography. The prompt scopes
    # african_terrorism to the Sahel / Horn of Africa, but the models reach for it on
    # generic insurgency copy regardless of where the event happened. Falls back to its
    # own catalog parent, `terrorism`, which carries the identical severity_base (95),
    # so this corrects the label without moving the event's priority.
    if event_type == GEO_SCOPED_EVENT_TYPE and not is_african(country_iso):
        logger.info(
            "Event %s reclassified %s→%s (country=%s is not African)",
            event_id[:8], event_type, GEO_SCOPED_FALLBACK, country_iso or "unknown",
        )
        event_type = GEO_SCOPED_FALLBACK
        parsed["event_type"] = event_type

    # Parse occurred_at from LLM output into a timestamp. The article's own date is
    # passed in so a stale-year estimate can be repaired rather than discarded.
    occurred_at_est = _parse_occurred_at(parsed.get("occurred_at"), event.get("published_at"))

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

    # Log telemetry (batched path logs once for the whole chunk instead)
    if log_telemetry:
        log_llm_telemetry(db_conn, result, router, success=True,
                          purpose="classify_single")

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


# Each batch item is instructed to lead with its "report" number (BATCH_SYSTEM_SUFFIX),
# and every corrupt sample observed in prod honors that. Anchoring salvage on this
# pattern rather than on brace depth is deliberate — see _salvage_batch_items.
_REPORT_OBJECT_START = re.compile(r'\{\s*"report"\s*:')


def _first_parseable_prefix(chunk: str) -> dict | None:
    """Longest-valid-object-from-the-left: the first `}` whose prefix json.loads.

    Walking closing braces left to right handles nested objects for free — a prefix
    ending at an INNER `}` is unbalanced and fails to parse, so the scan simply
    continues to the brace that actually closes the item.
    """
    for i, ch in enumerate(chunk):
        if ch != "}":
            continue
        try:
            obj = json.loads(chunk[:i + 1], strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _salvage_batch_items(content: str) -> list[dict]:
    """Recover the individually-valid report objects from a corrupt batch reply.

    A free-tier model can drop a garbage token mid-object (Nemotron emitted
    `"anchor_name": "Kyiv",",` with finish_reason=stop, 2026-08-05) — that breaks the
    OUTER json.loads and used to strand all six events in the chunk even though five
    objects were perfectly well-formed. Recovering them turns a 6-event loss into a
    1-event loss.

    Brace-depth scanning does NOT work here, which is why this splits on the
    `{"report":` marker instead: that stray `",` leaves an odd number of quotes, so
    every following `"` has inverted meaning and the item's real `}` reads as being
    inside a string. Parity corruption defeats any depth counter, but it cannot move
    the literal `{"report":` markers, so the chunk boundaries survive it. The cost is
    an assumption — if a model stops leading with "report", salvage finds nothing and
    the caller falls back to the old whole-batch failure. No regression, just no rescue.

    Only objects carrying an explicit integer "report" survive: a partial list can't
    be mapped by position without silently pinning classifications to the wrong events.
    """
    starts = [m.start() for m in _REPORT_OBJECT_START.finditer(content)]
    items: list[dict] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(content)
        obj = _first_parseable_prefix(content[start:end])
        # bool is an int subclass — a JSON `true` must not pass as report number 1.
        if obj is not None and isinstance(obj.get("report"), int) \
                and not isinstance(obj.get("report"), bool):
            items.append(obj)
    return items


def _parse_batch_response(content: str, expected: int) -> dict[int, dict]:
    """Parse a batch response into {report_number: parsed_item}.

    Outer-JSON failures fall back to per-object salvage; only a reply with nothing
    recoverable raises LLMParseError (whole batch stays queued). Per-item defects
    just drop that item — its event stays queued.
    """
    try:
        parsed = validate_and_parse(content)  # reuses markdown/trailing-comma repair
        results = parsed.get("results")
        if not isinstance(results, list):
            raise LLMParseError("Batch response missing 'results' array")
    except LLMParseError:
        results = _salvage_batch_items(content)
        if not results:
            raise
        logger.warning(
            "Batch outer JSON unparseable — salvaged %d/%d report objects individually",
            len(results), expected,
        )
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
    stats = {"classified": 0, "failed": 0, "high_signal_archived": 0}
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
        # TOTAL parse failure — nothing survived even per-object salvage. A reply that
        # was merely dented (one corrupt object among six) never reaches here: it is
        # salvaged, and the surviving events are applied. That asymmetry is deliberate.
        # The slot penalty below sidelines the model for the rest of the run, which
        # costs far more than one requeued event — worth it when the slot returns pure
        # garbage, not worth it for a single bad token.
        # Leave the events queued for the pacing retry / next run rather than
        # mislabeling all of them.
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
        # Record the failure so a model's true garbage-JSON rate is measurable —
        # successes already log telemetry, so without this a degrading :free slot
        # (e.g. Nemotron) stays invisible until it starves the run.
        log_llm_telemetry(db_conn, result, router, success=False,
                          purpose="classify_batch")
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

    # One call covered the whole chunk — log it once, here, rather than once per
    # event inside the apply loop (which recorded the same call N times over).
    log_llm_telemetry(db_conn, result, router, success=True, purpose="classify_batch")

    for i, event in enumerate(llm_events, 1):
        item = items.get(i)
        try:
            if item is None:
                logger.warning("Batch response missing report %d (event %s) — left queued",
                               i, event["id"][:8])
                stats["failed"] += 1
                continue
            applied = _apply_llm_classification(db_conn, router, event, event["_det"], item,
                                                result, worker_id, log_telemetry=False)
            if applied:
                stats["classified"] += 1
                if applied.get("_high_signal_archived"):
                    stats["high_signal_archived"] += 1
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
# the workflow timeout (55 min) kills the run before Pass D/E and alerting.
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
        # Events the classifier archived even though their HEADLINE claimed a hostile
        # act. This is the honest false-negative signal that replaced the old override
        # (642 rescues, 0 alerts, over the 7 days to 2026-08-10). If this climbs, the
        # classifier — not a keyword veto — is what needs fixing.
        "high_signal_archived": 0,
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
                    stats["high_signal_archived"] += batch.get("high_signal_archived", 0)
                    if batch.get("parse_error"):
                        consecutive_parse_errors += 1
                    else:
                        consecutive_parse_errors = 0
                else:
                    result = classify_single_event(db_conn, router, chunk[0], worker_id)
                    if result:
                        stats["events_classified"] += 1
                        if result.get("_high_signal_archived"):
                            stats["high_signal_archived"] += 1
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
