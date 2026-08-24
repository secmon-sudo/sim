"""
SIM — Storyline Adjudicator (Layer 2)
Blueprint V20.1 §PASS D (hybrid storyline linking)

Deterministic linking (`should_link_storyline`) groups the easy cases for free, but
leaves a hard residue: multiple sources reporting the SAME real-world incident whose
paraphrased hints share almost no tokens (e.g. "Kyiv Russia drone strike" vs
"Ukrainian capital missile attack"). Coarse geo-assist flags these as same-place but
deliberately refuses to merge them on geography alone (two DISTINCT same-city events
must not collapse).

This adjudicator resolves exactly that residue with a bounded LLM call:
  1. Only fires when deterministic linking found NO match.
  2. Only considers candidates that share country + coarse location within a tight
     window (the plausibly-same set) — never the whole storyline table.
  3. Runs on the BULK router (gpt-oss-20b), so it never competes with Pass C
     classification for smart-model quota.

If anything goes wrong (no candidates, LLM error, unparseable reply) it returns None
and the caller creates a fresh storyline — i.e. it can only ever MERGE, never lose an
event, and it fails safe toward "new storyline".
"""

import json
import logging
import re

from src.core.geo import geo_key, trusted_anchor
from src.core.llm_client import call_llm, log_llm_telemetry
from src.core.storyline import lexical_kinship

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise event-deduplication assistant for an OSINT security pipeline. "
    "You decide whether a news event describes the SAME real-world incident as an "
    "existing storyline, or a NEW distinct incident. Answer ONLY with strict JSON."
)


# A location key that carries no discriminating signal — geo_key's normalized-text
# fallback yields these for events whose location the model could not resolve
# ("Unknown", ""). Treated as "no usable geo" so such events take the country path.
_DEGENERATE_GEO = {"", "UNKNOWN"}


def _event_geo(ev: dict) -> str | None:
    """Coarse location key for an event: precise IATA anchor, else geo_key of raw text.

    Returns None when the only key available is degenerate ("Unknown"), so a genuinely
    location-less event is routed to the country-level fallback rather than being matched
    against every other unresolved-location event as if they shared a real place.
    """
    g = trusted_anchor(ev) or geo_key(
        ev.get("anchor_name_raw"), ev.get("country_iso")
    )
    if g and g.strip().upper() in _DEGENERATE_GEO:
        return None
    return g


def find_geo_candidates(
    event: dict,
    recent_events: list[dict],
    window_hours: float = 48.0,
    max_candidates: int = 6,
    lexical_floor: float = 0.10,
) -> list[dict]:
    """Candidate storylines that plausibly describe the same incident.

    Two candidate nets, both within a tight time window, one representative hint per
    storyline_id, and both deliberately LLM-adjudicated afterwards (the caller only ever
    MERGES on an explicit same-incident verdict):

    - **Geo net** (event has a resolvable location): same country + same coarse location.
      Intentionally the SAME set the deterministic geo-assist saw but could not confirm
      lexically — the ambiguous residue the LLM should judge.
    - **Country net** (everything else, including located events whose location matched
      nothing): same country plus a minimal lexical-overlap floor, ranked by that overlap
      so the most plausible duplicates fill the bounded candidate slots. It carries the
      location-less national news (missile tests, nuclear announcements) that would
      otherwise bypass every dedup layer, and the cases where one report names a region
      and another the town inside it ("Jammu Kashmir" vs "Kulgam") — coarse keys are not
      a containment test, so a geo-only net never offered those to the model at all.
    """
    dt = event.get("occurred_at_est")
    if dt is None:
        return []
    ev_geo = _event_geo(event)
    iso = event.get("country_iso")
    # Geo net needs a location; country fallback needs a country. With neither, there is
    # nothing coarse enough to gather a plausibly-same set from.
    if not ev_geo and not iso:
        return []
    ev_hint = event.get("storyline_hint") or ""

    scored: list[tuple[float, str, str]] = []  # (lexical_overlap, storyline_id, hint)
    seen_storylines: set[str] = set()
    for r in recent_events:
        sid = r.get("storyline_id")
        if not sid or sid in seen_storylines:
            continue
        r_dt = r.get("occurred_at_est")
        if r_dt is None:
            continue
        try:
            if abs((dt - r_dt).total_seconds()) > window_hours * 3600:
                continue
        except Exception:
            continue
        r_iso = r.get("country_iso")
        r_hint = r.get("storyline_hint") or ""
        if ev_geo and _event_geo(r) == ev_geo:
            # Geo net: same country (when both known) + same coarse location.
            if iso and r_iso and iso != r_iso:
                continue
            overlap = 1.0
        else:
            # Country net: same country required (both sides known), and some lexical
            # kinship — a wholly unrelated same-country incident is not a candidate.
            #
            # This also runs for events that DO have a location, because a coarse
            # key is not a containment test: "Kashmir" and "Kulgam" are the same
            # incident reported at region and at town level, and a geo-only net
            # meant the town's storyline was never even offered to the model.
            # Geo-net hits still outrank these (1.0 vs < 1.0).
            if not r_iso or r_iso != iso:
                continue
            # The floor also bounds LLM volume: every candidate list that is not
            # empty costs a bulk-model call, and this net now sees located events
            # too. One incidental shared word is not worth a call.
            overlap = lexical_kinship(ev_hint, r_hint)
            if overlap < lexical_floor:
                continue
        seen_storylines.add(sid)
        scored.append((overlap, sid, r_hint))

    # Highest lexical overlap first so the real duplicate lands inside max_candidates even
    # when the country net is broad; geo-net ties keep insertion (recency) order.
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"storyline_id": s, "hint": h} for _, s, h in scored[:max_candidates]]


def _build_prompt(event: dict, candidates: list[dict]) -> str:
    lines = [
        "NEW EVENT:",
        f"  location: {event.get('anchor_name_raw') or event.get('anchor_name_norm') or '?'}",
        f"  hint: {event.get('storyline_hint') or ''}",
        f"  title: {(event.get('source_title') or '')[:200]}",
        "",
        "EXISTING STORYLINES (same country and location, near in time):",
    ]
    for i, c in enumerate(candidates, 1):
        lines.append(f"  [{i}] {c['hint']}")
    lines += [
        "",
        "Does the NEW EVENT describe the SAME real-world incident as one of the existing "
        "storylines? Same location and same day do NOT automatically mean the same "
        "incident — match ONLY if it is genuinely the same event (same strike, same "
        "attack, same operation, same target). If it is a separate incident, answer NEW.",
        "Note: two reports that name the same place at different levels of detail are "
        "still the SAME incident — a state and a town inside it, or a city and a venue "
        "inside it, are not two different places.",
        'Reply with strict JSON only: {"match": <number of the matching storyline, or "NEW">}.',
    ]
    return "\n".join(lines)


def _parse_match_index(content: str, n_candidates: int) -> int | None:
    """Map the LLM reply to a 0-based candidate index, or None for NEW/unparseable."""
    if not content:
        return None
    val = None
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            val = json.loads(m.group(0)).get("match")
        except Exception:
            val = None
    if val is None:
        # Fallbacks: explicit NEW, else first standalone integer.
        if re.search(r"\bNEW\b", content, re.I):
            return None
        num = re.search(r"\b(\d+)\b", content)
        val = num.group(1) if num else None
    if isinstance(val, str) and val.strip().upper().startswith("NEW"):
        return None
    try:
        idx = int(val)
    except (TypeError, ValueError):
        return None
    if 1 <= idx <= n_candidates:
        return idx - 1
    return None


def _parse_decision(content: str, candidates: list[dict]) -> str | None:
    """Map the LLM reply to a storyline_id, or None for NEW/unparseable."""
    idx = _parse_match_index(content, len(candidates))
    return None if idx is None else candidates[idx]["storyline_id"]


_DUP_SYSTEM_PROMPT = (
    "You are a precise duplicate-detection assistant for an OSINT security alerting "
    "pipeline. You decide whether a candidate alert reports the SAME real-world incident "
    "as an alert that was already sent. Answer ONLY with strict JSON."
)


def _describe(ev: dict) -> str:
    """One line of what an alert is about — title first, since it is always present."""
    title = (ev.get("source_title") or "").strip()[:160]
    hint = (ev.get("storyline_hint") or "").strip()[:120]
    loc = (ev.get("anchor_name_raw") or ev.get("anchor_name_norm") or "").strip()
    parts = [p for p in (title, hint, loc) if p]
    return " · ".join(parts) if parts else "(no description)"


def _build_duplicate_prompt(event: dict, paged: list[dict]) -> str:
    lines = [
        "CANDIDATE ALERT (not yet sent):",
        f"  {_describe(event)}",
        "",
        "ALERTS ALREADY SENT (within the last few hours):",
    ]
    for i, p in enumerate(paged, 1):
        lines.append(f"  [{i}] {_describe(p)}")
    lines += [
        "",
        "Would sending the CANDIDATE be a DUPLICATE — does it report the same real-world "
        "incident as one of the alerts already sent? Two outlets covering one incident "
        "with different wording, different casualty counts, or different levels of "
        "geographic detail (a region and a town inside it) are the SAME incident.",
        "Different incidents in the same country on the same day are NOT duplicates, even "
        "when they are the same kind of event (two separate strikes, two separate "
        "bombings, an attack and the retaliation for it).",
        'Reply with strict JSON only: {"match": <number of the duplicated alert, or "NEW">}.',
    ]
    return "\n".join(lines)


def adjudicate_duplicate_page(
    event: dict,
    paged: list[dict],
    router,
    *,
    call_llm_fn=call_llm,
    db_conn=None,
) -> dict | None:
    """Return the already-sent alert this event duplicates, or None to let it page.

    The suppression keys collapse duplicates only when two reports agree on a machine
    identity — a storyline_id, or a normalized location string. Measured 2026-08-11
    (runs 1453/1456/1458) neither holds for the cases that actually flood the channel:
    one Colombian earthquake paged four times under four storyline_ids, twice inside a
    single run, because each outlet's paraphrase clustered separately and their location
    strings ("western Colombia", "San Jose Del Palmar, Chocó", absent) keyed differently.
    Of 922 tiered events in the preceding week only 49 carried an IATA anchor and 258 had
    no location at all, so the geo net could not have caught them.

    This is the same judgement `adjudicate_storyline` makes, asked at the moment it
    matters and against a different candidate set: not "which storyline is this" but
    "did we already page this". It runs on the bulk router, only for cards that have
    already cleared every cheap gate, so it costs roughly one call per card actually
    about to be sent.

    Fails safe in the OPPOSITE direction to storyline adjudication: any error, empty
    candidate set or unparseable reply yields None, meaning the alert is SENT. Losing a
    page is worse than repeating one.
    """
    if not paged:
        return None
    prompt = _build_duplicate_prompt(event, paged)
    try:
        result = call_llm_fn(router, prompt, system_prompt=_DUP_SYSTEM_PROMPT,
                             max_tokens=512)
    except Exception:
        logger.exception("Duplicate-page adjudication failed; sending the alert")
        return None
    if db_conn is not None:
        log_llm_telemetry(db_conn, result, router, success=True,
                          purpose="dedup_adjudication")
    idx = _parse_match_index(result.get("content", ""), len(paged))
    if idx is None:
        return None
    match = paged[idx]
    logger.info(
        "Adjudicator judged alert a duplicate of an already-sent %s card: %.80s",
        match.get("alert_tier"), _describe(match),
    )
    return match


def adjudicate_storyline(
    event: dict,
    recent_events: list[dict],
    router,
    *,
    call_llm_fn=call_llm,
    window_hours: float = 48.0,
    max_candidates: int = 6,
    lexical_floor: float = 0.10,
    db_conn=None,
) -> str | None:
    """Return an existing storyline_id if the LLM confirms the SAME incident, else None.

    Fails safe: any error or ambiguity yields None so the caller starts a new storyline.

    db_conn is optional purely so existing callers/tests keep working, but passing it
    matters: adjudication is the single largest consumer of LLM calls (~180/day against
    ~120 for classification) and none of it was reaching system_telemetry, so every
    quota, latency and cost figure read off that table understated real usage by more
    than half.
    """
    candidates = find_geo_candidates(event, recent_events, window_hours,
                                     max_candidates, lexical_floor)
    if not candidates:
        return None
    prompt = _build_prompt(event, candidates)
    try:
        # gpt-oss (the bulk model) still emits some low-effort reasoning tokens before the
        # answer even with reasoning_effort=low; too small a budget gets fully consumed by
        # reasoning, leaving an empty final message that trips Groq's json_object validator
        # (HTTP 400). Give enough headroom for reasoning + the tiny {"match": ...} reply.
        result = call_llm_fn(router, prompt, system_prompt=_SYSTEM_PROMPT, max_tokens=512)
    except Exception:
        logger.exception("Storyline adjudication LLM call failed; treating as NEW")
        return None
    if db_conn is not None:
        log_llm_telemetry(db_conn, result, router, success=True,
                          purpose="storyline_adjudication")
    decision = _parse_decision(result.get("content", ""), candidates)
    if decision:
        logger.info(
            "Adjudicator linked event to storyline %s among %d candidate(s)",
            decision[:8], len(candidates),
        )
    return decision
