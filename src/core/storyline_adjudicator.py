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

from src.core.geo import geo_key
from src.core.llm_client import call_llm
from src.core.storyline import jaccard_similarity

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
    g = ev.get("anchor_name_norm") or geo_key(
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
) -> list[dict]:
    """Candidate storylines that plausibly describe the same incident.

    Two candidate nets, both within a tight time window, one representative hint per
    storyline_id, and both deliberately LLM-adjudicated afterwards (the caller only ever
    MERGES on an explicit same-incident verdict):

    - **Geo net** (event has a resolvable location): same country + same coarse location.
      Intentionally the SAME set the deterministic geo-assist saw but could not confirm
      lexically — the ambiguous residue the LLM should judge.
    - **Country fallback** (event has NO usable location — e.g. missile tests, nuclear
      announcements and other national-level news that never resolves to a place): same
      country, ranked by lexical kinship so the most plausible duplicates fill the bounded
      candidate slots. Without this, location-less events bypass every dedup layer and each
      source pages separately. A minimal lexical-overlap floor stands in for the missing
      geo constraint so wholly unrelated same-country events are not offered to the LLM.
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
        if ev_geo:
            # Geo net: same country (when both known) + same coarse location.
            if iso and r_iso and iso != r_iso:
                continue
            if _event_geo(r) != ev_geo:
                continue
            overlap = 1.0
        else:
            # Country fallback: same country required (both sides known), and some lexical
            # kinship — a wholly unrelated same-country incident is not a candidate.
            if not r_iso or r_iso != iso:
                continue
            overlap = jaccard_similarity(ev_hint, r_hint)
            if overlap <= 0.0:
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
        'Reply with strict JSON only: {"match": <number of the matching storyline, or "NEW">}.',
    ]
    return "\n".join(lines)


def _parse_decision(content: str, candidates: list[dict]) -> str | None:
    """Map the LLM reply to a storyline_id, or None for NEW/unparseable."""
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
    if 1 <= idx <= len(candidates):
        return candidates[idx - 1]["storyline_id"]
    return None


def adjudicate_storyline(
    event: dict,
    recent_events: list[dict],
    router,
    *,
    call_llm_fn=call_llm,
    window_hours: float = 48.0,
    max_candidates: int = 6,
) -> str | None:
    """Return an existing storyline_id if the LLM confirms the SAME incident, else None.

    Fails safe: any error or ambiguity yields None so the caller starts a new storyline.
    """
    candidates = find_geo_candidates(event, recent_events, window_hours, max_candidates)
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
    decision = _parse_decision(result.get("content", ""), candidates)
    if decision:
        logger.info(
            "Adjudicator linked event to storyline %s among %d candidate(s)",
            decision[:8], len(candidates),
        )
    return decision
