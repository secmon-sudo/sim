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

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise event-deduplication assistant for an OSINT security pipeline. "
    "You decide whether a news event describes the SAME real-world incident as an "
    "existing storyline, or a NEW distinct incident. Answer ONLY with strict JSON."
)


def _event_geo(ev: dict) -> str | None:
    """Coarse location key for an event: precise IATA anchor, else geo_key of raw text."""
    return ev.get("anchor_name_norm") or geo_key(
        ev.get("anchor_name_raw"), ev.get("country_iso")
    )


def find_geo_candidates(
    event: dict,
    recent_events: list[dict],
    window_hours: float = 48.0,
    max_candidates: int = 6,
) -> list[dict]:
    """Candidate storylines that plausibly describe the same incident.

    Same country + same coarse location + within a tight time window, one representative
    hint per storyline_id. This is intentionally the SAME set the deterministic geo-assist
    saw but could not confirm lexically — the ambiguous residue the LLM should judge.
    """
    dt = event.get("occurred_at_est")
    if dt is None:
        return []
    ev_geo = _event_geo(event)
    if not ev_geo:
        return []
    iso = event.get("country_iso")

    candidates: list[dict] = []
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
        if iso and r_iso and iso != r_iso:
            continue
        if _event_geo(r) != ev_geo:
            continue
        seen_storylines.add(sid)
        candidates.append({"storyline_id": sid, "hint": r.get("storyline_hint") or ""})
        if len(candidates) >= max_candidates:
            break
    return candidates


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
        result = call_llm_fn(router, prompt, system_prompt=_SYSTEM_PROMPT, max_tokens=200)
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
