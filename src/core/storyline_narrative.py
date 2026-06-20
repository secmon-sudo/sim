"""
SIM — Storyline Narrative (zero-LLM)
Blueprint V20.1 §PASS D / Storyline

Builds a time-ordered "story so far" from the events sharing a storyline_id.
This is the deterministic backbone of the storyline-as-history feature: it orders
events chronologically and derives a factual summary (span, sources, escalation
trend, locations) WITHOUT spending any LLM tokens. An optional LLM prose layer can
sit on top later for high-value active storylines (budgeted).
"""

from datetime import datetime
from typing import Any, Dict, List

# Severity-trend classification keys (display strings are localized by the UI layer:
# escalating -> "Tırmanıyor", stable -> "Stabil", deescalating -> "Azalıyor").
TREND_ESCALATING = "escalating"
TREND_STABLE = "stable"
TREND_DEESCALATING = "deescalating"

# Minimum average-severity delta (later half vs earlier half) to call a trend.
_TREND_DELTA = 5.0


def _sort_key(event: Dict[str, Any]):
    return event.get("occurred_at_est") or datetime.min


def build_timeline(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return events sorted chronologically (oldest → newest) with a 1-based ``seq``.

    Events with a missing occurred_at_est sort to the front (treated as earliest
    known). The input list is not mutated.
    """
    ordered = sorted(events, key=_sort_key)
    timeline = []
    for i, ev in enumerate(ordered, start=1):
        item = dict(ev)
        item["seq"] = i
        timeline.append(item)
    return timeline


def _severity_trend(timeline: List[Dict[str, Any]]) -> str:
    """Compare average severity of the earlier half vs the later half of the story."""
    scored = [e for e in timeline if e.get("severity_score") is not None]
    if len(scored) < 2:
        return TREND_STABLE
    mid = len(scored) // 2
    earlier = scored[:mid] or scored[:1]
    later = scored[mid:]
    avg_earlier = sum(e["severity_score"] for e in earlier) / len(earlier)
    avg_later = sum(e["severity_score"] for e in later) / len(later)
    delta = avg_later - avg_earlier
    if delta >= _TREND_DELTA:
        return TREND_ESCALATING
    if delta <= -_TREND_DELTA:
        return TREND_DEESCALATING
    return TREND_STABLE


def summarize_timeline(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive a factual, zero-LLM "story so far" summary from storyline events.

    Returns counts, time span, distinct sources/countries/anchors, peak severity,
    and a severity trend — the structured history a reader (or a later LLM prose
    pass) can build a narrative from.
    """
    timeline = build_timeline(events)
    if not timeline:
        return {
            "event_count": 0,
            "source_count": 0,
            "started_at": None,
            "latest_at": None,
            "duration_hours": None,
            "countries": [],
            "anchors": [],
            "event_types": [],
            "peak_severity": 0,
            "severity_trend": TREND_STABLE,
        }

    times = [e["occurred_at_est"] for e in timeline if e.get("occurred_at_est")]
    started_at = min(times) if times else None
    latest_at = max(times) if times else None
    duration_hours = (
        round((latest_at - started_at).total_seconds() / 3600.0, 1)
        if started_at and latest_at else None
    )

    def _distinct(field: str) -> list:
        seen = []
        for e in timeline:
            v = e.get(field)
            if v and v not in seen:
                seen.append(v)
        return seen

    severities = [e.get("severity_score", 0) or 0 for e in timeline]

    return {
        "event_count": len(timeline),
        "source_count": len(_distinct("source_domain")),
        "started_at": started_at,
        "latest_at": latest_at,
        "duration_hours": duration_hours,
        "countries": _distinct("country_iso"),
        "anchors": _distinct("anchor_name_norm"),
        "event_types": _distinct("event_type"),
        "peak_severity": max(severities) if severities else 0,
        "severity_trend": _severity_trend(timeline),
    }
