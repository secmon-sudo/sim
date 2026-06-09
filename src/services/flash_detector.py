"""
SIM — Flash Detector
Blueprint V20.1 §PASS G / Phase 3

Critical Event Circuit Breaker (Flash Update) trigger logic:
1. Z-Score > 3.0 in the last 24h for any country.
2. Cross-domain convergence: TimeWindow < 6h, Same Location (anchor/50km), Domain Diversity (>= 2 categories).
3. 3+ verified high-confidence events in a country within a 6h window.
"""

import math
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple

from src.core.forecast_engine import get_source_credibility

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate geodesic distance between two points in km."""
    R = 6371.0  # Earth's radius in km
    try:
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    except Exception:
        return float('inf')


def is_same_location(ev1: Dict[str, Any], ev2: Dict[str, Any]) -> bool:
    """
    Check if two events occurred at the same location/anchor:
    - Same anchor_name_norm (non-null) OR
    - Same country_iso AND distance <= 50 km OR
    - Same country_iso AND identical normalized anchor_name_raw
    """
    country1 = (ev1.get("country_iso") or "").strip().upper()
    country2 = (ev2.get("country_iso") or "").strip().upper()
    
    if not country1 or country1 != country2:
        return False

    anchor_norm1 = (ev1.get("anchor_name_norm") or "").strip()
    anchor_norm2 = (ev2.get("anchor_name_norm") or "").strip()
    if anchor_norm1 and anchor_norm1 == anchor_norm2:
        return True

    raw1 = (ev1.get("anchor_name_raw") or "").strip().lower()
    raw2 = (ev2.get("anchor_name_raw") or "").strip().lower()
    if raw1 and raw1 == raw2:
        return True

    lat1, lon1 = ev1.get("latitude"), ev1.get("longitude")
    lat2, lon2 = ev2.get("latitude"), ev2.get("longitude")
    if lat1 is not None and lon1 is not None and lat2 is not None and lon2 is not None:
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        if dist <= 50.0:
            return True

    return False


def check_flash_triggers(
    recent_events: List[Dict[str, Any]],
    country_z_scores: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Checks recent events for Flash Update triggers.
    Returns a list of flash trigger details:
    [
      {
        "type": "Z-Score Exceeded" | "Cross-Domain Convergence" | "High Volume Escalation",
        "country_iso": "IL",
        "reason": "Description of trigger...",
        "events": [event_dict, ...]
      },
      ...
    ]
    """
    triggers = []
    
    # Trigger 1: Z-Score > 3.0 for any country
    for country, z in country_z_scores.items():
        if z > 3.0:
            triggers.append({
                "type": "Z-Score Exceeded",
                "country_iso": country,
                "reason": f"Tension Index Z-Score exceeded +3.0 threshold (Z={z:.2f}) in the country.",
                "events": [e for e in recent_events if (e.get("country_iso") or "").strip().upper() == country]
            })

    # Group events by country for local checks (Triggers 2 & 3)
    events_by_country: Dict[str, List[Dict[str, Any]]] = {}
    for ev in recent_events:
        c = (ev.get("country_iso") or "").strip().upper()
        if c:
            events_by_country.setdefault(c, []).append(ev)

    for country, country_evs in events_by_country.items():
        # Skip if already triggered by Z-score
        if any(t["country_iso"] == country and t["type"] == "Z-Score Exceeded" for t in triggers):
            continue

        # Sort events by occurred time
        sorted_evs = sorted(
            country_evs,
            key=lambda e: e.get("occurred_at_est") or datetime.min
        )

        n = len(sorted_evs)
        # Sliding time window checks (< 6 hours = 21600 seconds)
        for i in range(n):
            ev_i = sorted_evs[i]
            dt_i = ev_i.get("occurred_at_est")
            if not dt_i:
                continue

            # Accumulate events in 6-hour window starting at i
            window_evs = [ev_i]
            for j in range(i + 1, n):
                ev_j = sorted_evs[j]
                dt_j = ev_j.get("occurred_at_est")
                if not dt_j:
                    continue
                if (dt_j - dt_i).total_seconds() <= 21600:
                    window_evs.append(ev_j)
                else:
                    break

            # Trigger 2: Cross-domain convergence within same location & <6h window
            # Check pairwise locations and domain diversity in window_evs
            for k in range(len(window_evs)):
                loc_group = [window_evs[k]]
                for m in range(k + 1, len(window_evs)):
                    if is_same_location(window_evs[k], window_evs[m]):
                        loc_group.append(window_evs[m])
                
                if len(loc_group) >= 2:
                    # Check Domain Diversity (at least 2 different event_type categories)
                    categories = {e.get("event_type") for e in loc_group if e.get("event_type")}
                    if len(categories) >= 2:
                        triggers.append({
                            "type": "Cross-Domain Convergence",
                            "country_iso": country,
                            "reason": (
                                f"Cross-domain convergence detected at location '{loc_group[0].get('anchor_name_raw') or 'Unknown'}' "
                                f"within a 6-hour window. Domains: {list(categories)}."
                            ),
                            "events": loc_group
                        })
                        break

            if any(t["country_iso"] == country and t["type"] == "Cross-Domain Convergence" for t in triggers):
                break

            # Trigger 3: 3+ verified high-confidence events in 6 hours
            verified_evs = []
            for ev in window_evs:
                cred = get_source_credibility(ev.get("source_domain"))
                # Consider verified if source credibility >= 0.8 or system confidence is high
                if cred >= 0.8 or float(ev.get("system_confidence") or 0.0) >= 0.7:
                    verified_evs.append(ev)
            
            if len(verified_evs) >= 3:
                triggers.append({
                    "type": "High Volume Escalation",
                    "country_iso": country,
                    "reason": f"Detected 3+ verified high-confidence events within a 6-hour window.",
                    "events": verified_evs
                })
                break

    return triggers
