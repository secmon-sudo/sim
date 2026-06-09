"""
SIM — Forecast Engine
Blueprint V20.1 §PASS G / Phase 3

Tension Index (TI) scoring, Z-Score anomaly detection, trajectory analysis,
and Watchlist / Emerging Concern classification.
"""

import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Profile-based modifiers for event types
PROFILE_MULTIPLIERS = {
    "PAX": {
        "airspace_restriction": 0.8,
        "ground_security": 1.5,
        "hotel_security": 0.5,
        "airport_security": 1.5,
        "cyber_infrastructure": 0.8,
        "civil_unrest": 1.2,
        "security_incident": 1.2
    },
    "CREW": {
        "airspace_restriction": 1.5,
        "ground_security": 1.2,
        "hotel_security": 1.8,
        "airport_security": 1.0,
        "cyber_infrastructure": 0.8,
        "civil_unrest": 1.4,
        "security_incident": 1.2
    },
    "DIPLOMAT": {
        "airspace_restriction": 1.3,
        "ground_security": 1.4,
        "hotel_security": 1.6,
        "airport_security": 1.2,
        "cyber_infrastructure": 0.9,
        "civil_unrest": 1.5,
        "security_incident": 1.3
    },
    "CARGO": {
        "airspace_restriction": 1.4,
        "ground_security": 1.1,
        "hotel_security": 0.7,
        "airport_security": 1.0,
        "cyber_infrastructure": 0.9,
        "civil_unrest": 1.2,
        "security_incident": 1.1
    }
}

DEFAULT_WEIGHTS = {
    "w_volume": 0.15,
    "w_diversity": 0.10,
    "w_severity_avg": 0.20,
    "w_severity_max": 0.15,
    "w_delta": 0.15,
    "w_recency": 0.10,
    "w_quality": 0.05,
    "w_cross_domain": 0.05,
    "w_critical": 0.05
}

# Credibility mapping for domain confidence
SOURCE_CREDIBILITY = {
    "reuters.com": 1.0,
    "apnews.com": 1.0,
    "afp.com": 1.0,
    "bbc.com": 1.0,
    "bbc.co.uk": 1.0,
    "nitter.net": 0.8,
    "reddit.com": 0.5
}


def get_source_credibility(domain: str) -> float:
    """Resolve credibility multiplier with sub-domain fallback."""
    if not domain:
        return 0.6
    domain = domain.lower().strip()
    if domain in SOURCE_CREDIBILITY:
        return SOURCE_CREDIBILITY[domain]
    
    parts = domain.split('.')
    if len(parts) > 2:
        parent = ".".join(parts[-2:])
        if parent in SOURCE_CREDIBILITY:
            return SOURCE_CREDIBILITY[parent]
            
    return 0.6


def calculate_tension_index(
    events: List[Dict[str, Any]],
    max_volume: int,
    prev_ti: Optional[float] = None,
    profile: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Calculate Tension Index (TI) for a single country in [0, 100] range.
    Formula components:
    - V: Normalised weekly volume (0 - 100)
    - D: Log-diversity based on storyline clusters (0 - 100)
    - S_avg: Average severity of events (0 - 100)
    - S_max: Max severity of events (0 - 100)
    - R: Recency weighted score with 24h decay (0 - 100)
    - Q: Average source credibility rating (0 - 100)
    - X: Cross-domain convergence bonus (0 or 50 or 100)
    - C: Critical event bonus (0 or 100)
    - Delta: TI change WoW (calculated using base TI difference)
    """
    w = weights or DEFAULT_WEIGHTS
    
    if not events:
        return {
            "ti": 0.0, "v": 0.0, "d": 0.0, "s_avg": 0.0, "s_max": 0.0,
            "r": 0.0, "q": 0.0, "x": 0.0, "c": 0.0, "delta": 0.0,
            "cluster_count": 0
        }

    # Apply profile modifiers to severity score if profile is set
    modified_events = []
    for ev in events:
        ev_copy = dict(ev)
        sev = float(ev.get("severity_score") or 0.0)
        
        if profile and profile in PROFILE_MULTIPLIERS:
            category = ev.get("event_type") or "other_aviation_related"
            mult = PROFILE_MULTIPLIERS[profile].get(category, 1.0)
            sev = sev * mult
            
        ev_copy["_mod_severity"] = min(max(sev, 0.0), 100.0)
        modified_events.append(ev_copy)

    # 1. Volume (V)
    v_score = 100.0 * (len(events) / max_volume) if max_volume > 0 else 0.0

    # 2. Diversity (D)
    from src.core.storyline_clusterer import greedy_centrist_cluster
    clusters = greedy_centrist_cluster(events, threshold=0.40)
    num_clusters = len(clusters)
    # Log-scaling: D = log(1 + N_clusters) / log(11)
    d_score = 100.0 * (math.log(1 + num_clusters) / math.log(11))
    d_score = min(d_score, 100.0)

    # 3. Severity Average (S_avg)
    s_avg_score = sum(e["_mod_severity"] for e in modified_events) / len(modified_events)

    # 4. Severity Max (S_max)
    s_max_score = max(e["_mod_severity"] for e in modified_events)

    # 5. Recency (R)
    # Calculate age relative to the most recent event's ingested_at or current UTC time
    now_utc = datetime.now(timezone.utc)
    # Filter valid times
    valid_times = [
        e.get("occurred_at_est").replace(tzinfo=timezone.utc)
        for e in modified_events if e.get("occurred_at_est")
    ]
    reference_time = max(valid_times) if valid_times else now_utc

    decay_sum = 0.0
    weight_sum = 0.0
    for e in modified_events:
        dt = e.get("occurred_at_est")
        if not dt:
            age_hours = 72.0  # default middle range
        else:
            dt_tz = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            age_hours = max((reference_time - dt_tz).total_seconds() / 3600.0, 0.0)
        
        # 24h half-life decay
        decay_w = 2.0 ** (-age_hours / 24.0)
        decay_sum += decay_w * e["_mod_severity"]
        weight_sum += decay_w
        
    r_score = decay_sum / weight_sum if weight_sum > 0 else 0.0

    # 6. Quality (Q)
    q_score = 100.0 * (sum(get_source_credibility(e.get("source_domain")) for e in modified_events) / len(modified_events))

    # 7. Cross-Domain Bonus (X)
    # Check for presence of physical, cyber, or airspace events
    has_cyber = any(
        e.get("event_type") == "cyber_infrastructure" or "cyber" in (e.get("event_type") or "")
        for e in modified_events
    )
    has_airspace = any(
        e.get("event_type") == "airspace_restriction" or "notam" in (e.get("storyline_hint") or "").lower()
        for e in modified_events
    )
    # Any other event type acts as physical/ground
    has_physical = any(
        e.get("event_type") not in ["cyber_infrastructure", "airspace_restriction"]
        for e in modified_events
    )

    domains_detected = sum([has_cyber, has_airspace, has_physical])
    if domains_detected == 3:
        x_score = 100.0  # Multi-domain Convergence
    elif domains_detected == 2:
        x_score = 50.0   # Dual-domain Convergence
    else:
        x_score = 0.0

    # 8. Critical Event Bonus (C)
    # Severity 5 event (severity >= 80) + cross-domain (at least 2 domains)
    has_critical_event = any(e["_mod_severity"] >= 80 for e in modified_events)
    if has_critical_event and domains_detected >= 2:
        c_score = 100.0
    else:
        c_score = 0.0

    # Calculate static TI score (without delta)
    static_components = {
        "w_volume": w.get("w_volume", 0.15) * v_score,
        "w_diversity": w.get("w_diversity", 0.10) * d_score,
        "w_severity_avg": w.get("w_severity_avg", 0.20) * s_avg_score,
        "w_severity_max": w.get("w_severity_max", 0.15) * s_max_score,
        "w_recency": w.get("w_recency", 0.10) * r_score,
        "w_quality": w.get("w_quality", 0.05) * q_score,
        "w_cross_domain": w.get("w_cross_domain", 0.05) * x_score,
        "w_critical": w.get("w_critical", 0.05) * c_score,
    }
    
    w_delta = w.get("w_delta", 0.15)
    # Sum of all components except delta
    ti_static = sum(static_components.values())
    
    # Normalize static score to what it would be if delta weight was 0
    ti_static_normalized = ti_static / (1.0 - w_delta)
    
    if prev_ti is not None:
        delta_score = ti_static_normalized - prev_ti
    else:
        delta_score = 0.0

    # Final Tension Index
    ti_final = ti_static_normalized * (1.0 - w_delta) + w_delta * delta_score
    ti_final = min(max(ti_final, 0.0), 100.0)

    return {
        "ti": round(ti_final, 2),
        "v": round(v_score, 2),
        "d": round(d_score, 2),
        "s_avg": round(s_avg_score, 2),
        "s_max": round(s_max_score, 2),
        "r": round(r_score, 2),
        "q": round(q_score, 2),
        "x": round(x_score, 2),
        "c": round(c_score, 2),
        "delta": round(delta_score, 2),
        "cluster_count": num_clusters
    }


def calculate_trajectory(
    current_ti: float,
    rolling_avg: float,
    z_score: float
) -> str:
    """Classify risk trajectory based on delta and z-score."""
    delta = current_ti - rolling_avg
    if delta > 8.0 and z_score > 1.0:
        return "Tırmanıyor"
    elif delta < -8.0:
        return "Azalıyor"
    else:
        return "Stabil"


def classify_watchlist_and_emergings(
    countries_data: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    Classify countries into Watchlist and Emerging Concerns based on metrics:
    - Watchlist: TI <= 50 (below major threat threshold), but positive delta and z-score > 0.5.
    - Emerging Concerns: event volume is low (<= 3 clusters) but contains at least one event with severity >= 60
      and high source credibility (Q >= 0.8 / source credibility >= 80).
    """
    watchlist = []
    emergings = []

    for c in countries_data:
        ti = c.get("ti", 0.0)
        delta = c.get("delta", 0.0)
        z_score = c.get("z_score", 0.0)
        cluster_count = c.get("cluster_count", 0)
        events = c.get("events", [])
        
        country_code = c.get("country_iso")
        if not country_code:
            continue

        # 1. Watchlist check
        if ti <= 50.0 and delta > 0.0 and z_score > 0.5:
            watchlist.append(country_code)
            continue
            
        # 2. Emerging Concern check
        if cluster_count <= 3 and cluster_count > 0:
            has_high_credibility_severe_event = False
            for ev in events:
                sev = ev.get("severity_score") or 0.0
                cred = get_source_credibility(ev.get("source_domain"))
                if sev >= 60.0 and cred >= 0.8:
                    has_high_credibility_severe_event = True
                    break
            if has_high_credibility_severe_event:
                emergings.append(country_code)

    return {
        "watchlist": watchlist,
        "emerging_concerns": emergings
    }
