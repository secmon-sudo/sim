"""
SIM — Storyline Clusterer
Blueprint V20.1 §PASS G / Phase 3

Two-stage clustering for events:
1. Rule-based exact pre-filtering (Same Title + Same Location + 24h Window)
2. Greedy Centrist Clustering (Bigram Jaccard, threshold = 0.40) to prevent chaining.
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
from src.core.storyline import jaccard_similarity


def clean_title(title: str) -> str:
    """Normalize title for exact match comparison."""
    if not title:
        return ""
    # Strip special chars, lowercase, collapse whitespace
    cleaned = re.sub(r"[^\w\s]", "", title.lower())
    return " ".join(cleaned.split())


def rule_based_pre_filter(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage 1: Rule-based exact pre-filtering.
    Merges events that have:
    - Same cleaned title
    - Same location (country_iso and anchor_name_norm)
    - Occurred within 24 hours of each other
    """
    if not events:
        return []

    # Sort events by occurred_at_est to process sequentially
    sorted_events = sorted(
        events, 
        key=lambda e: e.get("occurred_at_est") or datetime.min
    )
    
    merged: List[Dict[str, Any]] = []

    for event in sorted_events:
        title = event.get("source_title") or event.get("storyline_hint") or ""
        norm_title = clean_title(title)
        country = event.get("country_iso") or ""
        anchor = event.get("anchor_name_norm") or ""
        dt = event.get("occurred_at_est")

        # Attempt to find a matching event in the already merged list
        found_match = False
        for m_event in merged:
            m_title = m_event.get("source_title") or m_event.get("storyline_hint") or ""
            m_norm_title = clean_title(m_title)
            m_country = m_event.get("country_iso") or ""
            m_anchor = m_event.get("anchor_name_norm") or ""
            m_dt = m_event.get("occurred_at_est")

            if (norm_title == m_norm_title and 
                country == m_country and 
                anchor == m_anchor and 
                dt and m_dt and 
                abs((dt - m_dt).total_seconds()) <= 86400):
                
                # Merge: append event id to the representative
                if "merged_event_ids" not in m_event:
                    m_event["merged_event_ids"] = [m_event["id"]]
                m_event["merged_event_ids"].append(event["id"])
                
                # Keep the one with higher severity or confidence if needed
                if event.get("severity_score", 0) > m_event.get("severity_score", 0):
                    m_event["severity_score"] = event["severity_score"]
                    m_event["source_title"] = event.get("source_title")
                    m_event["storyline_hint"] = event.get("storyline_hint")
                
                found_match = True
                break
        
        if not found_match:
            # Create a copy to avoid mutating the original events
            new_m_event = dict(event)
            new_m_event["merged_event_ids"] = [event["id"]]
            merged.append(new_m_event)

    return merged


def greedy_centrist_cluster(events: List[Dict[str, Any]], threshold: float = 0.40) -> List[List[Dict[str, Any]]]:
    """
    Stage 2: Greedy Centrist Clustering.
    To prevent chaining:
    - Each cluster has a centroid text (title/storyline hint of the first item).
    - New items are only added if their Jaccard similarity to the cluster centroid >= threshold.
    - Otherwise, a new cluster is created.
    """
    if not events:
        return []

    # Pre-filter exact duplicates first
    pre_filtered = rule_based_pre_filter(events)

    clusters: List[List[Dict[str, Any]]] = []
    # Store cluster centroids separately
    centroids: List[str] = []

    for event in pre_filtered:
        title = event.get("storyline_hint") or event.get("source_title") or ""
        
        best_cluster_idx = -1
        best_similarity = -1.0

        for idx, centroid_text in enumerate(centroids):
            sim = jaccard_similarity(title, centroid_text)
            if sim >= threshold and sim > best_similarity:
                best_similarity = sim
                best_cluster_idx = idx

        if best_cluster_idx != -1:
            clusters[best_cluster_idx].append(event)
        else:
            # Start new cluster
            clusters.append([event])
            centroids.append(title)

    return clusters
