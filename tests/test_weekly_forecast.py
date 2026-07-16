"""
SIM — Test Weekly Forecast Module
Blueprint V20.1 §PASS G / Phase 3

Unit tests for Storyline Clustering, Tension Index, Trajectories,
LLM 3-Pass validation, and Flash Update circuit breaker.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from src.core.storyline_clusterer import greedy_centrist_cluster, rule_based_pre_filter
from src.core.forecast_engine import (
    calculate_tension_index,
    calculate_trajectory,
    classify_watchlist_and_emergings
)
from src.services.flash_detector import check_flash_triggers
from src.services.forecast_generator import validate_g2_assessment, G2CountryAssessment, G2Forecast, run_g3_global_assessment


# 1. Test Rule-based pre-filter & Greedy centrist clustering to prevent chaining
def test_rule_based_pre_filter_merges():
    dt = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
    events = [
        {
            "id": "ev1",
            "source_title": "Drone attack at Tel Aviv airport",
            "country_iso": "IL",
            "anchor_name_norm": "TLV",
            "occurred_at_est": dt,
            "severity_score": 60
        },
        # Same title, same location, within 24h -> merge
        {
            "id": "ev2",
            "source_title": "Drone attack at Tel Aviv airport",
            "country_iso": "IL",
            "anchor_name_norm": "TLV",
            "occurred_at_est": dt + timedelta(hours=3),
            "severity_score": 75  # Higher severity should update representative
        },
        # Same title, different location -> do not merge
        {
            "id": "ev3",
            "source_title": "Drone attack at Tel Aviv airport",
            "country_iso": "IL",
            "anchor_name_norm": "HFA",
            "occurred_at_est": dt,
            "severity_score": 50
        }
    ]

    merged = rule_based_pre_filter(events)
    assert len(merged) == 2
    # Find the TLV merged event
    tlv_event = next(e for e in merged if e["anchor_name_norm"] == "TLV")
    assert tlv_event["severity_score"] == 75
    assert set(tlv_event["merged_event_ids"]) == {"ev1", "ev2"}


def test_greedy_clustering_prevents_chaining():
    # Setup overlapping hints:
    # A matches B ("A Drone in TLV" & "B Drone in TLV") -> Jaccard similarity ~ 0.40
    # B matches C ("B Drone in TLV" & "C Drone in TLV") -> Jaccard similarity ~ 0.40
    # A matches C -> Jaccard similarity is low (they share very few words directly)
    # A connected components model would group A, B, C together.
    # Greedy Centrist should keep A and C in separate clusters if they don't match the centroid.
    events = [
        {
            "id": "ev1",
            "storyline_hint": "Drone strike hits terminal",  # centroid 1
            "country_iso": "IL",
            "anchor_name_norm": "TLV",
            "occurred_at_est": datetime.now()
        },
        {
            "id": "ev2",
            "storyline_hint": "Drone hits terminal building",  # Jaccard with ev1: "drone", "hits", "terminal" -> Matches
            "country_iso": "IL",
            "anchor_name_norm": "TLV",
            "occurred_at_est": datetime.now()
        },
        {
            "id": "ev3",
            "storyline_hint": "building security alert active",  # Jaccard with ev2 is high ("building"), Jaccard with ev1 is 0.0
            "country_iso": "IL",
            "anchor_name_norm": "TLV",
            "occurred_at_est": datetime.now()
        }
    ]

    clusters = greedy_centrist_cluster(events, threshold=0.30)
    # ev1 and ev2 should group together. ev3 should NOT group with ev1 (centroid 1), so it starts cluster 2.
    assert len(clusters) == 2
    cluster_ids = [[e["id"] for e in c] for c in clusters]
    assert "ev1" in cluster_ids[0]
    assert "ev2" in cluster_ids[0]
    assert "ev3" in cluster_ids[1]


# 2. Test Tension Index Calculations & Profile Multipliers
def test_tension_index_calculation_formulas():
    dt = datetime.now(timezone.utc)
    events = [
        # Physical domain
        {
            "id": "ev1",
            "event_type": "security_incident",
            "severity_score": 80,
            "occurred_at_est": dt - timedelta(hours=2),
            "source_domain": "reuters.com",
            "country_iso": "UA"
        },
        # Cyber domain -> triggers cross-domain
        {
            "id": "ev2",
            "event_type": "cyber_infrastructure",
            "severity_score": 70,
            "occurred_at_est": dt - timedelta(hours=10),
            "source_domain": "nitter.net",
            "country_iso": "UA"
        }
    ]

    # Max volume in the system is say 5
    res = calculate_tension_index(events, max_volume=5, prev_ti=50.0)
    
    assert res["ti"] > 0.0
    assert res["v"] == 100.0 * (2/5)
    # Check log diversity: N=2 clusters, log(3)/log(11) * 100
    assert res["d"] > 0.0
    # Average severity of 80 and 70 is 75
    assert res["s_avg"] == 75.0
    assert res["s_max"] == 80.0
    # Q rating: reuters=1.0, nitter=0.8. Avg = 0.9 * 100 = 90
    assert res["q"] == 90.0
    # Cyber + Physical domains = 2 domains -> cross-domain bonus (X) = 50.0
    assert res["x"] == 50.0
    # Critical event bonus (C): severity >= 80 ( Ukraine has ev1 with 80) + cross-domain -> 100.0
    assert res["c"] == 100.0
    # Delta check: result should have a calculated delta
    assert "delta" in res


def test_profile_tension_index_modifiers():
    dt = datetime.now(timezone.utc)
    events = [
        {
            "id": "ev1",
            "event_type": "ground_security",
            "severity_score": 50,
            "occurred_at_est": dt
        }
    ]

    # For PAX, ground_security modifier is 1.5. Severity becomes 50 * 1.5 = 75
    res_pax = calculate_tension_index(events, max_volume=1, profile="PAX")
    assert res_pax["s_max"] == 75.0

    # For CREW, hotel_security modifier would apply. ground_security modifier is 1.2. Severity becomes 50 * 1.2 = 60
    res_crew = calculate_tension_index(events, max_volume=1, profile="CREW")
    assert res_crew["s_max"] == 60.0


# 3. Test Z-Score Trajectory Calculations & Grouping
def test_z_score_trajectory_classification():
    # Case 1: High rising trajectory (delta > 8.0 and z-score > 1.0)
    traj_up = calculate_trajectory(current_ti=75.0, rolling_avg=60.0, z_score=1.5)
    assert traj_up == "Tırmanıyor"

    # Case 2: Falling trajectory (delta < -8.0)
    traj_down = calculate_trajectory(current_ti=40.0, rolling_avg=50.0, z_score=-1.1)
    assert traj_down == "Azalıyor"

    # Case 3: Stable
    traj_stable = calculate_trajectory(current_ti=52.0, rolling_avg=50.0, z_score=0.2)
    assert traj_stable == "Stabil"


def test_watchlist_and_emerging_classification():
    datetime.now(timezone.utc)
    countries_data = [
        # Country 1: Fits Watchlist (ti <= 50, delta > 0, z_score > 0.5)
        {
            "country_iso": "PL",
            "ti": 45.0,
            "delta": 5.0,
            "z_score": 0.8,
            "cluster_count": 2,
            "events": []
        },
        # Country 2: Fits Emerging Concerns (low volume <= 3, at least one severity >= 60 with credibility >= 0.8)
        {
            "country_iso": "GE",
            "ti": 35.0,
            "delta": 1.0,
            "z_score": 0.2,
            "cluster_count": 1,
            "events": [
                {
                    "severity_score": 65,
                    "source_domain": "reuters.com"  # credibility 1.0
                }
            ]
        }
    ]

    res = classify_watchlist_and_emergings(countries_data)
    assert "PL" in res["watchlist"]
    assert "GE" in res["emerging_concerns"]


# 4. Test 3-Pass LLM Validation
def test_g2_validation_contradiction_checks():
    metrics_rising = {"delta": 10.0, "z_score": 1.5}
    assessment_contradict_up = G2CountryAssessment(
        country="UA",
        summary="Rising clashes",
        key_drivers=["conflict"],
        forecast=G2Forecast(
            risk_direction="De-escalating",  # Contradicts rising metrics
            confidence="High",
            most_likely_scenario="Peace",
            escalation_scenario="None",
            de_escalation_scenario="None",
            watch_indicators=[]
        ),
        assessment_confidence="High",
        data_coverage="sufficient",
        primary_event_count=5,
        storyline_cluster_count=2,
        rationale="Aligned"
    )

    with pytest.raises(ValueError, match="Contradiction"):
        validate_g2_assessment(assessment_contradict_up, metrics_rising)

    metrics_falling = {"delta": -12.0, "z_score": -1.2}
    assessment_contradict_down = G2CountryAssessment(
        country="UA",
        summary="Calming situation",
        key_drivers=["peace"],
        forecast=G2Forecast(
            risk_direction="Escalating",  # Contradicts falling metrics
            confidence="High",
            most_likely_scenario="War",
            escalation_scenario="None",
            de_escalation_scenario="None",
            watch_indicators=[]
        ),
        assessment_confidence="High",
        data_coverage="sufficient",
        primary_event_count=5,
        storyline_cluster_count=2,
        rationale="Aligned"
    )

    with pytest.raises(ValueError, match="Contradiction"):
        validate_g2_assessment(assessment_contradict_down, metrics_falling)


@patch("src.services.forecast_generator.call_llm")
def test_g3_spillover_fallback(mock_call):
    # Mock LLM return with empty spillovers
    mock_call.return_value = {
        "content": '{"executive_summary": "Global summary", "global_risk_direction": "Stable", "critical_global_drivers": [], "spillovers": []}'
    }
    
    router = MagicMock()
    res = run_g3_global_assessment(router, [])
    # Verify that "no significant spillover" gets appended automatically
    assert len(res.spillovers) == 1
    assert "no significant regional spillover" in res.spillovers[0].description.lower()


# 5. Test Flash Update Trigger Conditions
def test_flash_update_triggers():
    dt = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
    
    # Setup events for Trigger 2: Cross-domain convergence within same location & <6h window
    recent_events_cross = [
        {
            "id": "e1",
            "country_iso": "LB",
            "anchor_name_norm": "BEY",
            "event_type": "security_incident",
            "occurred_at_est": dt,
            "severity_score": 85,
            "system_confidence": 0.8
        },
        {
            "id": "e2",
            "country_iso": "LB",
            "anchor_name_norm": "BEY",
            "event_type": "cyber_infrastructure",  # Different domain
            "occurred_at_est": dt + timedelta(hours=2),  # < 6h window
            "severity_score": 60,
            "system_confidence": 0.7
        }
    ]

    triggers_cross = check_flash_triggers(recent_events_cross, country_z_scores={})
    assert len(triggers_cross) == 1
    assert triggers_cross[0]["type"] == "Cross-Domain Convergence"
    assert triggers_cross[0]["country_iso"] == "LB"

    # Setup events for Trigger 3: 3+ verified high-confidence events in 6h
    recent_events_vol = [
        {
            "id": "v1",
            "country_iso": "SY",
            "event_type": "civil_unrest",
            "occurred_at_est": dt,
            "source_domain": "reuters.com",  # high cred
            "system_confidence": 0.9
        },
        {
            "id": "v2",
            "country_iso": "SY",
            "event_type": "civil_unrest",
            "occurred_at_est": dt + timedelta(hours=1),
            "source_domain": "apnews.com",  # high cred
            "system_confidence": 0.8
        },
        {
            "id": "v3",
            "country_iso": "SY",
            "event_type": "civil_unrest",
            "occurred_at_est": dt + timedelta(hours=3),
            "source_domain": "bbc.com",  # high cred
            "system_confidence": 0.85
        }
    ]

    triggers_vol = check_flash_triggers(recent_events_vol, country_z_scores={})
    assert len(triggers_vol) == 1
    assert triggers_vol[0]["type"] == "High Volume Escalation"
    assert triggers_vol[0]["country_iso"] == "SY"

    # Setup Z-Score Trigger (> 3.0)
    triggers_z = check_flash_triggers([], country_z_scores={"IR": 3.4})
    assert len(triggers_z) == 1
    assert triggers_z[0]["type"] == "Z-Score Exceeded"
    assert triggers_z[0]["country_iso"] == "IR"
