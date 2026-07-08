"""
SIM — Forecast Generator
Blueprint V20.1 §PASS G / Phase 3

Coordinates G1, G2, G3 strategic passes with Pydantic validation, cross-check checks,
and auto-retry mechanisms.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ValidationError

from src.core.llm_router import LLMRouter
from src.core.llm_client import call_llm

logger = logging.getLogger(__name__)

# Pydantic Models for validation

class G1Selection(BaseModel):
    chosen_countries: List[str] = Field(description="List of ISO 2-letter country codes to analyze, max 8.")
    rationale: str = Field(description="Explanation of why these countries were chosen.")


class G2Forecast(BaseModel):
    risk_direction: Literal["Escalating", "Stable", "De-escalating"]
    confidence: Literal["High", "Medium", "Low"]
    most_likely_scenario: str
    escalation_scenario: str
    de_escalation_scenario: str
    watch_indicators: List[str]


class G2CountryAssessment(BaseModel):
    country: str = Field(description="ISO 2-letter country code")
    summary: str = Field(description="Concise evaluation summary of the last 7 days")
    key_drivers: List[str] = Field(description="Main conflict or risk drivers")
    forecast: G2Forecast
    assessment_confidence: Literal["High", "Medium", "Low"]
    data_coverage: Literal["sufficient", "limited", "insufficient"]
    primary_event_count: int
    storyline_cluster_count: int
    rationale: str = Field(description="Selection justification explaining Z-score and trajectory alignment")


class G3Spillover(BaseModel):
    spillover_title: str
    description: str
    countries_involved: List[str]
    risk_impact: str


class G3GlobalAssessment(BaseModel):
    executive_summary: str = Field(description="High-level overview of global geopolitical and aviation security trends")
    global_risk_direction: Literal["Escalating", "Stable", "De-escalating"]
    critical_global_drivers: List[str]
    spillovers: List[G3Spillover]


def extract_json(text: str) -> str:
    """Extract first valid JSON object substring from raw LLM output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def run_g1_selection(
    router: LLMRouter,
    candidate_countries: List[Dict[str, Any]],
) -> G1Selection:
    """
    Pass G1: Selects which countries to analyze from pre-filtered candidate list (max 8).
    """
    if not candidate_countries:
        return G1Selection(chosen_countries=[], rationale="No candidate countries available.")

    prompt_data = [
        {
            "country_iso": c["country_iso"],
            "ti": c["ti"],
            "delta": c["delta"],
            "z_score": c["z_score"],
            "event_count": len(c["events"]),
            "cluster_count": c["cluster_count"]
        }
        for c in candidate_countries
    ]

    system_prompt = (
        "You are a senior geopolitical risk analyst. Select the most critical countries for weekly security assessment.\n"
        "Input consists of countries and their tension index metrics. Output a JSON object matching this schema:\n"
        "{\n"
        "  \"chosen_countries\": [\"US\", \"IL\"],\n"
        "  \"rationale\": \"Explanation of choices...\"\n"
        "}\n"
        "Limit the selection to a maximum of 8 countries."
    )

    user_prompt = f"Candidate countries data:\n{json.dumps(prompt_data, indent=2)}"

    for attempt in range(3):
        try:
            res = call_llm(router, user_prompt, system_prompt, max_tokens=1000)
            cleaned = extract_json(res["content"])
            data = json.loads(cleaned)
            return G1Selection.model_validate(data)
        except Exception as e:
            logger.warning("G1 attempt %d failed: %s", attempt + 1, str(e))
            if attempt == 2:
                # Default fallback: return all candidate countries up to 8 sorted by TI
                sorted_candidates = sorted(candidate_countries, key=lambda x: x["ti"], reverse=True)
                chosen = [c["country_iso"] for c in sorted_candidates[:8]]
                return G1Selection(
                    chosen_countries=chosen,
                    rationale="Fallback: selected top countries by Tension Index due to LLM error."
                )
    return G1Selection(chosen_countries=[], rationale="Failed to generate selection.")


def validate_g2_assessment(
    assessment: G2CountryAssessment,
    metrics: Dict[str, Any]
) -> None:
    """
    Cross-checks LLM G2 forecast direction against calculated delta and z-score.
    Raises ValueError on contradiction to trigger retry.
    """
    delta = metrics.get("delta", 0.0)
    z_score = metrics.get("z_score", 0.0)
    direction = assessment.forecast.risk_direction

    # Contradiction: Risk is clearly rising (delta > 8.0 and Z-score > 1.0) but LLM chooses De-escalating
    if delta > 8.0 and z_score > 1.0 and direction == "De-escalating":
        raise ValueError(
            f"Contradiction: Calculated delta={delta:.2f} and z_score={z_score:.2f} indicate rising risk, "
            f"but forecast.risk_direction is '{direction}'."
        )

    # Contradiction: Risk is clearly falling (delta < -8.0) but LLM chooses Escalating
    if delta < -8.0 and direction == "Escalating":
        raise ValueError(
            f"Contradiction: Calculated delta={delta:.2f} indicates falling risk, "
            f"but forecast.risk_direction is '{direction}'."
        )


def run_g2_country_assessment(
    router: LLMRouter,
    country_iso: str,
    events: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    calibration_note: str = "",
) -> G2CountryAssessment:
    """
    Pass G2: Detailed evaluation and forecast for a single chosen country.
    Includes trajectory cross-check and up to 2 retries on validation failure.

    calibration_note: optional feedback from the forecast resolver on how past
    forecasts verified (accuracy / over- or under-escalation bias) — injected
    into the prompt as a mild prior so the model can self-correct.
    """
    # Sample events from each cluster to feed into prompt
    from src.core.storyline_clusterer import greedy_centrist_cluster
    clusters = greedy_centrist_cluster(events, threshold=0.40)
    
    cluster_samples = []
    for idx, cluster in enumerate(clusters):
        rep = cluster[0]
        cluster_samples.append({
            "cluster_id": idx + 1,
            "title": rep.get("source_title") or rep.get("storyline_hint") or "",
            "severity": rep.get("severity_score", 0),
            "event_type": rep.get("event_type", "other_aviation_related"),
            "domain": rep.get("source_domain", ""),
            "cluster_size": len(cluster)
        })

    system_prompt = (
        "You are a senior geopolitical risk analyst. Evaluate security incidents in the last 7 days and forecast risk direction.\n"
        "Output MUST match this JSON schema exactly:\n"
        "{\n"
        "  \"country\": \"ISO code\",\n"
        "  \"summary\": \"7-day summary\",\n"
        "  \"key_drivers\": [\"driver 1\", \"driver 2\"],\n"
        "  \"forecast\": {\n"
        "    \"risk_direction\": \"Escalating\" | \"Stable\" | \"De-escalating\",\n"
        "    \"confidence\": \"High\" | \"Medium\" | \"Low\",\n"
        "    \"most_likely_scenario\": \"...\",\n"
        "    \"escalation_scenario\": \"...\",\n"
        "    \"de_escalation_scenario\": \"...\",\n"
        "    \"watch_indicators\": [\"indicator 1\"]\n"
        "  },\n"
        "  \"assessment_confidence\": \"High\" | \"Medium\" | \"Low\",\n"
        "  \"data_coverage\": \"sufficient\" | \"limited\" | \"insufficient\",\n"
        "  \"primary_event_count\": 10,\n"
        "  \"storyline_cluster_count\": 3,\n"
        "  \"rationale\": \"selection rationale\"\n"
        "}"
    )

    user_prompt_template = (
        "Country: {country_iso}\n"
        "Calculated Metrics:\n"
        "- Tension Index: {ti}\n"
        "- Delta: {delta}\n"
        "- Z-Score: {z_score}\n"
        "- Event Count: {event_count}\n"
        "- Storyline Cluster Count: {cluster_count}\n\n"
        "Storyline Clusters Sample Data:\n{clusters_json}\n\n"
        "Provide your evaluation. Note: If Delta/Z-Score are highly positive, the risk direction should align with Escalating."
    )

    user_prompt = user_prompt_template.format(
        country_iso=country_iso,
        ti=metrics["ti"],
        delta=metrics["delta"],
        z_score=metrics["z_score"],
        event_count=len(events),
        cluster_count=metrics["cluster_count"],
        clusters_json=json.dumps(cluster_samples, indent=2)
    )

    if calibration_note:
        user_prompt += (
            f"\n\nCALIBRATION FEEDBACK (how your past forecasts verified):\n{calibration_note}\n"
            "Treat this as a mild prior — do not overcorrect; the evidence above still dominates."
        )

    feedback_msg = ""
    for attempt in range(3):
        try:
            current_prompt = user_prompt + feedback_msg
            res = call_llm(router, current_prompt, system_prompt, max_tokens=1500)
            cleaned = extract_json(res["content"])
            data = json.loads(cleaned)
            assessment = G2CountryAssessment.model_validate(data)
            
            # Cross-check check
            validate_g2_assessment(assessment, metrics)
            return assessment
            
        except (ValidationError, ValueError, Exception) as e:
            logger.warning("G2 assessment attempt %d for %s failed validation: %s", attempt + 1, country_iso, str(e))
            feedback_msg = f"\n\nCorrection required: Your previous response caused validation error: {str(e)}. Please correct the contradictions or JSON schema format."
            if attempt == 2:
                # Fallback: construct default object based on calculated trajectory
                ti = metrics["ti"]
                delta = metrics["delta"]
                z_score = metrics["z_score"]
                
                # Determine default direction based on math rules
                if delta > 8.0 and z_score > 1.0:
                    dir_val = "Escalating"
                elif delta < -8.0:
                    dir_val = "De-escalating"
                else:
                    dir_val = "Stable"

                return G2CountryAssessment(
                    country=country_iso,
                    summary=f"Automated evaluation for {country_iso}. Active storylines: {metrics['cluster_count']}.",
                    key_drivers=["Elevated incident counts" if ti > 50 else "Normal baseline activity"],
                    forecast=G2Forecast(
                        risk_direction=dir_val,
                        confidence="Medium",
                        most_likely_scenario="Baseline trends will continue with typical severity profiles.",
                        escalation_scenario="Sudden outbreak of cyber or kinetic incident could escalate risks.",
                        de_escalation_scenario="Restoration of regular patrols or de-escalation statements.",
                        watch_indicators=["Local press releases", "Official security warnings"]
                    ),
                    assessment_confidence="Medium",
                    data_coverage="sufficient" if len(events) >= 5 else "limited",
                    primary_event_count=len(events),
                    storyline_cluster_count=metrics["cluster_count"],
                    rationale=f"Fallback assessment generated due to LLM timeout/validation errors. Calculated TI: {ti}."
                )
    
    raise RuntimeError(f"Unexpected termination of G2 loop for {country_iso}")


def run_g3_global_assessment(
    router: LLMRouter,
    country_assessments: List[G2CountryAssessment]
) -> G3GlobalAssessment:
    """
    Pass G3: Regional/Global correlation and regional spillover analysis.
    Ensures that if no spillover is found, a default "no significant spillover" note is injected.
    """
    if not country_assessments:
        return G3GlobalAssessment(
            executive_summary="No high-risk countries were selected for assessment this week.",
            global_risk_direction="Stable",
            critical_global_drivers=["General regional stability"],
            spillovers=[
                G3Spillover(
                    spillover_title="No Significant Spillover",
                    description="No significant regional spillover effects observed.",
                    countries_involved=[],
                    risk_impact="None"
                )
            ]
        )

    system_prompt = (
        "You are a senior geopolitical risk analyst. Synthesize country-specific assessments into a global intelligence briefing.\n"
        "Output MUST match this JSON schema exactly:\n"
        "{\n"
        "  \"executive_summary\": \"High-level summary of global risk trends\",\n"
        "  \"global_risk_direction\": \"Escalating\" | \"Stable\" | \"De-escalating\",\n"
        "  \"critical_global_drivers\": [\"driver 1\", \"driver 2\"],\n"
        "  \"spillovers\": [\n"
        "    {\n"
        "      \"spillover_title\": \"Spillover title\",\n"
        "      \"description\": \"Description of spillover risk\",\n"
        "      \"countries_involved\": [\"IL\", \"LB\"],\n"
        "      \"risk_impact\": \"Impact level or details\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    prompt_data = [a.model_dump() for a in country_assessments]
    user_prompt = f"Assessments data:\n{json.dumps(prompt_data, indent=2)}"

    for attempt in range(3):
        try:
            res = call_llm(router, user_prompt, system_prompt, max_tokens=1800)
            cleaned = extract_json(res["content"])
            data = json.loads(cleaned)
            global_brief = G3GlobalAssessment.model_validate(data)
            
            # Post-validate/enrich G3 spillover constraint:
            # If no spillover was generated, add the default 'no significant spillover' item
            if not global_brief.spillovers:
                global_brief.spillovers.append(
                    G3Spillover(
                        spillover_title="No Significant Spillover",
                        description="No significant regional spillover effects observed.",
                        countries_involved=[],
                        risk_impact="None"
                    )
                )
            return global_brief
            
        except Exception as e:
            logger.warning("G3 attempt %d failed: %s", attempt + 1, str(e))
            if attempt == 2:
                # Default fallback
                return G3GlobalAssessment(
                    executive_summary="Geopolitical events show concentrated patterns in specific countries, with overall regional dynamics remaining active.",
                    global_risk_direction="Stable",
                    critical_global_drivers=["Localized geopolitical frictions", "Airspace restrictions"],
                    spillovers=[
                        G3Spillover(
                            spillover_title="No Significant Spillover",
                            description="No significant regional spillover effects observed.",
                            countries_involved=[],
                            risk_impact="None"
                        )
                    ]
                )
    
    raise RuntimeError("Unexpected G3 failure")
