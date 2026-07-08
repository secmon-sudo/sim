"""
SIM — Forecast Resolver (automated forecast verification)
Blueprint V20.1 §PASS G / Phase 3 follow-up

Closes the loop the weekly forecast never had: last week's report predicted a
risk_direction per country for THIS week, and this week's run computes exactly
the metrics (TI / delta / Z-score) that decide what actually happened. So
resolution is pure math against numbers the pipeline already produced — same
thresholds as validate_g2_assessment / calculate_trajectory, zero LLM calls.

Two exports:
  - resolve_pending_reports(): grade last week's unresolved report(s) against the
    current run's country scores and persist a row in report_validations
    (resolution_kind='auto').
  - build_calibration_feedback(): aggregate recent resolutions into a short
    calibration note (plus per-country recent misses) that run_weekly_forecast
    injects into the G2 prompt — past accuracy becomes a mild prior for the
    next forecast, at zero token cost beyond the note itself.

Fails safe throughout: any error yields "no resolution / empty note" so the
weekly report itself is never blocked by its own scorekeeping.
"""

import json
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Same escalation thresholds as validate_g2_assessment (forecast_generator) and
# calculate_trajectory (forecast_engine) — resolution must judge forecasts by the
# exact rules the forecasts were validated against.
_DELTA_ESCALATION = 8.0
_Z_ESCALATION = 1.0

# Stated forecast confidence read as an implied probability of being right.
# Used for a Brier-style score: mean((p - outcome)^2), lower is better.
_CONFIDENCE_PROB = {"High": 0.9, "Medium": 0.7, "Low": 0.5}

# A report is only resolvable while the current run's score window still matches
# its forecast window (normal cadence: report_date == current week_start, both
# Sundays). The tolerance absorbs manual/late workflow_dispatch runs.
_RESOLUTION_TOLERANCE_DAYS = 3


def _actual_direction(delta: float, z_score: float) -> str:
    """What actually happened, by the same thresholds the forecast was held to."""
    if delta > _DELTA_ESCALATION and z_score > _Z_ESCALATION:
        return "Escalating"
    if delta < -_DELTA_ESCALATION:
        return "De-escalating"
    return "Stable"


def _actual_for_country(
    country: str,
    current_scores: Dict[str, Dict[str, Any]],
    report_scores: Dict[str, Any],
) -> Tuple[str, Dict[str, float]]:
    """Resolve a country's actual outcome from the current run's computed scores.

    A country absent from current_scores produced zero scored events this week —
    its TI collapsed to 0. That is a real outcome (quiet week), not missing data:
    judged as De-escalating when it had meaningful tension before, else Stable.
    """
    cur = current_scores.get(country)
    if cur is not None:
        delta = float(cur.get("delta") or 0.0)
        z = float(cur.get("z_score") or 0.0)
        return _actual_direction(delta, z), {
            "ti": float(cur.get("ti") or 0.0), "delta": delta, "z_score": z,
        }
    prev = report_scores.get(country) or {}
    prev_ti = float(prev.get("ti") or 0.0)
    delta = -prev_ti
    return _actual_direction(delta, 0.0), {"ti": 0.0, "delta": delta, "z_score": 0.0}


def _grade_report(
    report_scores: Dict[str, Any],
    assessments: List[Dict[str, Any]],
    deteriorating: List[str],
    watchlist: List[str],
    current_scores: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Grade one report's country forecasts against actual outcomes.

    Returns None when the report has nothing gradable (no G2 assessments).
    """
    records: List[Dict[str, Any]] = []
    brier_terms: List[float] = []

    for a in assessments:
        country = (a.get("country") or "").strip().upper()
        forecast = a.get("forecast") or {}
        direction = forecast.get("risk_direction")
        if not country or direction not in ("Escalating", "Stable", "De-escalating"):
            continue
        actual, metrics = _actual_for_country(country, current_scores, report_scores)
        correct = direction == actual
        confidence = forecast.get("confidence") or "Medium"
        p = _CONFIDENCE_PROB.get(confidence, 0.7)
        brier_terms.append((p - (1.0 if correct else 0.0)) ** 2)
        records.append({
            "country": country,
            "forecast_direction": direction,
            "confidence": confidence,
            "actual_direction": actual,
            "correct": correct,
            "actual": metrics,
        })

    if not records:
        return None

    n = len(records)
    accuracy = sum(1 for r in records if r["correct"]) / n
    brier = sum(brier_terms) / n

    # Escalation recall over ALL countries that actually escalated this week (not
    # just assessed ones): a country the report never flagged anywhere is a miss.
    flagged = {r["country"] for r in records if r["forecast_direction"] == "Escalating"}
    flagged |= {c.strip().upper() for c in (deteriorating or [])}
    flagged |= {c.strip().upper() for c in (watchlist or [])}
    actually_escalated = {
        c for c, m in current_scores.items()
        if _actual_direction(float(m.get("delta") or 0.0), float(m.get("z_score") or 0.0)) == "Escalating"
    }
    escalation_recall = (
        len(actually_escalated & flagged) / len(actually_escalated)
        if actually_escalated else None
    )

    # FP rate among explicit Escalating forecasts only.
    esc_forecasts = [r for r in records if r["forecast_direction"] == "Escalating"]
    fp_rate = (
        sum(1 for r in esc_forecasts if r["actual_direction"] != "Escalating") / len(esc_forecasts)
        if esc_forecasts else None
    )

    return {
        "accuracy": accuracy,
        "brier": brier,
        "direction_correct": accuracy >= 0.5,
        "escalation_recall": escalation_recall,
        "fp_rate": fp_rate,
        "records": records,
    }


def resolve_pending_reports(
    db_conn,
    current_scores: Dict[str, Dict[str, Any]],
    week_start: date,
) -> List[Dict[str, Any]]:
    """Grade unresolved weekly reports whose forecast window this run just measured.

    current_scores: this run's per-country {ti, delta, z_score, ...} map — the
    ground truth for last week's forecasts. Persists one report_validations row
    per graded report (resolution_kind='auto'); idempotent via the partial unique
    index. Returns the resolution summaries (empty list on any failure).
    """
    lo = week_start - timedelta(days=_RESOLUTION_TOLERANCE_DAYS)
    hi = week_start + timedelta(days=_RESOLUTION_TOLERANCE_DAYS)
    try:
        rows = db_conn.execute(
            """SELECT r.id, r.report_date, r.scores_json, r.llm_assessment_json,
                      r.deteriorating, r.watchlist
               FROM weekly_reports r
               WHERE r.is_flash = FALSE
                 AND r.report_date >= %s AND r.report_date <= %s
                 AND NOT EXISTS (
                     SELECT 1 FROM report_validations v
                     WHERE v.report_id = r.id AND v.resolution_kind = 'auto'
                 )
               ORDER BY r.report_date DESC""",
            (lo, hi),
        ).fetchall()
    except Exception:
        db_conn.rollback()
        logger.exception("Forecast resolution: failed to fetch pending reports")
        return []

    resolutions: List[Dict[str, Any]] = []
    for report_id, report_date, scores_json, assessment_json, deteriorating, watchlist in rows:
        try:
            report_scores = scores_json if isinstance(scores_json, dict) else {}
            assessment = assessment_json if isinstance(assessment_json, dict) else {}
            graded = _grade_report(
                report_scores,
                assessment.get("country_assessments") or [],
                deteriorating or [],
                watchlist or [],
                current_scores,
            )
            if graded is None:
                logger.info("Forecast resolution: report %s has no gradable assessments, skipping", report_id)
                continue

            db_conn.execute(
                """INSERT INTO report_validations
                       (report_id, resolution_kind, direction_correct,
                        escalation_recall, fp_rate, accuracy, brier, details_json)
                   VALUES (%s, 'auto', %s, %s, %s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                (
                    report_id,
                    graded["direction_correct"],
                    graded["escalation_recall"],
                    graded["fp_rate"],
                    graded["accuracy"],
                    graded["brier"],
                    json.dumps({"report_date": str(report_date), "countries": graded["records"]}),
                ),
            )
            db_conn.commit()
            graded["report_id"] = str(report_id)
            resolutions.append(graded)
            logger.info(
                "Forecast resolution: report %s graded — accuracy=%.2f brier=%.3f over %d countr(y/ies)",
                report_id, graded["accuracy"], graded["brier"], len(graded["records"]),
            )
        except Exception:
            db_conn.rollback()
            logger.exception("Forecast resolution failed for report %s", report_id)
    return resolutions


def build_calibration_feedback(db_conn, lookback: int = 4) -> Dict[str, Any]:
    """Aggregate recent auto-resolutions into calibration guidance for Pass G2.

    Returns {"note": str, "per_country": {iso: str}}. Empty note when there is no
    resolution history yet — the G2 prompt is then unchanged.
    """
    empty: Dict[str, Any] = {"note": "", "per_country": {}}
    try:
        rows = db_conn.execute(
            """SELECT accuracy, details_json
               FROM report_validations
               WHERE resolution_kind = 'auto'
               ORDER BY validated_at DESC
               LIMIT %s""",
            (lookback,),
        ).fetchall()
    except Exception:
        db_conn.rollback()
        logger.exception("Forecast calibration: failed to fetch resolution history")
        return empty

    total = correct = over = under = 0
    per_country_counts: Dict[str, List[int]] = {}  # iso -> [correct, total]
    for _accuracy, details in rows:
        details = details if isinstance(details, dict) else {}
        for rec in details.get("countries") or []:
            total += 1
            iso = rec.get("country", "??")
            stats = per_country_counts.setdefault(iso, [0, 0])
            stats[1] += 1
            if rec.get("correct"):
                correct += 1
                stats[0] += 1
            elif rec.get("forecast_direction") == "Escalating":
                over += 1
            elif rec.get("actual_direction") == "Escalating":
                under += 1

    if total == 0:
        return empty

    parts = [
        f"Over your last {len(rows)} weekly report(s), {correct}/{total} country "
        f"risk-direction forecasts verified against measured TI/Z outcomes."
    ]
    if over > under and over >= 2:
        parts.append(
            f"Bias detected: {over} forecast(s) said Escalating but the situation did not escalate "
            "— you have been over-escalating; require stronger evidence before choosing Escalating."
        )
    elif under > over and under >= 2:
        parts.append(
            f"Bias detected: {under} actual escalation(s) were forecast as Stable/De-escalating "
            "— you have been under-escalating; weigh escalation signals more seriously."
        )
    note = " ".join(parts)

    per_country = {
        iso: f"Your recent forecasts for {iso} verified {c}/{n} time(s)."
        for iso, (c, n) in per_country_counts.items()
        if n >= 1 and c < n  # only surface countries with at least one miss
    }
    return {"note": note, "per_country": per_country}
