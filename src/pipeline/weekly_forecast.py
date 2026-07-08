"""
SIM — Weekly Forecast Pipeline Pass
Blueprint V20.1 §PASS G / Phase 3

Coordinates weekly report generation:
1. Calculates Tension Index and Z-Score for all countries.
2. Identifies G1 candidates, runs G1, G2, G3 LLM passes.
3. Classifies Watchlist and Emerging Concerns.
4. Generates premium HTML payload.
5. Saves structured reports & mappings to DB.
6. Uploads to Cloudflare R2.
7. Dispatches alerts & documents to Telegram.
"""

import json
import logging
import math
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import boto3
from botocore.config import Config

from src.core.llm_router import LLMRouter
from src.core.forecast_engine import (
    calculate_tension_index,
    calculate_trajectory,
    classify_watchlist_and_emergings,
    get_source_credibility
)
from src.services.forecast_generator import (
    run_g1_selection,
    run_g2_country_assessment,
    run_g3_global_assessment
)
from src.services.flash_detector import check_flash_triggers
from src.services.forecast_resolver import (
    resolve_pending_reports,
    build_calibration_feedback,
)
from src.services.telegram_report_notifier import (
    send_weekly_report_telegram,
    generate_html_report_payload,
    send_flash_update_telegram,
)

logger = logging.getLogger(__name__)


def get_country_name(db_conn, country_iso: str) -> str:
    """Fetch country name from anchor_master or fallback to ISO."""
    if not country_iso:
        return "Unknown"
    try:
        # Since we use pg_trgm and anchor_master, we can fetch country names:
        # We can find any airport/anchor in that country and extract canonical name
        row = db_conn.execute(
            "SELECT canonical_name FROM anchor_master WHERE country_iso = %s LIMIT 1",
            (country_iso.upper(),)
        ).fetchone()
        if row:
            # e.g. "JFK Airport, New York, US" -> extract US or country name
            name_parts = row[0].split(",")
            if len(name_parts) >= 2:
                return name_parts[-2].strip()
        return country_iso.upper()
    except Exception:
        return country_iso.upper()


def upload_report_to_r2(filename: str, content: bytes, content_type: str) -> Optional[str]:
    """Uploads weekly forecast JSON/HTML to Cloudflare R2 bucket and returns public URL."""
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME") or "sim-archive"
    public_url_base = os.environ.get("R2_PUBLIC_URL_BASE") or "https://pub-default.r2.dev"

    if not all([account_id, access_key, secret_key]):
        logger.warning("Cloudflare R2 credentials missing, skipping weekly report R2 upload")
        return None

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        
        s3.put_object(
            Bucket=bucket_name,
            Key=filename,
            Body=content,
            ContentType=content_type
        )
        url = f"{public_url_base.rstrip('/')}/{filename}"
        logger.info("Uploaded %s to R2. Public URL: %s", filename, url)
        return url
    except Exception:
        logger.exception("Cloudflare R2 upload failed for report: %s", filename)
        return None


def run_flash_detection(
    db_conn,
    events: List[Dict[str, Any]],
    countries_data: List[Dict[str, Any]],
    max_flashes: int = 5,
) -> List[Dict[str, Any]]:
    """Run the Flash Detector over the last 24h of events and dispatch alerts.

    Uses the Z-scores already computed for countries_data. At most one flash per
    country per run (highest-priority trigger wins — check_flash_triggers already
    skips lower triggers for Z-score-triggered countries), capped at max_flashes
    Telegram messages. Each trigger is recorded in system_telemetry.
    """
    now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    recent_events = []
    for ev in events:
        dt = ev.get("occurred_at_est")
        if dt is None:
            continue
        dt_naive = dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt
        if (now_naive - dt_naive).total_seconds() <= 86400:
            recent_events.append(ev)

    if not recent_events:
        return []

    country_z_scores = {c["country_iso"]: c.get("z_score", 0.0) for c in countries_data}
    triggers = check_flash_triggers(recent_events, country_z_scores)
    if not triggers:
        return []

    logger.warning("Flash Detector: %d trigger(s) fired", len(triggers))
    seen_countries: set = set()
    dispatched = 0
    for trig in triggers:
        country = trig.get("country_iso")
        if country in seen_countries:
            continue
        seen_countries.add(country)

        try:
            db_conn.execute(
                "INSERT INTO system_telemetry(event_type, value_json) VALUES ('flash_trigger', %s)",
                (json.dumps({
                    "type": trig.get("type"),
                    "country_iso": country,
                    "reason": trig.get("reason"),
                    "event_ids": [str(e.get("id")) for e in trig.get("events", [])][:20],
                }),),
            )
            db_conn.commit()
        except Exception:
            db_conn.rollback()
            logger.exception("Failed to log flash trigger telemetry for %s", country)

        if dispatched < max_flashes:
            msg_id = send_flash_update_telegram(
                trigger_type=trig.get("type", "Flash"),
                country_iso=country or "??",
                reason=trig.get("reason", ""),
                events=trig.get("events", []),
            )
            if msg_id:
                dispatched += 1

    return triggers


def run_weekly_forecast(db_conn, router: LLMRouter) -> Dict[str, Any]:
    """
    Core weekly forecast coordinator.
    Run on demand or via scheduler command.
    """
    logger.info("Starting Weekly Geopolitical Intelligence & Forecast Generation...")
    
    # Calculate Date Window
    week_end = datetime.now(timezone.utc).date()
    week_start = week_end - timedelta(days=7)
    
    logger.info("Period: %s to %s", week_start, week_end)

    # 1. Fetch Events
    query_events = """
        SELECT id, source_title, source_url, source_domain, event_type, occurred_at_est, 
               anchor_name_raw, anchor_name_norm, country_iso, latitude, longitude, 
               severity_score, system_confidence, storyline_id, storyline_hint
        FROM events
        WHERE occurred_at_est >= %s AND occurred_at_est < %s
          AND severity_score IS NOT NULL
    """
    rows = db_conn.execute(query_events, (week_start, week_end)).fetchall()
    
    columns = [
        "id", "source_title", "source_url", "source_domain", "event_type", "occurred_at_est",
        "anchor_name_raw", "anchor_name_norm", "country_iso", "latitude", "longitude",
        "severity_score", "system_confidence", "storyline_id", "storyline_hint"
    ]
    
    events: List[Dict[str, Any]] = []
    for r in rows:
        events.append(dict(zip(columns, r)))

    if not events:
        logger.warning("No scored events found for the period: %s to %s. Aborting weekly report.", week_start, week_end)
        return {"success": False, "reason": "No events found."}

    logger.info("Found %d events to analyze.", len(events))

    # Group events by country
    events_by_country: Dict[str, List[Dict[str, Any]]] = {}
    for ev in events:
        c = (ev.get("country_iso") or "").strip().upper()
        if c:
            events_by_country.setdefault(c, []).append(ev)

    max_volume = max(len(evs) for evs in events_by_country.values()) if events_by_country else 0

    # 2. Fetch Historical TI scores for Z-Score calculations
    # Fetch past 8 reports
    query_reports = """
        SELECT report_date, scores_json
        FROM weekly_reports
        WHERE is_flash = FALSE AND report_date < %s
        ORDER BY report_date DESC
        LIMIT 8
    """
    past_reports = db_conn.execute(query_reports, (week_start,)).fetchall()
    
    # Compile past TI history per country
    past_ti_by_country: Dict[str, List[float]] = {}
    prev_ti_by_country: Dict[str, float] = {}
    
    for idx, (rep_date, scores_json) in enumerate(past_reports):
        if not scores_json or not isinstance(scores_json, dict):
            continue
        for country, metrics in scores_json.items():
            ti_val = metrics.get("ti")
            if ti_val is not None:
                past_ti_by_country.setdefault(country, []).append(float(ti_val))
                if idx == 0:
                    prev_ti_by_country[country] = float(ti_val)

    # 3. Calculate metrics for all countries
    countries_data: List[Dict[str, Any]] = []
    
    for country, country_events in events_by_country.items():
        prev_ti = prev_ti_by_country.get(country)
        ti_metrics = calculate_tension_index(country_events, max_volume, prev_ti=prev_ti)
        
        # Calculate Z-Score
        history = past_ti_by_country.get(country, [])
        if len(history) >= 2:
            mean = sum(history) / len(history)
            variance = sum((x - mean) ** 2 for x in history) / len(history)
            stddev = math.sqrt(variance)
            if stddev < 1.0:
                stddev = 1.0
            z_score = (ti_metrics["ti"] - mean) / stddev
            rolling_avg = mean
        else:
            z_score = 0.0
            rolling_avg = ti_metrics["ti"]

        trajectory = calculate_trajectory(ti_metrics["ti"], rolling_avg, z_score)
        
        countries_data.append({
            "country_iso": country,
            "country_name": get_country_name(db_conn, country),
            "ti": ti_metrics["ti"],
            "delta": ti_metrics["delta"],
            "z_score": z_score,
            "trajectory": trajectory,
            "cluster_count": ti_metrics["cluster_count"],
            "events": country_events,
            "metrics": ti_metrics
        })

    # Sort countries by TI descending
    countries_data = sorted(countries_data, key=lambda x: x["ti"], reverse=True)

    # 3b. Flash Detector — critical-event circuit breaker (zero-LLM).
    # Checks the last 24h for Z-score spikes, cross-domain convergence, and
    # high-volume escalation; dispatches Telegram flash updates. Failures here
    # must never break the weekly report.
    try:
        run_flash_detection(db_conn, events, countries_data)
    except Exception:
        logger.exception("Flash detection failed, continuing with weekly report")

    # 3c. Forecast Resolution — grade LAST week's forecasts against the TI/delta/Z
    # just computed for this week (they ARE the outcomes of that forecast window),
    # then turn the resolution history into a calibration prior for this week's G2.
    # Pure math + one SQL insert; must never block the report itself.
    current_scores_map = {
        c["country_iso"]: {"ti": c["ti"], "delta": c["delta"], "z_score": c["z_score"]}
        for c in countries_data
    }
    resolutions: List[Dict[str, Any]] = []
    calibration = {"note": "", "per_country": {}}
    try:
        resolutions = resolve_pending_reports(db_conn, current_scores_map, week_start)
        calibration = build_calibration_feedback(db_conn)
    except Exception:
        logger.exception("Forecast resolution failed, continuing with weekly report")

    # 4. Filter top 8 candidate countries for G1 selection (Top-8 Dinamik Filtresi)
    # Target countries with active movements: TI > 30 or Z-Score > 0.5
    candidate_countries = [
        c for c in countries_data
        if c["ti"] > 30.0 or c["z_score"] > 0.5
    ][:8]

    logger.info("Candidates selected for LLM G1 Selection: %s", [c["country_iso"] for c in candidate_countries])

    # Pass G1: Select final countries for assessment (max 8)
    g1_result = run_g1_selection(router, candidate_countries)
    chosen_isos = g1_result.chosen_countries
    
    logger.info("LLM G1 chosen countries: %s", chosen_isos)

    # Pass G2: Country Assessment
    g2_assessments: List[Any] = []
    for c in countries_data:
        if c["country_iso"] in chosen_isos:
            logger.info("Running G2 Assessment for %s...", c["country_iso"])
            note = calibration["note"]
            country_note = calibration["per_country"].get(c["country_iso"])
            if country_note:
                note = f"{note} {country_note}" if note else country_note
            ass = run_g2_country_assessment(
                router, c["country_iso"], c["events"], c, calibration_note=note
            )
            c["assessment"] = ass.model_dump()
            g2_assessments.append(ass)

    # Pass G3: Global Assessment
    logger.info("Running G3 Global & Spillover Assessment...")
    g3_result = run_g3_global_assessment(router, g2_assessments)
    global_brief = g3_result.model_dump()

    # Watchlist & Emerging Concerns Groupings
    groupings = classify_watchlist_and_emergings(countries_data)
    watchlist = groupings["watchlist"]
    emergings = groupings["emerging_concerns"]

    # Filter top countries analyzed in report
    top_countries_report = [c for c in countries_data if c["country_iso"] in chosen_isos]

    # Generate HTML payload
    html_payload = generate_html_report_payload(
        str(week_start),
        str(week_end),
        top_countries_report,
        watchlist,
        emergings,
        global_brief
    )

    # Compile scores json
    scores_json = {}
    for c in countries_data:
        scores_json[c["country_iso"]] = {
            "ti": c["ti"],
            "delta": c["delta"],
            "z_score": c["z_score"],
            "trajectory": c["trajectory"],
            "cluster_count": c["cluster_count"]
        }

    # Compile LLM assessment json
    llm_assessment_json = {
        "global_assessment": global_brief,
        "country_assessments": [c["assessment"] for c in top_countries_report if "assessment" in c]
    }

    # Fetch active weight config ID
    config_id = None
    row_config = db_conn.execute("SELECT config_id FROM ti_weight_configs WHERE is_active = TRUE LIMIT 1").fetchone()
    if row_config:
        config_id = row_config[0]

    # 5. Insert weekly_report into Database
    query_insert_report = """
        INSERT INTO weekly_reports (
            report_date, week_start, week_end, is_flash, top_countries, deteriorating, watchlist,
            scores_json, llm_assessment_json, html_payload, model_version, prompt_version, config_id
        ) VALUES (%s, %s, %s, FALSE, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    
    deteriorating_isos = [
        c["country_iso"] for c in countries_data 
        if c["trajectory"] == "Tırmanıyor"
    ]
    
    # We can fetch model name from the first router account
    model_version = router.accounts[0].model if router.accounts else "unknown-model"

    try:
        report_id = db_conn.execute(
            query_insert_report,
            (
                week_end,
                week_start,
                week_end,
                chosen_isos,
                deteriorating_isos,
                watchlist,
                json.dumps(scores_json),
                json.dumps(llm_assessment_json),
                html_payload,
                model_version,
                "v20.1",
                config_id
            )
        ).fetchone()[0]
        
        # Link all processed events during this week to mapping table
        for ev in events:
            db_conn.execute(
                "INSERT INTO report_event_mapping (report_id, event_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (report_id, ev["id"])
            )
        
        db_conn.commit()
        logger.info("Weekly report stored successfully in database. ID: %s", report_id)
        
    except Exception as e:
        logger.exception("Failed to store weekly report in database.")
        db_conn.rollback()
        return {"success": False, "reason": "Database insertion error."}

    # 6. Upload HTML and JSON/JSONL report to Cloudflare R2
    # JSONL data structure containing weekly report stats
    report_data_obj = {
        "report_id": str(report_id),
        "report_date": str(week_end),
        "week_start": str(week_start),
        "week_end": str(week_end),
        "top_countries": chosen_isos,
        "deteriorating": deteriorating_isos,
        "watchlist": watchlist,
        "emerging_concerns": emergings,
        "scores": scores_json,
        "assessments": llm_assessment_json
    }
    
    jsonl_bytes = (json.dumps(report_data_obj) + "\n").encode("utf-8")
    html_bytes = html_payload.encode("utf-8")
    
    file_prefix = f"weekly_report_{str(week_end).replace('-', '')}"
    
    r2_jsonl_url = upload_report_to_r2(f"reports/{file_prefix}.jsonl", jsonl_bytes, "application/jsonl")
    r2_html_url = upload_report_to_r2(f"reports/{file_prefix}.html", html_bytes, "text/html")

    # Update R2 URL in DB
    if r2_html_url or r2_jsonl_url:
        primary_url = r2_html_url or r2_jsonl_url
        db_conn.execute(
            "UPDATE weekly_reports SET r2_url = %s WHERE id = %s",
            (primary_url, report_id)
        )
        db_conn.commit()

    # 7. Dispatch to Telegram
    scorecard = None
    if resolutions:
        res = resolutions[0]
        n = len(res["records"])
        hits = sum(1 for r in res["records"] if r["correct"])
        scorecard = f"{hits}/{n} ülke tahmini doğrulandı (Brier {res['brier']:.2f})"
    telegram_message_id = send_weekly_report_telegram(
        str(week_start),
        str(week_end),
        top_countries_report,
        watchlist,
        emergings,
        global_brief,
        r2_url=r2_html_url,
        scorecard=scorecard,
    )

    if telegram_message_id:
        db_conn.execute(
            "UPDATE weekly_reports SET telegram_message_id = %s WHERE id = %s",
            (telegram_message_id, report_id)
        )
        db_conn.commit()

    logger.info("Weekly forecast run finished successfully!")
    return {
        "success": True,
        "report_id": str(report_id),
        "r2_url": r2_html_url,
        "telegram_message_id": telegram_message_id
    }
