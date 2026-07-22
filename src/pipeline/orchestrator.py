"""
SIM — Pipeline Orchestrator
Blueprint V20.1 §4

Main entry point that executes all pipeline passes in sequence.
Designed to run as a GitHub Actions job every 30 minutes.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.core.llm_router import build_llm_router, build_quality_router
from src.pipeline.pass_a_ingest import run_pass_a
from src.pipeline.pass_b_dedup import run_pass_b
from src.pipeline.pass_c_classify import run_pass_c
from src.pipeline.pass_d_score import run_pass_d
from src.pipeline.pass_e_reconcile import run_pass_e
from src.pipeline.pass_f_archive import run_pass_f, run_run_snapshot
from src.services.czib_client import sync_czib_to_db
from src.services.supabase_client import close_pool, get_connection, put_connection

# Ensure logs/ directory exists for GitHub Actions artifact upload
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging — console + file
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
)

# Add file handler so logs are persisted for artifact upload
_file_handler = logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
logging.getLogger().addHandler(_file_handler)

# httpx logs every request at INFO with the FULL url — and Telegram/Gemini carry
# their credentials in the url (bot<token>/sendMessage, ?key=...). These logs are
# uploaded as CI artifacts from a public repo, i.e. world-readable, so a live
# token would be published on every run. WARNING keeps transport failures visible
# without printing request lines.
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("sim.orchestrator")

# Double-trigger guard: the pipeline is fired by BOTH the GitHub `schedule` cron
# (unreliable, but often works) and cron-job.org via workflow_dispatch (the
# reliable backstop). When both fire, runs land back-to-back and burn LLM quota
# on a near-empty ingest window. A new run exits early if the last SUCCESSFUL
# run is fresher than this spacing; a failed last run never blocks (the second
# trigger then acts as a free retry). PIPELINE_FORCE_RUN=1 bypasses the guard.
MIN_RUN_SPACING_MINUTES = 90


def _last_successful_run_age_minutes(db_conn) -> float | None:
    """Minutes since the newest successful pipeline_run telemetry row, or None."""
    try:
        row = db_conn.execute(
            """SELECT EXTRACT(EPOCH FROM (NOW() - timestamp)) / 60.0
               FROM system_telemetry
               WHERE event_type = 'pipeline_run'
                 AND value_json ->> 'success' = 'true'
               ORDER BY timestamp DESC
               LIMIT 1"""
        ).fetchone()
        return float(row[0]) if row else None
    except Exception:
        # Guard must never block a run on a telemetry hiccup
        logger.exception("Run-spacing check failed; proceeding with the run")
        return None


def _log_geo_distribution(db_conn, run_started_at) -> dict:
    """
    Country histogram of events classified during this run.

    Answers "are we actually capturing geographic diversity, or drowning in one
    conflict?" with a number per run. Written to system_telemetry as
    'geo_distribution' so the trend can be queried over weeks.
    """
    rows = db_conn.execute(
        """SELECT COALESCE(country_iso, '??') AS country, COUNT(*) AS n
           FROM events
           WHERE updated_at >= %s
             AND status IN ('scored', 'reconciled', 'alerted')
           GROUP BY country
           ORDER BY n DESC""",
        (run_started_at,),
    ).fetchall()

    distribution = {row[0]: row[1] for row in rows}
    total = sum(distribution.values())
    top = ", ".join(f"{c}={n}" for c, n in list(distribution.items())[:10])
    logger.info("Geo distribution: %d events across %d countries [%s]",
                total, len(distribution), top)

    db_conn.execute(
        "INSERT INTO system_telemetry(event_type, value_json) VALUES ('geo_distribution', %s)",
        (json.dumps({"total": total, "countries": distribution}),),
    )
    db_conn.commit()
    return distribution


def run_pipeline():
    """
    Execute the full SIM pipeline: Pass A → B → C → D → E.
    Each pass logs its own telemetry and handles errors independently.
    """
    start_time = time.monotonic()
    run_started_at = datetime.now(timezone.utc)
    run_id = run_started_at.strftime("%Y%m%dT%H%M%S")

    logger.info("=" * 60)
    logger.info("SIM Pipeline Run %s — Starting", run_id)
    logger.info("=" * 60)

    results = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pass_a": None,
        "pass_b": None,
        "pass_c": None,
        "pass_d": None,
        "pass_e": None,
        "run_snapshot": None,
        "pass_f": None,
        "success": False,
        "duration_seconds": 0,
    }

    db_conn = None
    try:
        # Initialize
        db_conn = get_connection()

        # Double-trigger guard (see MIN_RUN_SPACING_MINUTES above)
        if os.environ.get("PIPELINE_FORCE_RUN") != "1":
            age_min = _last_successful_run_age_minutes(db_conn)
            if age_min is not None and age_min < MIN_RUN_SPACING_MINUTES:
                logger.info(
                    "Skipping run %s: last successful run was %.0f min ago "
                    "(< %d min spacing) — duplicate trigger absorbed",
                    run_id, age_min, MIN_RUN_SPACING_MINUTES,
                )
                results["success"] = True
                results["skipped"] = True
                results["skip_reason"] = f"last successful run {age_min:.0f} min ago"
                return results

        router = build_llm_router()

        logger.info("LLM Router: %d accounts, %d RPD total quota",
                     len(router.accounts), router.total_daily_quota)

        # CZIB Sync: Refresh EASA conflict zones before ingestion
        logger.info("--- CZIB Sync: EASA Conflict Zones ---")
        try:
            czib_result = sync_czib_to_db(db_conn)
            logger.info("CZIB sync: %d fetched, %d inserted, %d updated",
                        czib_result["fetched"], czib_result["inserted"], czib_result["updated"])
        except Exception:
            logger.warning("CZIB sync failed, continuing without updated conflict zones")

        # Pass A: Ingest & Canonicalization
        logger.info("--- PASS A: Ingest & Canonicalization ---")
        results["pass_a"] = run_pass_a(db_conn)

        # Pass B: Dedup, Maturation & Distributed Locks
        logger.info("--- PASS B: Dedup & Locks ---")
        results["pass_b"] = run_pass_b(db_conn)

        # Pass C: LLM Classification
        # limit=200: at the default 50 the queue saturated (Jul 6-9 backlog) and
        # fresh events slipped 1-2 runs behind. Pass C's TPM pacing keeps a bigger
        # batch inside the free-tier budget, and the 2h run window has time to spare.
        logger.info("--- PASS C: LLM Classification ---")
        results["pass_c"] = run_pass_c(db_conn, router, limit=200)

        # Pass D: Scoring & Storyline
        logger.info("--- PASS D: Scoring & Storyline ---")
        results["pass_d"] = run_pass_d(db_conn)

        # Pass E: Reconciliation
        logger.info("--- PASS E: Reconciliation ---")
        results["pass_e"] = run_pass_e(db_conn)

        # Geographic diversity telemetry — country histogram of this run's
        # classified events, so source-diversity drift is a weekly metric
        # instead of a gut feeling. Isolated: must never break the run.
        try:
            results["geo_distribution"] = _log_geo_distribution(db_conn, run_started_at)
        except Exception:
            logger.exception("Geo distribution telemetry failed, continuing")

        # Storyline quiet-closures — page a single "storyline quiet" note for alerted
        # storylines that have gone silent. Isolated so it can never break the run.
        try:
            from src.pipeline.pass_d_score import run_storyline_closures
            results["storyline_closures"] = run_storyline_closures(db_conn)
        except Exception:
            logger.exception("Storyline closure sweep failed, continuing")

        # Storyline narratives ("story so far") — budgeted, quality-router (user-facing
        # prose), cache-aware. Isolated failure must never break the pipeline.
        try:
            from src.services.storyline_narrator import (
                NARRATIVE_ENABLED,
                run_storyline_narratives,
            )
            if NARRATIVE_ENABLED:
                logger.info("--- STORYLINE NARRATIVES ---")
                results["narratives"] = run_storyline_narratives(db_conn, build_quality_router())
        except Exception:
            logger.exception("Storyline narration failed, continuing")

        # Per-run JSONL snapshot → Telegram + R2 (does not delete events).
        logger.info("--- RUN SNAPSHOT ---")
        results["run_snapshot"] = run_run_snapshot(db_conn, run_started_at)

        # Pass F: Cold Storage & Archive
        logger.info("--- PASS F: Archive ---")
        results["pass_f"] = run_pass_f(db_conn)

        results["success"] = True

    except Exception as e:
        logger.exception("Pipeline run %s failed", run_id)
        results["success"] = False
        results["error"] = f"{type(e).__name__}: {e}"

    finally:
        results["duration_seconds"] = round(time.monotonic() - start_time, 2)

        # Persist telemetry JSON to logs/ for artifact upload
        try:
            with open(LOGS_DIR / "telemetry.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
        except Exception:
            logger.exception("Failed to write telemetry JSON to logs/")

        # Log pipeline run telemetry to database. Skipped (double-trigger) runs
        # get their OWN event type: a 'pipeline_run' row for a skip would both
        # keep refreshing the spacing guard forever and fool the dead-man check
        # into seeing a healthy run that never actually ingested anything.
        run_event_type = "pipeline_run_skipped" if results.get("skipped") else "pipeline_run"
        if db_conn:
            try:
                db_conn.execute(
                    "INSERT INTO system_telemetry(event_type, value_json) VALUES (%s, %s)",
                    (run_event_type, json.dumps(results, default=str)),
                )
                db_conn.commit()
            except Exception:
                logger.exception("Failed to log pipeline run telemetry")

            try:
                put_connection(db_conn)
            except Exception:
                pass
            close_pool()

    logger.info("=" * 60)
    logger.info(
        "SIM Pipeline Run %s — %s in %.1fs",
        run_id,
        "SUCCESS" if results["success"] else "FAILED",
        results["duration_seconds"],
    )
    logger.info("=" * 60)

    # Operational health ping: a hard failure or a pass that returned an error stat
    # must reach a human — otherwise a silent pipeline death means no alerts and no
    # one knows. Best-effort and isolated so it can never break the run.
    try:
        _notify_health(results)
    except Exception:
        logger.exception("Failed to emit pipeline health notification")

    return results


# Ordered stages we expect to complete; used to name the failure point in a health ping.
_PIPELINE_STAGES = ["pass_a", "pass_b", "pass_c", "pass_d", "pass_e", "run_snapshot", "pass_f"]


def _collect_degradations(results: dict) -> list[str]:
    """Human-readable problems found in a run's per-pass stats (empty if all clean)."""
    problems: list[str] = []
    for stage, stats in results.items():
        if not isinstance(stats, dict):
            continue
        if stats.get("error"):
            problems.append(f"{stage}: {stats['error']}")
        failed = stats.get("events_failed")
        if isinstance(failed, int) and failed > 0:
            problems.append(f"{stage}: {failed} event(s) failed")
    return problems


def _notify_health(results: dict) -> None:
    """Send an ops alert when the run failed hard or a pass reported an error stat."""
    from src.services.ops_notifier import send_ops_alert

    degradations = _collect_degradations(results)
    hard_failure = not results.get("success")
    # `error` keys are real pass failures; per-event `events_failed` alone is routine
    # noise and should not page on its own — only surface it alongside a real problem.
    has_pass_error = any(": " in d and "event(s) failed" not in d for d in degradations)
    if not hard_failure and not has_pass_error:
        return

    if hard_failure:
        # The first stage still None is where we stopped making progress.
        failed_stage = next(
            (s for s in _PIPELINE_STAGES if results.get(s) is None), "init/teardown"
        )
        header = f"❌ Run {results.get('run_id')} FAILED at {failed_stage}"
    else:
        header = f"⚠️ Run {results.get('run_id')} completed DEGRADED"

    lines = [header, f"duration: {results.get('duration_seconds')}s"]
    if results.get("error"):
        lines.append(f"error: {results['error']}")
    if degradations:
        lines.append("issues:")
        lines.extend(f"  • {d}" for d in degradations)
    send_ops_alert("\n".join(lines))
if __name__ == "__main__":
    if "--weekly" in sys.argv:
        logger.info("Weekly forecast execution triggered via CLI parameter.")
        db_conn = None
        success = False
        try:
            db_conn = get_connection()
            router = build_quality_router()
            from src.pipeline.weekly_forecast import run_weekly_forecast
            weekly_result = run_weekly_forecast(db_conn, router)
            success = weekly_result.get("success", False)
        except Exception:
            logger.exception("CLI weekly forecast run failed")
        finally:
            if db_conn:
                try:
                    put_connection(db_conn)
                except Exception:
                    pass
                close_pool()
        sys.exit(0 if success else 1)
    elif "--sitrep" in sys.argv:
        # Daily 24h country SITREP. Optional ISO2 args after the flag
        # (e.g. `--sitrep IR IQ`); without args, auto-selects by event volume.
        iso_args = [
            a.upper() for a in sys.argv[sys.argv.index("--sitrep") + 1:]
            if len(a) == 2 and a.isalpha()
        ]
        logger.info("Daily SITREP execution triggered via CLI (countries=%s).",
                    iso_args or "auto")
        db_conn = None
        success = False
        try:
            db_conn = get_connection()
            router = build_quality_router()
            from src.pipeline.daily_sitrep import run_daily_sitrep
            sitrep_result = run_daily_sitrep(db_conn, router, countries=iso_args or None)
            success = sitrep_result.get("success", False)
        except Exception:
            logger.exception("CLI daily SITREP run failed")
        finally:
            if db_conn:
                try:
                    put_connection(db_conn)
                except Exception:
                    pass
                close_pool()
        sys.exit(0 if success else 1)
    else:
        result = run_pipeline()
        sys.exit(0 if result.get("success") else 1)
