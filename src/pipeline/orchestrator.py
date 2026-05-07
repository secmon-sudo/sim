"""
SIM — Pipeline Orchestrator
Blueprint V20.1 §4

Main entry point that executes all pipeline passes in sequence.
Designed to run as a GitHub Actions job every 30 minutes.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.core.llm_router import build_llm_router
from src.pipeline.pass_a_ingest import run_pass_a
from src.pipeline.pass_b_dedup import run_pass_b
from src.pipeline.pass_c_classify import run_pass_c
from src.pipeline.pass_d_score import run_pass_d
from src.pipeline.pass_e_reconcile import run_pass_e
from src.pipeline.pass_f_archive import run_pass_f
from src.services.czib_client import sync_czib_to_db
from src.services.supabase_client import close_pool, get_connection, put_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sim.orchestrator")


def run_pipeline():
    """
    Execute the full SIM pipeline: Pass A → B → C → D → E.
    Each pass logs its own telemetry and handles errors independently.
    """
    start_time = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

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
        "pass_f": None,
        "success": False,
        "duration_seconds": 0,
    }

    db_conn = None
    try:
        # Initialize
        db_conn = get_connection()
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
        logger.info("--- PASS C: LLM Classification ---")
        results["pass_c"] = run_pass_c(db_conn, router)

        # Pass D: Scoring & Storyline
        logger.info("--- PASS D: Scoring & Storyline ---")
        results["pass_d"] = run_pass_d(db_conn)

        # Pass E: Reconciliation
        logger.info("--- PASS E: Reconciliation ---")
        results["pass_e"] = run_pass_e(db_conn)

        # Pass F: Cold Storage & Archive
        logger.info("--- PASS F: Archive ---")
        results["pass_f"] = run_pass_f(db_conn)

        results["success"] = True

    except Exception:
        logger.exception("Pipeline run %s failed", run_id)
        results["success"] = False

    finally:
        results["duration_seconds"] = round(time.monotonic() - start_time, 2)

        # Log pipeline run telemetry
        if db_conn:
            try:
                db_conn.execute(
                    "INSERT INTO system_telemetry(event_type, value_json) VALUES ('pipeline_run', %s)",
                    (json.dumps(results, default=str),),
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

    return results


if __name__ == "__main__":
    result = run_pipeline()
    sys.exit(0 if result.get("success") else 1)
