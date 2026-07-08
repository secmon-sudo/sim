"""
SIM — Dead-man's switch.

Runs on its own lightweight cron, INDEPENDENT of the main pipeline, so it can catch
the one failure mode the pipeline can never report itself: not running at all
(workflow disabled, repo suspended, DB unreachable at launch, GitHub Actions outage).

It reads the newest `pipeline_run` telemetry row and pages ops if the pipeline has
not produced one within DEADMAN_MAX_AGE_HOURS. A recent-but-failed run is left to the
orchestrator's own health ping; here we only care about silence.

Exit code is 0 even when it pages, so the cron job itself stays green — the signal is
the Telegram message, not the job status.
"""

import logging
import os
import sys

import psycopg

from src.services.ops_notifier import send_ops_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sim.deadman")

DEFAULT_MAX_AGE_HOURS = 3.0


def check(db_url: str, max_age_hours: float) -> bool:
    """Return True if healthy (recent run found), False if a stale/no-run alert was sent."""
    with psycopg.connect(db_url) as conn:
        row = conn.execute(
            """SELECT timestamp,
                      EXTRACT(EPOCH FROM (NOW() - timestamp)) / 3600.0 AS age_hours,
                      value_json ->> 'success'                        AS success
               FROM system_telemetry
               WHERE event_type = 'pipeline_run'
               ORDER BY timestamp DESC
               LIMIT 1""",
        ).fetchone()

    if row is None:
        logger.warning("No pipeline_run telemetry found at all")
        send_ops_alert(
            "🚨 DEAD-MAN: no pipeline run has ever been recorded. "
            "The pipeline may have never started successfully.",
            title="SIM DEAD-MAN'S SWITCH",
        )
        return False

    ts, age_hours, success = row
    age_hours = float(age_hours or 0.0)
    if age_hours > max_age_hours:
        logger.warning("Last pipeline run was %.1fh ago (threshold %.1fh)", age_hours, max_age_hours)
        send_ops_alert(
            f"🚨 DEAD-MAN: no pipeline run in {age_hours:.1f}h "
            f"(threshold {max_age_hours:.0f}h). Last run at {ts} UTC, "
            f"success={success}. The pipeline appears to have stopped.",
            title="SIM DEAD-MAN'S SWITCH",
        )
        return False

    logger.info("Healthy: last pipeline run %.1fh ago (success=%s)", age_hours, success)
    return True


def main() -> int:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set")
        # Can't check the DB, but we can still shout about it.
        send_ops_alert(
            "🚨 DEAD-MAN: DATABASE_URL is not set — cannot verify pipeline health.",
            title="SIM DEAD-MAN'S SWITCH",
        )
        return 0

    try:
        max_age = float(os.environ.get("DEADMAN_MAX_AGE_HOURS", DEFAULT_MAX_AGE_HOURS))
    except ValueError:
        max_age = DEFAULT_MAX_AGE_HOURS

    try:
        check(db_url, max_age)
    except Exception as e:
        logger.exception("Dead-man check itself failed")
        send_ops_alert(
            f"🚨 DEAD-MAN: health check crashed while querying the DB: {type(e).__name__}: {e}",
            title="SIM DEAD-MAN'S SWITCH",
        )
    # Always green — the alert is the payload, not the exit code.
    return 0


if __name__ == "__main__":
    sys.exit(main())
