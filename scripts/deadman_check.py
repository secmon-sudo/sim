"""
SIM — Dead-man's switch.

Runs on its own lightweight cron, INDEPENDENT of the main pipeline, so it can catch
the one failure mode the pipeline can never report itself: not running at all
(workflow disabled, repo suspended, DB unreachable at launch, GitHub Actions outage).

It reads the newest `pipeline_run` telemetry row and pages ops if the pipeline has
not produced one within DEADMAN_MAX_AGE_HOURS. A recent-but-failed run is left to the
orchestrator's own health ping; here we only care about silence.

It then checks the SECOND thing that can be silently absent: the cron-job.org dispatch
backstop (see the comment on check_dispatch_backstop).

Exit code is 0 even when it pages, so the cron job itself stays green — the signal is
the Telegram message, not the job status.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

import httpx
import psycopg

from src.services.ops_notifier import send_ops_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sim.deadman")

DEFAULT_MAX_AGE_HOURS = 3.0

# ── Dispatch backstop ──────────────────────────────────────────────────────
#
# The pipeline has two triggers: GitHub's own schedule (every 2h, documented as
# best-effort and known to drop firings under load) and a cron-job.org job that calls
# workflow_dispatch. The second exists precisely because the first is unreliable.
#
# It rotted silently for five days. Measured 2026-08-17: dispatch-triggered runs stopped
# on 12 Aug at 06:56Z — 8 a day on 10-11 Aug, 4 on the 12th, then zero — and cron-job.org
# was returning 422 from GitHub the whole time. Nothing noticed, because the staleness
# check above only asks whether the pipeline ran, and 12 scheduled runs a day answered
# yes. A backstop nobody watches is not a backstop; it is a belief.
DISPATCH_WORKFLOW_FILE = "osint-pipeline.yml"
DEFAULT_DISPATCH_MAX_AGE_HOURS = 6.0  # cron-job.org fires every 3h — one miss of slack.

# This condition is not urgent (the pipeline is running) and the switch runs hourly, so
# an uncooled alert would post ~24 identical cards a day until someone fixed a web form.
# The marker row is written to system_telemetry, whose retention only ages 'llm_call'.
DISPATCH_ALERT_COOLDOWN_HOURS = 12.0
DISPATCH_ALERT_EVENT = "deadman_dispatch_alert"


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


def latest_dispatch_age_hours(repo: str, token: str | None = None) -> float | None:
    """Hours since the newest workflow_dispatch-triggered pipeline run.

    None means the API answered but listed no such run at all, which is itself the
    alarm condition — not an error. Errors raise, and the caller treats them as
    "unknown" rather than as a failure, because a rate limit or a GitHub blip must
    never page about the backstop.

    The repo is public, so this works unauthenticated; the token is passed when one is
    available purely for the rate limit (60/h per IP vs 1000/h), and shared runner IPs
    make that worth doing.
    """
    url = (f"https://api.github.com/repos/{repo}/actions/workflows/"
           f"{DISPATCH_WORKFLOW_FILE}/runs")
    params = {"event": "workflow_dispatch", "per_page": 1}
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    resp = None
    if token:
        resp = httpx.get(url, params=params, timeout=15.0,
                         headers={**headers, "Authorization": f"Bearer {token}"})
        # A token that lacks actions:read answers 403, and swallowing that would leave a
        # monitor that reports nothing forever — the exact shape of the rot it is here to
        # catch. The public endpoint needs no token, so fall through to it.
        if resp.status_code in (401, 403):
            logger.warning("Token rejected for run history (%s) — retrying anonymously",
                           resp.status_code)
            resp = None
    if resp is None:
        resp = httpx.get(url, params=params, headers=headers, timeout=15.0)

    resp.raise_for_status()
    runs = resp.json().get("workflow_runs") or []
    if not runs:
        return None
    created = datetime.fromisoformat(runs[0]["created_at"].replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - created).total_seconds() / 3600.0


def _dispatch_alert_on_cooldown(conn, cooldown_hours: float) -> bool:
    row = conn.execute(
        """SELECT EXTRACT(EPOCH FROM (NOW() - timestamp)) / 3600.0
             FROM system_telemetry
            WHERE event_type = %s
            ORDER BY timestamp DESC
            LIMIT 1""",
        (DISPATCH_ALERT_EVENT,),
    ).fetchone()
    return bool(row and float(row[0] or 0.0) < cooldown_hours)


def check_dispatch_backstop(db_url: str, max_age_hours: float,
                            cooldown_hours: float = DISPATCH_ALERT_COOLDOWN_HOURS) -> bool:
    """Return True if the dispatch backstop looks alive, False if it was reported.

    Deliberately quiet about its own failures: this is a check on a redundancy, so an
    unreachable GitHub API logs and returns True rather than manufacturing a page about
    a system that is, as far as anyone knows, fine.
    """
    if not max_age_hours:
        logger.info("Dispatch backstop check disabled")
        return True

    repo = os.environ.get("GITHUB_REPOSITORY") or "secmon-sudo/sim"
    try:
        age = latest_dispatch_age_hours(repo, os.environ.get("GITHUB_TOKEN"))
    except Exception as e:
        logger.warning("Could not read dispatch history (%s: %s) — not paging",
                       type(e).__name__, e)
        return True

    if age is not None and age <= max_age_hours:
        logger.info("Dispatch backstop healthy: last dispatched run %.1fh ago", age)
        return True

    described = "has never fired" if age is None else f"last fired {age:.1f}h ago"
    with psycopg.connect(db_url) as conn:
        if _dispatch_alert_on_cooldown(conn, cooldown_hours):
            logger.info("Dispatch backstop %s; alert on cooldown (<%.0fh)",
                        described, cooldown_hours)
            return False
        logger.warning("Dispatch backstop %s (threshold %.1fh)", described, max_age_hours)
        send_ops_alert(
            f"⚠️ DISPATCH BACKSTOP: the cron-job.org workflow_dispatch trigger "
            f"{described} (threshold {max_age_hours:.0f}h). Scheduled runs are "
            f"unaffected — this is the redundancy for GitHub dropping them, so the "
            f"pipeline is running blind on one trigger. Check the job's response body: "
            f"GitHub names the reason in the 422 message.",
            title="SIM DISPATCH BACKSTOP",
        )
        conn.execute(
            "INSERT INTO system_telemetry(event_type, value_json) VALUES (%s, %s)",
            (DISPATCH_ALERT_EVENT,
             json.dumps({"age_hours": age, "threshold_hours": max_age_hours,
                         "repo": repo})),
        )
        conn.commit()
    return False


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


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

    max_age = _env_float("DEADMAN_MAX_AGE_HOURS", DEFAULT_MAX_AGE_HOURS)

    try:
        check(db_url, max_age)
    except Exception as e:
        logger.exception("Dead-man check itself failed")
        send_ops_alert(
            f"🚨 DEAD-MAN: health check crashed while querying the DB: {type(e).__name__}: {e}",
            title="SIM DEAD-MAN'S SWITCH",
        )

    # Separately, and after the primary check on purpose: the backstop is the lesser
    # signal, and a crash here must not cost the staleness page that actually matters.
    try:
        check_dispatch_backstop(
            db_url,
            _env_float("DEADMAN_DISPATCH_MAX_AGE_HOURS", DEFAULT_DISPATCH_MAX_AGE_HOURS),
        )
    except Exception:
        logger.exception("Dispatch backstop check failed — not paging")

    # Always green — the alert is the payload, not the exit code.
    return 0


if __name__ == "__main__":
    sys.exit(main())
