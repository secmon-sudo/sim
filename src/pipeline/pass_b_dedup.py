"""
SIM — Pass B: Dedup, Maturation & Distributed Locks
Blueprint V20.1 §4 PASS B

Handles stale lock detection, maturation window enforcement,
and lock acquisition for LLM classification.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

STALE_LOCK_THRESHOLD_MINUTES = 15
MATURATION_WINDOW_HOURS = 2


def clear_stale_locks(db_conn, worker_id: uuid.UUID) -> int:
    """
    Find and clear locks where heartbeat is older than threshold.
    Telemetry MUST be committed BEFORE lock is cleared (same transaction).

    Returns: number of stale locks cleared.
    """
    cleared = 0

    try:
        rows = db_conn.execute(
            """SELECT id, lock_owner, last_heartbeat_at,
                      classification_lock
               FROM events
               WHERE classification_lock = TRUE
                 AND last_heartbeat_at < NOW() - INTERVAL '%s minutes'""",
            (STALE_LOCK_THRESHOLD_MINUTES,),
        ).fetchall()

        for row in rows:
            event_id, lock_owner, last_hb, _ = row

            # Step 1: Write telemetry BEFORE clearing lock
            stale_duration = (datetime.now(timezone.utc) - last_hb).total_seconds() if last_hb else 0
            telemetry_payload = {
                "event_type": "stale_lock_cleared",
                "event_id": str(event_id),
                "lock_owner": str(lock_owner),
                "last_heartbeat_at": last_hb.isoformat() if last_hb else None,
                "cleared_by_worker": str(worker_id),
                "stale_duration_seconds": stale_duration,
            }
            db_conn.execute(
                "INSERT INTO system_telemetry(event_type, value_json) VALUES ('stale_lock_cleared', %s)",
                (json.dumps(telemetry_payload),),
            )

            # Step 2: Clear the lock (same transaction)
            db_conn.execute(
                """UPDATE events
                   SET classification_lock = FALSE,
                       lock_owner = NULL,
                       last_heartbeat_at = NULL
                   WHERE id = %s AND lock_owner = %s""",
                (event_id, lock_owner),
            )
            cleared += 1

        if cleared > 0:
            db_conn.commit()
            logger.warning("Cleared %d stale locks", cleared)

    except Exception:
        db_conn.rollback()
        logger.exception("Error clearing stale locks")

    return cleared


def acquire_lock(db_conn, event_id: str, worker_id: uuid.UUID) -> bool:
    """
    Try to acquire classification lock for an event.
    Uses atomic UPDATE with WHERE guard to prevent race conditions.

    Returns True if lock was acquired.
    """
    try:
        result = db_conn.execute(
            """UPDATE events
               SET classification_lock = TRUE,
                   lock_owner = %s,
                   last_heartbeat_at = NOW(),
                   status = 'locked'
               WHERE id = %s
                 AND classification_lock = FALSE
                 AND status = 'deduped'""",
            (str(worker_id), event_id),
        )
        db_conn.commit()
        return result.rowcount > 0
    except Exception:
        db_conn.rollback()
        logger.exception("Lock acquisition error for event %s", event_id)
        return False


def release_lock(db_conn, event_id: str, worker_id: uuid.UUID):
    """
    Idempotent lock release — 0 rows updated is NOT an error.
    """
    try:
        result = db_conn.execute(
            """UPDATE events
               SET classification_lock = FALSE,
                   lock_owner = NULL
               WHERE id = %s AND lock_owner = %s""",
            (event_id, str(worker_id)),
        )
        db_conn.commit()
        if result.rowcount == 0:
            logger.info("Lock release: 0 rows for event %s — already released", event_id)
    except Exception:
        db_conn.rollback()
        logger.exception("Lock release failed for event %s", event_id)


def get_events_for_classification(db_conn, limit: int = 50) -> list[dict]:
    """
    Get raw events that have passed the maturation window
    and are ready for LLM classification.
    """
    try:
        rows = db_conn.execute(
            """SELECT id, canonical_text, source_url, source_domain,
                      anchor_name_raw, country_iso
               FROM events
               WHERE status = 'deduped'
                 AND classification_lock = FALSE
                 AND ingested_at < NOW() - INTERVAL '%s hours'
               ORDER BY ingested_at ASC
               LIMIT %s""",
            (MATURATION_WINDOW_HOURS, limit),
        ).fetchall()

        return [
            {
                "id": str(row[0]),
                "canonical_text": row[1],
                "source_url": row[2],
                "source_domain": row[3],
                "anchor_name_raw": row[4],
                "country_iso": row[5],
            }
            for row in rows
        ]
    except Exception:
        logger.exception("Error fetching events for classification")
        return []


def mark_as_deduped(db_conn) -> int:
    """
    Move raw events to 'deduped' status.
    In a full implementation, this would include URL-hash dedup checks,
    content similarity, and maturation window enforcement.
    """
    try:
        result = db_conn.execute(
            """UPDATE events
               SET status = 'deduped', updated_at = NOW()
               WHERE status = 'raw'"""
        )
        db_conn.commit()
        count = result.rowcount
        if count > 0:
            logger.info("Marked %d events as deduped", count)
        return count
    except Exception:
        db_conn.rollback()
        logger.exception("Error marking events as deduped")
        return 0


def run_pass_b(db_conn) -> dict:
    """
    Execute Pass B: Dedup, Maturation & Distributed Locks.

    1. Mark raw events as deduped
    2. Clear stale locks
    3. Return stats

    Returns: stats dict
    """
    worker_id = uuid.uuid4()

    stats = {
        "worker_id": str(worker_id),
        "events_deduped": 0,
        "stale_locks_cleared": 0,
    }

    # Step 1: Move raw → deduped
    stats["events_deduped"] = mark_as_deduped(db_conn)

    # Step 2: Clear stale locks
    stats["stale_locks_cleared"] = clear_stale_locks(db_conn, worker_id)

    # Log telemetry
    try:
        db_conn.execute(
            "INSERT INTO system_telemetry(event_type, value_json) VALUES ('pass_b', %s)",
            (json.dumps(stats),),
        )
        db_conn.commit()
    except Exception:
        logger.exception("Failed to log Pass B telemetry")

    logger.info("Pass B complete: %s", stats)
    return stats
