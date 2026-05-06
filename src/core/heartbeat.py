"""
SIM — HeartbeatWorker
Blueprint V20.1 §PASS C

Context-manager that runs a background heartbeat update thread.
Keeps the event lock alive during long-running LLM calls.
Stops gracefully on exit, lock loss, or consecutive DB errors.
"""

import logging
import threading

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 5


class HeartbeatWorker:
    """
    Context-manager that runs a background heartbeat update thread.

    Usage:
        with HeartbeatWorker(db, event_id, lock_owner, interval=60) as hb:
            result = call_llm(router, text)
        # On 'with' block exit, worker stops automatically — success OR exception.
    """

    def __init__(self, db_conn, event_id: str, lock_owner: str, interval: int = 60):
        self._db = db_conn
        self._event_id = event_id
        self._lock_owner = lock_owner
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"hb-{event_id[:8]}",
        )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()           # Signal the loop to exit
        self._thread.join(timeout=10)    # Wait at most 10 s
        if self._thread.is_alive():
            logger.warning(
                "Heartbeat thread %s did not terminate cleanly for event %s",
                self._thread.name,
                self._event_id,
            )
        return False  # Always re-raise any exception from the caller

    def _run(self):
        """
        Writes a heartbeat every `interval` seconds.
        Exits immediately when _stop_event is set OR lock ownership is lost.
        Stops after MAX_CONSECUTIVE_ERRORS consecutive DB failures.
        """
        consecutive_errors = 0

        while not self._stop_event.wait(timeout=self._interval):
            try:
                cursor = self._db.execute(
                    """UPDATE events
                       SET    last_heartbeat_at = NOW()
                       WHERE  id = %s AND lock_owner = %s""",
                    (self._event_id, self._lock_owner),
                )
                rowcount = cursor.rowcount if hasattr(cursor, "rowcount") else 0

                if rowcount == 0:
                    # Lock was stolen or released externally — stop silently
                    logger.warning(
                        "Heartbeat: lock lost for event %s (owner %s). Stopping.",
                        self._event_id,
                        self._lock_owner,
                    )
                    return

                self._db.commit()
                consecutive_errors = 0  # Reset on success

            except Exception as exc:
                consecutive_errors += 1
                logger.error(
                    "Heartbeat DB error #%d for event %s: %s",
                    consecutive_errors,
                    self._event_id,
                    exc,
                )
                # Stop after too many consecutive DB failures
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.critical(
                        "Heartbeat: %d consecutive DB failures — stopping for event %s",
                        consecutive_errors,
                        self._event_id,
                    )
                    return
