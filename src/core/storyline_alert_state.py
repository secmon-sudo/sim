"""
SIM — Storyline alert state (escalation + quiet-closure).

Small state layer over `storyline_alert_state` (migration 013) so alerting can be
storyline-aware over time rather than treating each card in isolation:

  - ESCALATION: when a storyline that already paged at a lower tier crosses into a
    higher one, the new card is labelled "⬆️ Escalated WATCH→CRITICAL" instead of
    reading as an unrelated fresh alert.
  - CLOSURE: when an alerted storyline goes quiet (no new page within a window), a
    single "storyline quiet" note is emitted and the storyline is marked closed so it
    never double-closes.

All functions are best-effort at the call sites; failures here must never break the
event alert itself.
"""

import logging

logger = logging.getLogger(__name__)

TIER_RANK = {"WATCH": 1, "ALERT": 2, "CRITICAL": 3}


def _rank(tier: str | None) -> int:
    return TIER_RANK.get(tier or "", 0)


def get_peak_tier(db_conn, storyline_id: str) -> str | None:
    """Highest tier ever paged for this storyline, or None if it has never paged."""
    if not storyline_id:
        return None
    row = db_conn.execute(
        "SELECT peak_tier FROM storyline_alert_state WHERE storyline_id = %s",
        (storyline_id,),
    ).fetchone()
    return row[0] if row else None


def is_escalation(prev_peak: str | None, current_tier: str) -> bool:
    """True when the current tier is strictly higher than any tier paged before."""
    return prev_peak is not None and _rank(current_tier) > _rank(prev_peak)


def register_alert(db_conn, storyline_id: str, tier: str, severity: int, label: str) -> None:
    """Record that `storyline_id` paged at `tier`. Keeps the highest-ever peak_tier and
    re-opens the storyline (closed=FALSE) so renewed activity can close again later."""
    if not storyline_id:
        return
    # peak_tier is stored as the tier STRING; compare via a CASE→rank map on both the
    # incoming (EXCLUDED) and stored value so we keep whichever tier is higher.
    _rc = "CASE {col} WHEN 'CRITICAL' THEN 3 WHEN 'ALERT' THEN 2 WHEN 'WATCH' THEN 1 ELSE 0 END"
    try:
        db_conn.execute(
            f"""INSERT INTO storyline_alert_state
                   (storyline_id, last_tier, last_severity, peak_tier, label,
                    last_alerted_at, closed, closed_at)
               VALUES (%s, %s, %s, %s, %s, NOW(), FALSE, NULL)
               ON CONFLICT (storyline_id) DO UPDATE SET
                   last_tier       = EXCLUDED.last_tier,
                   last_severity   = EXCLUDED.last_severity,
                   peak_tier       = CASE
                                        WHEN {_rc.format(col='EXCLUDED.peak_tier')}
                                           > {_rc.format(col='storyline_alert_state.peak_tier')}
                                        THEN EXCLUDED.peak_tier
                                        ELSE storyline_alert_state.peak_tier
                                     END,
                   label           = EXCLUDED.label,
                   last_alerted_at = NOW(),
                   closed          = FALSE,
                   closed_at       = NULL""",
            (storyline_id, tier, severity, tier, label),
        )
        db_conn.commit()
    except Exception:
        db_conn.rollback()
        logger.exception("Failed to register alert state for storyline %s", storyline_id[:8])


def find_and_close_quiet(db_conn, quiet_hours: float) -> list[dict]:
    """Close open storylines with no page within `quiet_hours`.

    Atomically flips them to closed and returns their context so the caller can emit one
    closure note each. The UPDATE...RETURNING makes claiming-and-listing a single step, so
    concurrent pipeline runs can't both close the same storyline.
    """
    try:
        rows = db_conn.execute(
            """UPDATE storyline_alert_state
                   SET closed = TRUE, closed_at = NOW()
               WHERE closed = FALSE
                 AND last_alerted_at < NOW() - (%s * interval '1 hour')
               RETURNING storyline_id, peak_tier, label, last_alerted_at""",
            (quiet_hours,),
        ).fetchall()
        db_conn.commit()
    except Exception:
        db_conn.rollback()
        logger.exception("Failed to sweep quiet storylines")
        return []
    return [
        {
            "storyline_id": str(r[0]),
            "peak_tier": r[1],
            "label": r[2],
            "last_alerted_at": r[3],
        }
        for r in rows
    ]
