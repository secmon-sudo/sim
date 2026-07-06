"""
SIM — Pass E: Targeted Reconciliation
Blueprint V20.1 §4 PASS E

Strictly NO LLM. Re-evaluates anchors on concatenated text,
clears Top-10 arrays on anchor upgrade, and recalculates scores.
"""

import json
import logging

from src.core.anchor import get_anchor_confidence_level, normalize_anchor
from src.pipeline.pass_d_score import (
    _safe_float,
    apply_safety_downrank,
    compute_confidence,
    compute_severity,
)

logger = logging.getLogger(__name__)


def reconcile_single_event(db_conn, event_id: str) -> bool:
    """
    Reconcile a single scored event.

    1. Re-evaluate anchor using concatenated text from all storyline events
    2. If anchor upgraded, recalculate severity and confidence
    3. Mark as reconciled

    Returns True if event was reconciled.
    """
    try:
        row = db_conn.execute(
            """SELECT id, event_type, anchor_name_raw, anchor_name_norm,
                      anchor_confidence, storyline_id, storyline_hint,
                      llm_parsed_output, severity_score, system_confidence
               FROM events WHERE id = %s AND status = 'scored'""",
            (event_id,),
        ).fetchone()

        if not row:
            return False

        event_id = str(row[0])
        event_type = row[1]
        raw_anchor = row[2]
        current_norm = row[3]
        current_conf_level = row[4]
        storyline_id = row[5]
        llm_parsed = row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}")

        # 1. Gather all text from storyline siblings
        concatenated_text = raw_anchor or ""
        if storyline_id:
            siblings = db_conn.execute(
                """SELECT anchor_name_raw, canonical_text
                   FROM events
                   WHERE storyline_id = %s AND anchor_name_raw IS NOT NULL""",
                (str(storyline_id),),
            ).fetchall()
            for sib in siblings:
                if sib[0]:
                    concatenated_text += f" {sib[0]}"

        # 2. Re-evaluate anchor with enriched text
        if concatenated_text.strip():
            new_norm, new_conf = normalize_anchor(concatenated_text.strip(), db_conn)
            new_level = get_anchor_confidence_level(new_conf)

            # Check if this is an upgrade
            confidence_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            old_rank = confidence_order.get(current_conf_level or "LOW", 0)
            new_rank = confidence_order.get(new_level, 0)

            if new_rank > old_rank and new_norm:
                logger.info(
                    "Anchor upgrade for event %s: %s→%s (%s→%s)",
                    event_id[:8], current_norm, new_norm, current_conf_level, new_level,
                )

                # Get czib data for new anchor
                czib = False
                lat = None
                lon = None
                country = None
                try:
                    anchor_row = db_conn.execute(
                        "SELECT czib_flag, latitude, longitude, country_iso FROM anchor_master WHERE iata_code = %s",
                        (new_norm,),
                    ).fetchone()
                    if anchor_row:
                        czib, lat, lon, country = anchor_row
                except Exception:
                    pass

                # Recalculate severity (keep safety de-prioritization consistent)
                anchor_data = {"confidence": new_conf, "czib_flag": czib}
                new_severity = compute_severity(event_type, anchor_data, db_conn)
                new_severity, is_safety = apply_safety_downrank(event_type, new_severity, llm_parsed)

                # Recalculate confidence
                llm_conf = _safe_float(llm_parsed.get("confidence", 0.5))
                new_system_conf = compute_confidence(llm_conf, new_conf)

                # Update with upgraded anchor
                with db_conn.transaction():
                    db_conn.execute(
                        """UPDATE events
                           SET anchor_name_norm = %s,
                               anchor_confidence = %s,
                               latitude = COALESCE(%s, latitude),
                               longitude = COALESCE(%s, longitude),
                               country_iso = COALESCE(%s, country_iso),
                               severity_score = %s,
                               system_confidence = %s,
                               is_safety = %s,
                               status = 'reconciled',
                               updated_at = NOW()
                           WHERE id = %s""",
                        (new_norm, new_level, lat, lon, country,
                         new_severity, new_system_conf, is_safety, event_id),
                    )
                db_conn.commit()
                return True

        # No upgrade — just mark as reconciled
        with db_conn.transaction():
            db_conn.execute(
                """UPDATE events
                   SET status = 'reconciled', updated_at = NOW()
                   WHERE id = %s""",
                (event_id,),
            )
        db_conn.commit()
        return True

    except Exception:
        try:
            db_conn.rollback()
        except Exception:
            pass
        logger.exception("Error reconciling event %s", event_id)
        return False


def run_pass_e(db_conn) -> dict:
    """
    Execute Pass E: Targeted Reconciliation.
    Strictly NO LLM calls.

    Returns: stats dict
    """
    stats = {
        "events_reconciled": 0,
        "anchor_upgrades": 0,
        "events_failed": 0,
    }

    try:
        rows = db_conn.execute(
            "SELECT id FROM events WHERE status = 'scored' ORDER BY ingested_at ASC",
        ).fetchall()

        for row in rows:
            result = reconcile_single_event(db_conn, str(row[0]))
            if result:
                stats["events_reconciled"] += 1
            else:
                stats["events_failed"] += 1

    except Exception:
        logger.exception("Error in Pass E")

    # Log telemetry
    try:
        db_conn.execute(
            "INSERT INTO system_telemetry(event_type, value_json) VALUES ('pass_e', %s)",
            (json.dumps(stats),),
        )
        db_conn.commit()
    except Exception:
        logger.exception("Failed to log Pass E telemetry")

    logger.info("Pass E complete: %s", stats)
    return stats
