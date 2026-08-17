"""
SIM — Pass E: Targeted Reconciliation
Blueprint V20.1 §4 PASS E

Strictly NO LLM. Re-evaluates anchors on concatenated text,
clears Top-10 arrays on anchor upgrade, and recalculates scores.
"""

import json
import logging

from src.core.alerts import evaluate_alert_tier, tier_rank
from src.core.anchor import get_anchor_confidence_level, normalize_anchor
from src.pipeline.pass_d_score import (
    _safe_float,
    apply_safety_downrank,
    compute_confidence,
    compute_severity,
)

logger = logging.getLogger(__name__)


def _sibling_anchor_texts(db_conn, storyline_id) -> list[str]:
    """Raw location texts from the other reports of the same storyline."""
    if not storyline_id:
        return []
    try:
        rows = db_conn.execute(
            """SELECT anchor_name_raw
               FROM events
               WHERE storyline_id = %s AND anchor_name_raw IS NOT NULL""",
            (str(storyline_id),),
        ).fetchall()
    except Exception:
        logger.exception("Sibling anchor lookup failed for storyline %s", storyline_id)
        return []
    return [r[0] for r in rows if r and r[0]]


def _anchor_country(db_conn, iata_code: str) -> str | None:
    """country_iso for a resolved anchor, or None when it cannot be read."""
    try:
        row = db_conn.execute(
            "SELECT country_iso FROM anchor_master WHERE iata_code = %s", (iata_code,)
        ).fetchone()
    except Exception:
        logger.exception("Anchor country lookup failed for %s", iata_code)
        return None
    return row[0] if row and row[0] else None


def reconcile_single_event(db_conn, event_id: str) -> tuple[bool, bool]:
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
                      llm_parsed_output, severity_score, system_confidence,
                      alert_tier, source_title
               FROM events WHERE id = %s AND status = 'scored'""",
            (event_id,),
        ).fetchone()

        if not row:
            return False, False

        event_id = str(row[0])
        event_type = row[1]
        raw_anchor = row[2]
        current_norm = row[3]
        current_conf_level = row[4]
        storyline_id = row[5]
        llm_parsed = row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}")
        current_tier = row[10]
        # Needed by the aftermath gate in evaluate_alert_tier — without it an anchor
        # upgrade would re-promote a roundup that Pass D correctly refused to page.
        source_title = row[11]

        # 1. Gather each sibling's location text as a SEPARATE candidate.
        #
        # This used to concatenate them into one string and normalize that. It could
        # never upgrade anything: trigram similarity is a ratio over the whole string,
        # so every sibling appended drove the score DOWN, and the exact and alias paths
        # need a 3-4 letter code or a whole-string alias hit that a concatenation is
        # incapable of producing. anchor_upgrades was 0 on every run ever recorded —
        # not because nothing needed upgrading, but because the mechanism was
        # arithmetically incapable of firing. Scoring candidates one at a time is what
        # "enriched by siblings" was supposed to mean: a sibling that names the airport
        # plainly can now resolve an event whose own text was vague.
        candidates: list[str] = []
        seen: set[str] = set()
        for text in [raw_anchor] + _sibling_anchor_texts(db_conn, storyline_id):
            if text and text.strip() and text.strip().lower() not in seen:
                seen.add(text.strip().lower())
                candidates.append(text.strip())

        # 2. Re-evaluate the anchor, keeping the most confident single candidate whose
        #    country does not contradict this event's own.
        #
        # Scoring siblings separately means one bad resolution can be adopted by every
        # member of the storyline, which is strictly worse than the concatenation it
        # replaced: that could only fail to upgrade, this can actively mislabel.
        # Observed 2026-08-17 — a sibling reading "Russian capital" resolved to PEK
        # (Beijing) and three Moscow events inherited CN, each of them already paging.
        # The stopword fix removes that particular match, but the amplification is the
        # structural risk, so the country the classifier extracted acts as a veto.
        own_iso = (llm_parsed.get("country_iso") or llm_parsed.get("country") or "")
        own_iso = own_iso.strip().upper()[:2]
        if candidates:
            new_norm, new_conf = None, 0.0
            for candidate in candidates:
                cand_norm, cand_conf = normalize_anchor(candidate, db_conn)
                if not cand_norm or cand_conf <= new_conf:
                    continue
                if own_iso and _anchor_country(db_conn, cand_norm) not in (None, own_iso):
                    logger.info(
                        "Pass E rejected sibling anchor %s for event %s: country "
                        "disagrees with classifier (%s)", cand_norm, event_id[:8], own_iso,
                    )
                    continue
                new_norm, new_conf = cand_norm, cand_conf
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

                # Re-evaluate the alert tier against the values we just rewrote.
                # Without this the row kept a tier derived from the PRE-upgrade
                # anchor/severity/confidence — an invariant break that stayed
                # invisible only because anchor_upgrades has been 0 on every
                # observed run. It matters more now that resolving a location is
                # itself a tier gate: an upgrade is exactly the event that turns an
                # unlocated event into a located one.
                new_tier = evaluate_alert_tier({
                    "severity_score": new_severity,
                    "system_confidence": new_system_conf,
                    "anchor_confidence": new_level,
                    "time_certainty": llm_parsed.get("time_certainty", "unknown"),
                    "event_type": event_type,
                    "anchor_name_norm": new_norm,
                    "latitude": lat,
                    "source_title": source_title,
                })

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
                               alert_tier = %s,
                               is_safety = %s,
                               status = 'reconciled',
                               updated_at = NOW()
                           WHERE id = %s""",
                        (new_norm, new_level, lat, lon, country,
                         new_severity, new_system_conf, new_tier, is_safety, event_id),
                    )
                db_conn.commit()

                # An upgrade that RAISES the tier is a real escalation that Pass D
                # already declined to page. Pass E deliberately does not dispatch —
                # suppression/escalation state lives in Pass D — so surface it loudly
                # instead of deciding silently. This path has never executed in
                # production; if it starts to, the log is the signal to wire paging.
                if tier_rank(new_tier) > tier_rank(current_tier):
                    logger.warning(
                        "Event %s escalated %s→%s on anchor upgrade but was NOT paged "
                        "(Pass E does not dispatch)",
                        event_id[:8], current_tier or "none", new_tier,
                    )
                return True, True

        # No upgrade — just mark as reconciled
        with db_conn.transaction():
            db_conn.execute(
                """UPDATE events
                   SET status = 'reconciled', updated_at = NOW()
                   WHERE id = %s""",
                (event_id,),
            )
        db_conn.commit()
        return True, False

    except Exception:
        try:
            db_conn.rollback()
        except Exception:
            pass
        logger.exception("Error reconciling event %s", event_id)
        return False, False


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
            # anchor_upgrades was initialised and then never touched: the function
            # returned a bare bool, so the counter read 0 on every run since Pass E
            # existed. That is the same shape of blindness the upgrade path itself
            # had — the mechanism was repaired on 2026-08-17 and would still have
            # reported nothing.
            ok, upgraded = reconcile_single_event(db_conn, str(row[0]))
            if ok:
                stats["events_reconciled"] += 1
                if upgraded:
                    stats["anchor_upgrades"] += 1
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
