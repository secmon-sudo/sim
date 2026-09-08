"""
SIM — Pass E: Targeted Reconciliation
Blueprint V20.1 §4 PASS E

Strictly NO LLM. Re-evaluates anchors on concatenated text,
clears Top-10 arrays on anchor upgrade, and recalculates scores.

An upgrade rewrites severity, confidence and alert_tier, which means Pass E has to
recompute them with the SAME recipe Pass D used. It did not: it skipped the casualty
and aviation contributions to severity, the diversity and credibility weights on
confidence, and — the one that changed pages — four of the twelve fields the tier gates
read. Measured over the 17 production runs to 2026-09-07, 13 of 55 escalations were
that gap alone: articles Pass D had already refused to page as commentary or followup,
handed the tier back because report_kind never reached the gate.
"""

import json
import logging

from src.core.alerts import evaluate_alert_tier_verbose, tier_rank
from src.core.anchor import get_anchor_confidence_level, normalize_anchor
from src.pipeline.pass_d_score import (
    MAX_SEVERITY,
    _safe_float,
    apply_planned_closure_downrank,
    apply_safety_downrank,
    compute_aviation_bonus,
    compute_confidence,
    compute_diversity_score,
    compute_severity,
    dispatch_alert,
    source_credibility_multiplier,
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


def reconcile_single_event(db_conn, event_id: str) -> tuple[bool, bool, str | None]:
    """
    Reconcile a single scored event.

    1. Re-evaluate anchor using concatenated text from all storyline events
    2. If anchor upgraded, recalculate severity and confidence
    3. Dispatch when the upgrade RAISES the tier
    4. Mark as reconciled

    Returns (reconciled, anchor_upgraded, dispatch_result). dispatch_result is None
    unless the tier rose, in which case it is dispatch_alert's own verdict.
    """
    try:
        row = db_conn.execute(
            """SELECT id, event_type, anchor_name_raw, anchor_name_norm,
                      anchor_confidence, storyline_id, storyline_hint,
                      llm_parsed_output, severity_score, system_confidence,
                      alert_tier, source_title, country_iso, source_url,
                      source_domain, occurred_at_est, ingested_at,
                      published_at, date_verified, corroborating_sources
               FROM events WHERE id = %s AND status = 'scored'""",
            (event_id,),
        ).fetchone()

        if not row:
            return False, False, None

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

        # The same event dict Pass D built, for the same three consumers: the aviation
        # bonus reads the headline and hint, the tier gates read article shape and date
        # provenance, and dispatch_alert reads the notification fields. Assembling it
        # here rather than passing bare columns is what keeps the two passes honest —
        # a field Pass D adds to the gate is one Pass E cannot silently omit.
        event = {
            "id": event_id,
            "event_type": event_type,
            "anchor_name_raw": raw_anchor,
            "country_iso": row[12],
            "llm_parsed": llm_parsed,
            "storyline_hint": row[6],
            "storyline_id": str(storyline_id) if storyline_id else None,
            "occurred_at_est": row[15],
            "occurred_at_is_fallback": row[15] is None,
            "ingested_at": row[16],
            "source_title": source_title,
            "source_url": row[13],
            "source_domain": row[14],
            "date_verified": bool(row[18]),
            "corroborating_sources": row[19],
        }

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

                # Recompute severity with Pass D's full chain. Passing llm_parsed is
                # what restores the casualty bonus; the aviation bonus and the planned
                # closure cap were missing outright. Every one of them was a term Pass D
                # had already applied and Pass E then overwrote with a number computed
                # without it — an anchor upgrade could LOWER a stored severity.
                anchor_data = {"confidence": new_conf, "czib_flag": czib}
                new_severity = compute_severity(event_type, anchor_data, db_conn, llm_parsed)
                new_severity = min(new_severity + compute_aviation_bonus(event, anchor_data),
                                   MAX_SEVERITY)
                new_severity = apply_planned_closure_downrank(event_type, new_severity,
                                                             llm_parsed)
                new_severity, is_safety = apply_safety_downrank(event_type, new_severity, llm_parsed)

                # ...and the same for confidence: source diversity and publisher
                # credibility are both inputs Pass D weights and Pass E dropped.
                llm_conf = _safe_float(llm_parsed.get("confidence", 0.5))
                diversity = compute_diversity_score(db_conn, storyline_id)
                new_system_conf = compute_confidence(llm_conf, new_conf, diversity)
                new_system_conf = float(new_system_conf * source_credibility_multiplier(
                    event.get("source_domain")))

                # Re-evaluate the alert tier against the values we just rewrote.
                # Without this the row kept a tier derived from the PRE-upgrade
                # anchor/severity/confidence — an invariant break that stayed
                # invisible only because anchor_upgrades has been 0 on every
                # observed run. It matters more now that resolving a location is
                # itself a tier gate: an upgrade is exactly the event that turns an
                # unlocated event into a located one.
                new_tier, veto = evaluate_alert_tier_verbose({
                    "severity_score": new_severity,
                    "system_confidence": new_system_conf,
                    "anchor_confidence": new_level,
                    "time_certainty": llm_parsed.get("time_certainty", "unknown"),
                    "event_type": event_type,
                    "anchor_name_norm": new_norm,
                    "latitude": lat,
                    "source_title": source_title,
                    # The four fields this dict used to be missing. report_kind is the
                    # one that cost pages: 13 of 55 escalations over the 17 runs to
                    # 2026-09-07 were commentary or followup articles that Pass D had
                    # vetoed and Pass E promoted back, because an absent report_kind
                    # reads as new_incident by design (the gate fails open).
                    "report_kind": llm_parsed.get("report_kind"),
                    "date_verified": event["date_verified"],
                    "published_at": row[17],
                    "corroborating_sources": event["corroborating_sources"],
                })
                if veto:
                    logger.info("Pass E tier vetoed for event %s after upgrade: %s",
                                event_id[:8], veto)

                # NOTE: events.alert_tier is what this event QUALIFIES for, not a
                # record that a card was sent. Two things write a tier here without a
                # dispatch — the travel-advisory path, which returns before the
                # article-shape gates by design, and (until 2026-09-07) this pass
                # promoting past a report_kind veto. Measured 2026-09-08: 150 of
                # 13,856 reconciled rows carry a tier with a not-news report_kind,
                # 148 of them from before that fix and 2 advisories.
                #
                # They are inert, and it is worth writing down why so the next reader
                # does not go looking: purge_expired_archived only touches
                # status='archived', and recent_paged_alerts takes its tier from the
                # alert_suppression CLAIM rather than from this column. A claim exists
                # only where a card actually went. Anything that needs "did this page"
                # must read alert_suppression, never this.
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

                # An upgrade that RAISES the tier is a real escalation. This used to
                # log and stop, on the reasoning that suppression state lives in Pass D
                # — but the state lives in the alert_suppression TABLE, and dispatch_alert
                # is the function that reads it, so calling it here reuses the same
                # ledger rather than opening a second one. The path is no longer
                # theoretical: over the 17 runs to 2026-09-07 it fired 55 times, 42 of
                # them genuine once the gates above see their full inputs, 17 of those
                # ALERT→CRITICAL — refinery strikes and airport closures that reached
                # the SITREP while nobody was paged.
                #
                # Volume is the suppression keys' problem, which is what they are for:
                # a claim already at or above the new tier mutes the card, so a storyline
                # that merely re-resolves its anchor across six reports pages once.
                # No dup_adjudicator is passed — Pass E is strictly NO LLM — which means
                # a fragmented storyline can still reach a second card; that is the
                # direction this pass is allowed to be wrong in.
                dispatch_result = None
                if tier_rank(new_tier) > tier_rank(current_tier):
                    event.update({
                        "severity_score": new_severity,
                        "system_confidence": new_system_conf,
                        "anchor_confidence": new_level,
                        "anchor_name_norm": new_norm,
                        "country_iso": country or event["country_iso"],
                        "alert_tier": new_tier,
                    })
                    dispatch_result = dispatch_alert(db_conn, event, event_id)
                    logger.info(
                        "Event %s escalated %s→%s on anchor upgrade: dispatch=%s",
                        event_id[:8], current_tier or "none", new_tier, dispatch_result,
                    )
                return True, True, dispatch_result

        # No upgrade — just mark as reconciled
        with db_conn.transaction():
            db_conn.execute(
                """UPDATE events
                   SET status = 'reconciled', updated_at = NOW()
                   WHERE id = %s""",
                (event_id,),
            )
        db_conn.commit()
        return True, False, None

    except Exception:
        try:
            db_conn.rollback()
        except Exception:
            pass
        logger.exception("Error reconciling event %s", event_id)
        return False, False, None


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
        # An upgrade that raised the tier, and what dispatch_alert did with it. Split
        # the same way Pass D splits its own: 'sent' is the only value that means a
        # card exists, and the suppressed/skipped counts are how the suppression
        # ledger is shown to be doing the collapsing rather than the pass staying quiet.
        "tier_escalations": 0,
        "escalation_dispatch": {},
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
            ok, upgraded, dispatch_result = reconcile_single_event(db_conn, str(row[0]))
            if ok:
                stats["events_reconciled"] += 1
                if upgraded:
                    stats["anchor_upgrades"] += 1
            else:
                stats["events_failed"] += 1
            if dispatch_result:
                stats["tier_escalations"] += 1
                stats["escalation_dispatch"][dispatch_result] = (
                    stats["escalation_dispatch"].get(dispatch_result, 0) + 1
                )

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
