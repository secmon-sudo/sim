"""SIM — Iran theatre bulletin: the run.

Fetch the theatre window, attribute direction, narrate, render, save, dispatch.
Every piece it stands on already existed and is reused rather than reimplemented:
sitrep_html for the page, weekly_forecast for the R2 upload, telegram_report_notifier
for the push. The only new machinery is in src/services/iran_bulletin.py, which is
where the direction extraction and its measurements live.

Runs as its own workflow rather than its own repository. That was considered and
rejected: everything the bulletin reads lives in SIM's database, and everything it
uses to reason — the LLM router with its quota buckets and cooldowns, the Turkish
narration rules, the label vocabulary, the R2 and Telegram plumbing — lives in
SIM's modules. A second repository would fork all of it. The repo already carries
the cost of that mistake in source_credibility.py, whose docstring opens "There
were two of these" and goes on to record that the smaller of the two silently
governed the trend for months. A separate workflow buys the isolation that was
actually wanted (its own cadence, its own chat, its own failure, a clean deletion
when the war ends) without buying the fork.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from src.core.llm_router import LLMRouter, build_bulletin_router
from src.pipeline.weekly_forecast import upload_report_to_r2
from src.services.iran_bulletin import (
    SECTION_FROM_IRAN,
    SECTION_ON_IRAN,
    SECTION_REGIONAL,
    SECTION_TITLES,
    STANDING_LABELS,
    build_bulletin,
)
from src.services.sitrep_html import render_sitrep_html
from src.services.telegram_report_notifier import send_sitrep_telegram

logger = logging.getLogger(__name__)

WINDOW_HOURS = 24


def _save(db_conn, window_start, window_end, status: str,
          result: Optional[Dict[str, Any]] = None, r2_url: Optional[str] = None,
          error: Optional[str] = None) -> None:
    """Persist the bulletin. Never raises — a storage failure must not lose a
    report that has already been dispatched."""
    result = result or {}
    sections = result.get("sections") or {}
    summary = {
        key: [
            {"title": ev.get("title"), "country_iso": ev.get("country_iso"),
             "actor": ev.get("actor"), "standing": ev.get("standing"),
             "severity": ev.get("severity"), "domain": ev.get("domain")}
            for ev in sections.get(key, [])
        ]
        for key in (SECTION_ON_IRAN, SECTION_FROM_IRAN, SECTION_REGIONAL)
    }
    try:
        import json
        with db_conn.transaction():
            db_conn.execute(
                """INSERT INTO iran_bulletins (window_start, window_end, report_text,
                                               sections_json, event_count, status,
                                               llm_model, r2_url, error_message)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (window_start, window_end, result.get("narrative"),
                 json.dumps(summary, ensure_ascii=False),
                 len(result.get("events") or []), status,
                 result.get("model"), r2_url, error),
            )
    except Exception:
        logger.exception("Iran bulletin: could not persist the run")


def _clusters_for_render(result: Dict[str, Any]) -> list:
    """Adapt events to the shape render_sitrep_html's stat cards and highlights
    read. The bulletin's standing vocabulary is mapped onto the SITREP's
    verification labels so the two reports count the same things the same way.
    """
    from src.core.sitrep_verify import LABEL_MULTI, LABEL_SINGLE

    clusters = []
    for section in (SECTION_ON_IRAN, SECTION_FROM_IRAN, SECTION_REGIONAL):
        for ev in (result.get("sections") or {}).get(section, []):
            corroborated = len(ev.get("corroborating_sources") or []) > 0
            clusters.append({
                "headline": ev.get("title"),
                "severity": ev.get("severity") or 0,
                "verification": LABEL_MULTI if corroborated else LABEL_SINGLE,
                "sources": [{"domain": ev.get("domain"), "url": ev.get("url")}],
            })
    return clusters


def run_iran_bulletin(db_conn, router: Optional[LLMRouter] = None,
                      window_hours: int = WINDOW_HOURS) -> Dict[str, Any]:
    """Entry point. Returns a slim result for the caller's exit status."""
    window_end = datetime.now(timezone.utc).replace(tzinfo=None)
    window_start = window_end - timedelta(hours=window_hours)

    # Direction extraction has its own measured router; the narrative rides the
    # same one rather than the quality cascade, because the bulletin's prose is a
    # structured account of supplied facts, not the free narration the quality
    # slots exist for — and because those slots are already saturated by five
    # country SITREPs at the same hour (measured 3 Sep: Mistral 429 on the fifth).
    router = router or build_bulletin_router()

    try:
        result = build_bulletin(db_conn, router, window_start, window_end)
    except Exception as exc:
        logger.exception("Iran bulletin failed")
        _save(db_conn, window_start, window_end, "failed", error=str(exc)[:500])
        return {"success": False, "error": str(exc)}

    if result["status"] == "empty":
        _save(db_conn, window_start, window_end, "empty", result)
        logger.info("Iran bulletin: nothing in the window, nothing dispatched")
        return {"success": True, "status": "empty", "events": 0}

    clusters = _clusters_for_render(result)
    html_doc = render_sitrep_html(
        country_name="İran Tiyatrosu", country_iso="IR",
        window_start=str(window_start)[:16], window_end=str(window_end)[:16],
        report_text=result["narrative"], clusters=clusters,
    )

    stamp = window_end.strftime("%Y%m%d")
    r2_url = None
    try:
        r2_url = upload_report_to_r2(f"iran_bulletin_{stamp}.html",
                                     html_doc.encode("utf-8"), "text/html")
    except Exception:
        logger.warning("Iran bulletin: R2 upload failed; dispatching anyway",
                       exc_info=True)

    try:
        send_sitrep_telegram(
            country_iso="IR", country_name="İran Tiyatrosu Bülteni",
            window_start=str(window_start)[:16], window_end=str(window_end)[:16],
            clusters=clusters, html_doc=html_doc, r2_url=r2_url,
        )
    except Exception:
        logger.exception("Iran bulletin: Telegram dispatch failed")

    _save(db_conn, window_start, window_end, "completed", result, r2_url)
    sections = result["sections"]
    logger.info(
        "Iran bulletin completed: %d events (on Iran %d, from Iran %d, regional %d)",
        len(result["events"]), len(sections[SECTION_ON_IRAN]),
        len(sections[SECTION_FROM_IRAN]), len(sections[SECTION_REGIONAL]),
    )
    return {
        "success": True, "status": "completed",
        "events": len(result["events"]),
        "sections": {SECTION_TITLES[k]: len(sections[k]) for k in SECTION_TITLES},
        "standings": {
            label: sum(1 for ev in result["events"]
                       if STANDING_LABELS.get(ev.get("standing")) == label)
            for label in STANDING_LABELS.values()
        },
        "r2_url": r2_url,
    }
