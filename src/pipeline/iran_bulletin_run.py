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

from src.core import counters
from src.core.llm_router import (
    LLMRouter,
    build_bulletin_router,
    build_quality_router,
)
from src.pipeline.weekly_forecast import upload_report_to_r2
from src.services.iran_bulletin import (
    SECTION_FROM_IRAN,
    SECTION_ON_IRAN,
    SECTION_REGIONAL,
    SECTION_TITLES,
    STANDING_LABELS,
    THEATRE_NAMES,
    build_bulletin,
)
from src.services.sitrep_html import render_sitrep_html
from src.services.telegram_report_notifier import send_sitrep_telegram

logger = logging.getLogger(__name__)

WINDOW_HOURS = 24

# Named after the report it reproduces rather than after a theatre. "İran
# Tiyatrosu" read as jargon and told a reader nothing about what was inside.
REPORT_TITLE = "BÖLGESEL ASKERİ VE JEOPOLİTİK GELİŞMELER"
REPORT_SUBJECT = "İran, Körfez ve Doğu Akdeniz hattı"
REPORT_SUBJECT_SUFFIX = "12 ülke"


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
             "actor": ev.get("actor"), "target_country": ev.get("target_country"),
             "standing": ev.get("standing"), "outlet_count": ev.get("outlet_count"),
             "severity": ev.get("severity"), "domain": ev.get("domain"),
             # The heading the bullet was filed under, and how many filings it
             # stands for. Both were needed to audit the 11 Sep bulletin and
             # neither was saved, so the audit had to re-derive them.
             "place": ev.get("place"), "merged_filings": ev.get("merged_filings")}
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


# What each section contributes to an appendix row, so a reader scanning the full
# log can see the direction without re-reading the headline.
_SECTION_TAGS = {
    SECTION_ON_IRAN: "İran\u2019a yönelik",
    SECTION_FROM_IRAN: "İran\u2019dan",
    SECTION_REGIONAL: "Bölgesel",
}


def _clusters_for_render(result: Dict[str, Any]) -> list:
    """Adapt events to the record render_sitrep_html's appendix row reads.

    Every field that row reads has to be supplied, and the first version of this
    function supplied four of seven. The result shipped: 182 appendix rows drew a
    bold em-dash where `location` should have been, an empty meta line, an empty
    snippet and a source chip labelled "kaynak" — separator rules with nothing
    between them. The mapping is spelled out per field for that reason.
    """
    from src.core.sitrep_verify import (
        LABEL_MULTI, LABEL_SINGLE, is_independent_publisher, registrable_domain,
    )

    clusters = []
    for section in (SECTION_ON_IRAN, SECTION_FROM_IRAN, SECTION_REGIONAL):
        for ev in (result.get("sections") or {}).get(section, []):
            iso = ev.get("country_iso") or ""
            standing = STANDING_LABELS.get(ev.get("standing"), "")
            occurred = ev.get("occurred_at")
            # Every outlet that filed this story, the representative first. Since
            # the events are collapsed by storyline the siblings ARE the record of
            # how widely it was carried, and dropping them would trade twenty
            # repeated rows for one row that hides nineteen sources.
            siblings = ev.get("sibling_sources") or []
            outlets = [{"name": ev.get("domain"), "url": ev.get("url")}] + [
                {"name": s.get("name"), "url": s.get("url")} for s in siblings
            ]
            # Corroboration now has two ways to be true, and the second one only
            # became visible with the collapse: distinct independent publishers
            # carrying the same story is what "Çoklu kaynak" has always meant in
            # the SITREP. Carriers are excluded here for the reason they are
            # excluded there — a syndication feed republishing one wire is one
            # source, however many domains it wears.
            independent = {
                registrable_domain(s.get("name") or "")
                for s in outlets
                if s.get("name") and is_independent_publisher(s.get("name") or "")
            }
            corroborated = (len(ev.get("corroborating_sources") or []) > 0
                            or len(independent) > 1)
            clusters.append({
                # The bold line: where it landed, and which way it was going.
                "location": f"{THEATRE_NAMES.get(iso, iso)} · {_SECTION_TAGS[section]}",
                # The row's actual content. The headline IS the record here — the
                # bulletin has no separate summary per event.
                "snippet": ev.get("title") or "",
                # The grey meta line. Standing rides here rather than in the badge,
                # because the badge already carries corroboration and the two are
                # different claims: one is how many outlets, the other is whether
                # anybody stands behind it.
                # Standing plus, when the story was carried by more than one
                # outlet, how many — the number the collapsed row stands for.
                "date": (f"{standing} · {ev['outlet_count']} yayıncı"
                         if (ev.get("outlet_count") or 1) > 1 else standing),
                "event_type": ev.get("event_type") or "",
                "severity": ev.get("severity") or 0,
                "verification": LABEL_MULTI if corroborated else LABEL_SINGLE,
                # `name` is what the chip prints; without it every source read
                # "kaynak" and the publisher was invisible.
                # Capped: one story ran in 79 outlets on 9 Sep, and 79 chips is
                # not a record, it is a wall. The count above keeps the scale.
                "sources": outlets[:6],
                "occurred_at": occurred,
                # Splits the full log into the same three blocks as the
                # narrative; without it the bulletin's organising idea
                # survives only in the prose.
                "group": SECTION_TITLES[section],
            })
    return clusters


def run_iran_bulletin(db_conn, router: Optional[LLMRouter] = None,
                      window_hours: int = WINDOW_HOURS) -> Dict[str, Any]:
    """Entry point. Returns a slim result for the caller's exit status."""
    window_end = datetime.now(timezone.utc).replace(tzinfo=None)
    window_start = window_end - timedelta(hours=window_hours)

    # Two routers, because they are two different jobs.
    #
    # Direction extraction gets the slots measured for direction accuracy, in
    # small batches. The narrative gets the QUALITY cascade — it is prose, it
    # sends the whole day (11,064 tokens on 6 Sep), and Groq's per-request size
    # ceiling refuses that outright.
    #
    # The narrative used to ride the direction router, on the reasoning that the
    # quality slots were "already saturated by five country SITREPs at the same
    # hour (measured 3 Sep: Mistral 429 on the fifth)". Both halves of that have
    # expired: Mistral is gone from the cascade entirely, its replacement is a
    # paid slot with room, and the SITREP finishes at 07:32 against this run's
    # 08:00. What the old arrangement did instead was couple the two, so
    # narrowing the direction slots to one Groq model on 5 Sep also handed Groq's
    # size ceiling the narrative — and the 6 Sep bulletin did not publish at all.
    router = router or build_bulletin_router()
    narrative_router = build_quality_router()

    try:
        result = build_bulletin(db_conn, router, window_start, window_end,
                                narrative_router=narrative_router)
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
        country_name=REPORT_SUBJECT, country_iso="IR",
        window_start=str(window_start)[:16], window_end=str(window_end)[:16],
        report_text=result["narrative"], clusters=clusters,
        report_title=REPORT_TITLE, subject_suffix=REPORT_SUBJECT_SUFFIX,
    )

    stamp = window_end.strftime("%Y%m%d")
    r2_url = None
    try:
        # None when no public base is configured. The suppression used to live here,
        # as a string match on the invented host — and this path was written later
        # than daily_sitrep's and inherited the bug rather than the fix, so the
        # 4 Sep card carried pub-default.r2.dev/iran_bulletin_*. It lives in
        # upload_report_to_r2 now, once, for every caller.
        r2_url = upload_report_to_r2(f"iran_bulletin_{stamp}.html",
                                     html_doc.encode("utf-8"), "text/html")
    except Exception:
        logger.warning("Iran bulletin: R2 upload failed; dispatching anyway",
                       exc_info=True)

    try:
        send_sitrep_telegram(
            country_iso="IR", country_name=REPORT_SUBJECT,
            window_start=str(window_start)[:16], window_end=str(window_end)[:16],
            clusters=clusters, html_doc=html_doc, r2_url=r2_url,
            heading=REPORT_TITLE,
            # Not sitrep_IR_*: Iran gets its own country SITREP on the same
            # morning, into the same chat, and the second file to arrive would
            # overwrite the first on the reader's phone.
            filename_stem="iran_bulletin",
        )
    except Exception:
        logger.exception("Iran bulletin: Telegram dispatch failed")

    _save(db_conn, window_start, window_end, "completed", result, r2_url)
    taken = counters.snapshot()
    if taken:
        # Chiefly bulletin_direction_batch_failed, which fails OPEN: those events
        # keep the unattributed default and land in the regional section, so the
        # report looks entirely normal while having quietly stopped saying which
        # way anything was going. The number is the only trace.
        logger.warning("Iran bulletin degradation counters: %s", taken)
    sections = result["sections"]
    logger.info(
        "Iran bulletin completed: %d events (on Iran %d, from Iran %d, regional %d)",
        len(result["events"]), len(sections[SECTION_ON_IRAN]),
        len(sections[SECTION_FROM_IRAN]), len(sections[SECTION_REGIONAL]),
    )
    return {
        "success": True, "status": "completed",
        "degradation_counters": taken,
        "events": len(result["events"]),
        "sections": {SECTION_TITLES[k]: len(sections[k]) for k in SECTION_TITLES},
        "standings": {
            label: sum(1 for ev in result["events"]
                       if STANDING_LABELS.get(ev.get("standing")) == label)
            for label in STANDING_LABELS.values()
        },
        "r2_url": r2_url,
    }
