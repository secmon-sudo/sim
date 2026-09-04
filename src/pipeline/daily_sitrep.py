"""
SIM — Daily Country SITREP Pipeline
24-hour Turkish situation reports per country. Runs daily via GitHub Actions
or on demand: `python -m src.pipeline.orchestrator --sitrep [IR IQ ...]`.

Fail-soft per country: one failing country never kills the run.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.core.airspace import build_airspace_assessment
from src.core.llm_client import log_llm_telemetry
from src.core import counters
from src.core.llm_router import LLMRouter
from src.pipeline.weekly_forecast import get_country_name, upload_report_to_r2
from src.services.czib_client import fetch_active_czib_by_country
from src.services.sitrep_generator import (
    NARRATIVE_MAX_TOKENS,
    TRUNCATION_NOTICE,
    WINDOW_HOURS,
    build_sitrep_clusters,
    cap_for_prompt,
    drop_safety_clusters,
    fetch_aviation_spillover_events,
    fetch_penalized_domains,
    fetch_sitrep_events,
    fetch_spillover_events,
    is_truncated,
    relabel_cluster,
    run_sitrep_llm,
    select_sitrep_countries,
    split_strategic,
    validate_sitrep,
)
from src.services.sitrep_digest import build_digest
from src.services.sitrep_digest_html import render_digest_html
from src.services.sitrep_html import render_sitrep_html
from src.services.google_news_resolver import resolve_cluster_urls
from src.services.telegram_report_notifier import send_digest_telegram, send_sitrep_telegram

logger = logging.getLogger(__name__)

EMPTY_REPORT_TEXT = (
    "BÖLÜM I — SAHA OLAYLARI\n"
    "Son 24 saatte kayda değer, puanlanmış bir güvenlik olayı tespit edilmedi.\n"
)


def _save_sitrep(db_conn, country_iso: str, window_start, window_end,
                 status: str, report_text: Optional[str], clusters: List[Dict[str, Any]],
                 llm_provider: Optional[str] = None, llm_model: Optional[str] = None,
                 r2_url: Optional[str] = None, error_message: Optional[str] = None) -> None:
    db_conn.execute(
        """
        INSERT INTO sitreps (country_iso, window_start, window_end, report_text,
                             events_json, event_count, status, llm_provider,
                             llm_model, r2_url, error_message)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (country_iso, window_start, window_end, report_text,
         json.dumps(clusters, ensure_ascii=False, default=str), len(clusters),
         status, llm_provider, llm_model, r2_url, error_message),
    )
    db_conn.commit()


def run_country_sitrep(db_conn, router: LLMRouter, country_iso: str,
                       window_end: Optional[datetime] = None) -> Dict[str, Any]:
    """Generate, persist, and dispatch the SITREP for one country."""
    window_end = window_end or datetime.now(timezone.utc).replace(tzinfo=None)
    window_start = window_end - timedelta(hours=WINDOW_HOURS)
    country_iso = country_iso.strip().upper()
    country_name = get_country_name(db_conn, country_iso)

    logger.info("SITREP %s (%s): window %s — %s", country_iso, country_name, window_start, window_end)

    events = fetch_sitrep_events(db_conn, country_iso, window_start, window_end)
    penalized = fetch_penalized_domains(db_conn)
    clusters = build_sitrep_clusters(events, penalized)
    # The narrative covers SECURITY events; technical/safety occurrences (a
    # diverted flight, a bird strike) are excluded by the prompt's own rule and
    # the model still wrote them up as report bullets on 1 Aug. Enforce it here
    # instead of asking again. They stay in `clusters`, so the appendix log and
    # the stat cards remain a complete record of the day.
    #
    # `clusters` is the full ranked day; only the prompt is capped (cap_for_prompt).
    # Safety clusters are dropped BEFORE the cap so they no longer spend narrative
    # slots the reader will never see them in.
    narrated = cap_for_prompt(drop_safety_clusters(clusters))
    if len(clusters) > len(narrated):
        logger.info(
            "SITREP %s: %d clusters in the record, top %d narrated (prompt cap)",
            country_iso, len(clusters), len(narrated),
        )
    field, strategic = split_strategic(narrated)
    spillover_events = fetch_spillover_events(db_conn, country_iso, country_name,
                                              window_start, window_end)
    # Spillover rides in the same prompt, so it is capped too.
    spillover = cap_for_prompt(
        build_sitrep_clusters(spillover_events, penalized)) if spillover_events else []

    # Regional aviation disruptions relevant to this country but attributed to
    # the region/neighbours (null or other country_iso). Rendered as its own
    # deterministic block so aviation — the priority domain — is never lost to
    # per-country attribution or to the LLM narrative dropping it.
    aviation_events = fetch_aviation_spillover_events(db_conn, country_iso, country_name,
                                                      window_start, window_end)
    aviation_spill = build_sitrep_clusters(aviation_events, penalized) if aviation_events else []

    # Deterministic airspace exposure: which FIR each kinetic/aviation event sits
    # in, which neighbouring airspaces carry an active EASA CZIB restriction, and
    # which commercial airports are nearby. Pure geometry over a static
    # reference, so the narrative can be handed it as fact — and it is rendered
    # separately below, so a narrative that ignores it cannot lose it.
    try:
        czib_by_iso = fetch_active_czib_by_country(db_conn)
        airspace = build_airspace_assessment(clusters + aviation_spill,
                                             country_iso, czib_by_iso)
    except Exception:
        logger.exception("SITREP %s: airspace assessment failed", country_iso)
        airspace = None

    # Replace Google News redirect links with the real publisher URLs so the
    # report's sources are directly usable.
    resolve_cluster_urls(clusters + spillover + aviation_spill)

    # Drop any aviation cluster already covered by the country's own clusters
    # (all its resolved source URLs already appear there) so nothing shows twice.
    if aviation_spill:
        _main_urls = {s.get("url") for c in clusters for s in c.get("sources", []) if s.get("url")}

        def _already_shown(c: Dict[str, Any]) -> bool:
            urls = [s.get("url") for s in c.get("sources", []) if s.get("url")]
            return bool(urls) and all(u in _main_urls for u in urls)

        aviation_spill = [c for c in aviation_spill if not _already_shown(c)]

    # Verification labels are re-derived after URL resolution: a resolved link
    # reveals the real publisher domain, which is what label_cluster counts.
    # Over the whole record, not just the narrated slice: `clusters` is what the
    # appendix, the stat cards and the digest's confirmed-severe count read, and
    # those beyond the prompt cap would otherwise keep their pre-resolution labels.
    for cluster in clusters:
        relabel_cluster(cluster, penalized)

    if not clusters:
        logger.info("SITREP %s: no events in window — saving empty report", country_iso)
        _save_sitrep(db_conn, country_iso, window_start, window_end,
                     status="empty", report_text=EMPTY_REPORT_TEXT, clusters=[])
        return {"country_iso": country_iso, "status": "empty", "event_count": 0}

    # Built before the call, not after: it is both the allowlist validate_sitrep
    # enforces and the contract run_sitrep_llm holds each model to.
    allowed_urls = [
        s.get("url") for c in (clusters + spillover) for s in c["sources"] if s.get("url")
    ]

    try:
        res = run_sitrep_llm(router, country_iso, country_name,
                             window_start, window_end, field, strategic, spillover,
                             airspace=airspace, allowed_urls=allowed_urls)
        log_llm_telemetry(db_conn, res, router, success=True,
                          purpose="sitrep_country")
        report_text = validate_sitrep(res["content"], allowed_urls)
        # A completion cut off at max_tokens still passes every guardrail above —
        # the header is there, the URLs are allowlisted — so it used to ship as a
        # normal report. Mark it instead of pretending it finished.
        if is_truncated(res):
            logger.warning(
                "SITREP %s: narrative hit the %d-token ceiling and was cut off "
                "(%d chars, %d clusters) — shipping it marked; raise "
                "sitrep.narrative_max_tokens if this keeps happening",
                country_iso, NARRATIVE_MAX_TOKENS, len(report_text), len(field),
            )
            report_text += TRUNCATION_NOTICE
    except Exception as e:
        logger.exception("SITREP %s: LLM generation failed", country_iso)
        _save_sitrep(db_conn, country_iso, window_start, window_end,
                     status="failed", report_text=None, clusters=clusters,
                     error_message=str(e)[:1000])
        return {"country_iso": country_iso, "status": "failed", "error": str(e)}

    # Delivery is best-effort; the report row is the source of truth.
    # `clusters`, not `narrated`: the stat cards and the appendix log cover the
    # complete day — every cluster, field and strategic, above and below the prompt
    # cap. This comment used to claim as much while the list it named was itself
    # capped at 25 (fixed 2026-08-10, see cap_for_prompt).
    html_doc = render_sitrep_html(
        country_name, country_iso,
        f"{window_start:%Y-%m-%d %H:%M}", f"{window_end:%Y-%m-%d %H:%M}",
        report_text, clusters, aviation_spill, airspace,
    )
    r2_url = None
    try:
        filename = f"sitrep_{country_iso}_{window_end:%Y%m%d}.html"
        r2_url = upload_report_to_r2(filename, html_doc.encode("utf-8"), "text/html")
        # upload_report_to_r2 falls back to a placeholder public base when
        # R2_PUBLIC_URL_BASE is unset — that URL doesn't exist (SSL error in
        # Telegram), so suppress the link rather than publish a dead one.
        if r2_url and "pub-default.r2.dev" in r2_url:
            logger.warning("SITREP %s: R2_PUBLIC_URL_BASE not configured; omitting R2 link", country_iso)
            r2_url = None
    except Exception:
        logger.exception("SITREP %s: R2 upload failed", country_iso)

    _save_sitrep(db_conn, country_iso, window_start, window_end,
                 status="completed", report_text=report_text, clusters=clusters,
                 llm_provider=res.get("provider"), llm_model=res.get("model"),
                 r2_url=r2_url)

    try:
        send_sitrep_telegram(
            country_iso=country_iso,
            country_name=country_name,
            window_start=f"{window_start:%Y-%m-%d %H:%M}",
            window_end=f"{window_end:%Y-%m-%d %H:%M}",
            clusters=clusters,
            html_doc=html_doc,
            r2_url=r2_url,
        )
    except Exception:
        logger.exception("SITREP %s: Telegram dispatch failed", country_iso)

    logger.info("SITREP %s: completed (%d clusters, model=%s)",
                country_iso, len(clusters), res.get("model"))
    # report_text/clusters ride along for the run-level digest; run_daily_sitrep
    # strips them before returning so the pipeline result stays small.
    return {"country_iso": country_iso, "country_name": country_name,
            "status": "completed", "event_count": len(events),
            "cluster_count": len(clusters), "r2_url": r2_url,
            "report_text": report_text, "clusters": clusters,
            "airspace": airspace}


def _save_digest(db_conn, window_start, window_end, status: str,
                 digest: Optional[Dict[str, Any]] = None,
                 r2_url: Optional[str] = None,
                 error_message: Optional[str] = None) -> None:
    db_conn.execute(
        """
        INSERT INTO sitrep_digests (window_start, window_end, country_isos,
                                    digest_text, digest_json, status,
                                    llm_provider, llm_model, r2_url, error_message)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (window_start, window_end,
         (digest or {}).get("country_isos"),
         (digest or {}).get("raw_text"),
         json.dumps(digest, ensure_ascii=False, default=str) if digest else None,
         status,
         (digest or {}).get("provider"), (digest or {}).get("model"),
         r2_url, error_message),
    )
    db_conn.commit()


def run_digest(db_conn, router: LLMRouter, results: List[Dict[str, Any]],
               window_start: datetime, window_end: datetime) -> Optional[str]:
    """
    Run-level executive briefing: one short cross-country synthesis of the
    country SITREPs of this run. Fail-soft — the country reports are already
    delivered, so a digest failure never fails the run.
    """
    ws = f"{window_start:%Y-%m-%d %H:%M}"
    we = f"{window_end:%Y-%m-%d %H:%M}"
    try:
        digest = build_digest(router, results, ws, we, db_conn=db_conn)
    except Exception as e:
        logger.exception("Digest generation failed")
        try:
            _save_digest(db_conn, window_start, window_end, status="failed",
                         error_message=str(e)[:1000])
        except Exception:
            logger.exception("Digest failure row could not be saved")
        return None

    if digest is None:
        return None

    html_doc = render_digest_html(digest, ws, we)

    r2_url = None
    try:
        r2_url = upload_report_to_r2(f"brifing_{window_end:%Y%m%d}.html",
                                     html_doc.encode("utf-8"), "text/html")
        if r2_url and "pub-default.r2.dev" in r2_url:
            r2_url = None
    except Exception:
        logger.exception("Digest R2 upload failed")

    try:
        _save_digest(db_conn, window_start, window_end, status="completed",
                     digest=digest, r2_url=r2_url)
    except Exception:
        logger.exception("Digest row could not be saved")

    try:
        send_digest_telegram(digest, ws, we, html_doc)
    except Exception:
        logger.exception("Digest Telegram dispatch failed")

    logger.info("Digest completed (%d countries, model=%s)",
                len(digest.get("country_isos") or []), digest.get("model"))
    return r2_url


def run_daily_sitrep(db_conn, router: LLMRouter,
                     countries: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Entry point. With explicit `countries` (ISO2 list) runs on demand; otherwise
    auto-selects the highest-activity countries of the last 24h.
    """
    window_end = datetime.now(timezone.utc).replace(tzinfo=None)
    window_start = window_end - timedelta(hours=WINDOW_HOURS)

    if not countries:
        countries = select_sitrep_countries(db_conn, window_start, window_end)
        logger.info("SITREP auto-selection: %s", countries or "none above threshold")

    results = []
    for iso in countries or []:
        try:
            results.append(run_country_sitrep(db_conn, router, iso, window_end=window_end))
        except Exception as e:
            logger.exception("SITREP run failed hard for %s", iso)
            results.append({"country_iso": iso, "status": "failed", "error": str(e)})

    digest_r2_url = run_digest(db_conn, router, results, window_start, window_end)

    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    slim = [{k: v for k, v in r.items()
             if k not in ("report_text", "clusters", "airspace")}
            for r in results]

    # A SITREP run's own degradation record, written whether or not anything
    # failed. On 2026-09-04 this run reported five completed countries and five
    # unusable reports; a stage count cannot tell those apart, and this row is
    # where the difference now shows. Best-effort — the reports are already
    # delivered and a telemetry write must not undo that.
    taken = counters.snapshot()
    if taken:
        logger.warning("SITREP run degradation counters: %s", taken)
    try:
        db_conn.execute(
            "INSERT INTO system_telemetry(event_type, value_json) VALUES (%s, %s)",
            ("sitrep_run", json.dumps(
                {"completed": completed, "failed": failed,
                 "countries": [r.get("country_iso") for r in results],
                 "degradation_counters": taken}, default=str)),
        )
        db_conn.commit()
    except Exception:
        logger.exception("SITREP: could not log run telemetry")

    return {"success": failed == 0, "countries": slim, "completed": completed,
            "digest_r2_url": digest_r2_url, "degradation_counters": taken}
