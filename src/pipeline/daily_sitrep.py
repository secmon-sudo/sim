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

from src.core.llm_router import LLMRouter
from src.pipeline.weekly_forecast import get_country_name, upload_report_to_r2
from src.services.sitrep_generator import (
    MAX_WEB_ENRICH_CLUSTERS,
    WINDOW_HOURS,
    build_sitrep_clusters,
    fetch_aviation_spillover_events,
    fetch_penalized_domains,
    fetch_sitrep_events,
    fetch_spillover_events,
    relabel_cluster,
    run_sitrep_llm,
    select_sitrep_countries,
    split_strategic,
    validate_sitrep,
)
from src.services.sitrep_digest import build_digest
from src.services.sitrep_digest_html import render_digest_html
from src.services.sitrep_html import render_sitrep_html
from src.services.sitrep_web_enrich import apply_web_enrichment, resolve_cluster_urls
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
    field, strategic = split_strategic(clusters)
    spillover_events = fetch_spillover_events(db_conn, country_iso, country_name,
                                              window_start, window_end)
    spillover = build_sitrep_clusters(spillover_events, penalized) if spillover_events else []

    # Regional aviation disruptions relevant to this country but attributed to
    # the region/neighbours (null or other country_iso). Rendered as its own
    # deterministic block so aviation — the priority domain — is never lost to
    # per-country attribution or to the LLM narrative dropping it.
    aviation_events = fetch_aviation_spillover_events(db_conn, country_iso, country_name,
                                                      window_start, window_end)
    aviation_spill = build_sitrep_clusters(aviation_events, penalized) if aviation_events else []

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

    # Optional Gemini Google-Search grounding: extra corroborated detail per top
    # cluster, discovery of incidents the ingest pipeline missed, and a strategic
    # sweep for BÖLÜM III. Labels are re-derived afterwards so newly found
    # independent domains upgrade single-source events.
    enrichment = apply_web_enrichment(field, country_name, MAX_WEB_ENRICH_CLUSTERS)
    strategic_web = enrichment["strategic"]
    field = field + enrichment["discovered"]
    clusters = clusters + enrichment["discovered"]
    for cluster in field:
        relabel_cluster(cluster, penalized)

    # Only genuinely quiet after BOTH the events table and web discovery came
    # back empty (discovery is a no-op without GEMINI_API_KEY).
    if not clusters and not strategic_web:
        logger.info("SITREP %s: no events and no web findings — saving empty report", country_iso)
        _save_sitrep(db_conn, country_iso, window_start, window_end,
                     status="empty", report_text=EMPTY_REPORT_TEXT, clusters=[])
        return {"country_iso": country_iso, "status": "empty", "event_count": 0}

    try:
        res = run_sitrep_llm(router, country_iso, country_name,
                             window_start, window_end, field, strategic, spillover,
                             strategic_web=strategic_web)
        allowed_urls = [
            s.get("url") for c in (clusters + spillover) for s in c["sources"] if s.get("url")
        ]
        if strategic_web:
            allowed_urls += [s.get("url") for s in strategic_web.get("sources", []) if s.get("url")]
        report_text = validate_sitrep(res["content"], allowed_urls)
    except Exception as e:
        logger.exception("SITREP %s: LLM generation failed", country_iso)
        _save_sitrep(db_conn, country_iso, window_start, window_end,
                     status="failed", report_text=None, clusters=clusters,
                     error_message=str(e)[:1000])
        return {"country_iso": country_iso, "status": "failed", "error": str(e)}

    # Delivery is best-effort; the report row is the source of truth.
    # Full cluster list (field + strategic + discovered) so the stat cards and
    # the appendix log cover the complete day, not just field events.
    html_doc = render_sitrep_html(
        country_name, country_iso,
        f"{window_start:%Y-%m-%d %H:%M}", f"{window_end:%Y-%m-%d %H:%M}",
        report_text, clusters, aviation_spill,
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
            "report_text": report_text, "clusters": clusters}


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
        digest = build_digest(router, results, ws, we)
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
    slim = [{k: v for k, v in r.items() if k not in ("report_text", "clusters")}
            for r in results]
    return {"success": failed == 0, "countries": slim, "completed": completed,
            "digest_r2_url": digest_r2_url}
