"""
SIM — Storyline Narrator (budgeted LLM "story so far")
Blueprint V20.1 §PASS D / Storyline

Generates a short prose narrative ("story so far") for ACTIVE, high-severity
storylines on top of the zero-LLM structural summary (core.storyline_narrative).

Token discipline:
  - Only storylines with >= min_events and peak severity >= min_severity.
  - A content `signature` (event ids + latest time) is cached; unchanged storylines
    are skipped (no LLM call).
  - Runs on the bulk router (gpt-oss-20b, pooled across Groq keys A+B for ~2K RPD) so
    it never competes with Pass C for smart-model quota.
  - Capped at max_per_run generations per pipeline run.
"""

import hashlib
import json
import logging
from pathlib import Path

from src.core.llm_client import call_llm
from src.core.storyline_narrative import build_timeline, summarize_timeline

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
try:
    with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as _f:
        _NARRATIVE = json.load(_f).get("narrative", {})
except (OSError, json.JSONDecodeError):
    _NARRATIVE = {}

NARRATIVE_ENABLED = _NARRATIVE.get("enabled", True)
NARRATIVE_MIN_SEVERITY = _NARRATIVE.get("min_severity", 65)
NARRATIVE_MIN_EVENTS = _NARRATIVE.get("min_events", 2)
NARRATIVE_MAX_PER_RUN = _NARRATIVE.get("max_per_run", 10)
NARRATIVE_LOOKBACK_DAYS = _NARRATIVE.get("lookback_days", 14)

NARRATIVE_SYSTEM_PROMPT = (
    "You are an intelligence analyst writing a concise factual brief. Given a "
    "chronological list of related security incident reports that form one storyline, "
    "write a 2-4 sentence 'story so far' summary: how it started, how it developed, and "
    "the current situation. Be factual and neutral, name locations and actors, and do "
    "NOT invent details beyond what is given. Output plain prose only — no markdown, no "
    "preamble."
)


def compute_signature(events: list[dict]) -> str:
    """Stable content fingerprint of a storyline — changes only when events change."""
    ids = sorted(str(e.get("id")) for e in events)
    latest = max((str(e.get("occurred_at_est")) for e in events if e.get("occurred_at_est")), default="")
    raw = "|".join(ids) + "#" + latest
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_narrative_prompt(events: list[dict]) -> str:
    """Compact prompt: structural summary + the chronological event headlines."""
    summary = summarize_timeline(events)
    timeline = build_timeline(events)

    lines = []
    for e in timeline:
        when = e.get("occurred_at_est")
        when_s = when.strftime("%Y-%m-%d %H:%M") if hasattr(when, "strftime") else "unknown"
        title = (e.get("source_title") or e.get("storyline_hint") or "").strip()[:160]
        sev = e.get("severity_score", 0)
        lines.append(f"- [{when_s}] (sev {sev}) {title}")

    countries = ", ".join(summary["countries"]) or "unknown"
    anchors = ", ".join(summary["anchors"]) or "n/a"
    return (
        f"Storyline facts: {summary['event_count']} reports from "
        f"{summary['source_count']} sources over {summary['duration_hours']}h; "
        f"countries: {countries}; locations: {anchors}; "
        f"peak severity {summary['peak_severity']}; trend: {summary['severity_trend']}.\n\n"
        f"Chronological reports:\n" + "\n".join(lines)
    )


def fetch_active_storylines(db_conn) -> list[dict]:
    """Storylines worth narrating: recent, multi-event, high peak severity."""
    rows = db_conn.execute(
        """SELECT storyline_id,
                  COUNT(*) AS event_count,
                  MAX(severity_score) AS peak_severity
           FROM events
           WHERE storyline_id IS NOT NULL
             AND status IN ('scored', 'reconciled')
             AND occurred_at_est > NOW() - (%s * INTERVAL '1 day')
           GROUP BY storyline_id
           HAVING COUNT(*) >= %s AND MAX(severity_score) >= %s
           ORDER BY MAX(severity_score) DESC, COUNT(*) DESC
           LIMIT %s""",
        (NARRATIVE_LOOKBACK_DAYS, NARRATIVE_MIN_EVENTS, NARRATIVE_MIN_SEVERITY, NARRATIVE_MAX_PER_RUN),
    ).fetchall()
    return [{"storyline_id": str(r[0]), "event_count": r[1], "peak_severity": r[2]} for r in rows]


def fetch_storyline_events(db_conn, storyline_id: str) -> list[dict]:
    rows = db_conn.execute(
        """SELECT id, source_title, storyline_hint, occurred_at_est, severity_score,
                  source_domain, country_iso, anchor_name_norm, event_type
           FROM events
           WHERE storyline_id = %s
           ORDER BY occurred_at_est ASC""",
        (storyline_id,),
    ).fetchall()
    return [
        {
            "id": str(r[0]), "source_title": r[1], "storyline_hint": r[2],
            "occurred_at_est": r[3], "severity_score": r[4] or 0,
            "source_domain": r[5], "country_iso": r[6], "anchor_name_norm": r[7],
            "event_type": r[8],
        }
        for r in rows
    ]


def get_cached_signature(db_conn, storyline_id: str) -> str | None:
    row = db_conn.execute(
        "SELECT signature FROM storyline_narratives WHERE storyline_id = %s",
        (storyline_id,),
    ).fetchone()
    return row[0] if row else None


def upsert_narrative(db_conn, storyline_id: str, narrative: str, summary: dict,
                     signature: str, result: dict) -> None:
    db_conn.execute(
        """INSERT INTO storyline_narratives
               (storyline_id, narrative, summary_json, signature, event_count,
                peak_severity, severity_trend, llm_provider, llm_model, updated_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
           ON CONFLICT (storyline_id) DO UPDATE SET
               narrative = EXCLUDED.narrative,
               summary_json = EXCLUDED.summary_json,
               signature = EXCLUDED.signature,
               event_count = EXCLUDED.event_count,
               peak_severity = EXCLUDED.peak_severity,
               severity_trend = EXCLUDED.severity_trend,
               llm_provider = EXCLUDED.llm_provider,
               llm_model = EXCLUDED.llm_model,
               updated_at = NOW()""",
        (
            storyline_id, narrative, json.dumps(summary, default=str), signature,
            summary.get("event_count", 0), summary.get("peak_severity", 0),
            summary.get("severity_trend"), result.get("provider"), result.get("model"),
        ),
    )


def run_storyline_narratives(db_conn, router) -> dict:
    """Generate/refresh "story so far" prose for active high-severity storylines.

    Cache-aware (skips unchanged storylines) and capped per run. Never raises into
    the pipeline — failures are logged and counted.
    """
    stats = {"candidates": 0, "generated": 0, "skipped_cached": 0, "failed": 0}
    if not NARRATIVE_ENABLED:
        return stats

    try:
        candidates = fetch_active_storylines(db_conn)
    except Exception:
        logger.exception("Narrator: failed to fetch active storylines")
        return stats
    stats["candidates"] = len(candidates)

    for cand in candidates:
        sid = cand["storyline_id"]
        try:
            events = fetch_storyline_events(db_conn, sid)
            if len(events) < NARRATIVE_MIN_EVENTS:
                continue

            signature = compute_signature(events)
            if get_cached_signature(db_conn, sid) == signature:
                stats["skipped_cached"] += 1
                continue

            result = call_llm(
                router,
                prompt=build_narrative_prompt(events),
                system_prompt=NARRATIVE_SYSTEM_PROMPT,
                max_tokens=400,
            )
            narrative = (result.get("content") or "").strip()
            if not narrative:
                stats["failed"] += 1
                continue

            summary = summarize_timeline(events)
            upsert_narrative(db_conn, sid, narrative, summary, signature, result)
            db_conn.commit()
            stats["generated"] += 1
            logger.info("Narrator: storyline %s narrated (%d events)", sid[:8], len(events))

        except RuntimeError as e:
            logger.warning("Narrator: LLM exhausted, stopping: %s", e)
            break
        except Exception:
            db_conn.rollback()
            logger.exception("Narrator: failed for storyline %s", sid[:8])
            stats["failed"] += 1

    logger.info("Narrator complete: %s", stats)
    return stats
