"""SIM — did the reports we shipped actually say anything?

The dead-man's switch answers "did the pipeline run". Every LLM incident this
project has had answered YES to that question and was still a failure:

  * 2026-09-04 — five country SITREPs, all "completed", all five narrated by a
    model that shortened every citation URL to a bare domain. 108 of 108 links
    blanked. Not one report contained a working source, and nothing anywhere
    said so; it was found by a person reading a report.
  * 2026-09-04 — the Iran bulletin printed "us_coalition" at the reader eight
    times, in Turkish prose, in a delivered report.
  * 2026-07-23 — gemini-2.5-flash-lite had been retired early and every grounded
    call answered 404 for a fortnight. The aviation block was empty that whole
    time and the cause was miscredited to a quota.
  * 2026-08-10 — two weeks of SITREPs, a quarter of them ending mid-sentence at
    the token ceiling.

Each of those was visible in the database on the morning it happened, in one
query. Nobody ran the query. This module is the queries, with thresholds, so the
dead-man can page on a report that arrived and was hollow — not only on one that
never arrived.

Every check is deliberately about SHAPE, never about content quality. "Did any
citation survive", "did the narrative stop mid-sentence", "is a field name
showing through the prose", "did a different model write today's reports than
wrote last week's". A judgement about whether the analysis is any GOOD is a human
being's job and always will be; noticing that there is no analysis at all is not.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# How far back "today's reports" reaches. One SITREP cycle is daily, so a window
# a little over a day catches the latest run without dragging in the previous
# one when a run slips by an hour.
DEFAULT_WINDOW_HOURS = 30.0

# The trailing period a check compares against when it needs a baseline. Seven
# days is long enough that one bad day cannot move it and short enough that a
# deliberate change (a new model, a new prompt) stops looking like an anomaly
# within a week.
BASELINE_DAYS = 7


class Finding:
    """One thing that is wrong, in the words the ops channel will show."""

    def __init__(self, key: str, message: str, detail: str = ""):
        self.key = key
        self.message = message
        self.detail = detail

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Finding({self.key!r}, {self.message!r})"

    def render(self) -> str:
        return f"• {self.message}" + (f"\n  {self.detail}" if self.detail else "")


def _rows(conn, sql: str, params: tuple = ()) -> List[Tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def check_sitrep_citations(conn, window_hours: float) -> List[Finding]:
    """A completed SITREP with no surviving source link.

    This is the 2026-09-04 collapse, stated as a query. The bar is ZERO links,
    not "few" — six models over the preceding three weeks averaged 0.3 blanked
    citations per report and never once produced a report with none, so a report
    at zero is not a bad day, it is a different failure.
    """
    rows = _rows(conn, """
        SELECT country_iso, llm_model
          FROM sitreps
         WHERE status = 'completed'
           AND window_end > now() - (%s * interval '1 hour')
           AND report_text NOT LIKE %s
    """, (window_hours, "%https://%"))
    if not rows:
        return []
    listed = ", ".join(f"{iso.strip()} ({model or 'model?'})" for iso, model in rows)
    return [Finding(
        "sitrep_no_citations",
        f"{len(rows)} SITREP(s) shipped with NO working source link",
        listed,
    )]


def check_sitrep_truncation(conn, window_hours: float) -> List[Finding]:
    """Narratives cut off at the token ceiling.

    run_country_sitrep appends TRUNCATION_NOTICE when finish_reason says length,
    so the evidence is already in the text — it was simply never alarmed on.
    """
    rows = _rows(conn, """
        SELECT country_iso
          FROM sitreps
         WHERE status = 'completed'
           AND window_end > now() - (%s * interval '1 hour')
           AND report_text LIKE %s
    """, (window_hours, "%uzunluk sınırına takıldığı%"))
    if not rows:
        return []
    return [Finding(
        "sitrep_truncated",
        f"{len(rows)} SITREP(s) hit the token ceiling and were cut off",
        ", ".join(r[0].strip() for r in rows),
    )]


def check_narrator_changed(conn, window_hours: float,
                           baseline_days: int = BASELINE_DAYS,
                           min_baseline_reports: int = 5) -> List[Finding]:
    """Today's reports were written by a model that was not writing them before.

    Not an error on its own — the cascade is SUPPOSED to fall through, and a
    deliberate model change looks identical. It is a notice, and it earns its
    place because on 4 Sep it was the one fact that explained everything else:
    the primary slot 429'd every call and a fallback wrote the whole day. Seeing
    that in the morning would have turned a day of forensics into a glance.

    Two things keep it from crying wolf, both learned by writing it wrong first.
    It compares only the LATEST run — a 30-hour window spans two SITREP days, and
    against that the second day always looks new. And it abstains entirely until
    the baseline holds `min_baseline_reports`: on a table five days deep the
    first version flagged mistral-medium and laguna, the two most ordinary slots
    in the cascade, because there was nothing behind them to compare against. A
    check with no baseline has no finding, and saying nothing is the correct
    output for "I cannot tell yet".
    """
    latest = _rows(conn, """
        SELECT max(window_end) FROM sitreps
         WHERE status = 'completed'
           AND window_end > now() - (%s * interval '1 hour')
    """, (window_hours,))
    if not latest or latest[0][0] is None:
        return []
    run_at = latest[0][0]

    baseline = _rows(conn, """
        SELECT llm_model, count(*) FROM sitreps
         WHERE status = 'completed'
           AND window_end < %s - interval '6 hours'
           AND window_end > %s - (%s * interval '1 day')
           AND llm_model IS NOT NULL
         GROUP BY llm_model
    """, (run_at, run_at, baseline_days))
    if sum(n for _model, n in baseline) < min_baseline_reports:
        logger.info("Narrator-change check abstaining: baseline holds %d report(s)",
                    sum(n for _m, n in baseline))
        return []
    known = {model for model, _n in baseline}

    recent = _rows(conn, """
        SELECT DISTINCT llm_model FROM sitreps
         WHERE status = 'completed'
           AND window_end >= %s - interval '6 hours'
           AND llm_model IS NOT NULL
    """, (run_at,))
    fresh = sorted({m for (m,) in recent} - known)
    if not fresh:
        return []
    return [Finding(
        "narrator_changed",
        f"A model that has not narrated in the last {baseline_days} days wrote "
        "the newest SITREP(s)",
        ", ".join(fresh) + f" (usual: {', '.join(sorted(known))})",
    )]


def check_bulletin_attribution(conn, window_hours: float,
                               max_unattributed: float = 0.50) -> List[Finding]:
    """The bulletin stopped being able to say which way anything was going.

    Direction extraction fails OPEN: a batch that errors leaves its events
    unattributed, and unattributed events land in the regional section. The
    report renders perfectly. The only visible symptom is that the proportion of
    events with no actor climbs, so that proportion is the check.

    The ceiling is set from the record, not from taste: measured across every
    bulletin the report has produced, the unattributed share is 13.2%, 13.7% and
    19.7%. Half is two and a half times the worst of those, so this fires when
    most of the extraction has stopped working and not when a day is merely
    ambiguous. The precise signal is the counter — a failed batch increments
    BULLETIN_DIRECTION_BATCH_FAILED — and this is the backstop for a failure that
    increments nothing.
    """
    rows = _rows(conn, """
        SELECT window_end,
               count(*) FILTER (WHERE e->>'actor' = 'unattributed')::float
                 / NULLIF(count(*), 0) AS share,
               count(*) AS total
          FROM iran_bulletins b,
               LATERAL jsonb_array_elements(
                   coalesce(b.sections_json->'on_iran', '[]'::jsonb)
                   || coalesce(b.sections_json->'from_iran', '[]'::jsonb)
                   || coalesce(b.sections_json->'regional', '[]'::jsonb)) e
         WHERE b.status = 'completed'
           AND b.window_end > now() - (%s * interval '1 hour')
         GROUP BY b.window_end
    """, (window_hours,))
    findings = []
    for window_end, share, total in rows:
        if share is not None and share > max_unattributed and total >= 20:
            findings.append(Finding(
                "bulletin_unattributed",
                f"Iran bulletin: {share * 100:.0f}% of {total} events have no actor "
                f"(ceiling {max_unattributed * 100:.0f}%) — direction extraction "
                "may have failed",
                str(window_end),
            ))
    return findings


def check_degradation_counters(conn, window_hours: float) -> List[Finding]:
    """The counters the runs themselves recorded.

    src/core/counters.py exists so that a fallback path leaves evidence. This is
    the half of the job that reads it back.

    Thresholds, not "any non-zero" — that was the first version and it was wrong
    for a reason worth writing down. A single llm_contract_rejected means the
    citation guard caught a bad slot and rotated past it, which is the system
    working exactly as designed; paging about it teaches the reader that this
    channel reports non-events. The counters below fire only where the number
    means something the design did NOT already handle:

      * bulletin_direction_batch_failed at ANY count, because it fails open. Those
        events keep the unattributed default and the report renders perfectly
        while having quietly stopped saying which way anything was going.
      * llm_contract_rejected at 3+, which is no longer one bad slot rotated past
        but a pattern — the same slot failing all day, or most of a run's
        countries needing a second attempt.
      * llm_unusable_200 at 5+; below that it is ordinary provider weather.
    """
    rows = _rows(conn, """
        SELECT event_type, value_json->'degradation_counters'
          FROM system_telemetry
         WHERE event_type IN ('pipeline_run', 'sitrep_run')
           AND timestamp > now() - (%s * interval '1 hour')
           AND value_json ? 'degradation_counters'
    """, (window_hours,))
    totals: Dict[str, int] = {}
    for _event_type, counters in rows:
        for name, count in (counters or {}).items():
            try:
                totals[name] = totals.get(name, 0) + int(count)
            except (TypeError, ValueError):
                continue
    notable = {k: v for k, v in totals.items()
               if v >= COUNTER_ALARM_THRESHOLDS.get(k, DEFAULT_COUNTER_THRESHOLD)}
    if not notable:
        if totals:
            logger.info("Degradation counters below threshold, not paging: %s", totals)
        return []
    listed = ", ".join(f"{k}={v}" for k, v in sorted(notable.items()))
    return [Finding("degradation_counters",
                    "Degradation counters fired above their thresholds", listed)]


# Roughly what the paid floor spends in a day: 8 calls at $0.0055 each
# (gemini-3.1-flash-lite, priced against seven days of measured telemetry). Used
# only to turn a remaining balance into "about this many days left", which is the
# form a person can act on — "$0.84 remaining" is not.
FLOOR_USD_PER_DAY = 0.044
CREDIT_WARN_DAYS = 14.0


def check_openrouter_credit(conn, window_hours: float,
                            warn_days: float = CREDIT_WARN_DAYS) -> List[Finding]:
    """Is there enough credit left to keep the paid floor standing?

    OpenRouter is prepaid, so the balance is a hard ceiling and nothing can
    overspend it. That makes the risk not a surprise bill but a surprise
    SILENCE: the credit runs out, the floor drops away, and the free rungs
    beneath it quietly take over the reports — which is precisely the failure the
    paid slot was added on 2026-09-04 to end. A floor that can vanish without
    saying so is not a floor.

    Takes no database argument beyond the signature every check shares; the
    balance lives at the provider. Never raises on a network problem — an
    unreachable billing endpoint is not evidence of anything, and run_checks
    would report the exception as a finding of its own.
    """
    import os

    key = os.environ.get("OPENROUTER_API_KEY_A", "")
    if not key:
        return []
    try:
        import httpx

        resp = httpx.get("https://openrouter.ai/api/v1/key",
                         headers={"Authorization": f"Bearer {key}"}, timeout=15)
        data = (resp.json() or {}).get("data") or {}
    except Exception as exc:
        logger.warning("OpenRouter credit check could not reach the API: %s", exc)
        return []

    limit, usage = data.get("limit"), data.get("usage")
    if limit is None or usage is None:
        # An uncapped key reports limit=None. Nothing to warn about, and guessing
        # a ceiling would produce a daily false alarm.
        return []
    try:
        remaining = float(limit) - float(usage)
    except (TypeError, ValueError):
        return []
    days = remaining / FLOOR_USD_PER_DAY if FLOOR_USD_PER_DAY else 0.0
    if days > warn_days:
        return []
    return [Finding(
        "openrouter_credit_low",
        f"OpenRouter kredisi ~{days:.0f} gün sonra bitiyor — bitince ücretli "
        "zemin sessizce düşer ve raporları ücretsiz slotlar yazmaya başlar",
        f"kalan ${remaining:.2f} (limit ${float(limit):.2f}, "
        f"kullanılan ${float(usage):.2f})",
    )]


# See check_degradation_counters. A counter absent from this table has to reach
# the default before it is worth a person's attention.
COUNTER_ALARM_THRESHOLDS = {
    "bulletin_direction_batch_failed": 1,
    "bulletin_direction_short_reply": 5,
    "llm_contract_rejected": 3,
    "llm_unusable_200": 5,
}
DEFAULT_COUNTER_THRESHOLD = 3


CHECKS = (
    check_sitrep_citations,
    check_sitrep_truncation,
    check_narrator_changed,
    check_bulletin_attribution,
    check_degradation_counters,
    check_openrouter_credit,
)


def run_checks(conn, window_hours: float = DEFAULT_WINDOW_HOURS) -> List[Finding]:
    """Every check, with one failing check never costing the others.

    A check that raises is itself reported. The alternative — swallowing it — is
    how a health check quietly stops checking, which is the same class of silence
    this whole module exists to end.
    """
    findings: List[Finding] = []
    for check in CHECKS:
        try:
            findings.extend(check(conn, window_hours))
        except Exception as exc:
            logger.exception("Output-health check %s failed", check.__name__)
            findings.append(Finding(
                f"check_error:{check.__name__}",
                f"Health check {check.__name__} could not run",
                f"{type(exc).__name__}: {exc}",
            ))
    return findings


def format_report(findings: List[Finding]) -> Optional[str]:
    """The ops message, or None when there is nothing to say.

    Silence when healthy is the point: a check that pages every day is a check
    people stop reading, and then it is worth less than nothing.
    """
    if not findings:
        return None
    return ("⚠️ Raporlar çıktı ama içerikleri şüpheli:\n\n"
            + "\n".join(f.render() for f in findings))


def summarize(findings: List[Finding]) -> Dict[str, Any]:  # pragma: no cover - trivial
    return {"count": len(findings), "keys": [f.key for f in findings]}
