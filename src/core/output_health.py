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

# The share of a run's citations that may be blanked before it reads as a defect
# rather than a bad sentence. check_sitrep_citations is a CLIFF detector — its bar
# is zero surviving links, because that is the shape 2026-09-04 had — and a citation
# guard that fails halfway clears it comfortably. Measured over the ten days to
# 2026-09-07, excluding that collapse, the per-run rate has been:
#
#   0.0%  0.0%  0.0%  0.6%  1.1%  1.1%  2.2%  3.7%  4.1%  9.0%
#
# The 9.0% is gemini-3.5-flash-lite on 31 Aug, a model that is no longer in the
# cascade; everything the current floor has produced sits at or under 4.1%. 20% is
# therefore about twice the worst rate ever observed from a working model, which is
# the headroom a check needs if people are going to keep reading it.
CITATION_BLANK_RATE_MAX = 0.20


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

    Scoped to the LATEST run rather than the whole window, and that is not a
    detail. The window is 30 hours and a SITREP runs daily, so a window query
    spans two runs: on 5 Sep it reported the previous morning's five broken
    reports while that morning's five were perfect, twice, hours after the cause
    had been fixed and removed. A check that keeps announcing a problem you
    already solved is the fastest way to teach someone to ignore it.
    """
    rows = _rows(conn, """
        SELECT country_iso, llm_model
          FROM sitreps
         WHERE status = 'completed'
           AND window_end = (SELECT max(window_end) FROM sitreps
                              WHERE status = 'completed'
                                AND window_end > now() - (%s * interval '1 hour'))
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


def check_sitrep_citation_rate(conn, window_hours: float,
                               rate_max: float = CITATION_BLANK_RATE_MAX) -> List[Finding]:
    """Too many of a run's citations were invented, without all of them being.

    check_sitrep_citations asks whether a report has ANY surviving link, which is
    the question 2026-09-04 posed: minimax-m2.7 shortened all 108 of that morning's
    citations to bare domains and every report shipped with none. A guard that fails
    that completely is easy to see. One that fails halfway is not — 54 blanked
    citations out of 108 leaves every report full of working links and reads as
    healthy to every check in this module.

    Reports with no surviving link at all are excluded from the denominator rather
    than counted, so a total collapse pages once (as sitrep_no_citations) instead of
    twice. This check is only ever about the partial case.
    """
    rows = _rows(conn, """
        SELECT country_iso,
               (length(report_text) - length(replace(report_text, 'https://', ''))) / 8
                 AS kept,
               (length(report_text)
                - length(replace(report_text, '[kaynak listede]', ''))) / 16
                 AS blanked
          FROM sitreps
         WHERE status = 'completed'
           AND window_end = (SELECT max(window_end) FROM sitreps
                              WHERE status = 'completed'
                                AND window_end > now() - (%s * interval '1 hour'))
    """, (window_hours,))
    scored = [r for r in rows if (r[1] or 0) > 0]
    if not scored:
        return []
    kept = sum(r[1] or 0 for r in scored)
    blanked = sum(r[2] or 0 for r in scored)
    total = kept + blanked
    if total == 0:
        return []
    rate = blanked / total
    if rate <= rate_max:
        return []
    worst = sorted(scored, key=lambda r: -(r[2] or 0))[:3]
    return [Finding(
        "sitrep_citation_blank_rate",
        f"{blanked} of {total} SITREP citations ({rate:.0%}) were not in the "
        f"source list and were blanked",
        ", ".join(f"{(r[0] or '??').strip()} {r[2]}/{(r[1] or 0) + (r[2] or 0)}"
                  for r in worst),
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
           AND window_end = (SELECT max(window_end) FROM sitreps
                              WHERE status = 'completed'
                                AND window_end > now() - (%s * interval '1 hour'))
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

    Scoped to the LATEST bulletin, for the reason check_sitrep_citations already
    carries: this fires on the report a reader actually has, and every earlier
    attempt in the window has been superseded by it. On 6 Sep the burst fix landed
    between runs — 08:06 and 08:10 came out at 75% unattributed, 08:14 at 43% —
    and the unscoped query paged all three, hours later, two of them for a cause
    that had already been fixed and deployed. A check that keeps announcing a
    problem you solved is the fastest way to teach someone to ignore it.
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
           AND b.window_end = (SELECT max(window_end) FROM iran_bulletins
                                WHERE status = 'completed'
                                  AND window_end > now() - (%s * interval '1 hour'))
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


# Fallback burn rate, used only when OpenRouter itself will not say. The real
# rate comes from the provider's own usage_weekly/usage_daily fields, because a
# constant here is a number nobody re-checks: this one was written as $0.044 on
# 2026-09-04 from that week's telemetry, and by 9 Sep the floor was spending
# $0.070-0.086 a day — the bulletin had joined the paid slot in between. A stale
# rate does not merely mis-report, it under-reports: too small a divisor turns a
# balance into MORE days than there are, which is the wrong direction for the one
# number this check exists to produce.
FLOOR_USD_PER_DAY = 0.075
CREDIT_WARN_DAYS = 14.0


def _openrouter_data(path: str, key: str) -> Dict[str, Any]:
    """`data` from one OpenRouter account endpoint, or {} — never raises here."""
    import httpx

    resp = httpx.get(f"https://openrouter.ai/api/v1/{path}",
                     headers={"Authorization": f"Bearer {key}"}, timeout=15)
    return (resp.json() or {}).get("data") or {}


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_openrouter_credit(conn, window_hours: float,
                            warn_days: float = CREDIT_WARN_DAYS) -> List[Finding]:
    """Is there enough credit left to keep the paid floor standing?

    OpenRouter is prepaid, so the balance is a hard ceiling and nothing can
    overspend it. That makes the risk not a surprise bill but a surprise
    SILENCE: the credit runs out, the floor drops away, and the free rungs
    beneath it quietly take over the reports — which is precisely the failure the
    paid slot was added on 2026-09-04 to end. A floor that can vanish without
    saying so is not a floor.

    Two balances exist and they are not the same number. `/key` reports the
    per-key SPENDING CAP (limit, limit_remaining), which is null unless somebody
    set one; `/credits` reports the ACCOUNT balance actually funded. This project
    funded $10 into the account and set no per-key cap, so the first version of
    this check — limit minus usage, `[]` when limit was null — could never fire
    on the only account it was written for. It read as a working alarm for five
    days and was inert the whole time.

    So: cap first when there is one, account balance second, and when NEITHER
    can be read, say so. An unreadable balance is not the same as a healthy one,
    and the whole argument of this module is that a check which cannot check must
    not answer "fine". It clears itself the moment a spending limit is set on the
    key or a management key is supplied.

    The burn rate comes from the provider's own usage_weekly/usage_daily rather
    than from a constant here, so the "days left" figure stays true as the
    project's LLM volume changes. See FLOOR_USD_PER_DAY for what that cost when
    it was a constant.

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
        info = _openrouter_data("key", key)
    except Exception as exc:
        logger.warning("OpenRouter credit check could not reach the API: %s", exc)
        return []

    limit, usage = _as_float(info.get("limit")), _as_float(info.get("usage"))
    remaining = source = None
    if limit is not None and usage is not None:
        remaining, source = limit - usage, f"anahtar tavanı ${limit:.2f}"
    else:
        # No per-key cap, so the number that matters is the account balance.
        # /credits needs a management key and answers with no `data` when the
        # key is not one — which is a readable outcome, not an exception.
        try:
            credits = _openrouter_data("credits", key)
        except Exception as exc:
            logger.warning("OpenRouter credit check could not reach the API: %s", exc)
            return []
        total = _as_float(credits.get("total_credits"))
        spent = _as_float(credits.get("total_usage"))
        if total is not None and spent is not None:
            remaining, source = total - spent, f"hesap bakiyesi ${total:.2f}"

    if remaining is None:
        return [Finding(
            "openrouter_credit_unreadable",
            "OpenRouter bakiyesi OKUNAMIYOR — ücretli zemin haber vermeden "
            "düşebilir; anahtara harcama tavanı koy ya da management key ver",
            f"/key limit={info.get('limit')!r}, /credits yanıtsız"
            + (f", bu anahtarın toplam kullanımı ${usage:.2f}"
               if usage is not None else ""),
        )]

    # What it actually spends, from the provider's own counters. usage_weekly is
    # preferred because a single quiet day would otherwise read as a long runway.
    weekly = _as_float(info.get("usage_weekly"))
    daily = _as_float(info.get("usage_daily"))
    burn = FLOOR_USD_PER_DAY
    if weekly:
        burn = weekly / 7.0
    elif daily:
        burn = daily

    days = remaining / burn if burn else 0.0
    if days > warn_days:
        return []
    return [Finding(
        "openrouter_credit_low",
        f"OpenRouter kredisi ~{days:.0f} gün sonra bitiyor — bitince ücretli "
        "zemin sessizce düşer ve raporları ücretsiz slotlar yazmaya başlar",
        f"kalan ${remaining:.2f} ({source}), günlük ${burn:.3f}",
    )]


# See check_degradation_counters. A counter absent from this table has to reach
# the default before it is worth a person's attention.
COUNTER_ALARM_THRESHOLDS = {
    "bulletin_direction_batch_failed": 1,
    "bulletin_direction_short_reply": 5,
    "llm_contract_rejected": 3,
    # Measured 5 Sep across six consecutive pipeline runs: 1, 1, 1, 1, 4, 1 — and
    # every one of those runs completed successfully. This counter records the
    # router rotating past a slot that answered an empty 200, which is the system
    # healing itself; the daily total sits near 8-12 simply because there are
    # 8-12 runs. A threshold under that pages every day about nothing.
    "llm_unusable_200": 30,
}
DEFAULT_COUNTER_THRESHOLD = 3


CHECKS = (
    check_sitrep_citations,
    check_sitrep_citation_rate,
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
