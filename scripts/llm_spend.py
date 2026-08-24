"""SIM — LLM spend, broken down by the pipeline stage that spent it.

Why this exists
---------------
system_telemetry has logged one 'llm_call' row per call since May, but until
2026-08-24 the row named the provider, the model and the token count and never the
stage. The table could say "1332 calls in three days" and not "on what", so the
only spend question anyone could actually answer was the total.

Worse, the rows were not even a complete total: only pass_c and the storyline
adjudicator called log_llm_telemetry. The narrator, both SITREP stages and all
three weekly forecast passes ran on the quality router — the expensive one — and
wrote nothing, so the cheap bulk classifier accounted for ~100% of a bill it did
not own.

Both halves are fixed; this reads the result. Rows written before the fix carry no
'purpose' and are reported separately as unattributed rather than being folded in,
because guessing their stage would manufacture history.

Usage
-----
    python -m scripts.llm_spend                 # last 7 days
    python -m scripts.llm_spend --days 1
    python -m scripts.llm_spend --days 30 --by model
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List

from src.services.supabase_client import close_pool, get_connection, put_connection

logger = logging.getLogger(__name__)

_QUERY = """
    SELECT COALESCE(value_json->>'purpose', '(unattributed)') AS bucket,
           COUNT(*)                                            AS calls,
           SUM(COALESCE((value_json->>'tokens_used')::bigint, 0))       AS tokens,
           SUM(COALESCE((value_json->>'prompt_tokens')::bigint, 0))     AS prompt_tokens,
           SUM(COALESCE((value_json->>'completion_tokens')::bigint, 0)) AS completion_tokens,
           ROUND(AVG(COALESCE((value_json->>'latency_ms')::numeric, 0))) AS avg_ms,
           COUNT(*) FILTER (WHERE (value_json->>'success')::boolean IS FALSE) AS failures
      FROM system_telemetry
     WHERE event_type = 'llm_call'
       AND timestamp > NOW() - (%s * INTERVAL '1 day')
     GROUP BY 1
     ORDER BY tokens DESC, calls DESC
"""

_QUERY_BY_MODEL = _QUERY.replace(
    "COALESCE(value_json->>'purpose', '(unattributed)') AS bucket",
    "COALESCE(value_json->>'model', 'unknown') AS bucket",
)


def collect(db_conn, days: int, by: str = "purpose") -> List[Dict[str, Any]]:
    query = _QUERY_BY_MODEL if by == "model" else _QUERY
    rows = db_conn.execute(query, (days,)).fetchall()
    return [
        {"bucket": r[0], "calls": r[1], "tokens": int(r[2] or 0),
         "prompt_tokens": int(r[3] or 0), "completion_tokens": int(r[4] or 0),
         "avg_ms": int(r[5] or 0), "failures": r[6]}
        for r in rows
    ]


def render(rows: List[Dict[str, Any]], days: int, by: str) -> str:
    if not rows:
        return f"No LLM telemetry in the last {days} day(s)."
    total_calls = sum(r["calls"] for r in rows)
    total_tokens = sum(r["tokens"] for r in rows)
    width = max(len(r["bucket"]) for r in rows)

    out = [f"LLM spend by {by} — last {days} day(s)",
           f"{'stage'.ljust(width)}  {'calls':>7} {'%':>5}  {'tokens':>10} {'%':>5}  "
           f"{'out/call':>8} {'ms':>6} {'fail':>5}",
           "-" * (width + 52)]
    for r in rows:
        call_pct = 100.0 * r["calls"] / total_calls if total_calls else 0
        tok_pct = 100.0 * r["tokens"] / total_tokens if total_tokens else 0
        per_call = r["completion_tokens"] // r["calls"] if r["calls"] else 0
        out.append(
            f"{r['bucket'].ljust(width)}  {r['calls']:>7} {call_pct:>4.1f}%  "
            f"{r['tokens']:>10,} {tok_pct:>4.1f}%  {per_call:>8,} {r['avg_ms']:>6} "
            f"{r['failures']:>5}"
        )
    out.append("-" * (width + 52))
    out.append(f"{'TOTAL'.ljust(width)}  {total_calls:>7} {100.0:>4.1f}%  "
               f"{total_tokens:>10,} {100.0:>4.1f}%")

    unattributed = next((r for r in rows if r["bucket"] == "(unattributed)"), None)
    if unattributed:
        out += ["",
                f"note: {unattributed['calls']:,} call(s) predate the purpose label "
                f"(added 2026-08-24) and are not assigned to a stage."]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM spend by pipeline stage")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--by", choices=("purpose", "model"), default="purpose")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    db_conn = get_connection()
    try:
        print(render(collect(db_conn, args.days, args.by), args.days, args.by))
    finally:
        put_connection(db_conn)
        close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
