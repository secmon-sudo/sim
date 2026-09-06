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
import os
from typing import Any, Dict, List, Optional

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


# ── Money ──────────────────────────────────────────────────────────────────
#
# Token counts answer "what did we spend it on"; they do not answer "how much".
# Until 2026-09-04 nothing in this project cost money, so the difference did not
# matter. It does now, and the arithmetic was being done by hand — which is fine
# once and a liability every day after.
#
# Prices come from OpenRouter's PUBLIC model list, which needs no key. Two
# reasons that beats a table in this file: the table would drift silently the
# first time a provider repriced, and a hardcoded price is exactly the kind of
# number nobody re-checks. Free slots price at zero and cost nothing to include.
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"


def fetch_prices() -> Dict[str, tuple]:
    """{model_id: ($/M prompt, $/M completion)} from OpenRouter's public list.

    Never raises: a spend report that cannot reach the internet should still
    print the token table it was always able to print.
    """
    try:
        import httpx

        data = httpx.get(OPENROUTER_MODELS_URL, timeout=20).json().get("data", [])
    except Exception as exc:  # pragma: no cover - network
        logger.warning("Could not fetch OpenRouter prices: %s", exc)
        return {}
    prices = {}
    for m in data:
        p = m.get("pricing") or {}
        try:
            prices[m["id"]] = (float(p.get("prompt", 0)) * 1e6,
                               float(p.get("completion", 0)) * 1e6)
        except (TypeError, ValueError):
            continue
    return prices


def price_row(row: Dict[str, Any], prices: Dict[str, tuple]) -> float:
    """Dollars for one bucket, or 0.0 when the model is free or unknown.

    Only models we actually pay for appear in the list, so an unknown key is a
    free slot and costs nothing — which is why a missing price is 0.0 rather
    than an error. `--by model` gives an exact per-model figure; `--by purpose`
    cannot, because a purpose spreads across slots, so it is reported once at
    the bottom instead of being guessed per row.
    """
    rate = prices.get(row.get("model") or row.get("bucket") or "")
    if not rate:
        return 0.0
    return row["prompt_tokens"] / 1e6 * rate[0] + row["completion_tokens"] / 1e6 * rate[1]


def openrouter_actual() -> Optional[Dict[str, float]]:
    """What OpenRouter itself says this key has spent, for reconciliation.

    Our own figure is derived from token counts we recorded; theirs is the bill.
    On 2026-09-06 the two agreed to 2.4%, which is the only evidence that the
    derived number can be trusted between statements — and the credit alarm in
    output_health depends on it being trustworthy.
    """
    key = os.environ.get("OPENROUTER_API_KEY_A", "")
    if not key:
        return None
    try:
        import httpx

        d = (httpx.get(OPENROUTER_KEY_URL,
                       headers={"Authorization": f"Bearer {key}"},
                       timeout=20).json() or {}).get("data") or {}
        return {"usage": float(d.get("usage") or 0.0),
                "limit": float(d["limit"]) if d.get("limit") is not None else None}
    except Exception as exc:  # pragma: no cover - network
        logger.warning("Could not read OpenRouter usage: %s", exc)
        return None


def collect(db_conn, days: int, by: str = "purpose") -> List[Dict[str, Any]]:
    query = _QUERY_BY_MODEL if by == "model" else _QUERY
    rows = db_conn.execute(query, (days,)).fetchall()
    return [
        {"bucket": r[0], "model": r[0] if by == "model" else None,
         "calls": r[1], "tokens": int(r[2] or 0),
         "prompt_tokens": int(r[3] or 0), "completion_tokens": int(r[4] or 0),
         "avg_ms": int(r[5] or 0), "failures": r[6]}
        for r in rows
    ]


def render_money(rows: List[Dict[str, Any]], days: int, by: str) -> str:
    """The dollars, kept separate from the token table above it.

    Per-row only for `--by model`, where a row IS one price. A purpose spreads
    across slots at whatever mix the router chose that day, so pricing it per row
    would be a guess wearing a decimal point.
    """
    prices = fetch_prices()
    if not prices:
        return "\n(prices unavailable — token table above is unaffected)"
    total = sum(price_row(r, prices) for r in rows)
    out = ["", f"cost — last {days} day(s)"]
    if by == "model":
        paid = sorted((r for r in rows if price_row(r, prices) > 0),
                      key=lambda r: -price_row(r, prices))
        width = max((len(r["bucket"]) for r in paid), default=10)
        for r in paid:
            usd = price_row(r, prices)
            out.append(f"  {r['bucket'].ljust(width)}  ${usd:>8.4f}  "
                       f"({usd / total * 100 if total else 0:>4.1f}%)")
        free = len(rows) - len(paid)
        if free:
            out.append(f"  {free} free slot(s) at $0.0000")
    out.append(f"  TOTAL ${total:.4f}   →  ${total / max(days, 1) * 30:.2f}/month "
               f"at this rate")

    actual = openrouter_actual()
    if actual:
        out += ["", "reconciliation against OpenRouter's own figure:",
                f"  they say the key has spent ${actual['usage']:.4f} in total"]
        if actual.get("limit") is not None:
            left = actual["limit"] - actual["usage"]
            out.append(f"  credit left ${left:.2f}  "
                       f"(≈{left / (total / max(days, 1)):.0f} days at this rate)"
                       if total else f"  credit left ${left:.2f}")
        out.append("  ours is derived from recorded tokens; theirs is the bill. "
                   "They agreed to 2.4% on 2026-09-06.")
    return "\n".join(out)


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
    parser.add_argument("--no-cost", action="store_true",
                        help="token table only; skip the price fetch and the "
                             "reconciliation against OpenRouter's own figure")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    db_conn = get_connection()
    try:
        rows = collect(db_conn, args.days, args.by)
        print(render(rows, args.days, args.by))
        if not args.no_cost:
            print(render_money(rows, args.days, args.by))
    finally:
        put_connection(db_conn)
        close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
