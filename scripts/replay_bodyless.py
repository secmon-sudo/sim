#!/usr/bin/env python3
"""Would the article body have changed what Pass C decided?

A third of everything SIM classifies arrives with no body. Measured over the 7 days to
2026-09-09: 8,025 article fetches attempted, 5,405 succeeded — 2,620 failures, and Pass
C classifies those from the headline alone (see _batch_prompt: Headline, Source, Text).

The cohort looks worse on every axis. Archived 56.5% against 40.2%; tiered 23.7%
against 34.2%; mean severity 32.3 against 45.2; and time_certainty 'unknown' for 90.6%
of them against 69.2% — which matters more than the rest, because time_certainty is the
dominant alert gate.

That is CORRELATION and it has an innocent reading: articles that fail to fetch are not
a random sample, so perhaps they were the low-value ones all along. This script settles
which reading is right by re-running the REAL Pass C prompt and parser over the same
reports twice — once as the pipeline saw them, once with the body recovered — and
printing what moved.

SCOPE, because the first draft of this got it wrong. SIM is a SECURITY monitor, not a
safety one. A recovered body that turns a headline into an accident — a runway overrun,
an engine failure — has not recovered an alert, it has correctly declined one, and the
pipeline already says so: apply_safety_downrank caps SAFETY_EVENT_TYPES below the alert
floor unless there is mass casualty. So the tally below counts a recovery only when the
body lands the report on a SECURITY type; safety recoveries are printed, separately and
uncounted, because "we now know it was an accident" is a real improvement to the record
and no improvement at all to paging.

The bodies in db/replay/bodyless_sample.json were recovered externally (Parallel's
web_fetch, 26 of 30) and committed, so this run is reproducible and costs no fetch.

Keys live in GitHub Actions; run it there via bodyless-replay.yml.

  python -m scripts.replay_bodyless --provider groq --model qwen/qwen3.8-27b
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.core.llm_client import call_llm
from src.core.llm_router import LLMAccount, LLMRouter, build_llm_router
from src.core.token_bucket import TokenBucket
from src.pipeline.pass_d_score import SAFETY_EVENT_TYPES
from src.pipeline.pass_c_classify import (
    BATCH_SYSTEM_SUFFIX,
    CLASSIFICATION_SYSTEM_PROMPT,
    LLMParseError,
    _batch_prompt,
    _parse_batch_response,
)

FIXTURE = Path("db/replay/bodyless_sample.json")

# Small batches for the same reason Pass C uses them: the 8000-token per-request ceiling
# is shared with the reply, and a with-body batch is several times heavier than the
# headline-only one it is being compared against.
BATCH = 4
PACE_SECONDS = 4

_KEY_ENV = {"groq": "GROQ_API_KEY_A", "openrouter": "OPENROUTER_API_KEY_A",
            "cerebras": "CEREBRAS_API_KEY", "mistral": "MISTRAL_API_KEY"}


def _router(provider: str, model: str) -> LLMRouter:
    """The production cascade by default; a single pinned slot only on request.

    The first two runs of this script used a one-account router and died on their
    opening call — a Groq 429, then an OpenRouter "unusable HTTP 200" — because one
    failure puts the only account on cooldown and every later batch raises
    LLMAllThrottled. Production does not have that problem: it has a cascade, and
    falling down it is the normal state, not an error. Borrowing the real router makes
    the replay both sturdier AND more faithful, since the comparison it draws is
    between two prompts, never between two models.
    """
    if not provider:
        return build_llm_router()
    return LLMRouter([
        LLMAccount(
            provider=provider, account_id="A", model=model,
            api_key=os.environ.get(_KEY_ENV.get(provider, ""), ""),
            rpm=30, rpd=10_000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=10_000, burst=4),
        )
    ])


def _as_event(row: dict, with_body: bool) -> dict:
    """The shape _batch_prompt reads. `canonical_text` is the only field that differs."""
    return {
        "source_title": row["title"],
        "source_domain": row["domain"],
        "canonical_text": row["recovered_text"] if with_body else "",
    }


def _classify(router, rows: list[dict], with_body: bool) -> dict[int, dict]:
    """Pass C's own prompt and parser, so a difference here is a difference there."""
    out: dict[int, dict] = {}
    for i in range(0, len(rows), BATCH):
        if i:
            # The free slots are shared with the live pipeline, which runs every few
            # hours and is the higher-priority consumer. Pace rather than race it.
            time.sleep(PACE_SECONDS)
        chunk = rows[i:i + BATCH]
        events = [_as_event(r, with_body) for r in chunk]
        try:
            result = call_llm(
                router,
                prompt=_batch_prompt(events),
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT + BATCH_SYSTEM_SUFFIX,
                max_tokens=2048,
                json_mode=True,
            )
            items = _parse_batch_response(result.get("content", ""), expected=len(chunk))
        except (LLMParseError, Exception) as exc:   # noqa: B014 - report, never abort
            print(f"  batch {i // BATCH + 1} failed: {type(exc).__name__}: {str(exc)[:120]}")
            items = {}
        # _parse_batch_response keys by REPORT NUMBER within the batch, 1-based.
        for n, item in (items or {}).items():
            try:
                out[i + int(n) - 1] = item
            except (TypeError, ValueError):
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="",
                    help="pin one slot instead of the production cascade")
    ap.add_argument("--model", default="")
    ap.add_argument("--file", default=str(FIXTURE))
    args = ap.parse_args()

    rows = json.loads(Path(args.file).read_text(encoding="utf-8"))
    withb = [r for r in rows if len(r.get("recovered_text") or "") > 400]
    print(f"{len(rows)} reports, {len(withb)} with a recovered body — "
          f"the other {len(rows) - len(withb)} could not be read by anyone and are "
          f"reported but excluded from the verdict.\n")

    router = _router(args.provider, args.model)
    label = (f"{args.provider}:{args.model}" if args.provider
             else f"production cascade ({len(router.accounts)} accounts)")
    print(f"model: {label}\n")

    print("classifying headline-only …")
    before = _classify(router, rows, with_body=False)
    print("classifying with recovered body …")
    after = _classify(router, rows, with_body=True)

    # A replay that classified nothing must not print a verdict. The first run of this
    # script took a 429 on its opening call, cascaded every remaining batch into
    # LLMAllThrottled, and then printed a clean table of dashes under the heading
    # "of 26 reports whose body was recoverable: recovered onto a SECURITY type: 0".
    # That reads exactly like a measured negative and is nothing of the kind — the same
    # "it ran and was empty" failure this repo keeps a counters module for. Both sides
    # must have answered for a comparison to exist, and a thin sample is reported as
    # thin rather than averaged into confidence.
    graded = [i for i in range(len(rows))
              if len(rows[i].get("recovered_text") or "") > 400
              and before.get(i) and after.get(i)]
    if not graded:
        print("\nNO VERDICT: neither side produced classifications — see the batch "
              "errors above. This is a failed run, not a null result.")
        return 2
    if len(graded) < len(withb) // 2:
        print(f"\nWARNING: only {len(graded)} of {len(withb)} reports were classified "
              "on BOTH sides. Treat the tally below as a sample, not a rate.")

    print(f"\n{'domain':<22} {'headline-only':<34} {'with body':<34} moved")
    print("-" * 104)
    gained_type = gained_time = lost = gained_safety = 0
    SENTINELS = ("unclassified", "other_aviation_related", "—")
    for i, row in enumerate(rows):
        if i not in graded:
            continue
        b, a = before[i], after[i]
        bt, at = b.get("event_type") or "—", a.get("event_type") or "—"
        btc, atc = b.get("time_certainty") or "—", a.get("time_certainty") or "—"
        moved = []
        if bt != at:
            moved.append("type")
            # 'unclassified' is the sentinel for "the model could not place this", so
            # leaving it is a recovery and entering it is a loss — the asymmetry is the
            # whole question and must not be counted as a symmetric "changed". And
            # leaving it for a SAFETY type is neither: see the scope note above.
            if bt in SENTINELS and at not in SENTINELS:
                if at in SAFETY_EVENT_TYPES:
                    gained_safety += 1
                    moved.append("safety")
                else:
                    gained_type += 1
            elif at in SENTINELS:
                lost += 1
        if btc != atc:
            moved.append("time")
            if btc == "unknown" and atc != "unknown":
                gained_time += 1
        print(f"{row['domain'][:22]:<22} {bt[:20]:<20} {btc[:12]:<13} "
              f"{at[:20]:<20} {atc[:12]:<13} {','.join(moved) or '-'}")

    print("-" * 104)
    print(f"\nof {len(graded)} reports classified on BOTH sides "
          f"({len(withb)} had a recoverable body, {len(rows)} sampled):")
    print(f"  recovered onto a SECURITY type                : {gained_type}")
    print(f"  recovered onto a safety type (does NOT page)  : {gained_safety}")
    print(f"  time_certainty left 'unknown'                 : {gained_time}")
    print(f"  regressed into the sentinel                   : {lost}")
    print("\ntime_certainty is the dominant alert gate, so the middle line is the one\n"
          "that decides whether paying for extraction buys alerting recall.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
