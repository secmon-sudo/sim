#!/usr/bin/env python3
"""SIM — how many publishers is the "Tek kaynak" label hiding?

Measured 2026-09-05: 67% of CRITICAL alerts and 65% of ALERTs carry zero
corroborating sources, and are therefore labelled "Doğrulanmamış (Tek kaynak)"
in every report they appear in. In a corpus with 1,557 distinct domains a week,
that is a suspicious number, and it is the last unmeasured claim in the alert
path — three previous attempts at a source-quality signal (penalty_score,
earliness, corroboration rate) were each measured and each failed, two of them
pointing the wrong way.

The question this answers is narrow on purpose. NOT "is this publisher any
good", which is what failed three times. Just: at the moment we told a reader
"single source", how many OTHER publishers in our own corpus had filed the same
story?

Why the answer is not already known. Pass A credits corroboration by DROPPING a
duplicate at ingest and recording it against the survivor, so a story caught by
dedup leaves one row with N corroborating sources. A story dedup MISSED leaves N
separate rows, each looking single-sourced, each able to alert on its own. The
events table therefore cannot distinguish "nobody else reported this" from "we
failed to notice that everybody did" — the two look identical in every query.

So this replays the real matcher, `find_content_duplicate`, over pairs it never
got to see. At ingest a candidate is compared against a bounded window of recent
events; here every alerting single-source event is compared against the whole
corpus in a wide time band. Two possible outcomes, and they point at different
fixes:

  * the matcher DOES find siblings → the thresholds are fine and the miss is one
    of window or ordering, which is cheap to fix;
  * the matcher does NOT → the similarity thresholds are the binding constraint,
    and no amount of window widening helps.

Deliberately read-only and deliberately not a fix. Nothing here changes a
verdict, writes a row, or touches config. It exists so the fourth attempt at
this problem starts from a number instead of an intuition.

    python -m scripts.replay_corroboration --days 7
    python -m scripts.replay_corroboration --days 7 --band-hours 72 --examples 8
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.ingest_filters import find_content_duplicate  # noqa: E402

# How far either side of an event to look for the same story. Ingest compares
# against a much narrower window; the point of widening it here is to find out
# whether the siblings were ever reachable at all.
DEFAULT_BAND_HOURS = 48


def _fetch(days: int):
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL is not set — this reads production", file=sys.stderr)
        return None
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            """SELECT id, source_domain, source_title, canonical_text,
                      country_iso, alert_tier, created_at,
                      jsonb_array_length(COALESCE(corroborating_sources,'[]'::jsonb))
                 FROM events
                WHERE created_at > NOW() - (%s * INTERVAL '1 day')
                  AND status <> 'archived'
                  AND source_title IS NOT NULL
                ORDER BY created_at""",
            (days,),
        ).fetchall()
    return [
        {"id": str(r[0]), "domain": (r[1] or "").lower(), "title": r[2] or "",
         "text": r[3] or "", "iso": r[4], "tier": r[5], "at": r[6],
         "credited": r[7] or 0}
        for r in rows
    ]


def _siblings(target, corpus, band_hours):
    """Publishers, other than the target's own, whose story the real matcher
    calls the same story.

    One row per publisher, not per article: three rewrites from the same outlet
    are one voice, and counting them as three is exactly the error that made
    syndicated copy look like corroboration in August.
    """
    lo = target["at"].timestamp() - band_hours * 3600
    hi = target["at"].timestamp() + band_hours * 3600
    found = {}
    for other in corpus:
        if other["id"] == target["id"] or other["domain"] == target["domain"]:
            continue
        if not (lo <= other["at"].timestamp() <= hi):
            continue
        if find_content_duplicate([(other["title"], other["text"])],
                                  target["title"], target["text"]) is not None:
            found.setdefault(other["domain"], other["title"])
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--band-hours", type=float, default=DEFAULT_BAND_HOURS)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--tiers", default="ALERT,CRITICAL")
    args = ap.parse_args()

    corpus = _fetch(args.days)
    if corpus is None:
        return 2
    tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    targets = [e for e in corpus if e["tier"] in tiers and e["credited"] == 0]

    print(f"corpus: {len(corpus)} live events over {args.days} days")
    print(f"targets: {len(targets)} {'/'.join(sorted(tiers))} events labelled "
          f"single-source")
    print(f"band: ±{args.band_hours:g}h, matcher: find_content_duplicate "
          f"(production thresholds)\n")

    hist = Counter()
    examples = []
    for t in targets:
        sib = _siblings(t, corpus, args.band_hours)
        n = len(sib)
        hist[min(n, 5)] += 1
        if n and len(examples) < args.examples:
            examples.append((t, sib))

    total = len(targets) or 1
    hidden = sum(c for n, c in hist.items() if n > 0)
    print("publishers the label was hiding:")
    for n in sorted(hist):
        label = "5+" if n == 5 else str(n)
        print(f"  {label:>3} other publisher(s): {hist[n]:>5}  "
              f"({hist[n]/total*100:>5.1f}%)")
    print(f"\n{hidden} of {total} ({hidden/total*100:.1f}%) were NOT actually "
          f"single-source by our own matcher's judgement.")
    print("A miss the matcher can find is a window problem. One it cannot find "
          "is a threshold problem.\n")

    for t, sib in examples:
        print(f"— {t['tier']} {t['domain']}: {t['title'][:88]}")
        for dom, title in list(sib.items())[:4]:
            print(f"    also {dom}: {title[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
