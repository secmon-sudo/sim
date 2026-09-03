#!/usr/bin/env python3
"""SIM — dedup replay: prove a change to find_content_duplicate moves no verdict.

content_dedup_cpu is Pass A's largest phase — 132.1s of 284.1s on 3 Sep 2026, 46%
of it — and it is the one phase whose optimisation can change the PRODUCT. A faster
article fetch either returns the same page or it does not; a faster duplicate
matcher can quietly decide that two stories are no longer the same story, which
changes which events exist, which get a corroboration credit, and which reach a
report as a second card for something already sent.

The last dedup optimisation was verified this way and the verification was thrown
away: its test file records "800 candidates x 600 stored events (480K comparisons,
612 duplicates) with zero differing verdicts" as prose. The next change had to
rebuild that from nothing. This is that harness, kept.

Two commands, split on purpose along the line where the credentials are:

    dump    reads the production corpus and writes it to a file. Needs the
            database, so it runs in Actions where the secrets live.
    replay  runs find_content_duplicate over a dumped corpus and writes, or
            compares against, a verdict baseline. Pure CPU, no database, so it
            runs anywhere — including on a laptop with no credentials, which is
            the environment the person making the optimisation is actually in.

The flow a dedup change takes:

    (in Actions)   python -m scripts.replay_dedup dump --out corpus.json
    (locally, on HEAD)      python -m scripts.replay_dedup replay corpus.json \\
                                --out baseline.json
    (locally, on the change) python -m scripts.replay_dedup replay corpus.json \\
                                --against baseline.json

The last command exits non-zero if any verdict moved, and prints the pairs that
moved rather than only a count — a diff of one is a question to answer, not a
number to accept.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.ingest_filters import find_content_duplicate  # noqa: E402

# Mirrors _fetch_recent_events_for_dedup's own ceiling: replaying a corpus larger
# than production ever holds would measure a loop that never runs.
CORPUS_LIMIT = 2000


def dump(out_path: str, limit: int, days: int) -> int:
    """Write the real dedup corpus to a file. Requires DATABASE_URL."""
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL is not set — dump needs the database", file=sys.stderr)
        return 2
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            """SELECT id, source_domain, source_title, canonical_text,
                      anchor_name_raw
                 FROM events
                WHERE ingested_at > NOW() - (%s * INTERVAL '1 day')
                ORDER BY ingested_at DESC
                LIMIT %s""",
            (days, min(limit, CORPUS_LIMIT)),
        ).fetchall()

        # The real duplicates. events holds only what was INSERTED — a duplicate
        # was dropped and never became a row — so events-against-events cannot
        # exercise matching at all. corroborating_sources is where the discarded
        # side survives: each entry is a duplicate that WAS dropped, recorded
        # against the event it was merged into. That is ground truth, in
        # production, for the decision this harness has to protect.
        pairs = conn.execute(
            """SELECT ev.id, s->>'title'
                 FROM events ev, jsonb_array_elements(ev.corroborating_sources) s
                WHERE ev.ingested_at > NOW() - (%s * INTERVAL '1 day')
                  AND length(s->>'title') > 20""",
            (days,),
        ).fetchall()

    corpus = [
        {"id": str(r[0]), "domain": r[1] or "", "title": r[2] or "",
         "canonical": r[3] or "", "anchor": r[4] or ""}
        for r in rows
    ]
    pair_rows = [{"survivor": str(a), "title": b} for a, b in pairs]
    Path(out_path).write_text(
        json.dumps({"days": days, "rows": corpus, "pairs": pair_rows},
                   ensure_ascii=False),
        encoding="utf-8")
    print(f"dumped {len(corpus)} events and {len(pair_rows)} recorded duplicates "
          f"from the last {days} days → {out_path}")
    return 0


def _verdicts(corpus: list, candidates: list) -> tuple[dict, float]:
    """Every candidate's duplicate verdict, keyed by candidate id.

    The verdict is the matched event's ID, not its index: an index is a position
    in a list that a change could legitimately reorder, while the identity of the
    story a candidate was merged into is the thing that must not move.
    """
    stored = [(c["title"], c["canonical"], c["anchor"]) for c in corpus]
    ids = [c["id"] for c in corpus]
    out = {}
    started = time.perf_counter()
    for cand in candidates:
        idx = find_content_duplicate(stored, cand["title"], cand["canonical"])
        out[cand["id"]] = ids[idx] if idx is not None else None
    return out, time.perf_counter() - started


def _replay_pairs(rows: list, pairs: list, limit: int) -> tuple[dict, float, int]:
    """Re-run the matcher on duplicates production actually dropped.

    Each pair is a headline that WAS deduped and the event it was merged into, so
    this mode has something the other does not: a right answer. The verdict is
    whether the matcher still reaches the same survivor.

    What it does NOT prove, and this matters: corroborating_sources records the
    duplicate's TITLE and never its canonical_text, so the candidate goes in with
    its title as both. find_content_duplicate's third signal — body shingles —
    only engages above 100 characters of body, so this mode exercises the two
    title signals faithfully and the body signal barely. Pair it with the events
    mode, which runs full canonical text at production scale, or the coverage
    claim is overstated.
    """
    stored = [(r["title"], r["canonical"], r["anchor"]) for r in rows]
    ids = [r["id"] for r in rows]
    known = set(ids)
    # Only pairs whose survivor is inside this corpus can be judged; a survivor
    # that aged out would look like a regression that is really a missing row.
    usable = [p for p in pairs if p["survivor"] in known][:limit]
    out = {}
    started = time.perf_counter()
    for i, pair in enumerate(usable):
        idx = find_content_duplicate(stored, pair["title"], pair["title"])
        out[f"{pair['survivor']}#{i}"] = ids[idx] if idx is not None else None
    return out, time.perf_counter() - started, len(usable)


def replay(corpus_path: str, out_path: str | None, against: str | None,
           stored_n: int, candidate_n: int, mode: str = "events") -> int:
    data = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    rows = data["rows"]

    if mode == "pairs":
        verdicts, seconds, n = _replay_pairs(rows, data.get("pairs", []),
                                             candidate_n)
        hits = sum(1 for k, v in verdicts.items()
                   if v is not None and v == k.split("#")[0])
        other = sum(1 for v in verdicts.values() if v is not None) - hits
        print(f"{n} recorded duplicates x {len(rows)} stored, {seconds:.1f}s — "
              f"{hits} re-matched their survivor, {other} matched a different "
              f"event, {n - hits - other} no longer match anything")
        return _finish(verdicts, out_path, against)

    # The dump is ordered newest-first, and production compares TODAY's candidates
    # against an OLDER window — so candidates come off the front and the stored
    # side sits directly behind them. Disjoint, so nothing matches its own row,
    # but adjacent in time, because that is where duplicates actually live.
    #
    # The first version had this backwards, taking candidates from the oldest end
    # against the newest stored events. The two were then 14 days apart and the
    # replay found 8 duplicates in 800 candidates against production's ~32%: it
    # exercised the rejection path almost exclusively and would have reported
    # IDENTICAL while proving nothing about matching, which is the whole question.
    candidates = rows[:candidate_n]
    stored = rows[candidate_n:candidate_n + stored_n]
    verdicts, seconds = _verdicts(stored, candidates)

    dupes = sum(1 for v in verdicts.values() if v is not None)
    comparisons = len(stored) * len(candidates)
    print(f"{len(candidates)} candidates x {len(stored)} stored "
          f"= {comparisons:,} comparisons, {dupes} duplicates, {seconds:.1f}s")

    return _finish(verdicts, out_path, against)


def _finish(verdicts: dict, out_path: str | None, against: str | None) -> int:
    if out_path:
        Path(out_path).write_text(json.dumps(verdicts, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"baseline written → {out_path}")

    if not against:
        return 0

    base = json.loads(Path(against).read_text(encoding="utf-8"))
    shared = set(base) & set(verdicts)
    if len(shared) != len(base) or len(shared) != len(verdicts):
        print(f"WARNING: baseline covers {len(base)} candidates, this run "
              f"{len(verdicts)}; comparing the {len(shared)} in both",
              file=sys.stderr)
    moved = [(k, base[k], verdicts[k]) for k in sorted(shared)
             if base[k] != verdicts[k]]
    if not moved:
        print(f"IDENTICAL — {len(shared)} verdicts unchanged")
        return 0
    print(f"CHANGED — {len(moved)} of {len(shared)} verdicts moved")
    for cand_id, was, now in moved[:40]:
        print(f"  {cand_id}: {was or 'no-match'} → {now or 'no-match'}")
    if len(moved) > 40:
        print(f"  ... and {len(moved) - 40} more")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="read the corpus from the database")
    d.add_argument("--out", default="dedup_corpus.json")
    d.add_argument("--limit", type=int, default=CORPUS_LIMIT)
    d.add_argument("--days", type=int, default=14)

    r = sub.add_parser("replay", help="run the matcher over a dumped corpus")
    r.add_argument("corpus")
    r.add_argument("--out", help="write verdicts here as a baseline")
    r.add_argument("--against", help="compare verdicts against this baseline")
    r.add_argument("--stored", type=int, default=600)
    r.add_argument("--candidates", type=int, default=800)
    r.add_argument("--mode", choices=("events", "pairs"), default="events",
                   help="events: real non-duplicates at production scale, which "
                        "exercises the REJECTION path and its cost. pairs: "
                        "duplicates production actually dropped, which is the only "
                        "mode with a right answer. Run both.")

    args = ap.parse_args()
    if args.cmd == "dump":
        return dump(args.out, args.limit, args.days)
    return replay(args.corpus, args.out, args.against, args.stored,
                  args.candidates, args.mode)


if __name__ == "__main__":
    sys.exit(main())
