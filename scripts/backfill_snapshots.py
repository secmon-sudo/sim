"""
SIM — One-off snapshot backfill.

Between 2026-07-06 and the get_run_events fix, per-run snapshots exported raw
(unclassified) events: the Pass C backlog exceeded its per-run cap, so events
were classified 1-2 runs after ingestion and every snapshot row had a NULL
storyline_id — which the storyboard worker silently drops. Those events were
classified later in the DB but never re-exported, leaving a gap in D1.

This script re-exports all classified events updated since a cutoff as one or
more backfill JSONL files to R2. The worker cron picks them up automatically
(new filenames are absent from its processed_files ledger). Re-ingesting rows
that are already in D1 is safe: the worker inserts with ON CONFLICT DO NOTHING.

Usage (needs DATABASE_URL + R2_* env vars, same as the pipeline):
    python -m scripts.backfill_snapshots [--since 2026-07-06] [--dry-run]
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import psycopg

from src.pipeline.pass_f_archive import (
    _EVENT_COLUMNS,
    _rows_to_event_dicts,
    generate_jsonl_and_hash,
    upload_to_cloudflare_r2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sim.backfill")

# Keep each file well under the worker's per-cron time budget.
CHUNK_SIZE = 500


def fetch_classified_events(db_url: str, since: datetime) -> list[dict]:
    query = f"""
        SELECT {", ".join(_EVENT_COLUMNS)}
        FROM events
        WHERE updated_at >= %s
          AND storyline_id IS NOT NULL
        ORDER BY updated_at ASC
    """
    with psycopg.connect(db_url) as conn:
        rows = conn.execute(query, (since,)).fetchall()
    return _rows_to_event_dicts(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-07-06", help="Cutoff date (YYYY-MM-DD, UTC)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and report only, no upload")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL is not set")
        return 1

    since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    events = fetch_classified_events(db_url, since)
    logger.info("Fetched %d classified events updated since %s", len(events), args.since)
    if not events:
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    failures = 0
    for i in range(0, len(events), CHUNK_SIZE):
        chunk = events[i : i + CHUNK_SIZE]
        content, manifest_hash = generate_jsonl_and_hash(chunk)
        filename = f"sim_archive_backfill_{stamp}_p{i // CHUNK_SIZE + 1}_{len(chunk)}ev.jsonl"
        if args.dry_run:
            logger.info("[dry-run] Would upload %s (%d bytes, sha256=%s)", filename, len(content), manifest_hash[:12])
            continue
        if upload_to_cloudflare_r2(content, filename):
            logger.info("Uploaded %s (%d events)", filename, len(chunk))
        else:
            failures += 1
            logger.error("Upload FAILED for %s", filename)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
