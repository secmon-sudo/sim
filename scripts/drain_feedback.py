#!/usr/bin/env python3
"""Collect pending alert-card button presses. Standalone entry point.

The pipeline drains at the start of every run, but runs are ~3h apart, and a button
that shows no acknowledgement for three hours stops being pressed. This is the same
drain on a short cron, so the card's keyboard flips over within minutes.

Both callers are idempotent against each other (see feedback_drain's module docstring):
whichever one gets the window writes the rows, the other finds it empty.
"""

import argparse
import json
import logging
import sys

from src.services.feedback_drain import drain_feedback
from src.services.supabase_client import get_connection, put_connection

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print stats as JSON")
    args = parser.parse_args()

    conn = get_connection()
    try:
        stats = drain_feedback(conn)
    finally:
        put_connection(conn)

    if args.json:
        print(json.dumps(stats))
    else:
        print(f"fetched={stats['fetched']} recorded={stats['recorded']} "
              f"ignored={stats['ignored']} error={stats['error']}")
    # A Telegram hiccup is not a failed job — the presses stay in the 24h window and
    # the next tick collects them, and a workflow that goes red every time a network
    # call blips is a workflow whose red stops meaning anything. Only the permanent
    # failures (missing/revoked token, a webhook that has taken over the stream) go red.
    return 1 if stats.get("fatal") else 0


if __name__ == "__main__":
    sys.exit(main())
