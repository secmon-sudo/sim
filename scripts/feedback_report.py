#!/usr/bin/env python3
"""What the analyst said about the cards, sliced the ways a threshold argument needs.

Every gate in this pipeline was tuned against a proxy. This is the one report that is
not: it reads alert_feedback, which contains only button presses, and puts the noise
rate next to the volume it came from.

Read the RESPONSE RATE first. A 4% noise rate over three presses is not a measurement,
and the temptation to act on one is exactly why the coverage line is printed above the
verdicts rather than in a footnote.

  python -m scripts.feedback_report --days 14
"""

import argparse
import logging
import sys

from src.services.supabase_client import get_connection, put_connection

logging.basicConfig(level=logging.WARNING)

# Cards sent, by tier, as counted by Pass D's own telemetry. dispatch_result=="sent" is
# what that counter records — not events.alert_tier, which is not a record that a card
# was sent (ced1565) — so it is the honest denominator for a response rate.
_SENT_SQL = """
    SELECT COALESCE(SUM((value_json -> 'alerts_generated' ->> 'CRITICAL')::int), 0),
           COALESCE(SUM((value_json -> 'alerts_generated' ->> 'ALERT')::int), 0),
           COALESCE(SUM((value_json -> 'alerts_generated' ->> 'WATCH')::int), 0)
      FROM system_telemetry
     WHERE event_type = 'pass_d'
       AND timestamp > NOW() - make_interval(days => %s)
"""


def _rows(conn, sql, params):
    return conn.execute(sql, params).fetchall()


def _verdict_table(rows, label, min_n=1):
    """rows: (key, useful, noise). Prints noise share, widest first."""
    rows = [r for r in rows if (r[1] + r[2]) >= min_n]
    if not rows:
        print(f"  (no {label} with at least {min_n} press)")
        return
    print(f"\n  {label:<28} {'presses':>8} {'useful':>8} {'noise':>8} {'noise %':>9}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9}")
    for key, useful, noise in sorted(rows, key=lambda r: -(r[1] + r[2])):
        total = useful + noise
        print(f"  {str(key or '—')[:28]:<28} {total:>8} {useful:>8} {noise:>8} "
              f"{noise / total * 100:>8.0f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=14, help="lookback window (default 14)")
    parser.add_argument("--min-n", type=int, default=3,
                        help="hide slices with fewer presses than this (default 3)")
    args = parser.parse_args()

    conn = get_connection()
    try:
        totals = conn.execute(
            """SELECT COUNT(*),
                      COUNT(*) FILTER (WHERE verdict = 'useful'),
                      COUNT(*) FILTER (WHERE verdict = 'noise'),
                      COUNT(DISTINCT tg_user_id)
                 FROM alert_feedback
                WHERE created_at > NOW() - make_interval(days => %s)""",
            (args.days,),
        ).fetchone()
        presses, useful, noise, voters = totals or (0, 0, 0, 0)

        print(f"\n=== Alert feedback — last {args.days} days ===")
        if not presses:
            print("\n  No presses recorded.\n"
                  "  Either nobody is using the buttons or the drain is not running —\n"
                  "  check telegram_update_cursor.updated_at before reading anything\n"
                  "  into the silence.\n")
            return 0

        sent_c, sent_a, sent_w = conn.execute(_SENT_SQL, (args.days,)).fetchone()
        sent_total = (sent_c or 0) + (sent_a or 0) + (sent_w or 0)

        print(f"\n  cards sent      {sent_total:>6}   (C {sent_c} · A {sent_a} · W {sent_w})")
        print(f"  presses         {presses:>6}   from {voters} analyst(s)")
        if sent_total:
            print(f"  response rate   {presses / sent_total * 100:>5.1f}%   "
                  "← everything below is only as good as this")
        print(f"  overall noise   {noise / presses * 100:>5.0f}%   "
              f"({noise} noise / {useful} useful)")

        by_tier = _rows(conn, """
            SELECT card_tier,
                   COUNT(*) FILTER (WHERE verdict = 'useful'),
                   COUNT(*) FILTER (WHERE verdict = 'noise')
              FROM alert_feedback
             WHERE created_at > NOW() - make_interval(days => %s)
             GROUP BY card_tier""", (args.days,))
        _verdict_table(by_tier, "tier")

        by_type = _rows(conn, """
            SELECT event_type,
                   COUNT(*) FILTER (WHERE verdict = 'useful'),
                   COUNT(*) FILTER (WHERE verdict = 'noise')
              FROM alert_feedback
             WHERE created_at > NOW() - make_interval(days => %s)
             GROUP BY event_type""", (args.days,))
        _verdict_table(by_type, "event type", args.min_n)

        by_country = _rows(conn, """
            SELECT country_iso,
                   COUNT(*) FILTER (WHERE verdict = 'useful'),
                   COUNT(*) FILTER (WHERE verdict = 'noise')
              FROM alert_feedback
             WHERE created_at > NOW() - make_interval(days => %s)
             GROUP BY country_iso""", (args.days,))
        _verdict_table(by_country, "country", args.min_n)

        by_domain = _rows(conn, """
            SELECT source_domain,
                   COUNT(*) FILTER (WHERE verdict = 'useful'),
                   COUNT(*) FILTER (WHERE verdict = 'noise')
              FROM alert_feedback
             WHERE created_at > NOW() - make_interval(days => %s)
             GROUP BY source_domain""", (args.days,))
        _verdict_table(by_domain, "source domain", args.min_n)

        # Severity is the lever most likely to be reached for, so show whether the
        # verdicts actually separate along it. Four failed attempts at a domain-quality
        # signal (see the domain-penalty notes) say a plausible lever can be inert.
        bands = _rows(conn, """
            SELECT CASE WHEN severity_score IS NULL THEN 'unknown'
                        WHEN severity_score >= 90 THEN '90-100'
                        WHEN severity_score >= 75 THEN '75-89'
                        WHEN severity_score >= 60 THEN '60-74'
                        ELSE '<60' END,
                   COUNT(*) FILTER (WHERE verdict = 'useful'),
                   COUNT(*) FILTER (WHERE verdict = 'noise')
              FROM alert_feedback
             WHERE created_at > NOW() - make_interval(days => %s)
             GROUP BY 1""", (args.days,))
        _verdict_table(bands, "severity band")
        print()
    finally:
        put_connection(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
