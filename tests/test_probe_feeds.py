"""Tests for the candidate-feed probe.

The probe exists because a source list said "verified" about feeds that were
403ing, refusing connections, or serving items from 2013. Its whole value is
telling those apart, so the verdict function is where the tests are.
"""

from datetime import datetime, timedelta, timezone

from scripts import probe_feeds as pf

NOW = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


def _raw(status=200, items=10, newest=None):
    return {"status": status, "items": items,
            "newest": newest if newest is not None else NOW - timedelta(hours=6),
            "error": ""}


class TestVerdict:
    def test_fresh_items_with_a_keyword_hit_is_add(self):
        assert pf.verdict(_raw(), fresh=12, security=4, now=NOW) == "ADD"

    def test_fresh_items_with_no_hit_is_thin_not_dead(self):
        """A French state wire returns twenty current items and none of them pass
        an English keyword gate. The feed is fine; the gate is the problem, and
        calling that DEAD would send someone to fix the wrong thing."""
        assert pf.verdict(_raw(items=20), fresh=20, security=0, now=NOW) == "THIN"

    def test_a_feed_whose_newest_item_is_years_old_is_stale(self):
        raw = _raw(items=4, newest=datetime(2013, 12, 31, tzinfo=timezone.utc))
        assert pf.verdict(raw, fresh=0, security=0, now=NOW) == "STALE"

    def test_items_that_all_miss_the_ingest_window_are_stale(self):
        """Served fine, dated inside the staleness bound, but nothing recent
        enough for Pass A to take."""
        raw = _raw(items=10, newest=NOW - timedelta(days=9))
        assert pf.verdict(raw, fresh=0, security=0, now=NOW) == "STALE"

    def test_bot_protection_is_its_own_verdict(self):
        """403 is about the IP, not the publisher. It has to survive as a
        separate word or the feed gets dropped for the wrong reason."""
        for status in (401, 403, 429):
            assert pf.verdict(_raw(status=status, items=0), 0, 0, NOW) == "BLOCKED"

    def test_no_connection_is_dead(self):
        assert pf.verdict(_raw(status=0, items=0), 0, 0, NOW) == "DEAD"

    def test_a_200_with_no_items_is_dead(self):
        """Addis Admass answers 200 with a 961-byte document and no entries."""
        assert pf.verdict(_raw(items=0), 0, 0, NOW) == "DEAD"

    def test_blocked_wins_over_missing_items(self):
        """A 403 body has no items either; the status is the cause worth naming."""
        assert pf.verdict(_raw(status=403, items=0), 0, 0, NOW) == "BLOCKED"


class TestDateParsing:
    def test_rfc_822(self):
        parsed = pf._parse_date("Sun, 06 Sep 2026 03:00:00 +0000")
        assert parsed and parsed.year == 2026 and parsed.month == 9

    def test_iso_with_zulu(self):
        parsed = pf._parse_date("2026-09-10T14:22:00Z")
        assert parsed and parsed.day == 10

    def test_a_naive_stamp_is_read_as_utc(self):
        """Comparing a naive stamp against an aware now() raises, and a probe that
        raises on one feed reports nothing about the rest."""
        parsed = pf._parse_date("2026-09-10T14:22:00")
        assert parsed and parsed.tzinfo is not None

    def test_garbage_is_none_not_an_exception(self):
        assert pf._parse_date("whenever") is None
