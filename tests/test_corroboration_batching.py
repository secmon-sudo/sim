"""Corroborations are written once per run, not once per duplicate.

Measured 2026-09-03 over 30 runs, this write was 90 s/run — 20% of Pass A and 11%
of the whole pipeline — spent as ~730 separate UPDATEs against a remote pooler. The
round trips were the cost, not the work; the same finding, and the same fix, as
load_domain_penalties (210 s/run as a per-item query).

What must not change in the move: the refusals (an outlet republishing itself, a
carrier that is not a witness) still run per duplicate, and the statements still
execute one per duplicate in order, because the idempotency guard only works if the
previous append is already visible.
"""

from unittest.mock import MagicMock

from src.pipeline.pass_a_ingest import _corroboration_params, _flush_corroborations


def _cursor(rowcounts):
    """A cursor whose executemany yields one result set per rowcount given."""
    counts = list(rowcounts)
    cur = MagicMock()
    state = {"i": 0}

    def _rowcount():
        return counts[state["i"]] if state["i"] < len(counts) else -1

    def _nextset():
        state["i"] += 1
        return True if state["i"] < len(counts) else None

    type(cur).rowcount = property(lambda s: _rowcount())
    cur.nextset.side_effect = _nextset
    return cur


def _db(cur):
    db = MagicMock()
    db.transaction.return_value.__enter__ = lambda s: None
    db.transaction.return_value.__exit__ = lambda s, *a: False
    db.cursor.return_value.__enter__ = lambda s: cur
    db.cursor.return_value.__exit__ = lambda s, *a: False
    return db


class TestParamsApplyTheSameRefusals:
    def test_valid_duplicate_produces_params(self):
        params = _corroboration_params("evt-1", "reuters.com", "bbc.co.uk",
                                       "https://bbc.co.uk/x", "Strike reported")
        assert params is not None
        assert params[1] == "evt-1"

    def test_self_republish_refused(self):
        """An outlet republishing itself proves nothing — it never reaches the batch."""
        assert _corroboration_params("evt-1", "www.reuters.com", "reuters.com",
                                     "https://reuters.com/x", "Strike") is None

    def test_carrier_refused(self):
        """A carrier is not a witness: Yahoo redistributes one newsroom's filing."""
        assert _corroboration_params("evt-1", "reuters.com", "news.yahoo.com",
                                     "https://news.yahoo.com/x", "Strike") is None

    def test_missing_event_or_domain_refused(self):
        assert _corroboration_params(None, "reuters.com", "bbc.co.uk",
                                     "https://bbc.co.uk/x", "Strike") is None
        assert _corroboration_params("evt-1", "reuters.com", "",
                                     "https://bbc.co.uk/x", "Strike") is None


class TestFlush:
    def test_empty_batch_touches_no_connection(self):
        db = MagicMock()
        assert _flush_corroborations(db, []) == 0
        db.cursor.assert_not_called()

    def test_one_round_trip_for_the_whole_run(self):
        """The point of the change: N duplicates, ONE executemany."""
        cur = _cursor([1, 1, 1])
        db = _db(cur)
        pending = [("e", "evt-1", 8, "p")] * 3
        _flush_corroborations(db, pending)
        cur.executemany.assert_called_once()
        assert cur.executemany.call_args.args[1] == pending

    def test_counts_only_rows_actually_appended(self):
        """A duplicate whose domain is already credited updates nothing (rowcount 0),
        and must not be counted as a new corroboration."""
        cur = _cursor([1, 0, 1])
        assert _flush_corroborations(_db(cur), [("e", "evt-1", 8, "p")] * 3) == 2

    def test_statements_run_in_order_one_per_duplicate(self):
        """Ordering is load-bearing: the `NOT @> probe` guard makes the second
        duplicate from a credited domain a no-op only if the first is visible. A
        single UPDATE ... FROM (VALUES ...) would be one round trip too, and would
        silently drop every duplicate after the first for the same event."""
        cur = _cursor([1, 0])
        pending = [("first", "evt-1", 8, "p"), ("second", "evt-1", 8, "p")]
        _flush_corroborations(_db(cur), pending)
        sent = cur.executemany.call_args.args[1]
        assert [p[0] for p in sent] == ["first", "second"]

    def test_failure_never_breaks_the_run(self):
        """Pre-migration DBs lack the column; corroboration is a bonus signal."""
        db = MagicMock()
        db.transaction.side_effect = RuntimeError("no such column")
        assert _flush_corroborations(db, [("e", "evt-1", 8, "p")]) == 0
