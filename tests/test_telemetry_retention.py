"""Retention for the per-call LLM telemetry.

Measured 2026-08-17: system_telemetry was 65 MB — the second largest table in a 367 MB
database on a 500 MB tier — and 50 MB of that was 44,680 'llm_call' rows kept since
9 May, one per LLM call, growing ~1.2 MB/day with nothing reading them. The aggregate
types in the same table (pipeline_run, pass_a..pass_f, archive_manifest) are 3 MB in
total and are the run history, so they are kept.

These tests pin the scope, because a purge that widened to the whole table would delete
the only record of what every past run did.
"""

from unittest.mock import MagicMock, patch

from src.pipeline import pass_f_archive as pf


def _db(rowcount=5000):
    db = MagicMock()
    db.execute.return_value.rowcount = rowcount
    return db


def _sql(db) -> str:
    return " ".join(str(c.args[0]) for c in db.execute.call_args_list if c.args)


class TestPurgeScope:
    def test_only_llm_call_rows(self):
        db = _db()
        pf.purge_expired_telemetry(db)
        assert "event_type = 'llm_call'" in _sql(db)

    def test_only_system_telemetry_table(self):
        db = _db()
        pf.purge_expired_telemetry(db)
        sql = _sql(db)
        assert "DELETE FROM system_telemetry" in sql
        assert "events" not in sql

    def test_ages_on_timestamp(self):
        db = _db()
        pf.purge_expired_telemetry(db)
        assert "timestamp <" in _sql(db)

    def test_batched_larger_than_the_events_purge(self):
        """~35,000 rows of backlog: at the events batch size this would take a week."""
        db = _db()
        pf.purge_expired_telemetry(db)
        assert db.execute.call_args.args[1][1] == pf.TELEMETRY_BATCH_SIZE
        assert pf.TELEMETRY_BATCH_SIZE > pf.BATCH_SIZE

    def test_returns_rows_removed(self):
        assert pf.purge_expired_telemetry(_db(rowcount=1234)) == 1234


class TestPurgeIsHousekeeping:
    def test_db_error_does_not_raise(self):
        db = _db()
        db.execute.side_effect = RuntimeError("connection reset")
        assert pf.purge_expired_telemetry(db) == 0

    def test_disabled_by_zero(self):
        db = _db()
        with patch.object(pf, "TELEMETRY_RETENTION_DAYS", 0):
            assert pf.purge_expired_telemetry(db) == 0
        db.execute.assert_not_called()

    def test_window_covers_a_fortnight_investigation(self):
        """Model and quota regressions in this project are measured over 14 days."""
        assert pf.TELEMETRY_RETENTION_DAYS >= 14


class TestWiredIntoPassF:
    def test_runs_when_there_is_nothing_to_archive(self):
        db = _db()
        with patch.object(pf, "get_archivable_events", return_value=[]), \
             patch.object(pf, "purge_expired_archived", return_value=0), \
             patch.object(pf, "purge_expired_telemetry", return_value=5000) as purge:
            stats = pf.run_pass_f(db)
        purge.assert_called_once()
        assert stats["telemetry_purged"] == 5000
