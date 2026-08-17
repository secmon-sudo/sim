"""Retention for status='archived' noise.

Nothing removed these rows before. get_archivable_events reads status='reconciled'
only, so the classifier's rejects and the prescreen's zero-signal drops were neither
exported to cold storage nor deleted: 26,484 of 55,226 rows (48%) on 2026-08-17, with
the database at 367 MB of the 500 MB free tier and growing 4.4 MB/day — about 31 days
of runway.

These tests pin the three things the purge must NOT do, because each of them would be
silent: delete a row that still pages, delete the corpus, or skip the purge on a run
where the export had nothing to do.
"""

from unittest.mock import MagicMock, patch

from src.pipeline import pass_f_archive as pf


def _db(rowcount=7):
    db = MagicMock()
    db.transaction.return_value.__enter__ = lambda s: None
    db.transaction.return_value.__exit__ = lambda s, *a: False
    db.execute.return_value.rowcount = rowcount
    return db


def _purge_sql(db) -> str:
    return " ".join(str(c.args[0]) for c in db.execute.call_args_list if c.args)


class TestPurgeScope:
    def test_only_archived_status(self):
        db = _db()
        pf.purge_expired_archived(db)
        assert "status = 'archived'" in _purge_sql(db)

    def test_never_touches_paged_rows(self):
        """An archived row that paged is the only record the card was sent."""
        db = _db()
        pf.purge_expired_archived(db)
        assert "alert_tier IS NULL" in _purge_sql(db)

    def test_ages_on_ingested_at_not_occurred_at(self):
        """occurred_at_est is the classifier's guess — 60% fabricated, measured
        2026-08-13. Ageing on it would delete rows that arrived yesterday."""
        db = _db()
        pf.purge_expired_archived(db)
        sql = _purge_sql(db)
        assert "ingested_at" in sql
        assert "occurred_at_est" not in sql

    def test_batched(self):
        db = _db()
        pf.purge_expired_archived(db)
        assert db.execute.call_args.args[1][1] == pf.BATCH_SIZE

    def test_returns_rows_removed(self):
        assert pf.purge_expired_archived(_db(rowcount=42)) == 42


class TestPurgeIsHousekeeping:
    def test_db_error_does_not_raise(self):
        """Retention must never be the reason a run reports failure."""
        db = _db()
        db.execute.side_effect = RuntimeError("connection reset")
        assert pf.purge_expired_archived(db) == 0

    def test_disabled_by_zero(self):
        db = _db()
        with patch.object(pf, "ARCHIVED_RETENTION_DAYS", 0):
            assert pf.purge_expired_archived(db) == 0
        db.execute.assert_not_called()


class TestRetentionWindow:
    def test_wider_than_every_consumer_of_archived_rows(self):
        """The widest reader is the weekly forecast at 7 days (no status filter).

        Pass A's content dedup looks back max_article_age_days and the SITREP window
        is hours, not days; storyline linking excludes archived entirely. If someone
        widens a consumer past the retention window, archived rows start vanishing
        from underneath it — so the relationship is asserted, not just documented.
        """
        import json
        from pathlib import Path
        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "config" / "settings.json").read_text())
        weekly_lookback_days = 7
        dedup_lookback = cfg["ingestion"]["max_article_age_days"]
        assert pf.ARCHIVED_RETENTION_DAYS > weekly_lookback_days * 2
        assert pf.ARCHIVED_RETENTION_DAYS > dedup_lookback


class TestPurgeRunsBeforeTheEarlyReturn:
    def test_purge_happens_with_nothing_to_archive(self):
        """run_pass_f returns early when there is nothing to export. The purge sits
        before that return on purpose — a quiet run is the cheapest time to do it."""
        db = _db()
        with patch.object(pf, "get_archivable_events", return_value=[]), \
             patch.object(pf, "purge_expired_archived", return_value=5) as purge:
            stats = pf.run_pass_f(db)
        purge.assert_called_once()
        assert stats["archived_purged"] == 5
