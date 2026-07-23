"""Tests for Pass F — cold storage archive and the per-run snapshot.

Pass F is the only place in the pipeline that DELETES events. The archive
upload is the sole evidence the data ever existed, so the property that matters
more than any other is: no delete unless the upload succeeded. Everything else
here supports that — the JSONL must be reconstructible, and the manifest that
points at the archive must land in the same transaction as the deletes.
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.pass_f_archive import (
    _EVENT_COLUMNS,
    _rows_to_event_dicts,
    generate_jsonl_and_hash,
    run_pass_f,
    run_run_snapshot,
)


def _row(**overrides):
    """A DB row tuple in _EVENT_COLUMNS order."""
    values = {
        "id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
        "source_url": "https://example.com/a",
        "source_title": "Blast at port",
        "canonical_text": "blast at port",
        "event_type": "explosion",
        "alert_tier": "ALERT",
        "severity_score": 70,
        "anchor_name_norm": "BND",
        "country_iso": "IR",
        "occurred_at_est": datetime(2026, 7, 20, 3, 30, tzinfo=timezone.utc),
        "ingested_at": datetime(2026, 7, 20, 4, 0, tzinfo=timezone.utc),
        "llm_parsed_output": '{"aviation_impact": "direct"}',
        "storyline_id": uuid.UUID("22222222-2222-2222-2222-222222222222"),
    }
    values.update(overrides)
    return tuple(values[c] for c in _EVENT_COLUMNS)


class TestRowSerialization:
    def test_datetimes_become_isoformat(self):
        event = _rows_to_event_dicts([_row()])[0]
        assert event["occurred_at_est"] == "2026-07-20T03:30:00+00:00"
        assert event["ingested_at"] == "2026-07-20T04:00:00+00:00"

    def test_null_datetimes_survive(self):
        # occurred_at_est is nullable — an event archived before reconciliation
        # must not crash the export.
        event = _rows_to_event_dicts([_row(occurred_at_est=None, ingested_at=None)])[0]
        assert event["occurred_at_est"] is None
        assert event["ingested_at"] is None

    def test_uuids_become_strings(self):
        event = _rows_to_event_dicts([_row()])[0]
        assert event["id"] == "11111111-1111-1111-1111-111111111111"
        assert event["storyline_id"] == "22222222-2222-2222-2222-222222222222"

    def test_json_column_is_parsed_not_double_encoded(self):
        # llm_parsed_output arrives as TEXT from some drivers; leaving it a
        # string would nest a JSON document inside a JSON string in the archive.
        event = _rows_to_event_dicts([_row()])[0]
        assert event["llm_parsed_output"] == {"aviation_impact": "direct"}

    def test_malformed_json_becomes_empty_dict(self):
        event = _rows_to_event_dicts([_row(llm_parsed_output="{not json")])[0]
        assert event["llm_parsed_output"] == {}

    def test_null_json_becomes_empty_dict(self):
        event = _rows_to_event_dicts([_row(llm_parsed_output=None)])[0]
        assert event["llm_parsed_output"] == {}

    def test_null_storyline_id_stays_null(self):
        event = _rows_to_event_dicts([_row(storyline_id=None)])[0]
        assert event["storyline_id"] is None


class TestJsonlAndHash:
    def test_one_line_per_event_and_roundtrips(self):
        events = _rows_to_event_dicts([_row(), _row(source_url="https://example.com/b")])
        content, _ = generate_jsonl_and_hash(events)
        lines = content.decode("utf-8").splitlines()
        assert len(lines) == 2
        assert [json.loads(ln)["source_url"] for ln in lines] == [
            "https://example.com/a", "https://example.com/b",
        ]

    def test_hash_is_of_the_uploaded_bytes(self):
        # The manifest hash is the only integrity check on the archive, so it
        # must cover exactly the bytes that get uploaded.
        import hashlib
        content, digest = generate_jsonl_and_hash(_rows_to_event_dicts([_row()]))
        assert digest == hashlib.sha256(content).hexdigest()

    def test_empty_input_produces_empty_payload(self):
        content, digest = generate_jsonl_and_hash([])
        assert content == b""
        assert len(digest) == 64


class TestArchiveDeletionSafety:
    """No delete without a successful archive upload."""

    @staticmethod
    def _db():
        db = MagicMock()
        db.transaction.return_value.__enter__ = lambda s: None
        db.transaction.return_value.__exit__ = lambda s, *a: False
        return db

    @staticmethod
    def _executed_sql(db):
        return " ".join(str(c.args[0]) for c in db.execute.call_args_list if c.args)

    def test_telegram_failure_leaves_events_in_place(self):
        db = self._db()
        with patch("src.pipeline.pass_f_archive.get_archivable_events",
                   return_value=_rows_to_event_dicts([_row()])), \
             patch("src.pipeline.pass_f_archive.upload_to_cloudflare_r2", return_value=True), \
             patch("src.pipeline.pass_f_archive.upload_to_telegram", return_value=None):
            stats = run_pass_f(db)
        assert stats["error"] == "Telegram upload failed"
        assert stats["events_archived"] == 0
        assert "DELETE" not in self._executed_sql(db).upper()

    def test_telegram_not_ok_response_leaves_events_in_place(self):
        db = self._db()
        with patch("src.pipeline.pass_f_archive.get_archivable_events",
                   return_value=_rows_to_event_dicts([_row()])), \
             patch("src.pipeline.pass_f_archive.upload_to_cloudflare_r2", return_value=True), \
             patch("src.pipeline.pass_f_archive.upload_to_telegram",
                   return_value={"ok": False, "description": "chat not found"}):
            stats = run_pass_f(db)
        assert stats["events_archived"] == 0
        assert "DELETE" not in self._executed_sql(db).upper()

    def test_successful_upload_deletes_and_records_manifest(self):
        db = self._db()
        with patch("src.pipeline.pass_f_archive.get_archivable_events",
                   return_value=_rows_to_event_dicts([_row()])), \
             patch("src.pipeline.pass_f_archive.upload_to_cloudflare_r2", return_value=True), \
             patch("src.pipeline.pass_f_archive.upload_to_telegram",
                   return_value={"ok": True, "result": {"message_id": 4242}}):
            stats = run_pass_f(db)
        sql = self._executed_sql(db)
        assert stats["events_archived"] == 1
        assert stats["telegram_message_id"] == 4242
        assert "DELETE FROM events" in sql
        assert "archive_manifest" in sql

    def test_manifest_records_the_ids_it_deleted(self):
        # The manifest is the only pointer from the DB to the archived rows.
        db = self._db()
        with patch("src.pipeline.pass_f_archive.get_archivable_events",
                   return_value=_rows_to_event_dicts([_row()])), \
             patch("src.pipeline.pass_f_archive.upload_to_cloudflare_r2", return_value=True), \
             patch("src.pipeline.pass_f_archive.upload_to_telegram",
                   return_value={"ok": True, "result": {"message_id": 1}}):
            run_pass_f(db)
        manifest = next(
            json.loads(c.args[1][0]) for c in db.execute.call_args_list
            if c.args and "archive_manifest" in str(c.args[0])
        )
        assert manifest["archived_event_ids"] == ["11111111-1111-1111-1111-111111111111"]
        assert manifest["event_count"] == 1

    def test_r2_failure_alone_does_not_block_archival(self):
        # Telegram is the durability gate; R2 is a convenience copy.
        db = self._db()
        with patch("src.pipeline.pass_f_archive.get_archivable_events",
                   return_value=_rows_to_event_dicts([_row()])), \
             patch("src.pipeline.pass_f_archive.upload_to_cloudflare_r2", return_value=False), \
             patch("src.pipeline.pass_f_archive.upload_to_telegram",
                   return_value={"ok": True, "result": {"message_id": 7}}):
            stats = run_pass_f(db)
        assert stats["events_archived"] == 1
        assert stats["r2_uploaded"] is False

    def test_nothing_to_archive_uploads_nothing(self):
        db = self._db()
        with patch("src.pipeline.pass_f_archive.get_archivable_events", return_value=[]), \
             patch("src.pipeline.pass_f_archive.upload_to_telegram") as tg:
            stats = run_pass_f(db)
        tg.assert_not_called()
        assert stats["events_archived"] == 0
        assert stats["error"] is None

    def test_db_failure_after_upload_is_reported_not_swallowed(self):
        # Upload succeeded but the delete transaction blew up: the run must
        # surface an error, otherwise the same events are re-archived forever
        # with no signal that anything is wrong.
        db = self._db()
        db.transaction.side_effect = RuntimeError("deadlock detected")
        with patch("src.pipeline.pass_f_archive.get_archivable_events",
                   return_value=_rows_to_event_dicts([_row()])), \
             patch("src.pipeline.pass_f_archive.upload_to_cloudflare_r2", return_value=True), \
             patch("src.pipeline.pass_f_archive.upload_to_telegram",
                   return_value={"ok": True, "result": {"message_id": 9}}):
            stats = run_pass_f(db)
        assert stats["events_archived"] == 0
        assert "deadlock detected" in stats["error"]


class TestRunSnapshot:
    """The per-run snapshot exports but must never delete."""

    def test_snapshot_never_deletes(self):
        db = MagicMock()
        with patch("src.pipeline.pass_f_archive.get_run_events",
                   return_value=_rows_to_event_dicts([_row()])), \
             patch("src.pipeline.pass_f_archive.upload_to_cloudflare_r2", return_value=True), \
             patch("src.pipeline.pass_f_archive.upload_to_telegram",
                   return_value={"ok": True, "result": {"message_id": 3}}):
            stats = run_run_snapshot(db, datetime(2026, 7, 23, tzinfo=timezone.utc))
        sql = " ".join(str(c.args[0]) for c in db.execute.call_args_list if c.args)
        assert "DELETE" not in sql.upper()
        assert stats["events"] == 1
        assert stats["telegram_message_id"] == 3

    def test_empty_run_skips_upload(self):
        db = MagicMock()
        with patch("src.pipeline.pass_f_archive.get_run_events", return_value=[]), \
             patch("src.pipeline.pass_f_archive.upload_to_telegram") as tg:
            stats = run_run_snapshot(db, datetime(2026, 7, 23, tzinfo=timezone.utc))
        tg.assert_not_called()
        assert stats["events"] == 0

    def test_telegram_failure_is_reported_but_not_fatal(self):
        db = MagicMock()
        with patch("src.pipeline.pass_f_archive.get_run_events",
                   return_value=_rows_to_event_dicts([_row()])), \
             patch("src.pipeline.pass_f_archive.upload_to_cloudflare_r2", return_value=True), \
             patch("src.pipeline.pass_f_archive.upload_to_telegram", return_value={"ok": False}):
            stats = run_run_snapshot(db, datetime(2026, 7, 23, tzinfo=timezone.utc))
        assert stats["error"] == "Telegram snapshot upload failed"
        assert stats["r2_uploaded"] is True


class TestUploadCredentialGuards:
    def test_telegram_upload_skipped_without_credentials(self, monkeypatch):
        from src.pipeline.pass_f_archive import upload_to_telegram
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_ARCHIVE_CHAT_ID", raising=False)
        with patch("src.pipeline.pass_f_archive._post_telegram_document") as post:
            assert upload_to_telegram(b"x", "f.jsonl") is None
        post.assert_not_called()

    def test_r2_upload_skipped_without_credentials(self, monkeypatch):
        from src.pipeline.pass_f_archive import upload_to_cloudflare_r2
        for var in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(var, raising=False)
        assert upload_to_cloudflare_r2(b"x", "f.jsonl") is False


@pytest.mark.parametrize("missing", ["occurred_at_est", "ingested_at"])
def test_serialization_is_total_over_nullable_timestamps(missing):
    event = _rows_to_event_dicts([_row(**{missing: None})])[0]
    content, _ = generate_jsonl_and_hash([event])
    assert json.loads(content.decode("utf-8"))[missing] is None
