"""Corroboration entries carry when the duplicate was observed.

The count was already the signal confidence is not: measured 2026-08-17, every
silenced event carrying >= 2 independent domains was real (the mass drone attack on
Moscow, the Benghazi car bombing) while every piece of junk carried zero — which is
why CORROBORATION_ALERT_MIN exists. seen_at turns that count into a rate for the same
write, so "three outlets within an hour" becomes distinguishable from "three outlets
over two days".
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.pipeline.pass_a_ingest import _record_corroboration


def _db(rowcount=1):
    db = MagicMock()
    db.transaction.return_value.__enter__ = lambda s: None
    db.transaction.return_value.__exit__ = lambda s, *a: False
    db.execute.return_value.rowcount = rowcount
    return db


def _entry(db) -> dict:
    """The JSON payload appended to corroborating_sources."""
    return json.loads(db.execute.call_args.args[1][0])[0]


class TestSeenAt:
    def test_entry_carries_seen_at(self):
        db = _db()
        _record_corroboration(db, "evt-1", "reuters.com",
                              "bbc.co.uk", "https://bbc.co.uk/x", "Strike reported")
        assert "seen_at" in _entry(db)

    def test_seen_at_is_iso_utc(self):
        db = _db()
        _record_corroboration(db, "evt-1", "reuters.com",
                              "bbc.co.uk", "https://bbc.co.uk/x", "Strike reported")
        parsed = datetime.fromisoformat(_entry(db)["seen_at"])
        assert parsed.tzinfo is not None
        assert abs((datetime.now(timezone.utc) - parsed).total_seconds()) < 60

    def test_existing_fields_survive(self):
        """seen_at is additive — the domain is what the alert floor counts."""
        db = _db()
        _record_corroboration(db, "evt-1", "reuters.com",
                              "bbc.co.uk", "https://bbc.co.uk/x", "Strike reported")
        entry = _entry(db)
        assert entry["domain"] == "bbc.co.uk"
        assert entry["url"] == "https://bbc.co.uk/x"
        assert entry["title"] == "Strike reported"


class TestDedupContractUnchanged:
    def test_same_publisher_still_rejected(self):
        """An outlet republishing itself proves nothing, timestamp or not."""
        db = _db()
        assert _record_corroboration(
            db, "evt-1", "bbc.co.uk", "news.bbc.co.uk", "u", "t") is False
        db.execute.assert_not_called()

    def test_idempotence_probe_excludes_seen_at(self):
        """The NOT-contains probe must match on domain alone. If it carried the
        timestamp, every re-observation would look like a new source and one outlet
        could manufacture corroboration by being fetched twice."""
        db = _db()
        _record_corroboration(db, "evt-1", "reuters.com",
                              "bbc.co.uk", "https://bbc.co.uk/x", "Strike")
        probe = json.loads(db.execute.call_args.args[1][3])[0]
        assert probe == {"domain": "bbc.co.uk"}

    def test_db_error_is_swallowed(self):
        db = _db()
        db.execute.side_effect = RuntimeError("no such column")
        assert _record_corroboration(
            db, "evt-1", "reuters.com", "bbc.co.uk", "u", "t") is False
