"""
Tests for Faz 5.1 — outbox-ordered alert dispatch.

Verifies the suppression record is committed BEFORE the Telegram send (so a crash
between send and record can't duplicate alerts), and that a failed send releases
the suppression claim.
"""

import src.pipeline.pass_d_score as d


class _MockDB:
    def __init__(self):
        self.deletes = 0

    def execute(self, sql, params=None):
        if "DELETE FROM alert_suppression" in sql:
            self.deletes += 1
        return self

    def fetchone(self):
        return None

    def commit(self):
        pass

    def rollback(self):
        pass


def _wire(monkeypatch, calls, *, suppressed=False, send_ok=True):
    monkeypatch.setattr(d, "build_suppression_key", lambda ev: "KEY")
    monkeypatch.setattr(d, "is_suppressed", lambda db, key: suppressed)
    monkeypatch.setattr(d, "record_suppression",
                        lambda *a, **k: calls.append("record"))
    monkeypatch.setattr(d, "send_telegram_alert",
                        lambda ev: calls.append("send") or send_ok)


class TestDispatchAlert:
    def test_skipped_below_threshold(self, monkeypatch):
        calls = []
        _wire(monkeypatch, calls)
        ev = {"severity_score": 50, "alert_tier": "WATCH"}
        assert d.dispatch_alert(_MockDB(), ev, "evt-1") == "skipped"
        assert calls == []

    def test_suppressed_does_not_send(self, monkeypatch):
        calls = []
        _wire(monkeypatch, calls, suppressed=True)
        ev = {"severity_score": 90, "alert_tier": "CRITICAL"}
        assert d.dispatch_alert(_MockDB(), ev, "evt-2") == "suppressed"
        assert "send" not in calls

    def test_record_before_send(self, monkeypatch):
        calls = []
        _wire(monkeypatch, calls, send_ok=True)
        ev = {"severity_score": 90, "alert_tier": "CRITICAL"}
        assert d.dispatch_alert(_MockDB(), ev, "evt-3") == "sent"
        # The suppression must be recorded BEFORE the send goes out.
        assert calls == ["record", "send"]

    def test_failed_send_releases_suppression(self, monkeypatch):
        calls = []
        _wire(monkeypatch, calls, send_ok=False)
        db = _MockDB()
        ev = {"severity_score": 90, "alert_tier": "CRITICAL"}
        assert d.dispatch_alert(db, ev, "evt-4") == "failed"
        assert calls == ["record", "send"]
        assert db.deletes == 1  # claim released for retry

    def test_default_tier_when_missing(self, monkeypatch):
        calls = []
        _wire(monkeypatch, calls)
        ev = {"severity_score": 85}  # no alert_tier
        assert d.dispatch_alert(_MockDB(), ev, "evt-5") == "sent"
        assert ev["alert_tier"] == "ALERT"
