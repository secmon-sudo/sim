"""Tests for operational health notifications (ops_notifier + orchestrator wiring)."""

from unittest.mock import patch

from src.pipeline.orchestrator import _collect_degradations, _notify_health


def _clean_run():
    return {
        "run_id": "20260708T120000",
        "success": True,
        "duration_seconds": 42.0,
        "pass_a": {"ingested": 10},
        "pass_b": {"deduped": 2},
        "pass_c": {"events_failed": 0},
        "pass_d": {"events_scored": 8, "events_failed": 0},
        "pass_e": {"events_failed": 0},
        "run_snapshot": {"events": 8, "error": None},
        "pass_f": {"events": 8, "error": None},
    }


class TestCollectDegradations:
    def test_clean_run_has_none(self):
        assert _collect_degradations(_clean_run()) == []

    def test_pass_error_surfaced(self):
        r = _clean_run()
        r["pass_f"]["error"] = "Telegram upload failed"
        assert any("Telegram upload failed" in d for d in _collect_degradations(r))

    def test_failed_events_surfaced(self):
        r = _clean_run()
        r["pass_d"]["events_failed"] = 3
        assert any("3 event(s) failed" in d for d in _collect_degradations(r))


class TestNotifyHealth:
    def test_clean_run_does_not_page(self):
        with patch("src.services.ops_notifier.send_ops_alert") as m:
            _notify_health(_clean_run())
            m.assert_not_called()

    def test_events_failed_alone_does_not_page(self):
        # Per-event failures are routine noise; they must not page on their own.
        r = _clean_run()
        r["pass_d"]["events_failed"] = 2
        with patch("src.services.ops_notifier.send_ops_alert") as m:
            _notify_health(r)
            m.assert_not_called()

    def test_hard_failure_pages_with_failed_stage(self):
        r = _clean_run()
        r["success"] = False
        r["error"] = "OperationalError: could not connect"
        r["pass_c"] = None  # progress stopped at pass_c
        r["pass_d"] = None
        with patch("src.services.ops_notifier.send_ops_alert") as m:
            _notify_health(r)
            m.assert_called_once()
            body = m.call_args[0][0]
            assert "FAILED at pass_c" in body
            assert "could not connect" in body

    def test_pass_error_pages_degraded(self):
        r = _clean_run()
        r["pass_f"]["error"] = "DB Delete error"
        with patch("src.services.ops_notifier.send_ops_alert") as m:
            _notify_health(r)
            m.assert_called_once()
            assert "DEGRADED" in m.call_args[0][0]


class TestSendOpsAlert:
    def test_skips_without_credentials(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_ALERTS_CHAT_ID", raising=False)
        from src.services.ops_notifier import send_ops_alert
        assert send_ops_alert("test") is False

    def test_posts_to_alerts_chat(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_ALERTS_CHAT_ID", "alerts123")
        sent = {}
        import src.services.ops_notifier as ops

        class FakeResp:
            def raise_for_status(self):
                pass

        def fake_post(url, json, timeout):
            sent.update(json)
            return FakeResp()

        monkeypatch.setattr(ops.httpx, "post", fake_post)
        assert ops.send_ops_alert("hello") is True
        assert sent["chat_id"] == "alerts123"
        assert "hello" in sent["text"]
