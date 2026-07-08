"""Tests for storyline escalation cues and quiet-closure notes."""

from unittest.mock import patch

from src.core.storyline_alert_state import is_escalation, TIER_RANK


class TestIsEscalation:
    def test_watch_to_critical_is_escalation(self):
        assert is_escalation("WATCH", "CRITICAL") is True

    def test_alert_to_critical_is_escalation(self):
        assert is_escalation("ALERT", "CRITICAL") is True

    def test_same_tier_is_not_escalation(self):
        assert is_escalation("ALERT", "ALERT") is False

    def test_downgrade_is_not_escalation(self):
        assert is_escalation("CRITICAL", "WATCH") is False

    def test_first_ever_alert_is_not_escalation(self):
        # No prior peak → the very first page is not an escalation.
        assert is_escalation(None, "CRITICAL") is False

    def test_rank_ordering(self):
        assert TIER_RANK["WATCH"] < TIER_RANK["ALERT"] < TIER_RANK["CRITICAL"]


class TestEscalationCardRendering:
    def _send(self, event):
        import src.services.telegram_notifier as t
        captured = {}

        def fake_post(api_url, payload):
            captured.update(payload)
            return object()

        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "x", "TELEGRAM_ALERTS_CHAT_ID": "y"}):
            with patch.object(t, "_post_telegram", fake_post):
                t.send_telegram_alert(event)
        return captured["text"]

    def _event(self, **over):
        base = dict(
            alert_tier="CRITICAL", source_title="Escalating strike on base",
            event_type="missile_strike", anchor_name_norm="KYIV", country_iso="UA",
            severity_score=90, system_confidence=0.9, storyline_hint="kyiv strike",
        )
        base.update(over)
        return base

    def test_escalation_line_present(self):
        text = self._send(self._event(escalation_from="WATCH"))
        assert "Escalated WATCH → CRITICAL" in text

    def test_no_escalation_line_when_absent(self):
        text = self._send(self._event())
        assert "Escalated" not in text


class TestClosureNote:
    def test_closure_message_shape(self):
        import src.services.telegram_notifier as t
        captured = {}

        def fake_post(api_url, payload):
            captured.update(payload)
            return object()

        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "x", "TELEGRAM_ALERTS_CHAT_ID": "y"}):
            with patch.object(t, "_post_telegram", fake_post):
                ok = t.send_storyline_closure("CRITICAL", "Strike on base · KYIV UA", 12)
        assert ok is True
        assert "STORYLINE QUIET" in captured["text"]
        assert "peaked at CRITICAL" in captured["text"]
        assert "12h" in captured["text"]
