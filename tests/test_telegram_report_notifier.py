"""Tests for the SITREP/briefing Telegram dispatch.

The briefing body is sent with parse_mode=HTML, so any malformed markup makes
Telegram reject the entire message. The send is wrapped in a try/except that
only logs, so a rejection is silent: the customer gets the attachment and no
text. Length handling is therefore a correctness concern, not cosmetics.
"""

import html
import re
from unittest.mock import MagicMock, patch

import pytest

from src.services.telegram_report_notifier import (
    _trim_partial_markup,
    send_digest_telegram,
)


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("TELEGRAM_ALERTS_CHAT_ID", "-100999")


def _digest(countries=None, aviation=None, overview="Genel durum."):
    return {
        "overview": overview,
        "countries": countries if countries is not None else [
            {"name": "İran", "iso": "IR", "risk": "Kritik", "text": "Saldırılar sürüyor."},
        ],
        "aviation": aviation if aviation is not None else [],
        "highlights": [],
        "watch": [],
    }


def _send(digest):
    """Run the dispatch with both HTTP calls stubbed; return the message text."""
    ok = MagicMock()
    ok.json.return_value = {"ok": True, "result": {"message_id": 5}}
    with patch("src.services.telegram_report_notifier._post_telegram", return_value=ok) as post, \
         patch("src.services.telegram_report_notifier.httpx.post", return_value=ok):
        send_digest_telegram(digest, "2026-07-22 09:59", "2026-07-23 09:59", "<html></html>")
    return post.call_args.args[1]["text"]


def _tags_balanced(text: str) -> bool:
    return len(re.findall(r"<b>", text)) == len(re.findall(r"</b>", text))


def _has_partial_markup(text: str) -> bool:
    return bool(re.search(r"(<[a-zA-Z/][^>]*|&[a-zA-Z#][a-zA-Z0-9]*)$", text))


class TestBodyContent:
    def test_overview_and_country_are_included(self):
        text = _send(_digest())
        assert "Genel durum." in text
        assert "İran" in text and "Saldırılar sürüyor." in text

    def test_risk_icons_map_per_level(self):
        for risk, icon in [("Kritik", "🔴"), ("Yüksek", "🟠"),
                           ("Yükseltilmiş", "🔵"), ("Normal", "🟢")]:
            text = _send(_digest(countries=[{"name": "X", "iso": "XX",
                                             "risk": risk, "text": "t"}]))
            assert icon in text, risk

    def test_unknown_risk_gets_a_neutral_icon(self):
        text = _send(_digest(countries=[{"name": "X", "iso": "XX",
                                         "risk": "Belirsiz", "text": "t"}]))
        assert "⚪" in text

    def test_aviation_block_only_when_present(self):
        assert "HAVACILIK OPERASYONLARINA ETKİ" not in _send(_digest(aviation=[]))
        text = _send(_digest(aviation=["Emirates Tahran uçuşlarını durdurdu."]))
        assert "HAVACILIK OPERASYONLARINA ETKİ" in text
        assert "Emirates Tahran" in text

    def test_user_content_is_escaped(self):
        text = _send(_digest(overview="Durum <kritik> & belirsiz"))
        assert "&lt;kritik&gt;" in text
        assert "<kritik>" not in text


class TestLengthHandling:
    """A blind slice at 3880 split tags and entities; cut on line boundaries."""

    @staticmethod
    def _long_digest(pad):
        return _digest(countries=[
            {"name": "A" * pad, "iso": "IR", "risk": "Kritik", "text": "z & y"},
            {"name": "B" * 400, "iso": "SA", "risk": "Normal", "text": "ikinci ülke"},
        ])

    def test_stays_under_the_telegram_cap(self):
        text = _send(self._long_digest(4000))
        assert len(text) <= 4096

    @pytest.mark.parametrize("pad", range(3780, 3900, 7))
    def test_truncation_never_splits_markup(self, pad):
        # Regression: 173 of 300 body lengths used to leave a split <b> pair
        # and 4 a bare "&amp", each of which Telegram rejects outright.
        text = _send(self._long_digest(pad))
        assert _tags_balanced(text), f"unbalanced <b> at pad={pad}"
        assert not _has_partial_markup(text.rstrip()), f"partial markup at pad={pad}"

    def test_short_body_is_untouched(self):
        text = _send(_digest())
        assert "…" not in text
        assert text.endswith("<i>Brifingin tamamı ekte gönderilmiştir.</i>")

    def test_truncated_body_still_points_at_the_attachment(self):
        text = _send(self._long_digest(4000))
        assert "ekte gönderilmiştir" in text
        assert "…" in text


class TestTrimPartialMarkup:
    @pytest.mark.parametrize("broken,expected", [
        ("abc</b", "abc"),
        ("abc<b", "abc"),
        ("abc &amp", "abc "),
        ("abc &#123", "abc "),
        ("abc<b>bold</b>", "abc<b>bold</b>"),
        ("abc &amp; def", "abc &amp; def"),
        ("plain text", "plain text"),
    ])
    def test_only_dangling_markup_is_removed(self, broken, expected):
        assert _trim_partial_markup(broken) == expected


class TestDispatchRobustness:
    def test_missing_credentials_skips_silently(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        with patch("src.services.telegram_report_notifier._post_telegram") as post:
            assert send_digest_telegram(_digest(), "a", "b", "<html>") is None
        post.assert_not_called()

    def test_message_failure_still_sends_the_document(self):
        # The attachment is the fallback when the summary is rejected, so it
        # must not be skipped when the message call raises.
        doc = MagicMock()
        doc.json.return_value = {"ok": True, "result": {"message_id": 9}}
        with patch("src.services.telegram_report_notifier._post_telegram",
                   side_effect=RuntimeError("bad request")), \
             patch("src.services.telegram_report_notifier.httpx.post",
                   return_value=doc) as post_doc:
            send_digest_telegram(_digest(), "2026-07-22", "2026-07-23", "<html></html>")
        post_doc.assert_called_once()

    def test_document_filename_carries_the_window_date(self):
        ok = MagicMock()
        ok.json.return_value = {"ok": True, "result": {"message_id": 1}}
        with patch("src.services.telegram_report_notifier._post_telegram", return_value=ok), \
             patch("src.services.telegram_report_notifier.httpx.post", return_value=ok) as doc:
            send_digest_telegram(_digest(), "2026-07-22 09:59", "2026-07-23 09:59", "<html></html>")
        filename = doc.call_args.kwargs["files"]["document"][0]
        assert filename == "brifing_20260723.html"

    def test_empty_countries_does_not_crash(self):
        text = _send(_digest(countries=[]))
        assert "ÜLKE DEĞERLENDİRMELERİ" not in text
        assert "Genel durum." in text

    def test_missing_fields_do_not_crash(self):
        text = _send({"overview": "sadece özet"})
        assert "sadece özet" in text


def test_escaped_content_survives_truncation_intact():
    """An escaped ampersand must never be cut mid-entity."""
    digest = _digest(countries=[
        {"name": "Ülke", "iso": "IR", "risk": "Kritik", "text": "x & y " * 700},
    ])
    text = _send(digest)
    assert "&amp" not in text.replace("&amp;", "")
    assert html.unescape(text)  # decodes without raising
