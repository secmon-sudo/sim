"""Tests for the daily executive briefing renderer.

The briefing is the sixth HTML of a SITREP run and the one an executive
actually reads. Two constraints shape it and are easy to break silently:
the page must carry NO links (R2 URLs are not public, so a link is a dead
end), and it must render as one self-contained file with no external assets,
because it is delivered as a Telegram document.
"""

import re

import pytest

from src.services.sitrep_digest import (
    RISK_CRITICAL,
    RISK_ELEVATED,
    RISK_HIGH,
    RISK_NORMAL,
)
from src.services.sitrep_digest_html import render_digest_html

WINDOW = ("2026-07-22 09:59", "2026-07-23 09:59")


def _digest(**overrides):
    base = {
        "overview": "Gerilim İran ve Körfez hattında tırmandı.",
        "countries": [
            {"name": "İran", "iso": "IR", "risk": RISK_CRITICAL,
             "text": "Tahran ve Bandar Abbas'a hava saldırıları sürüyor."},
            {"name": "Suudi Arabistan", "iso": "SA", "risk": RISK_NORMAL,
             "text": "Kayda değer gelişme yok."},
        ],
        "aviation": ["Emirates Tahran uçuşlarını 30 Temmuz'a kadar durdurdu."],
        "highlights": ["Bandar Abbas limanında patlama, 12 ölü."],
        "watch": ["Hürmüz Boğazı'nda seyrüsefer kısıtlaması ihtimali."],
    }
    base.update(overrides)
    return base


class TestStructure:
    def test_renders_all_five_sections(self):
        html = render_digest_html(_digest(), *WINDOW)
        for heading in ("GENEL DURUM DEĞERLENDİRMESİ", "ÜLKE DEĞERLENDİRMELERİ",
                        "HAVACILIK OPERASYONLARINA ETKİ", "KRİTİK GELİŞMELER",
                        "İZLEME VE BEKLENTİLER"):
            assert heading in html, heading

    def test_country_count_in_header(self):
        html = render_digest_html(_digest(), *WINDOW)
        assert "2 ülke" in html

    def test_window_is_shown(self):
        html = render_digest_html(_digest(), *WINDOW)
        assert "2026-07-22 09:59" in html and "2026-07-23 09:59" in html

    def test_content_is_present(self):
        html = render_digest_html(_digest(), *WINDOW)
        assert "Emirates Tahran uçuşlarını" in html
        assert "Bandar Abbas limanında patlama" in html
        assert "Hürmüz Boğazı" in html

    def test_empty_section_is_omitted_not_left_blank(self):
        html = render_digest_html(_digest(aviation=[], highlights=[], watch=[]), *WINDOW)
        assert "HAVACILIK OPERASYONLARINA ETKİ" not in html
        assert "KRİTİK GELİŞMELER" not in html

    def test_no_countries_omits_the_country_block(self):
        html = render_digest_html(_digest(countries=[]), *WINDOW)
        assert "ÜLKE DEĞERLENDİRMELERİ" not in html
        assert "0 ülke" in html


class TestNoLinksNoExternalAssets:
    """R2 URLs are not public and the file is read offline in Telegram."""

    def test_contains_no_anchor_tags(self):
        html = render_digest_html(_digest(), *WINDOW)
        assert "<a " not in html.lower()
        assert "href=" not in html.lower()

    def test_no_external_requests(self):
        html = render_digest_html(_digest(), *WINDOW)
        for tag in ("<script", "<link", "<img", "@import", "url(http"):
            assert tag not in html.lower(), tag

    def test_a_url_inside_content_is_still_only_text(self):
        # Even if the LLM leaks a URL into a bullet, it must not become a link.
        html = render_digest_html(
            _digest(highlights=["Kaynak https://pub-default.r2.dev/x.html"]), *WINDOW)
        assert "<a " not in html.lower()


class TestEscaping:
    def test_html_in_llm_output_is_escaped(self):
        # Report text is model output; an unescaped angle bracket would break
        # the document or inject markup into a file the customer opens.
        html = render_digest_html(
            _digest(overview="Durum <kritik> & \"belirsiz\""), *WINDOW)
        assert "&lt;kritik&gt;" in html
        assert "<kritik>" not in html

    def test_escaping_applies_to_country_lines(self):
        html = render_digest_html(
            _digest(countries=[{"name": "İran <b>", "iso": "IR",
                                "risk": RISK_HIGH, "text": "a & b"}]), *WINDOW)
        assert "&lt;b&gt;" in html
        assert "a &amp; b" in html

    def test_escaping_applies_to_bullets(self):
        html = render_digest_html(_digest(aviation=["<script>alert(1)</script>"]), *WINDOW)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_none_fields_do_not_render_the_word_none(self):
        html = render_digest_html(
            _digest(overview=None, countries=[{"name": "İran", "iso": "IR",
                                              "risk": RISK_NORMAL, "text": None}]), *WINDOW)
        assert ">None<" not in html


class TestRiskBadges:
    @pytest.mark.parametrize("risk", [RISK_CRITICAL, RISK_HIGH, RISK_ELEVATED, RISK_NORMAL])
    def test_every_risk_level_renders_its_label(self, risk):
        html = render_digest_html(
            _digest(countries=[{"name": "X", "iso": "XX", "risk": risk, "text": "l"}]), *WINDOW)
        assert risk in html

    def test_risk_levels_are_visually_distinct(self):
        # Same colour for two levels would make the badge decorative, not
        # informative — the badge is the fastest read on the page.
        colours = set()
        for risk in (RISK_CRITICAL, RISK_HIGH, RISK_ELEVATED, RISK_NORMAL):
            html = render_digest_html(
                _digest(countries=[{"name": "X", "iso": "XX", "risk": risk, "text": "l"}]),
                *WINDOW)
            badge = re.search(r'<span style="display:inline-block;padding:2px 9px[^"]*"', html)
            colours.add(badge.group(0))
        assert len(colours) == 4

    def test_unknown_risk_falls_back_without_crashing(self):
        html = render_digest_html(
            _digest(countries=[{"name": "X", "iso": "XX", "risk": "Belirsiz", "text": "l"}]),
            *WINDOW)
        assert "Belirsiz" in html


class TestDocumentShape:
    def test_is_a_complete_html_document(self):
        html = render_digest_html(_digest(), *WINDOW)
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")
        assert 'lang="tr"' in html

    def test_declares_utf8_and_viewport(self):
        # Turkish text in a Telegram document viewer needs both.
        html = render_digest_html(_digest(), *WINDOW)
        assert 'charset="utf-8"' in html or "charset=utf-8" in html
        assert "width=device-width" in html

    def test_title_carries_the_window_date(self):
        html = render_digest_html(_digest(), *WINDOW)
        assert "<title>Günlük Yönetici Brifingi — 2026-07-23</title>" in html

    def test_missing_keys_do_not_raise(self):
        # parse_digest can return a partial dict when the model skips a heading.
        assert render_digest_html({"overview": "x"}, *WINDOW)
        assert render_digest_html({}, *WINDOW)
