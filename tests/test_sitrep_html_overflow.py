"""Report chrome must survive a narrow phone, 3 Sep 2026.

Reported from the first real Iran bulletin: "DOĞRULANMAMIŞ yazısı kutudan taşmış".
Three places could push content past its container, and all three shared one cause
— a long, unbreakable string in a box with no wrapping rule:

  * _stat_card's caption. "Doğrulanmamış" is thirteen characters, uppercased and
    letter-spaced, in a 64px box. A single word has no break opportunity, so it
    rendered straight out of the card.
  * _highlights put the 72px severity meter and a white-space:nowrap badge on one
    line with no wrap.
  * _appendix_row put the location and that same badge on one line, and the
    bulletin's locations are longer than a SITREP's ("İran · İran'a yönelik").

These assert the CSS that prevents it rather than a rendered width, because the
tests cannot measure text. The contract is "there is a way for this to break",
which is exactly what was missing.
"""

from src.services.sitrep_html import (
    _appendix_row,
    _badge,
    _highlights,
    _stat_card,
    render_sitrep_html,
)

LONGEST_CAPTION = "Doğrulanmamış"


class TestStatCard:
    def test_a_single_long_word_can_break(self):
        html = _stat_card("137", LONGEST_CAPTION, "#fbbf24")
        assert "overflow-wrap:anywhere" in html

    def test_the_box_is_wide_enough_for_the_captions_in_use(self):
        html = _stat_card("137", LONGEST_CAPTION, "#fbbf24")
        assert "min-width:84px" in html

    def test_content_cannot_escape_the_card(self):
        assert "overflow:hidden" in _stat_card("1", LONGEST_CAPTION, "#fff")


class TestBadgeLines:
    def test_the_badge_itself_stays_on_one_line(self):
        """It should wrap as a unit, never break mid-label."""
        assert "white-space:nowrap" in _badge("Doğrulanmamış (Tek kaynak)")

    def test_the_appendix_row_lets_the_badge_drop_to_its_own_line(self):
        row = _appendix_row({
            "location": "İran · İran'a yönelik", "snippet": "x", "severity": 90,
            "verification": "Doğrulanmamış (Tek kaynak)",
            "sources": [{"name": "reuters.com", "url": "https://a"}],
        })
        assert "flex-wrap:wrap" in row
        assert "overflow-wrap:anywhere" in row

    def test_the_highlight_meter_and_badge_can_wrap(self):
        html = _highlights([{
            "location": "İran · İran'a yönelik", "severity": 95,
            "event_type": "airstrike", "date": "Doğrulandı",
            "verification": "Doğrulanmamış (Tek kaynak)",
        }])
        assert "flex-wrap:wrap" in html


class TestPage:
    def _page(self):
        return render_sitrep_html(
            country_name="İran, Körfez ve Doğu Akdeniz hattı", country_iso="IR",
            window_start="2026-09-02 10:05", window_end="2026-09-03 10:05",
            report_text=(
                "YÖNETİCİ ÖZETİ\nUzun bir bağlantı: "
                "https://example.com/a/very/long/path/that/would/not/break/anywhere/"
                "on/its/own/and/pushes/the/page/sideways\n\n"
                "İRAN TOPRAKLARINA YÖNELİK SALDIRILAR\nBandar Abbas\n- Bir şey."),
            clusters=[{"location": "İran · İran'a yönelik", "snippet": "x",
                       "severity": 90, "verification": "Doğrulanmamış (Tek kaynak)",
                       "sources": [{"name": "reuters.com", "url": "https://a"}]}],
        )

    def test_the_page_never_scrolls_sideways(self):
        assert "overflow-x:hidden" in self._page()

    def test_a_long_url_in_the_narrative_can_break(self):
        page = self._page()
        assert "overflow-wrap:anywhere" in page

    def test_the_sitrep_masthead_is_unchanged_by_default(self):
        """The bulletin parameterised this; the SITREP must not have moved."""
        assert "GÜNLÜK DURUM RAPORU" in self._page()
