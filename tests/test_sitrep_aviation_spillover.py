"""
Fix B — regional aviation spillover.

Flight-disruption headlines are usually regional ("Airlines suspend Middle East
flights to Dubai, Riyadh and Beirut") and so carry a null or neighbour
country_iso. The per-country SITREP query (WHERE country_iso = %s) never sees
them, so real "which carrier stopped flying where" news — the highest-value line
in an aviation SITREP — vanished. These tests cover the recovery sweep and the
dedicated render block that surfaces it.
"""

from datetime import datetime, timezone

from src.services.sitrep_generator import (
    _EVENT_COLUMNS,
    fetch_aviation_spillover_events,
)
from src.services.sitrep_html import render_sitrep_html

T0 = datetime(2026, 7, 23, 9, 56, tzinfo=timezone.utc)
T1 = datetime(2026, 7, 24, 9, 56, tzinfo=timezone.utc)


def _row(title: str, canonical: str = "", url: str = "http://x", iso=None, sev=60):
    """Build an events-table row tuple in _EVENT_COLUMNS order."""
    d = {c: None for c in _EVENT_COLUMNS}
    d.update(source_title=title, source_url=url, canonical_text=canonical,
             country_iso=iso, severity_score=sev)
    return tuple(d[c] for c in _EVENT_COLUMNS)


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.sql = None
        self.params = None

    def execute(self, sql, params):
        self.sql, self.params = sql, params
        rows = self._rows
        class R:
            def fetchall(self_inner):
                return rows
        return R()


class TestAviationSpilloverFetch:
    def test_keeps_disruption_drops_aviation_noun_only(self):
        rows = [
            _row("Airlines suspend Middle East flights to Riyadh amid conflict"),  # keep
            _row("Kuwait suspends flights after Iran attacks on airport"),         # keep
            _row("Saudi Arabia opens new airport terminal in Riyadh"),             # drop (no disruption verb)
            _row("Riyadh hosts regional aviation safety summit"),                  # drop (noun only)
        ]
        out = fetch_aviation_spillover_events(_FakeConn(rows), "SA", "Saudi Arabia", T0, T1)
        titles = {e["source_title"] for e in out}
        assert titles == {
            "Airlines suspend Middle East flights to Riyadh amid conflict",
            "Kuwait suspends flights after Iran attacks on airport",
        }

    def test_disruption_verb_may_sit_in_canonical_text(self):
        # Aviation noun in the title, the disruption verb only in the body —
        # the gate concatenates title + canonical_text, exactly like ingest.
        rows = [_row("Dubai flight update", canonical="All flights cancelled after strikes")]
        out = fetch_aviation_spillover_events(_FakeConn(rows), "AE", "United Arab Emirates", T0, T1)
        assert len(out) == 1

    def test_query_binds_mention_aviation_and_country_distinct(self):
        conn = _FakeConn([])
        fetch_aviation_spillover_events(conn, "IR", "Iran", T0, T1)
        assert "country_iso IS DISTINCT FROM" in conn.sql
        assert "~*" in conn.sql and "aviation" in conn.sql  # aviation-noun pre-filter
        assert "%Tehran%" in conn.params and "%IRGC%" in conn.params
        # every placeholder is bound (the aviation regex adds no %s)
        assert conn.sql.count("%s") == len(conn.params)

    def test_empty_when_no_country_name(self):
        conn = _FakeConn([_row("Airlines suspend flights to Riyadh")])
        assert fetch_aviation_spillover_events(conn, "SA", "", T0, T1) == []


def _cluster(location, title, sev=60, url="http://pub/x"):
    return {
        "location": location,
        "event_type": "travel_advisory",
        "date": "2026-07-23, saat belirsiz",
        "verification": None,
        "severity": sev,
        "snippet": title,
        "sources": [{"name": "thenationalnews.com", "url": url, "title": title}],
    }


class TestAviationSectionRender:
    _REPORT = "YÖNETİCİ ÖZETİ\nGünün özeti."

    def test_section_renders_with_sources(self):
        av = [_cluster("Riyadh", "Airlines suspend Middle East flights to Riyadh")]
        html = render_sitrep_html("Suudi Arabistan", "SA", "2026-07-23 09:56",
                                  "2026-07-24 09:56", self._REPORT, [], av)
        assert "BÖLGESEL HAVACILIK KESİNTİLERİ" in html
        assert "http://pub/x" in html
        assert "Riyadh" in html

    def test_section_absent_when_no_aviation(self):
        html = render_sitrep_html("Ürdün", "JO", "2026-07-23 09:56",
                                  "2026-07-24 09:56", self._REPORT, [], [])
        assert "BÖLGESEL HAVACILIK KESİNTİLERİ" not in html

    def test_backward_compatible_without_aviation_arg(self):
        # Existing callers pass only 6 positional args — must still render.
        html = render_sitrep_html("Ürdün", "JO", "2026-07-23 09:56",
                                  "2026-07-24 09:56", self._REPORT, [])
        assert "GÜNLÜK DURUM RAPORU" in html
        assert "BÖLGESEL HAVACILIK KESİNTİLERİ" not in html
