"""A citation URL is stored whole or not at all.

Pass A used to slice every duplicate's URL at 500 characters before recording it
as a corroborating source. Google News RSS links run past a thousand characters,
so the slice cut through the base64 article id: the resolver could no longer
decode it, and the SITREP appendix rendered a link that goes nowhere. Measured
2026-08-19, 19 of the 51 Google News URLs across four days of SITREPs were
mangled this way while events.source_url — never sliced — held all 533 intact.

These tests pin the replacement contract: whole URL or None, and the domain
(which is what the corroboration count actually keys on) survives either way.
"""

import json
from unittest.mock import MagicMock

from src.pipeline.pass_a_ingest import (
    _MAX_SOURCE_URL_CHARS,
    _citable_url,
    _record_corroboration,
)

GNEWS = "https://news.google.com/rss/articles/" + "A" * 1000 + "?oc=5"


class TestCitableUrl:
    def test_a_long_real_url_survives_whole(self):
        assert _citable_url(GNEWS) == GNEWS
        assert len(GNEWS) > 500  # the length the old slice would have cut at

    def test_over_the_ceiling_is_dropped_not_cut(self):
        monster = "https://x.example/" + "B" * (_MAX_SOURCE_URL_CHARS + 1)
        assert _citable_url(monster) is None

    def test_exactly_at_the_ceiling_is_kept(self):
        edge = "https://x.example/" + "B" * (_MAX_SOURCE_URL_CHARS - 18)
        assert len(edge) == _MAX_SOURCE_URL_CHARS
        assert _citable_url(edge) == edge

    def test_empty_and_none_are_none(self):
        assert _citable_url(None) is None
        assert _citable_url("") is None
        assert _citable_url("   ") is None

    def test_never_returns_a_prefix_of_its_input(self):
        """The whole point: no output may be a truncation of the input."""
        for url in (GNEWS, "https://x.example/" + "B" * 5000, "https://a.b/c"):
            out = _citable_url(url)
            assert out is None or out == url.strip()


def _conn():
    conn = MagicMock()
    tx = MagicMock()
    tx.__enter__ = lambda s: s
    tx.__exit__ = lambda s, *a: False
    conn.transaction.return_value = tx
    conn.execute.return_value.rowcount = 1
    return conn


def _recorded_entry(conn):
    entry_json = conn.execute.call_args.args[1][0]
    return json.loads(entry_json)[0]


class TestRecordCorroboration:
    def test_long_google_news_url_is_recorded_intact(self):
        conn = _conn()
        assert _record_corroboration(conn, "evt-1", "reuters.com",
                                     "news.google.com", GNEWS, "t") is True
        assert _recorded_entry(conn)["url"] == GNEWS

    def test_the_domain_is_kept_even_when_the_url_is_dropped(self):
        # The corroboration SIGNAL is the domain count (CORROBORATION_ALERT_MIN);
        # an unusable URL must not cost the event its corroboration.
        conn = _conn()
        monster = "https://x.example/" + "B" * (_MAX_SOURCE_URL_CHARS + 1)
        assert _record_corroboration(conn, "evt-2", "reuters.com",
                                     "apnews.com", monster, "t") is True
        entry = _recorded_entry(conn)
        assert entry["url"] is None
        assert entry["domain"] == "apnews.com"

    def test_title_is_still_capped(self):
        # Titles are prose: a cut title is degraded, not broken, so the cap stays.
        conn = _conn()
        _record_corroboration(conn, "evt-3", "reuters.com", "apnews.com",
                              "https://a.b/c", "T" * 400)
        assert len(_recorded_entry(conn)["title"]) == 200

    def test_same_registrable_domain_is_still_refused(self):
        conn = _conn()
        assert _record_corroboration(conn, "evt-4", "www.reuters.com",
                                     "reuters.com", GNEWS, "t") is False
        conn.execute.assert_not_called()
