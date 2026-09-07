"""A dead feed has to be countable, not just loggable.

insightcrime.org started answering 403 to every request on 2026-08-29 (an IP-reputation
block from its CDN, so it fails identically from every runner). The bot-protection
branch logged "drop the source if this persists" on 16 of 16 production runs, and
nothing anywhere measured that it had persisted — the source was removed only because a
human happened to read a warning line. The counter is the fix; dropping the feed is just
the consequence.
"""

import json
from pathlib import Path
from unittest.mock import patch

import src.pipeline.ingest_sources as isrc


def _fetch_failing(url, stats):
    with patch.object(isrc, "_http_get_with_retry", return_value=None):
        return isrc.fetch_rss_feed(url, is_direct_url=True, stats=stats)


def test_unreachable_feed_is_counted_by_host():
    stats: dict = {}
    assert _fetch_failing("https://insightcrime.org/feed/", stats) == []
    assert stats["feeds_unreachable"] == {"insightcrime.org": 1}


def test_repeated_failures_on_one_host_accumulate():
    # The Google News queries all share a host on purpose: 40 failures in a run is one
    # outage, and it should read as one number rather than 40 separate dead sources.
    stats: dict = {}
    for q in ("https://news.google.com/rss/search?q=a", "https://news.google.com/rss/search?q=b"):
        _fetch_failing(q, stats)
    assert stats["feeds_unreachable"] == {"news.google.com": 2}


def test_an_exception_counts_the_same_as_an_empty_response():
    stats: dict = {}
    with patch.object(isrc, "_http_get_with_retry", side_effect=RuntimeError("boom")):
        assert isrc.fetch_rss_feed("https://example.org/feed/", is_direct_url=True,
                                   stats=stats) == []
    assert stats["feeds_unreachable"] == {"example.org": 1}


def test_missing_stats_dict_is_not_an_error():
    # Pass A always passes one, but fetch_rss_feed's signature makes it optional and a
    # counter must never be the thing that breaks ingestion.
    assert _fetch_failing("https://example.org/feed/", None) == []


def test_the_blocked_feed_is_no_longer_configured():
    settings = json.loads(Path("config/settings.json").read_text())
    feeds = settings["sources"]["publisher_feeds"] + settings["sources"]["news_queries"]
    assert not any("insightcrime.org" in f for f in feeds)
