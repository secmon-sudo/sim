"""Google News link resolution must finish the busy countries, and must stop.

The budget was 20, chosen blind, and it bound precisely on the reports a reader
opens: Lebanon on 16 Aug 2026 and Ukraine on 19 Aug both carried 96 sources and
still shipped 23 and 15 raw news.google.com redirects in the appendix. Sampling
on 19 Aug resolved 5 of 5 links at a mean of 0.30s, so cost was never the reason
to stop at 20.

Raising a budget without bounding time is the mistake Pass C already made once
(a per-request timeout does not bound a sequence of requests), so a wall-clock
deadline is pinned here alongside it.
"""

from unittest.mock import patch

from src.services import google_news_resolver as gnr

GN = "https://news.google.com/rss/articles/ID{}?oc=5"


def _clusters(n_gnews, n_plain=0):
    sources = [{"url": GN.format(i)} for i in range(n_gnews)]
    sources += [{"url": f"https://real.example/{i}"} for i in range(n_plain)]
    return [{"sources": sources}]


class TestBudget:
    def test_a_busy_country_is_finished_not_cut_at_twenty(self):
        clusters = _clusters(40)
        with patch.object(gnr, "resolve_url", side_effect=lambda u, **k: "https://x.example/1"):
            gnr.resolve_cluster_urls(clusters)
        assert all("news.google.com" not in s["url"] for s in clusters[0]["sources"])

    def test_the_budget_still_bounds_an_absurd_day(self):
        clusters = _clusters(500)
        with patch.object(gnr, "resolve_url", side_effect=lambda u, **k: "https://x.example/1") as r:
            gnr.resolve_cluster_urls(clusters)
        assert r.call_count == gnr.DEFAULT_MAX_RESOLVE

    def test_non_google_urls_never_spend_budget(self):
        """A country of ordinary links must not exhaust the allowance."""
        clusters = _clusters(5, n_plain=200)
        with patch.object(gnr, "resolve_url", side_effect=lambda u, **k: "https://x.example/1") as r:
            gnr.resolve_cluster_urls(clusters)
        assert r.call_count == 5

    def test_untouched_urls_keep_their_value(self):
        clusters = _clusters(1, n_plain=1)
        original = clusters[0]["sources"][1]["url"]
        with patch.object(gnr, "resolve_url", side_effect=lambda u, **k: "https://x.example/1"):
            gnr.resolve_cluster_urls(clusters)
        assert clusters[0]["sources"][1]["url"] == original


class TestDeadline:
    def test_a_slow_google_stops_the_pass(self):
        """Each call lands under its own timeout; only the deadline bounds the sum."""
        clock = iter([0.0] + [i * 5.0 for i in range(1, 500)])
        with patch.object(gnr, "resolve_url", side_effect=lambda u, **k: "https://x.example/1") as r, \
             patch.object(gnr.time, "monotonic", side_effect=lambda: next(clock)):
            gnr.resolve_cluster_urls(_clusters(50), deadline_seconds=20.0)
        assert 0 < r.call_count < 50

    def test_a_resolution_that_fails_leaves_a_working_link(self):
        clusters = _clusters(3)
        with patch.object(gnr, "resolve_url", side_effect=lambda u, **k: u):
            gnr.resolve_cluster_urls(clusters)
        assert all(s["url"].startswith(GN[:38]) for s in clusters[0]["sources"])
