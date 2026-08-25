"""Carriers cannot corroborate, added 2026-08-25.

A verification label is the thing a SITREP reader trusts most, and the corroboration
count is also an ALERT floor. Both were counting redistribution as confirmation.

Measured over the 7 days to 2026-08-25 on live corroborating_sources: yahoo.com was
the most frequent corroborator in the entire corpus (93 credits, ahead of Al Jazeera),
reddit.com had 43, and aol.com + aol.co.uk 46 — Yahoo and AOL being one syndication
feed. 39 events carried >= 2 corroborating sources that collapse below 2 once carriers
are removed, 8 of which had paged as ALERT/CRITICAL.

The case that named the bug: the 2026-08-24 Ukraine SITREP published a Kherson strike
as "Onaylandı (Çoklu kaynak)" on UNITED24 Media plus a Reddit crosspost of that same
UNITED24 article. The Haiti church attack was credited to inbox.lv, modernghana.com
and yahoo.com under a byte-identical headline.
"""

from src.core.alerts import corroboration_count
from src.core.sitrep_verify import (
    LABEL_MULTI,
    LABEL_OFFICIAL,
    LABEL_SINGLE,
    is_independent_publisher,
    label_cluster,
)


class TestIsIndependentPublisher:
    def test_real_outlets_are_independent(self):
        for domain in ("reuters.com", "ukrinform.net", "kyivindependent.com",
                       "aljazeera.com", "modernghana.com"):
            assert is_independent_publisher(domain), domain

    def test_carriers_are_not(self):
        for domain in ("yahoo.com", "aol.com", "reddit.com", "msn.com",
                       "t.me", "x.com", "inbox.lv", "facebook.com"):
            assert not is_independent_publisher(domain), domain

    def test_cctld_editions_of_a_carrier_are_covered(self):
        """The brand is matched on the registrable domain's first label precisely so
        the country editions do not each need listing."""
        for domain in ("aol.co.uk", "uk.news.yahoo.com", "yahoo.co.jp"):
            assert not is_independent_publisher(domain), domain

    def test_google_news_collapses_to_the_brand(self):
        """news.google.com reduces to google.com under the public suffix list, so the
        entry has to be the brand or the check silently passes."""
        assert not is_independent_publisher("news.google.com")
        assert not is_independent_publisher("google.com")

    def test_unattributable_is_not_independent(self):
        assert not is_independent_publisher("")
        assert not is_independent_publisher(None or "")


class TestLabelCluster:
    def test_outlet_plus_carrier_is_single_source(self):
        """The exact Kherson case."""
        cluster = [{"source_domain": "united24media.com"}, {"source_domain": "reddit.com"}]
        assert label_cluster(cluster) == LABEL_SINGLE

    def test_two_real_outlets_still_multi(self):
        cluster = [{"source_domain": "united24media.com"}, {"source_domain": "pravda.com.ua"}]
        assert label_cluster(cluster) == LABEL_MULTI

    def test_yahoo_and_aol_are_one_feed_not_two_sources(self):
        cluster = [{"source_domain": "yahoo.com"}, {"source_domain": "aol.com"}]
        assert label_cluster(cluster) == LABEL_SINGLE

    def test_all_carriers_never_reaches_official(self):
        cluster = [{"source_domain": "yahoo.com"}, {"source_domain": "reddit.com"}]
        assert label_cluster(cluster) == LABEL_SINGLE

    def test_official_source_still_wins(self):
        """Carriers are removed before the official check too, but a real government
        source in the same cluster must be unaffected."""
        cluster = [{"source_domain": "yahoo.com"},
                   {"source_domain": "travel.state.gov", "country_iso": "US"}]
        assert label_cluster(cluster, ) == LABEL_OFFICIAL

    def test_penalized_and_carrier_exclusions_compose(self):
        cluster = [{"source_domain": "yahoo.com"}, {"source_domain": "junk.example"},
                   {"source_domain": "reuters.com"}]
        assert label_cluster(cluster, penalized_domains=["junk.example"]) == LABEL_SINGLE


class TestCorroborationCount:
    def test_carriers_do_not_count(self):
        event = {"corroborating_sources": [{"domain": "yahoo.com"}, {"domain": "reddit.com"}]}
        assert corroboration_count(event) == 0

    def test_real_outlets_count(self):
        event = {"corroborating_sources": [{"domain": "reuters.com"}, {"domain": "apnews.com"}]}
        assert corroboration_count(event) == 2

    def test_mixed_counts_only_the_real_ones(self):
        """This is what drops an event below CORROBORATION_ALERT_MIN."""
        event = {"corroborating_sources": [{"domain": "reuters.com"}, {"domain": "yahoo.com"},
                                           {"domain": "aol.com"}]}
        assert corroboration_count(event) == 1

    def test_json_string_column_is_still_parsed(self):
        event = {"corroborating_sources": '[{"domain": "reuters.com"}, {"domain": "yahoo.com"}]'}
        assert corroboration_count(event) == 1

    def test_malformed_entries_do_not_raise(self):
        event = {"corroborating_sources": [None, "reuters.com", {"no_domain": 1}, {"domain": None}]}
        assert corroboration_count(event) == 0
