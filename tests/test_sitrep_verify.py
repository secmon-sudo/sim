"""Tests for rule-based SITREP verification labeling (src/core/sitrep_verify.py)."""

from src.core.sitrep_verify import (
    LABEL_MULTI,
    LABEL_OFFICIAL,
    LABEL_SINGLE,
    fallback_cluster_key,
    is_official_domain,
    label_cluster,
    registrable_domain,
)


def ev(domain: str, **kwargs):
    return {"source_domain": domain, **kwargs}


class TestRegistrableDomain:
    def test_strips_www_and_subdomains(self):
        assert registrable_domain("www.reuters.com") == "reuters.com"
        assert registrable_domain("uk.reuters.com") == "reuters.com"

    def test_handles_full_urls(self):
        assert registrable_domain("https://edition.cnn.com/2026/07/16/x") == "cnn.com"

    def test_second_level_public_suffixes(self):
        assert registrable_domain("www.gov.uk") == "gov.uk"
        assert registrable_domain("assets.publishing.service.gov.uk") == "service.gov.uk"
        assert registrable_domain("haber.aa.com.tr") == "aa.com.tr"

    def test_empty(self):
        assert registrable_domain("") == ""


class TestIsOfficialDomain:
    def test_gov_mil_int_suffixes(self):
        assert is_official_domain("centcom.mil")
        assert is_official_domain("www.defense.gov")
        assert is_official_domain("travel.state.gov")
        assert is_official_domain("nato.int")

    def test_state_agencies(self):
        assert is_official_domain("en.irna.ir")
        assert is_official_domain("www.aa.com.tr")
        assert is_official_domain("tass.com")

    def test_regular_press_is_not_official(self):
        assert not is_official_domain("reuters.com")
        assert not is_official_domain("cnn.com")
        assert not is_official_domain("almayadeen.net")

    def test_lookalike_not_official(self):
        # "gov" must be a domain-boundary suffix, not a substring
        assert not is_official_domain("mygovnews.com")


class TestLabelCluster:
    def test_official_source_wins(self):
        events = [ev("reuters.com"), ev("centcom.mil")]
        assert label_cluster(events) == LABEL_OFFICIAL

    def test_two_independent_domains(self):
        events = [ev("reuters.com"), ev("bbc.co.uk")]
        assert label_cluster(events) == LABEL_MULTI

    def test_same_registrable_domain_counts_once(self):
        events = [ev("www.cnn.com"), ev("edition.cnn.com")]
        assert label_cluster(events) == LABEL_SINGLE

    def test_single_source(self):
        assert label_cluster([ev("almayadeen.net")]) == LABEL_SINGLE

    def test_penalized_domain_excluded(self):
        events = [ev("fakenews.example"), ev("reuters.com")]
        assert label_cluster(events, penalized_domains=["fakenews.example"]) == LABEL_SINGLE

    def test_all_penalized_stays_unverified(self):
        events = [ev("a.example"), ev("b.example2")]
        assert label_cluster(events, penalized_domains=["a.example", "b.example2"]) == LABEL_SINGLE

    def test_falls_back_to_source_url(self):
        events = [
            {"source_url": "https://www.centcom.mil/media/x"},
        ]
        assert label_cluster(events) == LABEL_OFFICIAL

    def test_empty_cluster(self):
        assert label_cluster([]) == LABEL_SINGLE


class TestFallbackClusterKey:
    def test_same_incident_same_key(self):
        a = {"anchor_name_norm": "BND", "event_type": "missile_strike",
             "occurred_at_est": "2026-07-16 03:30:00"}
        b = {"anchor_name_norm": "BND", "event_type": "missile_strike",
             "occurred_at_est": "2026-07-16 22:00:00"}
        assert fallback_cluster_key(a) == fallback_cluster_key(b)

    def test_different_location_differs(self):
        a = {"anchor_name_norm": "BND", "event_type": "missile_strike",
             "occurred_at_est": "2026-07-16 03:30:00"}
        b = {"anchor_name_norm": "AWZ", "event_type": "missile_strike",
             "occurred_at_est": "2026-07-16 03:30:00"}
        assert fallback_cluster_key(a) != fallback_cluster_key(b)

    def test_missing_fields(self):
        assert fallback_cluster_key({}) == ("", "", "")
