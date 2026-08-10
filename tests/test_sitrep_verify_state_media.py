"""
A state news agency is "official" about its own country, and an interested party
about anyone else's.

v1 treated state agencies as official everywhere: "the state said it happened" was
taken as the confirmation signal even when the state is a party to the conflict.
Measured over the 14 days to 2026-08-10, the rule fired almost entirely in the case
where it is wrong — Anadolu carried 143 events, 2 about Turkey and 81 about other
countries; TASS 126, of which 21 about Russia and 55 about others, overwhelmingly
Ukraine.

The concrete failure, from the 2026-08-09 Ukraine SITREP, is the regression test at
the bottom of this file: the report told its reader, labelled "Onaylandı (Resmî)",
that Russian forces had struck "military warehouses storing electronic warfare
equipment" in Odesa port — sourced to TASS quoting the Russian MoD. A belligerent's
targeting claim about the adversary's territory, presented as verified fact.

Downgrading it does not hide it: the cluster still appears with whatever label its
real corroboration earns, and the appendix still carries the source.
"""

import pytest

from src.core.sitrep_verify import (
    LABEL_MULTI,
    LABEL_OFFICIAL,
    LABEL_SINGLE,
    label_cluster,
    state_media_home_iso,
)


def _ev(domain, iso):
    return {"source_domain": domain, "country_iso": iso}


class TestHomeIsoLookup:
    @pytest.mark.parametrize("domain,iso", [
        ("tass.com", "RU"), ("www.tass.com", "RU"),
        ("en.irna.ir", "IR"), ("aa.com.tr", "TR"), ("sana.sy", "SY"),
        ("spa.gov.sa", "SA"), ("petra.gov.jo", "JO"),
    ])
    def test_known_agencies(self, domain, iso):
        assert state_media_home_iso(domain) == iso

    @pytest.mark.parametrize("domain", ["reuters.com", "bbc.co.uk", "kyivindependent.com"])
    def test_ordinary_press_has_no_home_state(self, domain):
        assert state_media_home_iso(domain) is None


class TestClusterLabelling:
    def test_state_agency_at_home_still_confirms(self):
        # TASS on a Russian event is the strongest available confirmation.
        assert label_cluster([_ev("tass.com", "RU")]) == LABEL_OFFICIAL

    def test_state_agency_abroad_is_a_single_source(self):
        assert label_cluster([_ev("tass.com", "UA")]) == LABEL_SINGLE

    def test_state_agency_abroad_still_counts_toward_corroboration(self):
        # Downgraded from "official", NOT excluded — two independent domains is
        # still two independent domains.
        cluster = [_ev("tass.com", "UA"), _ev("kyivindependent.com", "UA")]
        assert label_cluster(cluster) == LABEL_MULTI

    def test_a_real_official_source_in_the_cluster_still_wins(self):
        cluster = [_ev("tass.com", "UA"), _ev("mod.gov.ua", "UA")]
        assert label_cluster(cluster) == LABEL_OFFICIAL

    def test_multinational_bodies_are_official_anywhere(self):
        assert label_cluster([_ev("un.org", "UA")]) == LABEL_OFFICIAL
        assert label_cluster([_ev("reliefweb.int", "YE")]) == LABEL_OFFICIAL

    def test_government_portals_keep_cross_border_authority(self):
        # A travel advisory is a government officially speaking about somewhere else;
        # that IS the official act being reported.
        assert label_cluster([_ev("travel.state.gov", "NG")]) == LABEL_OFFICIAL
        assert label_cluster([_ev("gov.uk", "PK")]) == LABEL_OFFICIAL

    def test_unknown_country_does_not_grant_official(self):
        assert label_cluster([_ev("tass.com", None)]) == LABEL_SINGLE

    def test_agencies_on_gov_domains_follow_the_state_media_rule(self):
        # SPA and Petra are news agencies living on gov.sa / gov.jo. The generic
        # government-label rule must not hand them cross-border authority.
        assert label_cluster([_ev("spa.gov.sa", "SA")]) == LABEL_OFFICIAL
        assert label_cluster([_ev("spa.gov.sa", "YE")]) == LABEL_SINGLE
        assert label_cluster([_ev("petra.gov.jo", "IL")]) == LABEL_SINGLE
        # All India Radio / Doordarshan: general wire services on gov.in that report
        # abroad far more than at home — including on Pakistan.
        assert label_cluster([_ev("newsonair.gov.in", "IN")]) == LABEL_OFFICIAL
        assert label_cluster([_ev("newsonair.gov.in", "PK")]) == LABEL_SINGLE
        assert label_cluster([_ev("ddnews.gov.in", "IL")]) == LABEL_SINGLE

    def test_every_countrys_government_now_counts(self):
        # The old suffix test only matched US-style .gov/.mil plus hard-coded
        # gov.uk/gov.il, so these were never official while TASS always was.
        for domain in ("mod.gov.ua", "mha.gov.in", "police.gov.za"):
            assert label_cluster([_ev(domain, "UA")]) == LABEL_OFFICIAL, domain

    def test_penalized_domain_is_still_excluded_entirely(self):
        assert label_cluster([_ev("tass.com", "RU")], penalized_domains=["tass.com"]) \
            == LABEL_SINGLE


class TestOdesaRegression:
    """2026-08-09 UA SITREP, verbatim shape."""

    def test_russian_mod_claim_about_odesa_is_no_longer_official(self):
        cluster = [_ev("tass.com", "UA")]  # "Rus kuvvetleri Odesa limanında … vurdu"
        assert label_cluster(cluster) != LABEL_OFFICIAL

    def test_the_same_agency_on_a_russian_target_is_unaffected(self):
        # TASS reporting a Ukrainian drone strike on Russian soil.
        assert label_cluster([_ev("tass.com", "RU")]) == LABEL_OFFICIAL
