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
        # gov.uk and service.gov.uk are themselves public suffixes, so the eTLD+1 keeps
        # the label in front of them. The old hand-rolled version returned "gov.uk" here
        # by treating the suffix as the whole domain; both answers satisfy every consumer
        # (is_official_domain sees the "gov" label either way), and this one is what the
        # public suffix list actually says.
        assert registrable_domain("www.gov.uk") == "www.gov.uk"
        assert registrable_domain("assets.publishing.service.gov.uk") == "service.gov.uk"
        assert registrable_domain("haber.aa.com.tr") == "aa.com.tr"

    def test_cctld_publishers_keep_their_name(self):
        """The hand-maintained suffix set covered 28 entries, so every ccTLD outside it
        lost the publisher's name to the bare suffix — measured 2026-08-13, 202 events
        across 65 domains in 7 days. label_cluster() counts distinct registrable domains,
        so three Korean outlets collapsing to "co.kr" published a three-source cluster as
        "Doğrulanmamış (Tek kaynak)"."""
        assert registrable_domain("nst.com.my") == "nst.com.my"
        assert registrable_domain("abc.net.au") == "abc.net.au"
        assert registrable_domain("nhk.or.jp") == "nhk.or.jp"
        assert registrable_domain("thefinancialexpress.com.bd") == "thefinancialexpress.com.bd"
        assert registrable_domain("kenyans.co.ke") == "kenyans.co.ke"
        assert registrable_domain("saudigazette.com.sa") == "saudigazette.com.sa"

    def test_distinct_cctld_outlets_are_independent_sources(self):
        korean = [ev("mk.co.kr"), ev("asiae.co.kr"), ev("koreatimes.co.kr")]
        assert len({registrable_domain(e["source_domain"]) for e in korean}) == 3
        assert label_cluster(korean) == LABEL_MULTI

    def test_state_media_on_a_second_level_gov_suffix_still_resolves(self):
        """spa.gov.sa / petra.gov.jo are matched by exact registrable domain, so the
        suffix change must not move them."""
        assert registrable_domain("www.spa.gov.sa") == "spa.gov.sa"
        assert registrable_domain("petra.gov.jo") == "petra.gov.jo"

    def test_empty(self):
        assert registrable_domain("") == ""


class TestIsOfficialDomain:
    def test_gov_mil_int_suffixes(self):
        assert is_official_domain("centcom.mil")
        assert is_official_domain("www.defense.gov")
        assert is_official_domain("travel.state.gov")
        assert is_official_domain("nato.int")

    def test_state_agencies_are_official_at_home(self):
        assert is_official_domain("en.irna.ir", {"IR"})
        assert is_official_domain("www.aa.com.tr", {"TR"})
        assert is_official_domain("tass.com", {"RU"})

    def test_state_agencies_are_not_official_abroad(self):
        # A belligerent's agency reporting the adversary's territory is a claim,
        # not a confirmation — TASS on Ukraine, Anadolu on Iran.
        assert not is_official_domain("tass.com", {"UA"})
        assert not is_official_domain("www.aa.com.tr", {"IR"})
        assert not is_official_domain("en.irna.ir", {"IL"})

    def test_state_agency_without_country_context_is_not_official(self):
        assert not is_official_domain("tass.com")

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


class TestGoogleNewsDecode:
    def test_legacy_id_decodes_to_publisher_url(self):
        import base64
        from src.services.google_news_resolver import decode_google_news_url
        # legacy format: base64 payload embeds the article URL directly
        inner = b"\x08\x13\x22" + bytes([len(b"https://example.com/story-1")]) \
            + b"https://example.com/story-1" + b"\xd2\x01\x00"
        token = base64.urlsafe_b64encode(inner).decode().rstrip("=")
        url = f"https://news.google.com/rss/articles/{token}?oc=5"
        assert decode_google_news_url(url) == "https://example.com/story-1"

    def test_non_google_url_returns_none(self):
        from src.services.google_news_resolver import decode_google_news_url
        assert decode_google_news_url("https://reuters.com/a") is None

    def test_opaque_new_format_returns_none_or_url(self):
        from src.services.google_news_resolver import decode_google_news_url
        # new AU_yq… IDs are not offline-decodable; must not crash or return garbage
        res = decode_google_news_url(
            "https://news.google.com/rss/articles/CBMihwJBVV95cUxNYTc5?oc=5")
        assert res is None or res.startswith("http")


class TestRelabelCluster:
    def test_web_source_upgrades_single_to_multi(self):
        from src.services.sitrep_generator import relabel_cluster
        cluster = {
            "verification": LABEL_SINGLE,
            "sources": [
                {"name": "almayadeen.net", "url": "https://almayadeen.net/x"},
                {"name": "reuters.com", "url": "https://vertexaisearch.cloud.google.com/r/1"},
            ],
        }
        relabel_cluster(cluster, [])
        assert cluster["verification"] == LABEL_MULTI

    def test_official_web_source_upgrades_to_official(self):
        from src.services.sitrep_generator import relabel_cluster
        cluster = {
            "verification": LABEL_SINGLE,
            "sources": [
                {"name": "almayadeen.net", "url": "https://almayadeen.net/x"},
                {"name": "centcom.mil", "url": "https://vertexaisearch.cloud.google.com/r/2"},
            ],
        }
        relabel_cluster(cluster, [])
        assert cluster["verification"] == LABEL_OFFICIAL


class TestHtmlRenderer:
    def _render(self):
        from src.services.sitrep_html import render_sitrep_html
        report = (
            "YÖNETİCİ ÖZETİ\n"
            "Gerilim tırmanıyor & durum <kritik>.\n"
            "BÖLÜM I — SAHA OLAYLARI\n"
            "Bandar Abbas\n"
            "• [2026-07-16, saat belirsiz] Komuta merkezleri vuruldu — "
            "Doğruluk Durumu: Onaylandı (Resmî) — Kaynak: centcom.mil (https://centcom.mil/a)\n"
            "BÖLÜM III — STRATEJİK VE SİYASİ GELİŞMELER\n"
            "Bu bölüm için doğrulanmış veri bulunmamaktadır."
        )
        clusters = [{"verification": "Onaylandı (Resmî)", "sources": []}]
        return render_sitrep_html("İran", "IR", "2026-07-15 10:30",
                                  "2026-07-16 10:30", report, clusters)

    def test_structure_and_badges(self):
        html_out = self._render()
        assert "<!DOCTYPE html>" in html_out
        assert "viewport" in html_out                      # mobile-first
        assert "BÖLÜM I — SAHA OLAYLARI" in html_out
        assert "Onaylandı (Resmî)" in html_out             # badge text
        assert 'href="https://centcom.mil/a"' in html_out  # source chip link
        assert "📍 Bandar Abbas" in html_out               # location subheader

    def test_escapes_html_in_content(self):
        html_out = self._render()
        assert "<kritik>" not in html_out
        assert "&lt;kritik&gt;" in html_out

    def test_decorative_separator_lines_are_dropped(self):
        from src.services.sitrep_html import render_sitrep_html
        report = "YÖNETİCİ ÖZETİ\nDurum sakin.\n---\n***\nBandar Abbas\nDetay yok."
        html_out = render_sitrep_html("İran", "IR", "2026-07-15 10:30",
                                      "2026-07-16 10:30", report, [])
        assert "📍 ---" not in html_out
        assert ">---<" not in html_out and ">***<" not in html_out

    def test_exec_summary_rendered_as_callout(self):
        html_out = self._render()
        assert "📌 YÖNETİCİ ÖZETİ" in html_out
        assert "Gerilim tırmanıyor" in html_out

    def test_highlights_rank_clusters_by_severity(self):
        from src.services.sitrep_html import render_sitrep_html
        clusters = [
            {"location": "Az Önemli", "severity": 40, "event_type": "protest",
             "date": "2026-07-17", "verification": "Doğrulanmamış (Tek kaynak)",
             "snippet": "x", "sources": []},
            {"location": "Çok Önemli", "severity": 95, "event_type": "missile_strike",
             "date": "2026-07-17", "verification": "Onaylandı (Resmî)",
             "snippet": "y", "sources": []},
        ]
        html_out = render_sitrep_html("İran", "IR", "2026-07-15 10:30",
                                      "2026-07-16 10:30", "YÖNETİCİ ÖZETİ\nÖzet.", clusters)
        assert "GÜNÜN ÖNE ÇIKANLARI" in html_out
        assert html_out.index("Çok Önemli") < html_out.index("Az Önemli")
        assert "Füze Saldırısı" in html_out          # TR event-type label
        assert ">95<" in html_out                     # severity value as text
        assert "Maks. Şiddet" in html_out             # KPI tile

    def test_appendix_is_collapsible(self):
        from src.services.sitrep_html import render_sitrep_html
        clusters = [{"location": "X", "severity": 10, "event_type": "protest",
                     "date": "2026-07-17", "verification": None,
                     "snippet": "s", "sources": []}]
        html_out = render_sitrep_html("İran", "IR", "2026-07-15 10:30",
                                      "2026-07-16 10:30", "YÖNETİCİ ÖZETİ\nÖzet.", clusters)
        assert "<details" in html_out and "<summary" in html_out

    def test_appendix_lists_every_cluster_with_sources(self):
        from src.services.sitrep_html import render_sitrep_html
        clusters = [
            {"location": "Çabahar Limanı", "date": "2026-07-17, saat belirsiz",
             "event_type": "missile_strike", "verification": "Onaylandı (Çoklu kaynak)",
             "snippet": "Watchtower destroyed at the port.",
             "sources": [{"name": "reuters.com", "url": "https://reuters.com/x"}]},
            {"location": "Yezd", "date": "2026-07-17, saat belirsiz",
             "event_type": "explosion", "verification": "Doğrulanmamış (Tek kaynak)",
             "snippet": "Blast reported near the city.", "sources": []},
        ]
        html_out = render_sitrep_html("İran", "IR", "2026-07-15 10:30",
                                      "2026-07-16 10:30", "YÖNETİCİ ÖZETİ\nÖzet.", clusters)
        assert "GÜNLÜK OLAY KÜNYESİ" in html_out
        assert "Çabahar Limanı" in html_out and "Yezd" in html_out
        assert 'href="https://reuters.com/x"' in html_out
        assert "Watchtower destroyed" in html_out


class TestCorroborationLabeling:
    def _members(self):
        return [{
            "source_domain": "almayadeen.net", "source_url": "https://almayadeen.net/a",
            "source_title": "strike reported", "canonical_text": "strike reported",
            "anchor_name_raw": "Basra", "event_type": "missile_strike",
            "severity_score": 70, "time_certainty": "same_day",
            "occurred_at_est": "2026-07-17 03:00:00",
        }]

    def test_corroborating_source_upgrades_single_to_multi(self):
        from src.services.sitrep_generator import build_sitrep_clusters
        members = self._members()
        members[0]["corroborating_sources"] = [
            {"domain": "reuters.com", "url": "https://reuters.com/x", "title": "same strike"}]
        clusters = build_sitrep_clusters(members, [])
        assert clusters[0]["verification"] == LABEL_MULTI
        assert any(s["name"] == "reuters.com" for s in clusters[0]["sources"])

    def test_official_corroborating_source_upgrades_to_official(self):
        from src.services.sitrep_generator import build_sitrep_clusters
        members = self._members()
        members[0]["corroborating_sources"] = [
            {"domain": "centcom.mil", "url": "https://centcom.mil/x", "title": "statement"}]
        clusters = build_sitrep_clusters(members, [])
        assert clusters[0]["verification"] == LABEL_OFFICIAL

    def test_no_corroboration_stays_single(self):
        from src.services.sitrep_generator import build_sitrep_clusters
        clusters = build_sitrep_clusters(self._members(), [])
        assert clusters[0]["verification"] == LABEL_SINGLE


class TestSpilloverAliases:
    def test_us_search_covers_common_forms(self):
        from src.services.sitrep_generator import _country_mention_terms
        terms = _country_mention_terms("US", "United States")
        assert "U.S." in terms and "American forces" in terms
        assert "United States" in terms

    def test_no_bare_short_tokens(self):
        # %US% / %IR% substring-match everything — 1-2 char forms must never be
        # search terms. 3-char acronyms (IDF, UAE) are allowed: within the
        # security-filtered events table their substring collision rate is ~0.
        from src.services.sitrep_generator import _COUNTRY_ALIASES
        for aliases in _COUNTRY_ALIASES.values():
            assert all(len(a) >= 3 for a in aliases), aliases

    def test_unknown_country_falls_back_to_display_name(self):
        from src.services.sitrep_generator import _country_mention_terms
        assert _country_mention_terms("XX", "Wakanda") == ["Wakanda"]

    def test_spillover_query_binds_all_aliases(self):
        from datetime import datetime, timezone
        from src.services.sitrep_generator import fetch_spillover_events

        captured = {}
        class FakeConn:
            def execute(self, sql, params):
                captured["sql"], captured["params"] = sql, params
                class R:
                    def fetchall(self): return []
                return R()

        t0 = datetime(2026, 7, 16, tzinfo=timezone.utc)
        t1 = datetime(2026, 7, 17, tzinfo=timezone.utc)
        fetch_spillover_events(FakeConn(), "IR", "Iran", t0, t1)
        assert "%Tehran%" in captured["params"]
        assert "%IRGC%" in captured["params"]
        # placeholder count matches parameter count (2 per term + 3 fixed)
        assert captured["sql"].count("%s") == len(captured["params"])

