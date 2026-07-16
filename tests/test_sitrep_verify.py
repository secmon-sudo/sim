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


class TestGoogleNewsDecode:
    def test_legacy_id_decodes_to_publisher_url(self):
        import base64
        from src.services.sitrep_web_enrich import decode_google_news_url
        # legacy format: base64 payload embeds the article URL directly
        inner = b"\x08\x13\x22" + bytes([len(b"https://example.com/story-1")]) \
            + b"https://example.com/story-1" + b"\xd2\x01\x00"
        token = base64.urlsafe_b64encode(inner).decode().rstrip("=")
        url = f"https://news.google.com/rss/articles/{token}?oc=5"
        assert decode_google_news_url(url) == "https://example.com/story-1"

    def test_non_google_url_returns_none(self):
        from src.services.sitrep_web_enrich import decode_google_news_url
        assert decode_google_news_url("https://reuters.com/a") is None

    def test_opaque_new_format_returns_none_or_url(self):
        from src.services.sitrep_web_enrich import decode_google_news_url
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


class TestDiscoverIncidents:
    def _fake_gemini(self, text, sources, supports):
        return {"text": text, "sources": sources, "supports": supports}

    def test_supported_lines_become_clusters(self, monkeypatch):
        import src.services.sitrep_web_enrich as enr
        text = (
            "LOKASYON: Erbil | OLAY: ABD Konsolosluğu yakınına füze saldırısı düzenlendi.\n"
            "LOKASYON: Basra | OLAY: Petrol tesisinde patlama meydana geldi."
        )
        sources = [
            {"name": "reuters.com", "url": "https://v.example/1", "title": "reuters.com"},
            {"name": "centcom.mil", "url": "https://v.example/2", "title": "centcom.mil"},
        ]
        supports = [
            ("LOKASYON: Erbil | OLAY: ABD Konsolosluğu yakınına füze saldırısı", [0, 1]),
        ]
        monkeypatch.setattr(enr, "_call_gemini",
                            lambda *a, **k: self._fake_gemini(text, sources, supports))
        clusters = enr.discover_incidents("Iraq", "fake-key", [])
        # Basra line has no grounding support → dropped; Erbil kept with both sources
        assert len(clusters) == 1
        assert clusters[0]["location"] == "Erbil"
        assert {s["name"] for s in clusters[0]["sources"]} == {"reuters.com", "centcom.mil"}

    def test_no_findings(self, monkeypatch):
        import src.services.sitrep_web_enrich as enr
        monkeypatch.setattr(enr, "_call_gemini",
                            lambda *a, **k: self._fake_gemini("EK_BILGI_YOK", [], []))
        assert enr.discover_incidents("Iraq", "fake-key", []) == []


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
