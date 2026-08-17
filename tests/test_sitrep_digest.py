"""Tests for the daily cross-country SITREP digest (src/services/sitrep_digest.py)."""

import pytest

from src.services.sitrep_digest import (
    RISK_CRITICAL,
    RISK_ELEVATED,
    RISK_HIGH,
    RISK_NORMAL,
    build_digest,
    build_digest_inputs,
    compute_risk_level,
    parse_digest,
    validate_digest,
)


def country(iso, status="completed", severities=(50,), text="rapor metni",
            verification=None):
    return {
        "country_iso": iso,
        "country_name": f"Ülke-{iso}",
        "status": status,
        "report_text": text if status == "completed" else None,
        "clusters": [{"severity": s, "verification": verification} for s in severities],
    }


class TestComputeRiskLevel:
    """Run #22 (1 Aug 2026): all five countries came out Kritik, because country
    selection picks BY severity and a single unverified 100 was enough. The tier
    now turns on CORROBORATED severe volume, so the scale can vary again."""

    def test_unverified_severity_alone_is_not_critical(self):
        assert compute_risk_level(90, 1, confirmed_severe=0) == RISK_HIGH
        assert compute_risk_level(100, 3, confirmed_severe=0) == RISK_HIGH

    def test_corroborated_severe_volume_is_critical(self):
        assert compute_risk_level(95, 5, confirmed_severe=3) == RISK_CRITICAL
        assert compute_risk_level(95, 10, confirmed_severe=2) == RISK_CRITICAL
        assert compute_risk_level(95, 9, confirmed_severe=2) == RISK_HIGH
        assert compute_risk_level(95, 20, confirmed_severe=1) == RISK_HIGH

    def test_bands(self):
        """Points re-derived 2026-08-17 with the severity catalog compression.

        These were 80/65/65/59 on the saturated scale, where an event type could
        reach the ceiling on its label alone. The bands moved with the scale, so the
        test moved with them — same positions relative to the thresholds, mapped
        through the same compression (80->68, 65->62, 59->59).
        """
        assert compute_risk_level(68, 1) == RISK_ELEVATED
        assert compute_risk_level(62, 5) == RISK_ELEVATED
        assert compute_risk_level(62, 4) == RISK_NORMAL
        assert compute_risk_level(59, 20) == RISK_NORMAL
        assert compute_risk_level(0, 0) == RISK_NORMAL

    def test_severe_band_still_separates(self):
        """The compression must not collapse HIGH into ELEVATED."""
        assert compute_risk_level(71, 1) == RISK_HIGH
        assert compute_risk_level(70, 1) == RISK_ELEVATED


class TestBuildDigestInputs:
    def test_drops_failed_and_empty_countries(self):
        rows = build_digest_inputs([
            country("IR"),
            country("IQ", status="failed"),
            country("SY", status="empty"),
        ])
        assert [r["iso"] for r in rows] == ["IR"]

    def test_sorted_by_risk_then_severity(self):
        rows = build_digest_inputs([
            country("AA", severities=(20,)),
            country("BB", severities=(95,)),
            country("CC", severities=(70,)),
        ])
        assert [r["iso"] for r in rows] == ["BB", "CC", "AA"]

    def test_report_text_is_capped(self):
        rows = build_digest_inputs([country("IR", text="x" * 99_999)])
        assert len(rows[0]["report_text"]) == 3500


SAMPLE = """GENEL DURUM DEĞERLENDİRMESİ
Bölgede gerilim tırmandı. Havayolları uçuşlarını askıya aldı.

ÜLKE DEĞERLENDİRMELERİ
- IR | Çok sayıda tesis vuruldu.
- BH | Üsse İHA saldırısı düzenlendi.
- ZZ | Uydurma ülke.

HAVACILIK OPERASYONLARINA ETKİ
- Emirates Tahran uçuşlarını askıya aldı.
- Manama havalimanı kapandı. Kaynak: reuters.com
- Lufthansa rota değiştirdi. https://example.com/haber

KRİTİK GELİŞMELER
- **Bandar Abbas** limanında yangın çıktı.

İZLEME VE BEKLENTİLER
- Hürmüz Boğazı'nda seyrüsefer güvenliği.
"""


class TestParseDigest:
    def test_sections_are_split(self):
        p = parse_digest(SAMPLE, ["IR", "BH"])
        assert "gerilim tırmandı" in p["overview"]
        assert len(p["aviation"]) == 3
        assert len(p["highlights"]) == 1
        assert len(p["watch"]) == 1

    def test_unknown_country_is_dropped(self):
        p = parse_digest(SAMPLE, ["IR", "BH"])
        assert [c["iso"] for c in p["countries"]] == ["IR", "BH"]

    def test_source_attribution_and_urls_are_stripped(self):
        p = parse_digest(SAMPLE, ["IR", "BH"])
        joined = " ".join(p["aviation"])
        assert "Kaynak:" not in joined
        assert "http" not in joined
        assert "Manama havalimanı kapandı." in p["aviation"]

    def test_markdown_is_stripped(self):
        p = parse_digest(SAMPLE, ["IR", "BH"])
        assert p["highlights"][0].startswith("Bandar Abbas")

    def test_empty_marker_section_yields_no_items(self):
        text = "GENEL DURUM DEĞERLENDİRMESİ\nSakin bir gün.\n\nHAVACILIK OPERASYONLARINA ETKİ\nYOK\n"
        assert parse_digest(text, [])["aviation"] == []

    def test_verification_labels_are_stripped(self):
        text = ("GENEL DURUM DEĞERLENDİRMESİ\nDurum.\n\nKRİTİK GELİŞMELER\n"
                "- Üsse saldırı — Doğruluk Durumu: Onaylandı\n")
        assert parse_digest(text, [])["highlights"] == ["Üsse saldırı"]


class TestValidateDigest:
    def test_rejects_missing_overview(self):
        with pytest.raises(ValueError, match="overview"):
            validate_digest(parse_digest("KRİTİK GELİŞMELER\n- bir şey\n", []))

    def test_accepts_overview_only(self):
        assert validate_digest(parse_digest("GENEL DURUM DEĞERLENDİRMESİ\nDurum sakin.\n", []))


class TestBuildDigest:
    def test_skipped_below_two_countries(self):
        def boom(*a, **kw):
            raise AssertionError("LLM must not be called")

        assert build_digest(boom, [country("IR")], "s", "e") is None
        assert build_digest(boom, [], "s", "e") is None

    def test_uncovered_country_still_listed(self, monkeypatch):
        # Model narrated only IR; BH had a report and must not vanish silently.
        monkeypatch.setattr(
            "src.services.sitrep_digest.run_digest_llm",
            lambda *a, **kw: {"content": "GENEL DURUM DEĞERLENDİRMESİ\nDurum.\n\nÜLKE DEĞERLENDİRMELERİ\n- IR | Vuruldu.\n",
                              "provider": "p", "model": "m"},
        )
        d = build_digest(None, [country("IR", severities=(95,)), country("BH")], "s", "e")
        assert [c["iso"] for c in d["countries"]] == ["IR", "BH"]
        assert d["countries"][0]["risk"] == RISK_HIGH
        assert d["countries"][1]["text"]  # placeholder, not empty

    def test_risk_levels_come_from_severity_not_llm(self, monkeypatch):
        monkeypatch.setattr(
            "src.services.sitrep_digest.run_digest_llm",
            lambda *a, **kw: {"content": "GENEL DURUM DEĞERLENDİRMESİ\nDurum.\n\nÜLKE DEĞERLENDİRMELERİ\n"
                                         "- IR | Sakin bir gün yaşandı.\n- BH | Kritik durum.\n",
                              "provider": "p", "model": "m"},
        )
        d = build_digest(None, [country("IR", severities=(95, 95, 95),
                                        verification="Onaylandı (Çoklu kaynak)"),
                                country("BH", severities=(10,))], "s", "e")
        by_iso = {c["iso"]: c["risk"] for c in d["countries"]}
        assert by_iso == {"IR": RISK_CRITICAL, "BH": RISK_NORMAL}
