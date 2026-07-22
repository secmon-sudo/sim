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


def country(iso, status="completed", severities=(50,), text="rapor metni"):
    return {
        "country_iso": iso,
        "country_name": f"Ülke-{iso}",
        "status": status,
        "report_text": text if status == "completed" else None,
        "clusters": [{"severity": s} for s in severities],
    }


class TestComputeRiskLevel:
    def test_critical_on_severity_alone(self):
        assert compute_risk_level(90, 1) == RISK_CRITICAL
        assert compute_risk_level(97, 1) == RISK_CRITICAL

    def test_critical_on_high_severity_plus_volume(self):
        assert compute_risk_level(82, 8) == RISK_CRITICAL
        assert compute_risk_level(82, 7) == RISK_HIGH

    def test_bands(self):
        assert compute_risk_level(80, 1) == RISK_HIGH
        assert compute_risk_level(60, 3) == RISK_ELEVATED
        assert compute_risk_level(59, 20) == RISK_NORMAL
        assert compute_risk_level(0, 0) == RISK_NORMAL


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


SAMPLE = """GÜNÜN TABLOSU
Bölgede gerilim tırmandı. Havayolları uçuşlarını askıya aldı.

ÜLKE DURUMU
- IR | Çok sayıda tesis vuruldu.
- BH | Üsse İHA saldırısı düzenlendi.
- ZZ | Uydurma ülke.

HAVACILIK ETKİSİ
- Emirates Tahran uçuşlarını askıya aldı.
- Manama havalimanı kapandı. Kaynak: reuters.com
- Lufthansa rota değiştirdi. https://example.com/haber

ÖNE ÇIKANLAR
- **Bandar Abbas** limanında yangın çıktı.

İZLEME
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
        text = "GÜNÜN TABLOSU\nSakin bir gün.\n\nHAVACILIK ETKİSİ\nYOK\n"
        assert parse_digest(text, [])["aviation"] == []

    def test_verification_labels_are_stripped(self):
        text = ("GÜNÜN TABLOSU\nDurum.\n\nÖNE ÇIKANLAR\n"
                "- Üsse saldırı — Doğruluk Durumu: Onaylandı\n")
        assert parse_digest(text, [])["highlights"] == ["Üsse saldırı"]


class TestValidateDigest:
    def test_rejects_missing_overview(self):
        with pytest.raises(ValueError, match="overview"):
            validate_digest(parse_digest("ÖNE ÇIKANLAR\n- bir şey\n", []))

    def test_accepts_overview_only(self):
        assert validate_digest(parse_digest("GÜNÜN TABLOSU\nDurum sakin.\n", []))


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
            lambda *a, **kw: {"content": "GÜNÜN TABLOSU\nDurum.\n\nÜLKE DURUMU\n- IR | Vuruldu.\n",
                              "provider": "p", "model": "m"},
        )
        d = build_digest(None, [country("IR", severities=(95,)), country("BH")], "s", "e")
        assert [c["iso"] for c in d["countries"]] == ["IR", "BH"]
        assert d["countries"][0]["risk"] == RISK_CRITICAL
        assert d["countries"][1]["text"]  # placeholder, not empty

    def test_risk_levels_come_from_severity_not_llm(self, monkeypatch):
        monkeypatch.setattr(
            "src.services.sitrep_digest.run_digest_llm",
            lambda *a, **kw: {"content": "GÜNÜN TABLOSU\nDurum.\n\nÜLKE DURUMU\n"
                                         "- IR | Sakin bir gün yaşandı.\n- BH | Kritik durum.\n",
                              "provider": "p", "model": "m"},
        )
        d = build_digest(None, [country("IR", severities=(95,)),
                                country("BH", severities=(10,))], "s", "e")
        by_iso = {c["iso"]: c["risk"] for c in d["countries"]}
        assert by_iso == {"IR": RISK_CRITICAL, "BH": RISK_NORMAL}
