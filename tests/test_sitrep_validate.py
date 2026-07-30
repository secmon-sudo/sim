"""
validate_sitrep tolerance — a verification label the LLM editorialized (extra
words inside/after the parentheses) is normalized to the nearest canonical label
instead of failing the whole country report. Run #19 (29 Jul) lost IR's entire
SITREP to a single stray label: "Onaylandı (Çoklu kaynak, ancak detaylar
doğrulanmamış)". Normalization never raises the claimed confidence tier.
"""

import pytest

from src.core.sitrep_verify import LABEL_MULTI, LABEL_OFFICIAL, LABEL_SINGLE
from src.services.sitrep_generator import validate_sitrep

_HDR = "YÖNETİCİ ÖZETİ\nGünün özeti.\n"


def _label_line(label: str) -> str:
    return _HDR + f"— Bir olay. Doğruluk Durumu: {label}"


def _tail(out: str) -> str:
    return out.splitlines()[-1].split("Doğruluk Durumu:", 1)[1].strip()


class TestVerificationLabelNormalization:
    def test_run19_regression_multi_editorialized(self):
        out = validate_sitrep(
            _label_line("Onaylandı (Çoklu kaynak, ancak detaylar doğrulanmamış)"), [])
        assert _tail(out) == LABEL_MULTI

    def test_official_editorialized(self):
        out = validate_sitrep(_label_line("Onaylandı (Resmî, teyit sürüyor)"), [])
        assert _tail(out) == LABEL_OFFICIAL

    def test_single_editorialized(self):
        out = validate_sitrep(_label_line("Doğrulanmamış (Tek kaynak, şüpheli)"), [])
        assert _tail(out) == LABEL_SINGLE

    def test_canonical_unchanged(self):
        for lbl in (LABEL_OFFICIAL, LABEL_MULTI, LABEL_SINGLE):
            out = validate_sitrep(_label_line(lbl), [])
            assert _tail(out) == lbl

    def test_uninferrable_degrades_to_most_conservative(self):
        # Never invent verification: an unrecognisable label → single-source.
        out = validate_sitrep(_label_line("Teyit edilemedi bir şekilde"), [])
        assert _tail(out) == LABEL_SINGLE

    def test_confirmed_without_tier_stays_conservative(self):
        # "Onaylandı" with no Resmî/Çoklu keyword must not become OFFICIAL.
        out = validate_sitrep(_label_line("Onaylandı"), [])
        assert _tail(out) == LABEL_MULTI

    def test_source_remainder_preserved(self):
        line = (_HDR + "— Olay. Doğruluk Durumu: Onaylandı (Çoklu kaynak, ekstra)"
                " — Kaynak: reuters (http://r)")
        out = validate_sitrep(line, ["http://r"])
        assert LABEL_MULTI in out
        assert "— Kaynak: reuters" in out
        assert "ekstra" not in out

    def test_never_raises_on_bad_label(self):
        # The whole point: a stray label no longer fails the report.
        validate_sitrep(_label_line("garbage xyz"), [])

    def test_missing_header_still_raises(self):
        with pytest.raises(ValueError):
            validate_sitrep("no header at all", [])

    def test_unknown_url_still_masked(self):
        out = validate_sitrep(_HDR + "Bkz http://evil.example/x", [])
        assert "evil.example" not in out
        assert "[kaynak listede]" in out
