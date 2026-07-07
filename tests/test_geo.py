"""Tests for the coarse geo_key resolver (src/core/geo.py)."""

from src.core.geo import geo_key


class TestTransliterationAliases:
    def test_kyiv_variants_collapse(self):
        keys = {geo_key(v) for v in ("Kyiv", "Kiev", "KYEV", "kyiv")}
        assert keys == {"KYIV"}

    def test_kharkiv_variants_collapse(self):
        assert geo_key("Kharkov") == geo_key("Kharkiv") == "KHARKIV"

    def test_multiword_alias(self):
        assert geo_key("Gaza Strip") == "GAZA"

    def test_alias_inside_phrase(self):
        assert geo_key("near Kyiv suburb") == "KYIV"


class TestAdminSuffixStripping:
    def test_city_suffix_stripped(self):
        assert geo_key("Kyiv city") == "KYIV"

    def test_oblast_suffix_stripped(self):
        assert geo_key("Kharkiv Oblast") == "KHARKIV"

    def test_gaza_city_alias(self):
        assert geo_key("Gaza City") == "GAZA"


class TestCapitalResolution:
    def test_country_capital_phrase(self):
        assert geo_key("Ukrainian capital", country_iso="UA") == "KYIV"

    def test_capital_word_with_iso(self):
        assert geo_key("the capital", country_iso="RU") == "MOSCOW"

    def test_capital_without_iso_is_not_resolved(self):
        # No ISO hint -> cannot know which capital; falls back to text key.
        assert geo_key("capital") == "CAPITAL"

    def test_same_event_paraphrase_links(self):
        # The core bug: "Kyiv" and "Ukraine capital" must share a key.
        assert geo_key("Kyiv") == geo_key("Ukraine capital", country_iso="UA")


class TestFallbackAndGuards:
    def test_unknown_city_falls_back_to_normalized_text(self):
        assert geo_key("Timbuktu") == "TIMBUKTU"

    def test_identical_unknown_strings_match(self):
        assert geo_key("Some Town") == geo_key("some town") == "SOME TOWN"

    def test_none_input(self):
        assert geo_key(None) is None

    def test_empty_input(self):
        assert geo_key("") is None
        assert geo_key("   ") is None

    def test_punctuation_only(self):
        assert geo_key("!!!") is None

    def test_non_string(self):
        assert geo_key(123) is None
