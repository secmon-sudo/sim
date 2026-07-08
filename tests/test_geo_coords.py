"""Tests for the curated city coordinate gazetteer (geo_coords)."""

from src.core.geo import geo_coords


class TestGeoCoords:
    def test_canonical_city(self):
        lat, lon, iso = geo_coords("Kyiv")
        assert round(lat) == 50 and round(lon) == 31 and iso == "UA"

    def test_transliteration_alias_resolves(self):
        # "Kiev" → KYIV via the same alias table geo_key uses.
        assert geo_coords("Kiev") == geo_coords("Kyiv")

    def test_admin_suffix_stripped(self):
        # "Aleppo province" → ALEPPO
        assert geo_coords("Aleppo province") is not None

    def test_capital_phrasing(self):
        # "Ukrainian capital" with iso hint → KYIV coordinates.
        assert geo_coords("Ukrainian capital", "UA") == geo_coords("Kyiv")

    def test_unknown_place_is_none(self):
        assert geo_coords("Springfield") is None

    def test_none_input(self):
        assert geo_coords(None) is None

    def test_iso_hint_rejects_name_collision(self):
        # Aleppo is Syrian; a hint of a different country rejects the coordinate rather
        # than planting a misplaced one.
        assert geo_coords("Aleppo", "US") is None
        assert geo_coords("Aleppo", "SY") is not None

    def test_iso_backfilled_from_gazetteer(self):
        _, _, iso = geo_coords("Mogadishu")
        assert iso == "SO"


class TestAnchorResolutionUsesCoords:
    def test_city_event_gets_coordinates(self, monkeypatch):
        # A city event with no IATA anchor should still get lat/lon from the gazetteer.
        import src.pipeline.pass_d_score as pd

        monkeypatch.setattr(pd, "normalize_anchor", lambda raw, db: (None, 0.0))

        class FakeConn:
            def execute(self, *a, **k):
                raise AssertionError("anchor_master should not be queried when norm is None")

        anchor = pd.resolve_anchor_for_event(FakeConn(), {
            "anchor_name_raw": "Kharkiv", "country_iso": "UA",
        })
        assert anchor["latitude"] is not None
        assert anchor["longitude"] is not None
        assert anchor["country_iso"] == "UA"
