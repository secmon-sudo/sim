"""A weak anchor must not be used as a location key.

Regression for the 2026-08-16 finding: the Varanasi airport shooting paged TWICE
because two reports of one incident fuzzy-matched to different wrong airports (VAR and
SHJ), producing different suppression keys. geo_key on the raw text ("Varanasi
airport") would have collapsed them — so the bad anchor was worse than no anchor.
"""

from src.core.alerts import build_geo_suppression_key, build_suppression_key
from src.core.geo import trusted_anchor
from src.core.storyline_adjudicator import _event_geo


def _report(anchor_norm, level, raw="Varanasi airport", iso="IN", storyline="s-1"):
    return {
        "id": f"ev-{anchor_norm}",
        "storyline_id": storyline,
        "anchor_name_norm": anchor_norm,
        "anchor_confidence": level,
        "anchor_name_raw": raw,
        "country_iso": iso,
    }


class TestTrustedAnchor:
    def test_low_confidence_anchor_is_not_trusted(self):
        assert trusted_anchor(_report("VAR", "LOW")) is None

    def test_medium_and_high_are_trusted(self):
        assert trusted_anchor(_report("KBP", "MEDIUM")) == "KBP"
        assert trusted_anchor(_report("KBP", "HIGH")) == "KBP"

    def test_missing_column_reads_as_trusted(self):
        """Fail-open: a query that omits anchor_confidence must not strip every anchor."""
        assert trusted_anchor({"anchor_name_norm": "KBP"}) == "KBP"

    def test_no_anchor_is_none(self):
        assert trusted_anchor({"anchor_name_raw": "Varanasi airport"}) is None


class TestSuppressionKeysCollapse:
    def test_two_bad_anchors_for_one_incident_share_a_geo_key(self):
        """The exact double-page that shipped: VAR/Bulgaria and SHJ/UAE, one incident."""
        varna = _report("VAR", "LOW")
        sharjah = _report("SHJ", "LOW")
        assert build_geo_suppression_key(varna) == build_geo_suppression_key(sharjah)

    def test_collapsed_key_uses_the_raw_location_not_the_wrong_airport(self):
        key = build_geo_suppression_key(_report("VAR", "LOW"))
        assert key.split("|")[-1] != "VAR"
        assert "VARANASI" in key.upper()
        assert key.split("|")[1] == "IN"  # the classifier's country, not Bulgaria's

    def test_trusted_anchor_still_keys_precisely(self):
        assert build_geo_suppression_key(
            _report("KBP", "HIGH", raw="Boryspil", iso="UA")
        ) == "geofp|UA|KBP"

    def test_primary_key_does_not_fragment_on_weak_anchors(self):
        varna = _report("VAR", "LOW")
        sharjah = _report("SHJ", "LOW")
        assert build_suppression_key(varna) == build_suppression_key(sharjah)


class TestAdjudicatorGeo:
    def test_weak_anchor_falls_back_to_raw_text(self):
        assert _event_geo(_report("VAR", "LOW")) == _event_geo(_report("SHJ", "LOW"))

    def test_trusted_anchor_is_kept(self):
        assert _event_geo(_report("KBP", "HIGH", raw="Boryspil", iso="UA")) == "KBP"
