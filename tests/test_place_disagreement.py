"""One incident's headline is not another incident's evidence.

Measured 2026-08-20. Wire copy about a single war shares one headline scaffold —
"Russian missile strike on X kills N" — and the ingest deduper scores headlines on
character overlap at a 0.65 threshold. "Russian missile strike on Kharkiv region
kills ten" scored 0.667 against "Russian Missile Attack on Kyiv Kills 12": the
Kharkiv filing was dropped as a duplicate and credited to the Kyiv event as
corroboration. That day's UA SITREP then cited a Kharkiv link, under a Kyiv
cluster, next to Bloomberg and Al Jazeera, labelled "Onaylandı (Çoklu kaynak)".

The threshold is not the fix: over a labelled sample of that day's merges the
similarity scores of the true and false pairs overlap completely (bad: .66–.77,
good: .67–.83). The place claim is what separates them.
"""

from src.core.geo import place_keys, places_disagree
from src.pipeline.ingest_filters import find_content_duplicate
from src.services.sitrep_generator import build_sitrep_clusters, _EVENT_COLUMNS

KYIV = "Russian Missile Attack on Kyiv Kills 12, Wounds 33 - Bloomberg.com"
KHARKIV = "Russian missile strike on Kharkiv region kills ten - Apa.az"


class TestPlaceKeys:
    def test_reads_places_out_of_a_headline(self):
        assert place_keys(KYIV) == {"KYIV"}
        assert place_keys(KHARKIV) == {"KHARKIV"}

    def test_transliterations_collapse(self):
        assert place_keys("Explosions rock Kiev overnight") == {"KYIV"}

    def test_kherson_is_known(self):
        assert place_keys("Drone strike on minibus in Kherson kills four") == {"KHERSON"}

    def test_unknown_places_yield_no_opinion(self):
        assert place_keys("Blast at Pechenihy village") == set()

    def test_disagreement_needs_two_positive_claims(self):
        assert places_disagree(KYIV, KHARKIV) is True
        # One side naming no city is a legitimate retelling, not a contradiction.
        assert places_disagree("Russian strike kills 12", KYIV) is False
        assert places_disagree(KYIV, KYIV) is False


class TestIngestDedup:
    def _recent(self):
        return [(KYIV, KYIV), (KHARKIV, KHARKIV)]

    def test_kharkiv_filing_is_not_credited_to_the_kyiv_event(self):
        idx = find_content_duplicate(self._recent(), KHARKIV, KHARKIV)
        assert idx == 1, "must match the Kharkiv event, not the Kyiv one"

    def test_same_place_duplicates_still_collapse(self):
        idx = find_content_duplicate(
            self._recent(),
            "Russian missile attack on Kyiv kills 12, wounds dozens - NV",
            "Russian missile attack on Kyiv kills 12, wounds dozens")
        assert idx == 0

    def test_placeless_rewording_still_collapses(self):
        idx = find_content_duplicate(
            [("Twelve killed as missiles hit residential blocks", "x" * 20)],
            "Twelve killed as missiles hit residential block", "y" * 20)
        assert idx == 0


class TestClusterSources:
    """Second guard: rows corroborated before the ingest fix stay in the column
    for the retention window, and the SITREP appendix is where a reader sees them.
    """

    def _event(self, **over):
        d = {c: None for c in _EVENT_COLUMNS}
        d.update(id="e1", source_title=KYIV, source_domain="bloomberg.com",
                 source_url="https://bloomberg.com/a", event_type="missile_strike",
                 country_iso="UA", severity_score=100, anchor_name_raw="Kyiv",
                 canonical_text="Missiles hit Kyiv.", corroborating_sources=[])
        d.update(over)
        return d

    def test_off_place_corroboration_is_dropped(self):
        ev = self._event(corroborating_sources=[
            {"domain": "apa.az", "url": "https://apa.az/x", "title": KHARKIV},
            {"domain": "nv.ua", "url": "https://nv.ua/y",
             "title": "Russian missile attack on Kyiv kills 12, wounds dozens"},
        ])
        [cluster] = build_sitrep_clusters([ev], [])
        names = [s["name"] for s in cluster["sources"]]
        assert "apa.az" not in names
        assert "nv.ua" in names
