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
        assert place_keys("Blast at Hlukhiv village") == set()

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


class TestContainment:
    """Measured 2026-08-21: 172 of the ~330 anchored events a day name a place the
    curated city table does not, and the recurring ones are villages inside a
    region. Added flat they would read as different places, and the veto would
    split one strike in two — so the table records what contains what.
    """

    def test_village_agrees_with_its_region(self):
        assert places_disagree(
            "Russia kills 10 in missile strike on Pechenihy",
            "Russian missile strike on Kharkiv region kills ten") is False

    def test_village_still_disagrees_with_another_oblast(self):
        assert places_disagree(
            "10 killed in Russian missile strike on Pechenihy",
            "One killed, seven injured in Russian strike on Kyiv region") is True

    def test_west_bank_village_agrees_with_the_territory(self):
        assert places_disagree("Settlers attack Qusra village",
                               "Israeli raid in the West Bank kills two") is False

    def test_west_bank_village_disagrees_with_gaza(self):
        assert places_disagree("Settlers attack Qusra village",
                               "Israeli strike on Gaza kills six") is True

    def test_district_agrees_with_its_province(self):
        assert places_disagree("Five terrorists killed in Panjgur",
                               "Five terrorists killed in Balochistan IBO") is False


class TestAnchorAsPlaceEvidence:
    """The wire headline omits the town its own body names — "Russian missile
    strike on Ukrainian town" — and the anchor is where the classifier recorded
    it. Only the stored side can carry one: Pass C has not seen the incoming item.
    """

    STORED = "10 killed, 8 injured in Russian missile strike on Ukrainian town"
    INCOMING = "One killed, seven injured in Russian strike on Kyiv region"

    def test_headline_alone_cannot_tell_them_apart(self):
        assert places_disagree(self.STORED, self.INCOMING) is False
        assert find_content_duplicate([(self.STORED, self.STORED)],
                                      self.INCOMING, self.INCOMING) == 0

    def test_anchor_supplies_the_missing_place(self):
        recent = [(self.STORED, self.STORED, "Pechenihy")]
        assert find_content_duplicate(recent, self.INCOMING, self.INCOMING) is None

    def test_anchor_does_not_break_a_true_duplicate(self):
        """The anchor names the village; a duplicate naming the oblast it sits in
        still collapses, because containment is what the comparison expands."""
        recent = [(self.STORED, self.STORED, "Pechenihy")]
        dupe = "10 killed, 8 injured in Russian missile strike on Kharkiv region town"
        assert find_content_duplicate(recent, dupe, dupe) == 0

    def test_two_tuple_entries_still_work(self):
        assert find_content_duplicate([("A blast hit a market", "x")],
                                      "A blast hit a market", "y") == 0


class TestAgainstProductionPairs:
    """Ten corroboration pairs sampled from the 2026-08-21 corpus, run through the
    anchor-augmented rule. The two Pechenihy rows are the ones the headline-only
    veto let through the day before: a strike on a village in Kharkiv oblast filed
    as evidence for the Kyiv barrage.
    """

    CASES = [
        ("10 killed, 8 injured in Russian missile strike on Ukrainian town", "Pechenihy",
         "12 killed, 33 injured in Russian strikes on Kyiv and surrounding region", True),
        ("10 killed, 8 injured in Russian missile strike on Ukrainian town", "Pechenihy",
         "5 killed in Russian missile strikes on Kyiv - Pajhwok Afghan News", True),
        ("Russian missile barrage across Ukraine's capital kills at least 14",
         "Ukraine's capital",
         "Russian missile barrage across Ukraine's capital kills at least 12", False),
        ("Private plane from D.C. makes emergency landing at CAK", "CAK",
         "Plane makes emergency landing at Akron-Canton Airport - WKYC", False),
        ("Harrowing moment Russian ballistic missile hits Kyiv skyline: Killing 8",
         "Kyiv skyline", "Russian ballistic missile attack sets Kyiv skyline ablaze", False),
        ("Hamas Police Commanders Killed in Israeli Gaza Strike", "Gaza",
         "Hamas Commander Killed in Nuseirat", False),
        ("US Envoy Visits Ladakh, Signals Possible Travel Advisory Shift", "Ladakh",
         "US Envoy Sergio Gor Signals Possible Easing of Travel Advisory for J&K", False),
        ("Israeli strikes hit southern Lebanon", "southern Lebanon",
         "Israeli warplanes carry out overnight strikes in southern Lebanon", False),
        ("Gaza: Palestinians hold funeral for 50 people killed in Israeli attacks",
         "Al-Shifa Hospital",
         "Gaza holds mass funeral for 50 Palestinians killed in Israeli genocidal war", False),
        ("Four killed in Russian drone attack on a bus in Kherson region", "Kherson",
         "Four killed, five injured in Russian drone attack on bus in Kherson", False),
    ]

    def test_every_sampled_pair_lands_the_right_way(self):
        wrong = [
            (anchor, other) for title, anchor, other, expect in self.CASES
            if places_disagree(other, f"{title} {anchor}") is not expect
        ]
        assert wrong == []
