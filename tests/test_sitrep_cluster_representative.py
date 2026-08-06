"""
Regression cover for the SITREP clustering defects seen in run #26 (2026-08-05).

A 32-event Kyiv storyline was rendered as a single cluster headlined "Russian
missiles strike enterprise in Bohodukhiv, three people injured", citing Odesa, a
Kyiv high-rise fire and a Reuters Kyiv report as corroboration, and labelled
"Onaylandı (Çoklu kaynak)". The report that reached the reader stated the Kyiv
casualty toll "henüz netleşmedi" while the same window held ~15 filings of
"15 killed, 51 injured".

Two independent causes, one test file:

1. Representative selection could not discriminate. The sort key was
   (not official, -severity); no member was official and Pass D saturates
   severity at 100, so every key tied and Python's stable sort simply kept
   whatever order Postgres returned — the representative was arbitrary.
2. Clusters were storyline-scoped. A storyline tracks a running campaign and
   legitimately spans cities, but every source in a cluster is presented as
   corroborating one incident.
"""

from datetime import datetime, timezone

from src.core.sitrep_verify import LABEL_MULTI, LABEL_SINGLE
from src.services.sitrep_generator import build_sitrep_clusters

KYIV_STORYLINE = "2c09a725-8220-467c-a90d-b49d14331156"


def _event(title, domain, place, hour, storyline=KYIV_STORYLINE, severity=100, text=None):
    return {
        "source_title": title,
        "source_url": f"https://{domain}/a",
        "source_domain": domain,
        "event_type": "missile_strike",
        "occurred_at_est": datetime(2026, 8, 5, hour, tzinfo=timezone.utc),
        "published_at": datetime(2026, 8, 5, hour, tzinfo=timezone.utc),
        "time_certainty": "same_day",
        "anchor_name_raw": place,
        "anchor_name_norm": None,
        "country_iso": "UA",
        "severity_score": severity,
        "storyline_id": storyline,
        "canonical_text": text or title,
        "corroborating_sources": [],
        "latitude": None,
        "longitude": None,
    }


def _production_order():
    """Members in the order Postgres returned them — Bohodukhiv first."""
    return [
        _event("Russian missiles strike enterprise in Bohodukhiv, three people injured",
               "ukrinform.net", "Bohodukhiv", 11),
        _event("Odesa Suburb Missile Strike Wounds Three, Blast Caused by Gas Pipeline Impact",
               "112.ua", "Odesa Suburb", 13),
        _event("Explosions reported in Kyiv as Russia launches ballistic missile attack",
               "kyivindependent.com", "Kyiv", 21),
        _event("Russian ballistic missile attack kills 15, injures 51 in Kyiv and surrounding area",
               "kyivindependent.com", "Kyiv", 22),
        _event("Russian Attack Kills 17 in Ukraine", "wsj.com", "Kyiv", 7),
        _event("Massive Russian Strike Leaves At Least 15 Dead in Kyiv Oblast",
               "novinite.com", "Kyiv Oblast", 7),
        _event("Russian massive attack on Kyiv region kills 14, injures nearly 30",
               "rbc.ua", "Kyiv region", 6),
    ]


def _by_location(clusters):
    return {c["location"]: c for c in clusters}


class TestLocationScopedClusters:
    def test_other_cities_do_not_join_the_kyiv_cluster(self):
        by_loc = _by_location(build_sitrep_clusters(_production_order(), []))
        assert "Bohodukhiv" in by_loc
        assert "Odesa Suburb" in by_loc
        kyiv_sources = {s["name"] for s in by_loc["Kyiv"]["sources"]}
        assert "ukrinform.net" not in kyiv_sources
        assert "112.ua" not in kyiv_sources

    def test_single_source_city_is_not_labelled_multi_source(self):
        # The production bug handed Bohodukhiv four unrelated sources and
        # promoted it to "Onaylandı (Çoklu kaynak)".
        by_loc = _by_location(build_sitrep_clusters(_production_order(), []))
        assert by_loc["Bohodukhiv"]["verification"] == LABEL_SINGLE
        assert len(by_loc["Bohodukhiv"]["sources"]) == 1

    def test_administrative_suffixes_do_not_fragment_one_strike(self):
        # "Kyiv", "Kyiv Oblast" and "Kyiv region" are the same strike.
        locations = [c["location"] for c in build_sitrep_clusters(_production_order(), [])]
        assert locations.count("Kyiv") == 1
        assert "Kyiv Oblast" not in locations
        assert "Kyiv region" not in locations

    def test_genuinely_multi_source_city_keeps_its_label(self):
        by_loc = _by_location(build_sitrep_clusters(_production_order(), []))
        assert by_loc["Kyiv"]["verification"] == LABEL_MULTI


class TestRepresentativeSelection:
    def test_best_informed_member_headlines_the_cluster(self):
        by_loc = _by_location(build_sitrep_clusters(_production_order(), []))
        assert "17" in by_loc["Kyiv"]["snippet"]

    def test_representative_is_not_decided_by_row_order(self):
        forward = build_sitrep_clusters(_production_order(), [])
        reversed_rows = build_sitrep_clusters(list(reversed(_production_order())), [])
        assert _by_location(forward)["Kyiv"]["snippet"] == \
            _by_location(reversed_rows)["Kyiv"]["snippet"]

    def test_toll_free_filing_does_not_outrank_a_counted_one(self):
        events = [
            _event("Kyiv under ballistic missile attack", "breakingthenews.net", "Kyiv", 22),
            _event("Russian Attack Kills 17 in Ukraine", "wsj.com", "Kyiv", 7),
        ]
        clusters = build_sitrep_clusters(events, [])
        assert "17" in clusters[0]["snippet"]

    def test_mass_casualty_cluster_leads_the_report(self):
        clusters = build_sitrep_clusters(_production_order(), [])
        assert clusters[0]["location"] == "Kyiv"


class TestCasualtyMagnitude:
    def test_reads_verb_first_headlines(self):
        from src.services.sitrep_generator import _casualty_magnitude
        assert _casualty_magnitude({"source_title": "Kyiv strike kills 15"}) == (15, 15)

    def test_reads_number_first_headlines(self):
        from src.services.sitrep_generator import _casualty_magnitude
        assert _casualty_magnitude({"source_title": "At least 15 killed in Kyiv"}) == (15, 15)

    def test_separates_deaths_from_total_casualties(self):
        from src.services.sitrep_generator import _casualty_magnitude
        assert _casualty_magnitude(
            {"source_title": "Attack kills 15, injures 51 in Kyiv"}) == (15, 51)

    def test_death_toll_outranks_a_larger_injury_count(self):
        # "one killed and 26 injured" must not headline over "kills 17".
        from src.services.sitrep_generator import _casualty_magnitude
        assert _casualty_magnitude({"source_title": "Russian Attack Kills 17"}) > \
            _casualty_magnitude(
                {"source_title": "Russia attacks Kyiv: one killed and 26 injured"})

    def test_non_casualty_counts_are_ignored(self):
        from src.services.sitrep_generator import _casualty_magnitude
        assert _casualty_magnitude(
            {"source_title": "475 UAVs launched at Russian regions overnight"}) == (0, 0)

    def test_headline_without_figures(self):
        from src.services.sitrep_generator import _casualty_magnitude
        assert _casualty_magnitude({"source_title": "Explosions reported in Kyiv"}) == (0, 0)


class TestCountryLevelAnchors:
    """Outlets disagree on how to place one incident.

    The 2026-08-05 Kyiv strike was filed by WSJ as "Russian Attack Kills 17 in
    Ukraine" — anchored at country level, it became a second cluster for the same
    event and split the day's lead story in two.
    """

    @staticmethod
    def _mixed_anchors():
        return [
            _event("Russian ballistic missile attack kills 15, injures 51 in Kyiv",
                   "kyivindependent.com", "Kyiv", 22),
            _event("Explosions reported in Kyiv as Russia launches ballistic missile attack",
                   "unn.ua", "Kyiv", 21),
            _event("Russian Attack Kills 17 in Ukraine", "wsj.com", "Ukraine", 7,
                   text="Russian Attack Kills 17 in Ukraine The impact of ballistic "
                        "missiles in Kyiv could be heard across the city."),
        ]

    def test_country_anchored_filing_joins_the_city_cluster(self):
        clusters = build_sitrep_clusters(self._mixed_anchors(), [])
        assert len(clusters) == 1
        assert clusters[0]["location"] == "Kyiv"

    def test_cluster_is_not_relabelled_as_nationwide(self):
        # The folded member is the best-informed one and becomes the
        # representative; the place must still come from a located member.
        clusters = build_sitrep_clusters(self._mixed_anchors(), [])
        assert "17" in clusters[0]["snippet"]
        assert clusters[0]["location"] != "Ukraine"

    def test_nationwide_item_that_never_names_the_city_stays_separate(self):
        # "Over 8,300 glide bombs dropped ... on Ukraine in July" is a monthly
        # figure, not evidence for one night's strike on Kyiv.
        events = self._mixed_anchors() + [
            _event("Over 8,300 glide bombs dropped by Russia on Ukraine in July in record figure",
                   "kyivindependent.com", "Ukraine", 17),
        ]
        locations = [c["location"] for c in build_sitrep_clusters(events, [])]
        assert "Ukraine" in locations
        assert "Kyiv" in locations

    def test_other_cities_are_never_absorbed(self):
        events = self._mixed_anchors() + [
            _event("Two men killed in Druzhkivka in Russian drone strike",
                   "ukrinform.net", "Druzhkivka", 16),
        ]
        by_loc = _by_location(build_sitrep_clusters(events, []))
        assert "Druzhkivka" in by_loc
        assert len(by_loc["Druzhkivka"]["sources"]) == 1

    def test_folding_only_happens_inside_one_storyline(self):
        events = self._mixed_anchors() + [
            _event("Lithuania will demand compensation for embassy damage in Kyiv",
                   "kurs.com.ua", "Ukraine", 11,
                   storyline="b7b11b20-f995-47ee-9c3d-0b570f956145", severity=45),
        ]
        locations = [c["location"] for c in build_sitrep_clusters(events, [])]
        assert "Ukraine" in locations  # separate storyline keeps its own cluster

    def test_unlocated_events_alone_still_cluster(self):
        events = [
            _event("Ukraine and Russia exchange attacks after seven killed on beach",
                   "belfasttelegraph.co.uk", None, 10,
                   storyline="76d396bb-5f6a-4a52-977c-6690e30a6ae6"),
        ]
        clusters = build_sitrep_clusters(events, [])
        assert clusters[0]["location"] == "Ülke Geneli"


class TestPlaceIdentity:
    """Run #27 (2026-08-06): one night's strike on Kyiv became six clusters.

    "Kyiv" and "Kiev" keyed apart, and "Kyiv train station" keyed apart from both,
    so the narrator was handed the same incident three times and wrote up 21 dead,
    then 17, then 8 — in a report whose own summary put the country-wide toll at 21.
    Every cluster boundary below is an identity question: same place, different
    spelling. Semantic merging ("is this nationwide item ABOUT Kyiv?") is a
    separate gate and deliberately not covered here.
    """

    def _kyiv_night(self):
        # Verbatim anchor spellings and headlines from the run #27 UA window.
        return [
            _event("Russian ballistic missile attack kills 15, injures 51 in Kyiv",
                   "kyivindependent.com", "Kyiv", 6),
            _event("Russian missile barrage on Kyiv kills 17 as Zelensky speaks",
                   "reuters.com", "Kyiv", 8),
            _event("At least 17 killed in Russian missile strike as Kiev left defenceless",
                   "yahoo.com", "Kiev", 7),
            _event("Kyiv city workers remove bodies of eight killed following attack "
                   "on train station", "apnews.com", "Kyiv train station", 9),
            _event("Massive Russian Strike Leaves At Least 15 Dead in Kyiv Oblast",
                   "novinite.com", "Kyiv Oblast", 6),
        ]

    def test_exonym_does_not_split_the_cluster(self):
        clusters = build_sitrep_clusters(self._kyiv_night(), [])
        assert len(clusters) == 1, [c["location"] for c in clusters]

    def test_venue_inside_a_city_does_not_split_the_cluster(self):
        from src.services.sitrep_generator import _location_key
        assert _location_key({"anchor_name_raw": "Kyiv train station"}) == "kyiv"
        assert _location_key({"anchor_name_raw": "Odesa region enterprise"}) == "odesa"

    def test_all_spellings_collapse_onto_one_key(self):
        from src.services.sitrep_generator import _location_key
        keys = {_location_key({"anchor_name_raw": name})
                for name in ("Kyiv", "Kiev", "Kyiv Oblast", "Kiev region",
                             "Kyiv Region", "Kyiv train station")}
        assert keys == {"kyiv"}

    def test_distinct_places_stay_distinct(self):
        # The village station in Kyiv oblast is NOT the capital: over-merging is
        # the mirror-image bug and would attribute its dead to the city.
        from src.services.sitrep_generator import _location_key
        assert _location_key({"anchor_name_raw": "Kvitneve railway station"}) == "kvitneve"
        assert _location_key({"anchor_name_raw": "Kharkiv"}) != "kyiv"

    def test_strategic_sites_are_not_folded_into_their_namesake_city(self):
        # Zaporizhzhia NPP sits in Enerhodar and is its own story; folding it into
        # the city would merge two genuinely different events.
        from src.services.sitrep_generator import _location_key
        assert _location_key({"anchor_name_raw": "Zaporozhye Nuclear Power Plant"}) \
            != _location_key({"anchor_name_raw": "Zaporizhzhia"})

    def test_bare_venue_word_does_not_become_country_level(self):
        # Stripping "Airport" to "" would silently reclassify the event as
        # nationwide and hand it to the country-level folding path.
        from src.services.sitrep_generator import _location_key
        assert _location_key({"anchor_name_raw": "Airport"}) == "airport"

    def test_alias_table_is_canonical_and_acyclic(self):
        # A canonical name appearing as a variant would make normalization depend
        # on lookup order — "a -> b" plus "b -> c" leaves "a" stuck at "b".
        from src.services.sitrep_generator import _PLACE_ALIASES
        assert not set(_PLACE_ALIASES) & set(_PLACE_ALIASES.values())

    def test_country_level_item_absorbs_via_alias_spelling(self):
        # A wire item datelined "Ukraine" whose text says "Kiev" is about the Kyiv
        # group; matching only the canonical spelling would leave it a lone cluster.
        events = self._kyiv_night() + [
            _event("Massive Russian missile attack, 14 killed and 22 injured",
                   "cna.asia", "Ukraine", 10,
                   text="The barrage struck Kiev overnight, officials said."),
        ]
        clusters = build_sitrep_clusters(events, [])
        # Surviving as its own cluster would show up as a second, "Ülke Geneli"
        # entry re-reporting the same night's dead.
        assert len(clusters) == 1, [c["location"] for c in clusters]
        assert clusters[0]["location"] != "Ülke Geneli"

    def test_nationwide_item_naming_no_city_still_stands_alone(self):
        # The mirror-image risk: absorbing on the storyline alone would fold a
        # genuinely country-wide item into a single night's strike and let it
        # corroborate a death toll it never reported.
        events = self._kyiv_night() + [
            _event("Over 8,300 glide bombs dropped by Russia on Ukraine in July",
                   "pravda.com.ua", "Ukraine", 11,
                   text="Air Force figures for the month showed a record total."),
        ]
        locations = [c["location"] for c in build_sitrep_clusters(events, [])]
        assert sorted(locations) == ["Kyiv", "Ukraine"]


class TestCountryTermTables:
    def test_every_aliased_country_has_self_terms(self):
        # _COUNTRY_ALIASES mixes in capitals and groups; _COUNTRY_SELF_TERMS must
        # stay in step or a new country silently loses country-level folding.
        from src.services.sitrep_generator import _COUNTRY_ALIASES, _COUNTRY_SELF_TERMS
        assert set(_COUNTRY_ALIASES) <= set(_COUNTRY_SELF_TERMS)

    def test_self_terms_contain_no_capitals(self):
        from src.services.sitrep_generator import _COUNTRY_SELF_TERMS
        capitals = {"tehran", "moscow", "kyiv", "baghdad", "damascus", "beirut",
                    "sanaa", "riyadh", "doha", "dubai", "cairo", "ankara",
                    "kabul", "khartoum", "beijing", "taipei", "amman"}
        for iso, terms in _COUNTRY_SELF_TERMS.items():
            assert not (terms & capitals), f"{iso} lists a capital as a country term"

    def test_city_is_not_treated_as_country_level(self):
        from src.services.sitrep_generator import _is_country_level
        assert _is_country_level({"anchor_name_raw": "Kyiv", "country_iso": "UA"}) is False
        assert _is_country_level({"anchor_name_raw": "Ukraine", "country_iso": "UA"}) is True
        assert _is_country_level({"anchor_name_raw": None, "country_iso": "UA"}) is True

    def test_unknown_country_never_folds(self):
        from src.services.sitrep_generator import _is_country_level
        assert _is_country_level({"anchor_name_raw": "Somewhere", "country_iso": "ZZ"}) is False
