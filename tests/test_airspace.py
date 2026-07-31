"""
Airspace impact analysis — the geometry layer.

A SITREP could report "a drone hit eastern Poland" but never "that is Warszawa
FIR, and Lublin airport is 10 km away". src/core/airspace.py closes that gap by
resolving an event's location against a vendored FIR/airport reference. Since
the whole point is that the LLM cannot invent these facts, the facts themselves
have to be right — hence both a dataset-integrity suite and behavioural tests.
"""

import pytest

from src.core import airspace
from src.core.airspace import (
    AIRPORTS,
    FIRS,
    assess_cluster,
    build_airspace_assessment,
    country_airports,
    fir_for_point,
    firs_for_country,
    is_airspace_relevant,
    nearby_airports,
    neighbor_firs,
    resolve_cluster_point,
    summarize_assessment,
)
from src.pipeline.ingest_filters import _is_flight_disruption

CZIB = {
    "UA": [{"czib_id": "1", "name": "UKRAINE — CZIB-2022-01", "valid_until": "2026-12-31"}],
    "BY": [{"czib_id": "2", "name": "BELARUS — CZIB-2023-04", "valid_until": "2026-10-01"}],
}


class TestDatasetIntegrity:
    """config/airspace.json is hand-curated reference data. If it is internally
    inconsistent every downstream claim is too, so it gets checked like code."""

    def test_fir_codes_unique(self):
        codes = [f["icao"] for f in FIRS]
        assert len(codes) == len(set(codes))

    def test_every_neighbor_exists_and_is_mutual(self):
        by_icao = {f["icao"]: f for f in FIRS}
        for fir in FIRS:
            for n in fir["neighbors"]:
                assert n in by_icao, f"{fir['icao']} lists unknown neighbor {n}"
                assert fir["icao"] in by_icao[n]["neighbors"], (
                    f"{fir['icao']}→{n} is not mutual")

    def test_bboxes_are_well_formed(self):
        for fir in FIRS:
            min_lat, min_lon, max_lat, max_lon = fir["bbox"]
            assert min_lat < max_lat and min_lon < max_lon, fir["icao"]
            assert -90 <= min_lat <= 90 and -90 <= max_lat <= 90, fir["icao"]
            assert -180 <= min_lon <= 180 and -180 <= max_lon <= 180, fir["icao"]

    def test_every_fir_declares_at_least_one_country(self):
        for fir in FIRS:
            assert fir["countries"], fir["icao"]

    def test_airport_codes_unique(self):
        iatas = [a["iata"] for a in AIRPORTS]
        assert len(iatas) == len(set(iatas))
        icaos = [a["icao"] for a in AIRPORTS if a["icao"]]
        assert len(icaos) == len(set(icaos))

    def test_every_airport_falls_inside_some_fir(self):
        """An airport we cannot place in an airspace is an orphan in a
        FIR-keyed dataset — and caught three corrupt anchor records when added
        (Brussels geocoded to Ontario, Birmingham to Alabama)."""
        for ap in AIRPORTS:
            assert fir_for_point(ap["lat"], ap["lon"], ap["country"]) is not None, (
                f"{ap['iata']} ({ap['lat']},{ap['lon']}) is outside every FIR")

    def test_airport_coordinates_are_plausible(self):
        for ap in AIRPORTS:
            assert -90 <= ap["lat"] <= 90 and -180 <= ap["lon"] <= 180, ap["iata"]
            assert ap["tier"] in {"hub", "major", "regional"}, ap["iata"]


class TestFirResolution:
    @pytest.mark.parametrize("lat,lon,iso,expected", [
        (51.2465, 22.5684, "PL", "EPWW"),   # Lublin
        (50.0413, 21.9990, "PL", "EPWW"),   # Rzeszów
        (50.4501, 30.5234, "UA", "UKBV"),   # Kyiv
        (49.8397, 24.0297, "UA", "UKLV"),   # Lviv
        (41.0082, 28.9784, "TR", "LTBB"),   # Istanbul
        (39.9334, 32.8597, "TR", "LTAA"),   # Ankara
        (33.8938, 35.5018, "LB", "OLBB"),   # Beirut
        (55.7558, 37.6173, "RU", "UUWV"),   # Moscow
    ])
    def test_known_cities_resolve(self, lat, lon, iso, expected):
        assert fir_for_point(lat, lon, iso)["icao"] == expected

    def test_country_hint_wins_over_box_size_at_a_border(self):
        """Lublin sits inside Poland but near boxes that reach over the border.
        Without the hint the tightest box could win; with it, Polish airspace
        must — an event attributed to Poland is not in Ukrainian airspace."""
        lat, lon = 50.5, 23.0  # SE Poland, inside both EPWW and UKLV boxes
        assert fir_for_point(lat, lon, "PL")["icao"] == "EPWW"
        assert fir_for_point(lat, lon, "UA")["icao"] == "UKLV"

    def test_point_outside_reference_footprint_returns_none(self):
        assert fir_for_point(41.88, -87.63, "US") is None  # Chicago, no US FIRs

    def test_firs_for_country_handles_multi_state_firs(self):
        """Dakar FIR covers Mali; a FIR is not a country."""
        assert any(f["icao"] == "GOOO" for f in firs_for_country("ML"))

    def test_neighbor_firs_resolve_to_records(self):
        epww = fir_for_point(52.2297, 21.0122, "PL")
        icaos = {f["icao"] for f in neighbor_firs(epww)}
        assert {"UKLV", "UMMV", "EYVL"} <= icaos


class TestAirportProximity:
    def test_sorted_by_distance_and_within_radius(self):
        out = nearby_airports(51.2465, 22.5684, radius_km=300, limit=6)
        assert out[0]["iata"] == "LUZ"
        assert out[0]["distance_km"] < 20
        assert [a["distance_km"] for a in out] == sorted(a["distance_km"] for a in out)
        assert all(a["distance_km"] <= 300 for a in out)

    def test_radius_excludes_far_airports(self):
        near = {a["iata"] for a in nearby_airports(51.2465, 22.5684, radius_km=50)}
        assert near == {"LUZ"}

    def test_limit_is_honoured(self):
        assert len(nearby_airports(51.2465, 22.5684, radius_km=1000, limit=3)) == 3

    def test_crosses_borders(self):
        """Airspace exposure does not stop at a national border — the airports a
        Polish incident threatens include Lviv and Brest."""
        out = nearby_airports(51.2465, 22.5684, radius_km=300, limit=10)
        assert {a["country"] for a in out} > {"PL"}

    def test_country_airports_carry_no_distance(self):
        out = country_airports("PL", limit=4)
        assert out and all("distance_km" not in a for a in out)
        assert out[0]["tier"] == "hub"  # most significant first


class TestPointResolution:
    def test_event_coordinate_wins(self):
        cluster = {"latitude": 51.0, "longitude": 22.0, "location": "Warsaw",
                   "country_iso": "PL"}
        assert resolve_cluster_point(cluster) == (51.0, 22.0, "event")

    def test_falls_back_to_gazetteer(self):
        cluster = {"location": "Lublin", "country_iso": "PL"}
        lat, lon, source = resolve_cluster_point(cluster)
        assert source == "gazetteer"
        assert round(lat) == 51

    def test_falls_back_to_airport_name(self):
        """Bydgoszcz is not in the city gazetteer, but it is an airport we know —
        the anchor gazetteer is the last chance to place an event."""
        cluster = {"location": "Bydgoszcz Airport", "country_iso": "PL"}
        lat, lon, source = resolve_cluster_point(cluster)
        assert source == "airport"
        assert round(lat) == 53

    def test_unplaceable_location_returns_none(self):
        assert resolve_cluster_point(
            {"location": "Ülke Geneli", "country_iso": "PL"}) is None

    def test_unusable_coordinates_do_not_crash(self):
        cluster = {"latitude": "abc", "longitude": None, "location": "Lublin",
                   "country_iso": "PL"}
        assert resolve_cluster_point(cluster)[2] == "gazetteer"


class TestRelevanceGate:
    def test_kinetic_event_type_qualifies(self):
        assert is_airspace_relevant({"event_type": "drone_attack_critical_infra"})

    def test_non_kinetic_without_disruption_text_does_not(self):
        cluster = {"event_type": "political_event", "snippet": "Minister gave a speech."}
        assert not is_airspace_relevant(cluster, _is_flight_disruption)

    def test_flight_disruption_text_qualifies_any_event_type(self):
        """A misclassified event whose text says flights were suspended is still
        an aviation event — the gate mirrors the production ingest filter."""
        cluster = {"event_type": "political_event",
                   "snippet": "The carrier suspended all flights to the region."}
        assert is_airspace_relevant(cluster, _is_flight_disruption)


class TestAssessment:
    def test_point_scope_carries_distances_and_czib(self):
        cluster = {"location": "Lublin", "country_iso": "PL", "severity": 70,
                   "event_type": "drone_attack_critical_infra"}
        out = assess_cluster(cluster, CZIB)
        assert out["scope"] == "point"
        assert out["fir"]["icao"] == "EPWW"
        assert out["fir"]["czib_active"] is False
        restricted = {n["icao"] for n in out["neighbor_firs"] if n["czib_active"]}
        assert restricted == {"UKLV", "UMMV"}
        assert out["neighbor_firs"][0]["czib_active"] is True  # restricted lead
        assert all("distance_km" in a for a in out["airports"])

    def test_country_scope_when_location_cannot_be_placed(self):
        cluster = {"location": "Ülke Geneli", "country_iso": "PL", "severity": 50,
                   "event_type": "military_action"}
        out = assess_cluster(cluster, CZIB)
        assert out["scope"] == "country"
        assert out["point"] is None
        assert out["radius_km"] is None
        assert all("distance_km" not in a for a in out["airports"])

    def test_multi_fir_country_is_not_reduced_to_one_airspace(self):
        """India has four FIRs and Russia six. Naming one for an event we could
        not place is a guess dressed as a fact — and the guess used to be
        whichever code sorted first, so a Kashmir event came out as Mumbai FIR.
        Production anchors ('Iran', 'Iraqi Kurdistan', 'eastern Poland') hit this
        path constantly, so it has to be honest rather than confident."""
        out = assess_cluster({"location": "Ülke Geneli", "country_iso": "IN",
                              "event_type": "military_action"}, CZIB)
        assert out["scope"] == "country"
        assert {f["icao"] for f in out["firs"]} == {"VABF", "VECF", "VIDF", "VOMF"}
        # neighbours must exclude the country's own FIRs
        assert not ({f["icao"] for f in out["neighbor_firs"]}
                    & {f["icao"] for f in out["firs"]})

    def test_single_fir_country_still_names_it(self):
        out = assess_cluster({"location": "Ülke Geneli", "country_iso": "PL",
                              "event_type": "military_action"}, CZIB)
        assert [f["icao"] for f in out["firs"]] == ["EPWW"]

    def test_located_event_names_exactly_one_fir(self):
        out = assess_cluster({"location": "Kulgam", "country_iso": "IN",
                              "event_type": "missile_strike"}, CZIB)
        assert out["scope"] == "point"
        assert [f["icao"] for f in out["firs"]] == ["VIDF"]  # Kashmir → Delhi FIR

    def test_unknown_country_yields_nothing(self):
        cluster = {"location": "Somewhere", "country_iso": "US",
                   "event_type": "military_action"}
        assert assess_cluster(cluster, CZIB) is None

    def test_czib_flag_set_on_the_containing_fir(self):
        cluster = {"location": "Kyiv", "country_iso": "UA",
                   "event_type": "missile_strike"}
        out = assess_cluster(cluster, CZIB)
        assert out["fir"]["icao"] == "UKBV"
        assert out["fir"]["czib_active"] is True
        assert out["fir"]["czib"][0]["name"].startswith("UKRAINE")


class TestBuildAssessment:
    def _clusters(self):
        return [
            {"location": "Lublin", "country_iso": "PL", "severity": 78,
             "event_type": "drone_attack_critical_infra", "snippet": ""},
            {"location": "Ülke Geneli", "country_iso": "PL", "severity": 60,
             "event_type": "military_action", "snippet": ""},
            {"location": "Rzeszow", "country_iso": "PL", "severity": 55,
             "event_type": "missile_strike", "snippet": ""},
            {"location": "Warsaw", "country_iso": "PL", "severity": 30,
             "event_type": "political_event", "snippet": "A statement was issued."},
        ]

    def test_only_relevant_clusters_are_assessed(self):
        out = build_airspace_assessment(self._clusters(), "PL", CZIB)
        assert {a["location"] for a in out["assessments"]} == {"Lublin", "Rzeszow"}

    def test_country_card_suppressed_when_fir_already_located(self):
        """A country-scope card for EPWW adds nothing once Lublin already put a
        located card on EPWW — it would repeat the airspace with less detail."""
        out = build_airspace_assessment(self._clusters(), "PL", CZIB)
        assert all(a["scope"] == "point" for a in out["assessments"])

    def test_country_card_kept_when_nothing_else_located(self):
        clusters = [{"location": "Ülke Geneli", "country_iso": "PL", "severity": 60,
                     "event_type": "military_action", "snippet": ""}]
        out = build_airspace_assessment(clusters, "PL", CZIB)
        assert out["assessments"][0]["scope"] == "country"

    def test_ordered_by_severity_and_capped(self):
        out = build_airspace_assessment(self._clusters(), "PL", CZIB, max_clusters=1)
        assert len(out["assessments"]) == 1
        assert out["assessments"][0]["location"] == "Lublin"

    def test_returns_none_when_nothing_qualifies(self):
        clusters = [{"location": "Warsaw", "country_iso": "PL", "severity": 30,
                     "event_type": "political_event", "snippet": "A speech."}]
        assert build_airspace_assessment(clusters, "PL", CZIB) is None

    def test_missing_czib_index_is_tolerated(self):
        out = build_airspace_assessment(self._clusters(), "PL", None)
        assert out["assessments"][0]["fir"]["czib_active"] is False

    def test_disabled_by_config(self, monkeypatch):
        monkeypatch.setattr(airspace, "AIRSPACE_ENABLED", False)
        assert build_airspace_assessment(self._clusters(), "PL", CZIB) is None


class TestPromptCompaction:
    """The HTML block wants every field; the prompt cannot afford them. A full
    25-cluster SITREP prompt already exceeds the smallest request ceiling in the
    router cascade, so repeated neighbour lists are budget taken from events."""

    def _assessment(self):
        return build_airspace_assessment(
            [{"location": "Lublin", "country_iso": "PL", "severity": 78,
              "event_type": "drone_attack_critical_infra", "snippet": ""}], "PL", CZIB)

    def test_keeps_the_facts_the_narrator_needs(self):
        out = airspace.compact_for_prompt(self._assessment())
        item = out["assessments"][0]
        assert item["fir"]["icao"] == "EPWW"
        assert item["kapsam"] == "point"
        assert {n["icao"] for n in item["kisitlamali_komsu_firlar"]} == {"UKLV", "UMMV"}
        assert item["kisitlamali_komsu_firlar"][0]["easa_czib_aktif"] is True
        assert "EYVL" in item["diger_komsu_firlar"]
        assert item["havalimanlari"][0]["iata"] == "LUZ"
        assert item["havalimanlari"][0]["distance_km"] == 10

    def test_unrestricted_neighbours_collapse_to_codes(self):
        item = airspace.compact_for_prompt(self._assessment())["assessments"][0]
        assert all(isinstance(n, str) for n in item["diger_komsu_firlar"])

    def test_is_materially_smaller(self):
        import json
        full = self._assessment()
        big = len(json.dumps(full, ensure_ascii=False))
        small = len(json.dumps(airspace.compact_for_prompt(full), ensure_ascii=False))
        assert small < big * 0.6

    def test_country_scope_carries_no_radius(self):
        out = airspace.compact_for_prompt(build_airspace_assessment(
            [{"location": "Ülke Geneli", "country_iso": "PL", "severity": 60,
              "event_type": "military_action", "snippet": ""}], "PL", CZIB))
        assert "yaricap_km" not in out["assessments"][0]

    def test_empty_input_stays_empty(self):
        assert airspace.compact_for_prompt(None) is None
        assert airspace.compact_for_prompt({"assessments": []}) is None


class TestDigestSummary:
    def test_mentions_fir_restrictions_and_airports(self):
        out = build_airspace_assessment(
            [{"location": "Lublin", "country_iso": "PL", "severity": 78,
              "event_type": "drone_attack_critical_infra", "snippet": ""}], "PL", CZIB)
        line = summarize_assessment(out)
        assert "EPWW" in line
        assert "CZIB" in line and "UKLV" in line
        assert "LUZ" in line and "km" in line

    def test_country_scope_line_omits_distances(self):
        out = build_airspace_assessment(
            [{"location": "Ülke Geneli", "country_iso": "PL", "severity": 60,
              "event_type": "military_action", "snippet": ""}], "PL", CZIB)
        line = summarize_assessment(out)
        assert "EPWW" in line
        assert "km" not in line

    def test_empty_assessment_yields_empty_string(self):
        assert summarize_assessment(None) == ""
        assert summarize_assessment({"assessments": []}) == ""
