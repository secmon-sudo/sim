"""
Tests for Faz 3 — hybrid storyline linking (anchor-assist + date-token removal)
and the zero-LLM narrative timeline.
"""

from datetime import datetime, timedelta

from src.core.storyline import jaccard_similarity, should_link_storyline
from src.core.storyline_narrative import (
    TREND_ESCALATING,
    TREND_STABLE,
    build_timeline,
    summarize_timeline,
)

_T0 = datetime(2026, 6, 8, 10, 0)


def _ev(hint, iso="AF", anchor=None, when=_T0, sev=50):
    return {
        "storyline_hint": hint,
        "country_iso": iso,
        "anchor_name_norm": anchor,
        "occurred_at_est": when,
        "severity_score": sev,
    }


class TestDateTokenRemoval:
    def test_date_hint_does_not_distort(self):
        # Same event reported on consecutive days must stay maximally similar.
        sim = jaccard_similarity(
            "istanbul ataturk bomb threat jun8", "istanbul ataturk bomb threat jun9"
        )
        assert sim == 1.0

    def test_flight_number_preserved(self):
        # "dl54" is a strong identifier, must NOT be stripped as a date token.
        sim = jaccard_similarity("delta dl54 emergency atlanta jun7", "delta dl54 emergency atlanta")
        assert sim > 0.5


def _geo_ev(hint, raw, iso="UA", when=_T0, sev=50):
    """Event without an IATA anchor but with raw location text for geo-assist."""
    return {
        "storyline_hint": hint,
        "country_iso": iso,
        "anchor_name_norm": None,
        "anchor_name_raw": raw,
        "occurred_at_est": when,
        "severity_score": sev,
    }


class TestCoarseGeoAssist:
    def test_same_city_partial_overlap_links(self):
        # No IATA anchor; same city (KYIV); lexical sim ~0.27 — below the 0.4 main
        # threshold but above the 0.2 geo floor -> geo-assist links them.
        a = _geo_ev("kyiv power grid damaged", raw="Kyiv")
        b = _geo_ev("kyiv power station outage", raw="Kiev")
        assert 0.2 <= jaccard_similarity(a["storyline_hint"], b["storyline_hint"]) < 0.4
        assert should_link_storyline(a, b) is True

    def test_capital_paraphrase_zero_overlap_not_linked_here(self):
        # Same real place (KYIV via capital resolution) but ZERO lexical overlap.
        # Deterministic geo-assist deliberately does NOT link this — it is the LLM
        # adjudicator's job (Layer 2), so we don't merge on geography alone.
        a = _geo_ev("kyiv drone strike", raw="Kyiv")
        b = _geo_ev("ukrainian capital missile", raw="Ukrainian capital")
        from src.core.geo import geo_key
        assert geo_key("Kyiv") == geo_key("Ukrainian capital", "UA")  # same place
        assert jaccard_similarity(a["storyline_hint"], b["storyline_hint"]) < 0.2
        assert should_link_storyline(a, b) is False

    def test_different_city_low_overlap_not_linked(self):
        a = _geo_ev("kyiv power grid outage", raw="Kyiv")
        b = _geo_ev("moscow power station failure", raw="Moscow", iso="RU")
        assert should_link_storyline(a, b) is False

    def test_missing_raw_location_is_safe(self):
        # No raw text -> no geo key -> the geo path is skipped without crashing.
        a = _geo_ev("kyiv power grid outage", raw=None)
        b = _geo_ev("kyiv power station outage", raw=None)
        # These two still link, now via containment (three of four words shared):
        # a grid outage and a power-station outage in Kyiv hours apart are one
        # story — the same conclusion the geo path reaches when raw text exists.
        assert should_link_storyline(a, b) is True
        # A same-country pair with nothing meaningful in common still falls through.
        assert should_link_storyline(a, _geo_ev("lviv rail depot fire", raw=None)) is False


class TestHybridAnchorAssist:
    def test_paraphrase_same_airport_same_day_links(self):
        a = _ev("kabul airport explosion terminal", anchor="KBL", when=_T0)
        b = _ev("blast rocks kabul international departures", anchor="KBL", when=_T0 + timedelta(hours=4))
        # Lexical similarity alone is far too low to link...
        assert jaccard_similarity(a["storyline_hint"], b["storyline_hint"]) < 0.2
        # ...but same anchor within the tight window rescues it.
        assert should_link_storyline(a, b) is True

    def test_same_airport_far_apart_low_sim_does_not_link(self):
        a = _ev("kabul airport explosion terminal", anchor="KBL", when=_T0)
        b = _ev("kabul airport security drill announced", anchor="KBL", when=_T0 + timedelta(days=10))
        assert should_link_storyline(a, b) is False

    def test_different_airport_low_sim_does_not_link(self):
        a = _ev("kabul airport explosion terminal", anchor="KBL", iso="AF", when=_T0)
        b = _ev("blast rocks departures hall", anchor="JFK", iso="US", when=_T0)
        assert should_link_storyline(a, b) is False

    def test_country_mismatch_hard_gate(self):
        a = _ev("identical attack hint here", iso="AF", when=_T0)
        b = _ev("identical attack hint here", iso="US", when=_T0)
        assert should_link_storyline(a, b) is False


class TestGenericIncidentOverlap:
    """Run #22 (1 Aug 2026): a Seattle festival shooting and an unrelated Arkansas
    killing were merged into one storyline on a 0.43 Jaccard built entirely out of
    {mass, shooting}. The SITREP then presented the Arkansas article as Seattle's
    second source and stamped the bullet "Onaylandı (Çoklu kaynak)"."""

    def test_generic_only_overlap_does_not_link(self):
        a = _geo_ev("seattle mass shooting", raw="Bite of Seattle", iso="US")
        b = _geo_ev("breckenridge mass shooting", raw="Breckenridge", iso="US",
                    when=_T0 + timedelta(hours=7))
        assert jaccard_similarity(a["storyline_hint"], b["storyline_hint"]) > 0.4
        assert should_link_storyline(a, b) is False

    def test_shared_place_still_links(self):
        # Same overlap size, but the shared token names a place — that is evidence.
        a = _geo_ev("philippines high school shooting", raw=None, iso="PH")
        b = _geo_ev("philippines school shooting", raw=None, iso="PH")
        assert should_link_storyline(a, b) is True

    def test_venue_name_resolves_to_its_city(self):
        """"Bite of Seattle" is a festival, not a place name — until it resolved to
        SEATTLE the same shooting lived in two storylines that no layer rejoined."""
        from src.core.geo import geo_key
        assert geo_key("Bite of Seattle", "US") == geo_key("Seattle", "US")
        a = _geo_ev("seattle mass shooting", raw="Bite of Seattle", iso="US")
        b = _geo_ev("seattle festival shooting", raw="Seattle", iso="US",
                    when=_T0 - timedelta(hours=8))
        assert should_link_storyline(a, b) is True


class TestSpecificityContainment:
    """Run #24 (2-3 Aug 2026): the Twin Falls In-N-Out shooting ran as TWO
    storylines for two days — one keyed on the town, one on the state — because
    Jaccard penalizes a hint for carrying the extra word. Each half had its own
    sources, its own verification label and its own airspace card in the US
    SITREP."""

    def test_town_and_state_naming_link(self):
        a = _geo_ev("twin falls in-n-out shooting", raw=None, iso="US")
        b = _geo_ev("idaho in-n-out shooting", raw=None, iso="US",
                    when=_T0 + timedelta(hours=6))
        assert jaccard_similarity(a["storyline_hint"], b["storyline_hint"]) < 0.4
        assert should_link_storyline(a, b) is True

    def test_extra_word_does_not_break_the_match(self):
        # "burger" rewrites every bigram it touches, which is why containment is
        # measured on single words.
        a = _geo_ev("twin falls in-n-out shooting", raw=None, iso="US")
        b = _geo_ev("idaho in-n-out burger shooting", raw=None, iso="US",
                    when=_T0 + timedelta(hours=6))
        assert should_link_storyline(a, b) is True

    def test_one_shared_word_is_not_enough(self):
        a = _geo_ev("gaza airstrike children", raw=None, iso="PS")
        b = _geo_ev("gaza protest crackdown", raw=None, iso="PS",
                    when=_T0 + timedelta(hours=6))
        assert should_link_storyline(a, b) is False

    def test_generic_overlap_is_still_gated(self):
        a = _geo_ev("seattle mass shooting", raw=None, iso="US")
        b = _geo_ev("breckenridge mass shooting", raw=None, iso="US",
                    when=_T0 + timedelta(hours=6))
        assert should_link_storyline(a, b) is False

    def test_placeholder_location_hint_does_not_link(self):
        # The classifier emits "location ..." when it cannot name the place.
        a = _geo_ev("twin falls in-n-out shooting", raw=None, iso="US")
        b = _geo_ev("location mass shooting", raw=None, iso="US",
                    when=_T0 + timedelta(hours=6))
        assert should_link_storyline(a, b) is False

    def test_country_words_are_not_evidence(self):
        # Linking already requires a shared country, so "russian" is constant
        # across every candidate pair — these are two different cities.
        a = _geo_ev("kyiv russian airstrike", raw=None, iso="UA")
        b = _geo_ev("kherson region russian airstrike", raw=None, iso="UA",
                    when=_T0 + timedelta(hours=6))
        assert should_link_storyline(a, b) is False

    def test_words_the_week_is_about_are_not_evidence(self):
        """On 2-3 Aug 2026 "idf" named a dozen separate Gaza storylines; sharing
        it means "same conflict", not "same incident"."""
        from src.core.storyline import overexposed_tokens

        corpus = [("s1", "gaza idf missile strike"), ("s2", "gaza idf raid"),
                  ("s3", "gaza idf tank fire"), ("s4", "khan yunis clashes")]
        common = overexposed_tokens(corpus, min_storylines=3)
        assert "idf" in common and "yunis" not in common

        a = _geo_ev("gaza idf missile strike", raw=None, iso="PS")
        b = _geo_ev("gaza city idf terrorist killing", raw=None, iso="PS",
                    when=_T0 + timedelta(hours=6))
        assert should_link_storyline(a, b, common_tokens=common) is False
        # Without the corpus signal the same pair would have merged.
        assert should_link_storyline(a, b) is True

    def test_one_busy_storyline_cannot_make_its_own_words_generic(self):
        from src.core.storyline import overexposed_tokens

        corpus = [("s1", f"twin falls in-n-out shooting {i}") for i in range(20)]
        assert overexposed_tokens(corpus, min_storylines=3) == set()

    def test_containment_is_time_boxed(self):
        # Same shape, five days apart: two distinct incidents at one place read
        # exactly like one incident named twice.
        a = _geo_ev("twin falls in-n-out shooting", raw=None, iso="US")
        b = _geo_ev("idaho in-n-out shooting", raw=None, iso="US",
                    when=_T0 + timedelta(days=5))
        assert should_link_storyline(a, b) is False


class TestAdjudicatorCandidateNet:
    def test_region_and_town_reports_reach_the_model(self):
        """A coarse geo key is not a containment test: "Kashmir" and "Kulgam" are
        the same incident, and a geo-only net never offered one to the other."""
        from src.core.storyline_adjudicator import find_geo_candidates

        recent = [
            {"storyline_id": "kulgam", "storyline_hint": "kulgam terror attack",
             "anchor_name_raw": "Kulgam", "country_iso": "IN", "occurred_at_est": _T0},
            {"storyline_id": "other", "storyline_hint": "mumbai port fire",
             "anchor_name_raw": "Mumbai", "country_iso": "IN", "occurred_at_est": _T0},
        ]
        event = {"storyline_hint": "kashmir terrorist attack", "anchor_name_raw": "Kashmir",
                 "country_iso": "IN", "occurred_at_est": _T0 + timedelta(hours=15)}
        assert [c["storyline_id"] for c in find_geo_candidates(event, recent)] == ["kulgam"]

    def test_same_geo_still_outranks_lexical_candidates(self):
        from src.core.storyline_adjudicator import find_geo_candidates

        recent = [
            {"storyline_id": "lexical", "storyline_hint": "kashmir terror attack",
             "anchor_name_raw": "Jammu", "country_iso": "IN", "occurred_at_est": _T0},
            {"storyline_id": "same-place", "storyline_hint": "unrelated wording",
             "anchor_name_raw": "Kulgam", "country_iso": "IN", "occurred_at_est": _T0},
        ]
        event = {"storyline_hint": "kulgam terror attack", "anchor_name_raw": "Kulgam",
                 "country_iso": "IN", "occurred_at_est": _T0 + timedelta(hours=2)}
        assert find_geo_candidates(event, recent)[0]["storyline_id"] == "same-place"

    def test_unrelated_same_country_event_is_not_a_candidate(self):
        from src.core.storyline_adjudicator import find_geo_candidates

        recent = [{"storyline_id": "x", "storyline_hint": "mumbai port fire",
                   "anchor_name_raw": "Mumbai", "country_iso": "IN",
                   "occurred_at_est": _T0}]
        event = {"storyline_hint": "kulgam terror attack", "anchor_name_raw": "Kulgam",
                 "country_iso": "IN", "occurred_at_est": _T0 + timedelta(hours=1)}
        assert find_geo_candidates(event, recent) == []


class TestNarrative:
    def test_build_timeline_orders_and_sequences(self):
        evs = [
            _ev("c", when=_T0 + timedelta(hours=20)),
            _ev("a", when=_T0),
            _ev("b", when=_T0 + timedelta(hours=4)),
        ]
        tl = build_timeline(evs)
        assert [e["seq"] for e in tl] == [1, 2, 3]
        assert tl[0]["occurred_at_est"] < tl[1]["occurred_at_est"] < tl[2]["occurred_at_est"]

    def test_summary_detects_escalation(self):
        evs = [
            _ev("x", when=_T0, sev=40),
            _ev("x", when=_T0 + timedelta(hours=4), sev=70),
            _ev("x", when=_T0 + timedelta(days=1), sev=90),
        ]
        s = summarize_timeline(evs)
        assert s["event_count"] == 3
        assert s["peak_severity"] == 90
        assert s["severity_trend"] == TREND_ESCALATING

    def test_empty_summary_is_safe(self):
        s = summarize_timeline([])
        assert s["event_count"] == 0
        assert s["severity_trend"] == TREND_STABLE
