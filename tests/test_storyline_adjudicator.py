"""Tests for the Layer 2 LLM storyline adjudicator."""

from datetime import datetime, timedelta

from src.core.storyline_adjudicator import (
    adjudicate_storyline,
    find_geo_candidates,
    _parse_decision,
)

_T0 = datetime(2026, 6, 8, 10, 0)


def _ev(hint, raw="Kyiv", iso="UA", sid=None, when=_T0):
    return {
        "storyline_hint": hint,
        "storyline_id": sid,
        "country_iso": iso,
        "anchor_name_raw": raw,
        "anchor_name_norm": None,
        "occurred_at_est": when,
        "source_title": hint,
    }


class TestFindGeoCandidates:
    def test_same_city_same_window_is_candidate(self):
        event = _ev("kyiv drone strike")
        recent = [_ev("ukrainian capital missile", raw="Ukrainian capital", sid="S1")]
        cands = find_geo_candidates(event, recent)
        assert [c["storyline_id"] for c in cands] == ["S1"]

    def test_different_city_excluded(self):
        event = _ev("kyiv drone strike")
        recent = [_ev("moscow blast", raw="Moscow", iso="RU", sid="S9")]
        assert find_geo_candidates(event, recent) == []

    def test_outside_window_excluded(self):
        event = _ev("kyiv drone strike")
        recent = [_ev("kyiv earlier strike", sid="S2", when=_T0 - timedelta(days=5))]
        assert find_geo_candidates(event, recent, window_hours=48) == []

    def test_one_representative_per_storyline(self):
        event = _ev("kyiv drone strike")
        recent = [
            _ev("kyiv hit one", sid="S1"),
            _ev("kyiv hit two", sid="S1"),
            _ev("kyiv hit three", sid="S2"),
        ]
        sids = [c["storyline_id"] for c in find_geo_candidates(event, recent)]
        assert sids == ["S1", "S2"]

    def test_no_geo_no_country_no_candidates(self):
        # Nothing coarse enough to gather a plausibly-same set from.
        event = _ev("some event", raw=None, iso=None)
        recent = [_ev("kyiv thing", sid="S1")]
        assert find_geo_candidates(event, recent) == []

    def test_locationless_falls_back_to_country(self):
        # National-level news (no resolvable place) still gathers same-country,
        # lexically-related storylines so the LLM can adjudicate paraphrased duplicates.
        event = _ev("china missile test", raw=None, iso="CN")
        recent = [_ev("china pla missile_test", raw=None, iso="CN", sid="S1")]
        cands = find_geo_candidates(event, recent)
        assert [c["storyline_id"] for c in cands] == ["S1"]

    def test_country_fallback_excludes_unrelated(self):
        # Same country but zero lexical kinship → not a candidate.
        event = _ev("china missile test", raw=None, iso="CN")
        recent = [_ev("shanghai factory flood", raw=None, iso="CN", sid="S1")]
        assert find_geo_candidates(event, recent) == []

    def test_country_fallback_excludes_other_country(self):
        event = _ev("china missile test", raw=None, iso="CN")
        recent = [_ev("india missile test", raw=None, iso="IN", sid="S1")]
        assert find_geo_candidates(event, recent) == []

    def test_country_fallback_ranks_by_overlap(self):
        # The strongest lexical match must survive the max_candidates cap.
        event = _ev("china submarine missile test", raw=None, iso="CN")
        recent = [
            _ev("china economy trade", raw=None, iso="CN", sid="S_weak"),
            _ev("china submarine missile launch", raw=None, iso="CN", sid="S_strong"),
        ]
        cands = find_geo_candidates(event, recent, max_candidates=1)
        assert [c["storyline_id"] for c in cands] == ["S_strong"]

    def test_degenerate_unknown_geo_uses_country(self):
        # geo_key("Unknown") is degenerate; the event must not match other unresolved
        # events as if "UNKNOWN" were a shared place — it takes the country path instead.
        event = _ev("china missile test", raw="Unknown", iso="CN")
        recent = [_ev("china pla missile_test", raw="Unknown", iso="CN", sid="S1")]
        cands = find_geo_candidates(event, recent)
        assert [c["storyline_id"] for c in cands] == ["S1"]


class TestParseDecision:
    _cands = [{"storyline_id": "S1", "hint": "a"}, {"storyline_id": "S2", "hint": "b"}]

    def test_json_match_number(self):
        assert _parse_decision('{"match": 2}', self._cands) == "S2"

    def test_json_match_new(self):
        assert _parse_decision('{"match": "NEW"}', self._cands) is None

    def test_json_embedded_in_prose(self):
        assert _parse_decision('Sure! {"match": 1} done', self._cands) == "S1"

    def test_bare_number_fallback(self):
        assert _parse_decision("2", self._cands) == "S2"

    def test_new_word_fallback(self):
        assert _parse_decision("This looks NEW to me", self._cands) is None

    def test_out_of_range_is_new(self):
        assert _parse_decision('{"match": 9}', self._cands) is None

    def test_empty_is_new(self):
        assert _parse_decision("", self._cands) is None


class TestAdjudicateStoryline:
    def test_links_when_llm_says_same(self):
        event = _ev("kyiv drone strike")
        recent = [_ev("ukrainian capital missile", raw="Ukrainian capital", sid="S1")]
        fake_llm = lambda router, prompt, **kw: {"content": '{"match": 1}'}
        assert adjudicate_storyline(event, recent, router=None, call_llm_fn=fake_llm) == "S1"

    def test_new_when_llm_says_new(self):
        event = _ev("kyiv drone strike")
        recent = [_ev("ukrainian capital missile", raw="Ukrainian capital", sid="S1")]
        fake_llm = lambda router, prompt, **kw: {"content": '{"match": "NEW"}'}
        assert adjudicate_storyline(event, recent, router=None, call_llm_fn=fake_llm) is None

    def test_no_candidates_skips_llm(self):
        called = []
        fake_llm = lambda *a, **k: called.append(1) or {"content": "1"}
        event = _ev("kyiv drone strike")
        recent = [_ev("moscow blast", raw="Moscow", iso="RU", sid="S9")]
        assert adjudicate_storyline(event, recent, router=None, call_llm_fn=fake_llm) is None
        assert called == []  # never invoked when there is nothing to judge

    def test_llm_error_fails_safe_to_new(self):
        def boom(*a, **k):
            raise RuntimeError("all accounts exhausted")
        event = _ev("kyiv drone strike")
        recent = [_ev("ukrainian capital missile", raw="Ukrainian capital", sid="S1")]
        assert adjudicate_storyline(event, recent, router=None, call_llm_fn=boom) is None
