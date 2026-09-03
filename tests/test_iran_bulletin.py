"""Iran theatre bulletin: direction extraction and section assignment (3 Sep 2026).

The bulletin is organised by DIRECTION — strikes on Iran, strikes from Iran,
regional moves — and SIM stores no such field: Pass C records where an event
happened, never who acted. So the actor is extracted, and these tests pin the two
places that decision can go wrong.

Measured over the live corpus before the module was written: 411 of 474 theatre
headlines (87%) name an actor, so the information is there; report_kind already
removes commentary/followup/roundup (71 of 474); and inside what remains, 59 of
403 carry claim language and 7 are denials — which is why standing is a FIELD and
not a filter.
"""

import json

from src.services import iran_bulletin as ib


class TestSectionAssignment:
    def _ev(self, country, actor, standing=ib.STANDING_CONFIRMED, target=None):
        ev = {"country_iso": country, "actor": actor, "standing": standing}
        if target is not None:
            ev["target"] = target
        return ev

    def test_us_strike_on_iranian_soil_is_section_one(self):
        assert ib.assign_section(self._ev("IR", ib.US_SIDE)) == ib.SECTION_ON_IRAN

    def test_iranian_strike_on_a_neighbour_is_section_two(self):
        assert ib.assign_section(self._ev("JO", ib.IRAN_SIDE)) == ib.SECTION_FROM_IRAN
        assert ib.assign_section(self._ev("KW", ib.IRAN_SIDE)) == ib.SECTION_FROM_IRAN

    def test_an_internal_iranian_incident_is_not_an_exchange(self):
        """Iran acting on its own soil is not part of the war's exchange."""
        assert ib.assign_section(self._ev("IR", ib.IRAN_SIDE)) == ib.SECTION_REGIONAL

    def test_unattributed_never_enters_a_directional_section(self):
        """Filing it by direction would assert the thing that could not be read."""
        assert ib.assign_section(self._ev("IR", ib.UNATTRIBUTED)) == ib.SECTION_REGIONAL
        assert ib.assign_section(self._ev("JO", ib.UNATTRIBUTED)) == ib.SECTION_REGIONAL

    def test_a_third_party_actor_is_regional(self):
        assert ib.assign_section(self._ev("LB", ib.OTHER_SIDE)) == ib.SECTION_REGIONAL

    def test_a_denied_claim_keeps_its_direction(self):
        """Standing is reported, not used to re-file: a denied Iranian claim is
        still an Iranian claim, and the bulletin says so in its source line."""
        assert ib.assign_section(
            self._ev("JO", ib.IRAN_SIDE, ib.STANDING_DENIED)) == ib.SECTION_FROM_IRAN

    def test_us_strike_on_a_third_country_is_not_from_iran(self):
        assert ib.assign_section(self._ev("IQ", ib.US_SIDE)) == ib.SECTION_REGIONAL


class TestGrouping:
    def test_buckets_are_ordered_by_severity(self):
        events = [
            {"country_iso": "IR", "actor": ib.US_SIDE, "severity": 40},
            {"country_iso": "IR", "actor": ib.US_SIDE, "severity": 95},
            {"country_iso": "IR", "actor": ib.US_SIDE, "severity": 70},
        ]
        out = ib.group_into_sections(events)
        assert [e["severity"] for e in out[ib.SECTION_ON_IRAN]] == [95, 70, 40]

    def test_every_section_exists_even_when_empty(self):
        out = ib.group_into_sections([])
        assert set(out) == {ib.SECTION_ON_IRAN, ib.SECTION_FROM_IRAN,
                            ib.SECTION_REGIONAL}

    def test_a_missing_severity_does_not_crash_the_sort(self):
        events = [{"country_iso": "JO", "actor": ib.IRAN_SIDE, "severity": None},
                  {"country_iso": "JO", "actor": ib.IRAN_SIDE, "severity": 60}]
        out = ib.group_into_sections(events)
        assert [e["severity"] for e in out[ib.SECTION_FROM_IRAN]] == [60, None]


class TestExtractionParsing:
    def test_reads_a_clean_reply(self):
        body = json.dumps({"items": [
            {"n": 1, "actor": "iran", "standing": "claimed"},
            {"n": 2, "actor": "us_coalition", "standing": "confirmed"}]})
        out = ib._parse_extraction(body, 2)
        assert out[0]["actor"] == ib.IRAN_SIDE
        assert out[0]["standing"] == ib.STANDING_CLAIMED
        assert out[1]["actor"] == ib.US_SIDE
        assert out[1]["standing"] == ib.STANDING_CONFIRMED

    def test_tolerates_prose_around_the_json(self):
        """Bulk slots emit a reasoning preamble; the JSON still has to be found."""
        body = ('Here is my analysis.\n{"items":[{"n":1,"actor":"iran",'
                '"standing":"confirmed"}]}\nHope that helps.')
        assert ib._parse_extraction(body, 1)[0]["actor"] == ib.IRAN_SIDE

    def test_an_invented_label_is_treated_as_absent(self):
        """A hallucinated actor would move a real strike into the wrong half of
        the war, so an unrecognised value must not be trusted."""
        body = json.dumps({"items": [{"n": 1, "actor": "russia_side",
                                      "standing": "probably"}]})
        out = ib._parse_extraction(body, 1)
        assert out[0] == {"actor": ib.UNATTRIBUTED, "target": ib.UNATTRIBUTED,
                          "standing": ib.STANDING_UNKNOWN}

    def test_a_short_reply_leaves_the_rest_unattributed(self):
        body = json.dumps({"items": [{"n": 1, "actor": "iran",
                                      "standing": "confirmed"}]})
        out = ib._parse_extraction(body, 3)
        assert len(out) == 3
        assert out[1]["actor"] == ib.UNATTRIBUTED
        assert out[2]["actor"] == ib.UNATTRIBUTED

    def test_an_out_of_range_index_is_ignored_not_crashed(self):
        body = json.dumps({"items": [{"n": 9, "actor": "iran",
                                      "standing": "confirmed"}]})
        assert ib._parse_extraction(body, 2)[0]["actor"] == ib.UNATTRIBUTED

    def test_a_reply_with_no_json_raises(self):
        try:
            ib._parse_extraction("I cannot help with that.", 1)
        except ValueError:
            return
        raise AssertionError("expected ValueError")


class TestExtractionResilience:
    def test_a_failed_batch_leaves_events_unattributed_not_missing(self, monkeypatch):
        """A bad LLM day must cost the bulletin precision, never coverage."""
        def _boom(*a, **k):
            raise RuntimeError("all slots throttled")

        monkeypatch.setattr(ib, "call_llm", _boom)
        events = [{"title": "Iran strikes Ali Al Salem", "country_iso": "KW"},
                  {"title": "US hits IRGC targets", "country_iso": "IR"}]
        out = ib.extract_direction(None, events)
        assert len(out) == 2
        assert all(e["actor"] == ib.UNATTRIBUTED for e in out)
        assert all(ib.assign_section(e) == ib.SECTION_REGIONAL for e in out)

    def test_batching_covers_every_event(self, monkeypatch):
        seen = []

        def _fake(router, prompt, system_prompt, max_tokens):
            count = sum(1 for line in prompt.split("\n")
                        if line[:2].strip().rstrip(".").isdigit()
                        and line.strip()[0].isdigit())
            seen.append(count)
            return {"content": json.dumps({"items": [
                {"n": i + 1, "actor": "iran", "standing": "confirmed"}
                for i in range(count)]})}

        monkeypatch.setattr(ib, "call_llm", _fake)
        events = [{"title": f"Iran strikes site {i}", "country_iso": "JO"}
                  for i in range(25)]
        out = ib.extract_direction(None, events, batch_size=10)
        assert len(out) == 25
        assert all(e["actor"] == ib.IRAN_SIDE for e in out)
        assert sum(seen) == 25

    def test_the_call_is_labelled_for_spend_attribution(self, monkeypatch):
        """A stage that never logs looks free in the spend rollup, which is the
        exact regression tests/test_llm_spend_attribution.py exists to stop."""
        captured = {}

        monkeypatch.setattr(ib, "call_llm", lambda **k: {"content": json.dumps(
            {"items": [{"n": 1, "actor": "iran", "standing": "confirmed"}]})})
        monkeypatch.setattr(ib, "log_llm_telemetry",
                            lambda conn, res, router, success, purpose:
                            captured.update(purpose=purpose, success=success))
        ib.extract_direction(None, [{"title": "x", "country_iso": "JO"}],
                             db_conn=object())
        assert captured["purpose"] == "bulletin_direction"
        assert captured["success"] is True

    def test_no_database_means_no_telemetry_call(self, monkeypatch):
        monkeypatch.setattr(ib, "call_llm", lambda **k: {"content": json.dumps(
            {"items": [{"n": 1, "actor": "iran", "standing": "confirmed"}]})})

        def _must_not_run(*a, **k):
            raise AssertionError("telemetry needs a connection")

        monkeypatch.setattr(ib, "log_llm_telemetry", _must_not_run)
        ib.extract_direction(None, [{"title": "x", "country_iso": "JO"}])


class TestBulletinRouter:
    """The extraction router is a MEASURED subset, not the full bulk cascade.

    probe_models --bulletin, 3 Sep 2026, against the real extraction prompt:

        qwen/qwen3.8-27b        actor 8/8   520ms
        gemini-3.5-flash-lite   actor 8/8  1041ms
        nemotron-3-super        actor 8/8  2063ms
        openai/gpt-oss-20b      actor 6/8   976ms   ← excluded

    gpt-oss-20b returns actor=iran for "Iran says 18 killed, 142 injured in US
    strikes", filing an American strike as an Iranian one. Direction is the one
    thing this bulletin exists to state.
    """

    def test_the_inverting_slot_is_not_in_the_allowed_set(self):
        from src.core.llm_router import BULLETIN_MEASURED_MODELS
        assert "openai/gpt-oss-20b" not in BULLETIN_MEASURED_MODELS

    def test_measured_order_is_fastest_perfect_first(self):
        from src.core.llm_router import BULLETIN_MEASURED_MODELS
        assert BULLETIN_MEASURED_MODELS[0] == "qwen/qwen3.8-27b"

    def test_router_only_ever_contains_measured_models(self, monkeypatch):
        from src.core import llm_router as lr

        class _Acct:
            def __init__(self, model):
                self.model = model
                self.bucket = None

        class _FakeRouter:
            accounts = [_Acct("openai/gpt-oss-20b"),
                        _Acct("nvidia/nemotron-3-super-120b-a12b:free"),
                        _Acct("qwen/qwen3.8-27b"),
                        _Acct("some/unmeasured-model")]

        monkeypatch.setattr(lr, "build_llm_router", lambda: _FakeRouter())
        out = lr.build_bulletin_router()
        models = [a.model for a in out.accounts]
        assert models == ["qwen/qwen3.8-27b",
                          "nvidia/nemotron-3-super-120b-a12b:free"]

    def test_no_measured_key_yields_an_empty_router_not_a_fallback(self, monkeypatch):
        """An absent slot leaves events unattributed and the bulletin says so;
        falling back to the full cascade would silently reach the inverting one."""
        from src.core import llm_router as lr

        class _FakeRouter:
            accounts = []

        monkeypatch.setattr(lr, "build_llm_router", lambda: _FakeRouter())
        assert lr.build_bulletin_router().accounts == []


class TestNarrativePrompt:
    """The prompt carries the two rules the measurements made non-negotiable."""

    def _sections(self):
        return ib.group_into_sections([
            {"title": "US launches strikes on IRGC targets", "country_iso": "IR",
             "actor": ib.US_SIDE, "standing": ib.STANDING_CONFIRMED,
             "severity": 95, "domain": "reuters.com", "corroborating_sources": [{}]},
            {"title": "IRGC claims elimination of US personnel in Jordan",
             "country_iso": "JO", "actor": ib.IRAN_SIDE,
             "standing": ib.STANDING_CLAIMED, "severity": 75,
             "domain": "farsnews.ir", "corroborating_sources": []},
        ])

    def _prompt(self):
        from datetime import datetime, timezone
        return ib._narrative_prompt(
            self._sections(),
            datetime(2026, 9, 2, 7, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 3, 7, 0, tzinfo=timezone.utc))

    def test_it_forbids_inventing_a_time(self):
        """time_certainty='exact' was 0 across all 12 theatre countries, so any
        clock detail in the output would be fabricated."""
        assert "Saat verme" in self._prompt()

    def test_a_one_sided_claim_must_not_be_told_as_fact(self):
        prompt = self._prompt()
        assert "Tek taraflı iddia" in prompt
        assert "gerçekleşmiş gibi anlatma" in prompt

    def test_standing_reaches_the_model_per_event(self):
        prompt = self._prompt()
        assert '"durum": "Tek taraflı iddia"' in prompt
        assert '"durum": "Doğrulandı"' in prompt

    def test_sections_are_upper_case_so_the_renderer_sees_headers(self):
        """render_sitrep_html is shape-driven: an ALL-CAPS line is a section."""
        for title in ib.SECTION_TITLES.values():
            letters = [c for c in title if c.isalpha()]
            assert letters and all(c == c.upper() for c in letters), title

    def test_every_section_title_appears_even_when_empty(self):
        prompt = self._prompt()
        for title in ib.SECTION_TITLES.values():
            assert title in prompt


class TestBuildBulletin:
    def test_an_empty_window_is_not_an_error(self, monkeypatch):
        monkeypatch.setattr(ib, "fetch_theatre_events", lambda *a: [])

        def _must_not_run(*a, **k):
            raise AssertionError("no events means no model call")

        monkeypatch.setattr(ib, "call_llm", _must_not_run)
        from datetime import datetime, timezone
        out = ib.build_bulletin(None, None,
                                datetime(2026, 9, 2, tzinfo=timezone.utc),
                                datetime(2026, 9, 3, tzinfo=timezone.utc))
        assert out["status"] == "empty"
        assert out["narrative"] == ""

    def test_narrative_is_requested_as_prose_not_json(self, monkeypatch):
        """A reasoning model asked for JSON returns the report inside a string
        field, and the shape-driven renderer then sees one long line."""
        captured = {}
        monkeypatch.setattr(ib, "fetch_theatre_events", lambda *a: [
            {"title": "US strikes IRGC site", "country_iso": "IR",
             "corroborating_sources": [], "severity": 90, "domain": "x.com"}])
        monkeypatch.setattr(ib, "extract_direction",
                            lambda r, e, db_conn=None: [
                                x.update(actor=ib.US_SIDE,
                                         standing=ib.STANDING_CONFIRMED) or x
                                for x in e])

        def _fake(**kwargs):
            captured.update(kwargs)
            return {"content": "YÖNETİCİ ÖZETİ\nBir şeyler oldu."}

        monkeypatch.setattr(ib, "call_llm", _fake)
        from datetime import datetime, timezone
        out = ib.build_bulletin(None, None,
                                datetime(2026, 9, 2, tzinfo=timezone.utc),
                                datetime(2026, 9, 3, tzinfo=timezone.utc))
        assert captured["json_mode"] is False
        assert out["status"] == "ok"
        assert out["narrative"].startswith("YÖNETİCİ ÖZETİ")


class TestDirectionUsesTargetNotFiling:
    """country_iso is a fallback, not the signal. Measured 3 Sep 2026.

    The first real bulletin put 29 of one window's 74 "regional" events in the
    wrong section — 16% of the report — because assign_section read country_iso as
    "where it landed". Pass C files "Iran strikes bases in Bahrain, Iraq and
    Jordan" under IR: Iran is the dominant country in the text, not the country
    that was hit. Every one of those 29 was section-2 material, which is precisely
    what the bulletin exists to show.
    """

    def test_the_headline_that_exposed_it(self):
        """Iran striking neighbours, filed under IR."""
        ev = {"country_iso": "IR", "actor": ib.IRAN_SIDE, "target": ib.US_SIDE,
              "standing": ib.STANDING_CONFIRMED}
        assert ib.assign_section(ev) == ib.SECTION_FROM_IRAN

    def test_a_us_strike_on_iran_is_still_section_one(self):
        ev = {"country_iso": "IR", "actor": ib.US_SIDE, "target": ib.IRAN_SIDE}
        assert ib.assign_section(ev) == ib.SECTION_ON_IRAN

    def test_one_side_acting_on_itself_is_not_an_exchange(self):
        """Air defence over its own territory, an internal incident."""
        ev = {"country_iso": "IR", "actor": ib.IRAN_SIDE, "target": ib.IRAN_SIDE}
        assert ib.assign_section(ev) == ib.SECTION_REGIONAL

    def test_target_beats_country_iso_when_they_disagree(self):
        filed_in_iran = {"country_iso": "IR", "actor": ib.IRAN_SIDE,
                         "target": ib.OTHER_SIDE}
        assert ib.assign_section(filed_in_iran) == ib.SECTION_FROM_IRAN

    def test_a_missing_target_falls_back_to_the_filing(self):
        """The old rule survives for exactly the case it was right for."""
        assert ib.assign_section(
            {"country_iso": "IR", "actor": ib.US_SIDE}) == ib.SECTION_ON_IRAN
        assert ib.assign_section(
            {"country_iso": "JO", "actor": ib.IRAN_SIDE}) == ib.SECTION_FROM_IRAN

    def test_an_unreadable_actor_still_never_gets_a_direction(self):
        for target in (ib.IRAN_SIDE, ib.US_SIDE, ib.OTHER_SIDE):
            ev = {"country_iso": "IR", "actor": ib.UNATTRIBUTED, "target": target}
            assert ib.assign_section(ev) == ib.SECTION_REGIONAL

    def test_a_third_party_exchange_is_regional(self):
        ev = {"country_iso": "LB", "actor": ib.OTHER_SIDE, "target": ib.US_SIDE}
        assert ib.assign_section(ev) == ib.SECTION_REGIONAL

    def test_a_us_strike_on_a_third_country_is_not_section_one(self):
        ev = {"country_iso": "IQ", "actor": ib.US_SIDE, "target": ib.OTHER_SIDE}
        assert ib.assign_section(ev) == ib.SECTION_REGIONAL
