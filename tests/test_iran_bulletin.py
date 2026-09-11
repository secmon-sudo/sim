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


class TestIncidentClustering:
    """Six rows for one Iranian strike on a Jordanian airbase, four tagged
    "claimed" and two "confirmed", and a narrator that wrote six contradicting
    bullets. collapse_by_storyline could not reach it — the linker had given them
    six storyline_ids — and title similarity cannot either, since "Iran Strike On
    Jordan Base Damages US Warplanes" and "U.S. A-10 Warthog and F-15 Strike
    Eagles Damaged in Iranian Missile Attack" share almost no wording."""

    EVENTS = [
        {"title": "Iran strike on Jordan base damages US warplanes",
         "standing": ib.STANDING_CLAIMED, "outlet_count": 2, "severity": 80},
        {"title": "A-10 and F-15 damaged in Iranian missile attack",
         "standing": ib.STANDING_CONFIRMED, "outlet_count": 5, "severity": 90},
        {"title": "Kurdish activist dies during IRGC siege",
         "standing": ib.STANDING_CONFIRMED, "outlet_count": 1, "severity": 40},
    ]

    def _router(self):
        return object()

    def _reply(self, body):
        return lambda *_a, **_k: {"content": body}

    def test_a_merged_incident_keeps_the_weakest_standing(self):
        """The whole point: one claim among the members makes the merged row a
        claim. Taking the strongest is how a claim becomes a fact."""
        out = ib.merge_same_incident(self._router(), self.EVENTS,
                                     call_llm_fn=self._reply('{"groups":[[1,2],[3]]}'))
        assert len(out) == 2
        assert out[0]["standing"] == ib.STANDING_CLAIMED
        assert out[0]["merged_filings"] == 2

    def test_the_representative_is_the_most_corroborated_member(self):
        out = ib.merge_same_incident(self._router(), self.EVENTS,
                                     call_llm_fn=self._reply('{"groups":[[1,2],[3]]}'))
        assert "A-10" in out[0]["title"]
        assert out[0]["outlet_count"] == 7
        assert out[0]["severity"] == 90

    def test_an_unrelated_event_is_left_alone(self):
        out = ib.merge_same_incident(self._router(), self.EVENTS,
                                     call_llm_fn=self._reply('{"groups":[[1,2],[3]]}'))
        assert out[1]["title"].startswith("Kurdish activist")
        assert "merged_filings" not in out[1]

    def test_a_dropped_index_groups_nothing(self):
        """A reply that forgets an index would delete that event from the report.
        Refusing the whole reply is the only safe reading."""
        out = ib.merge_same_incident(self._router(), self.EVENTS,
                                     call_llm_fn=self._reply('{"groups":[[1,2]]}'))
        assert out == self.EVENTS

    def test_a_repeated_index_groups_nothing(self):
        out = ib.merge_same_incident(self._router(), self.EVENTS,
                                     call_llm_fn=self._reply('{"groups":[[1,2],[2,3]]}'))
        assert out == self.EVENTS

    def test_an_unparseable_reply_fails_open(self):
        out = ib.merge_same_incident(self._router(), self.EVENTS,
                                     call_llm_fn=self._reply("sorry, no JSON here"))
        assert out == self.EVENTS

    def test_a_raising_model_fails_open(self):
        def boom(*_a, **_k):
            raise RuntimeError("router exhausted")

        assert ib.merge_same_incident(self._router(), self.EVENTS,
                                      call_llm_fn=boom) == self.EVENTS

    def test_a_single_event_is_not_worth_a_call(self):
        called = []

        def spy(*_a, **_k):
            called.append(1)
            return {"content": '{"groups":[[1]]}'}

        assert ib.merge_same_incident(self._router(), self.EVENTS[:1],
                                      call_llm_fn=spy) == self.EVENTS[:1]
        assert not called


class TestNonKineticRouting:
    """The nuclear file joined the report on 2026-09-11; the sections are titled
    SALDIRILAR. A Security Council referral belongs in the bulletin — it is what
    this war is fought over — but not under a heading that says someone was
    struck."""

    def test_a_sanctions_move_does_not_enter_section_one(self):
        ev = {"actor": ib.US_SIDE, "target": ib.IRAN_SIDE, "country_iso": "IR",
              ib.KINETIC: False}
        assert ib.assign_section(ev) == ib.SECTION_REGIONAL

    def test_a_ceasefire_condition_does_not_enter_section_two(self):
        ev = {"actor": ib.IRAN_SIDE, "target": ib.US_SIDE, "country_iso": "IR",
              ib.KINETIC: False}
        assert ib.assign_section(ev) == ib.SECTION_REGIONAL

    def test_a_strike_still_routes_by_direction(self):
        ev = {"actor": ib.IRAN_SIDE, "target": ib.US_SIDE, "country_iso": "JO",
              ib.KINETIC: True}
        assert ib.assign_section(ev) == ib.SECTION_FROM_IRAN

    def test_a_missing_flag_changes_nothing(self):
        """Every default in this parser fails open; a model that never answers the
        field must leave the report exactly as it was."""
        ev = {"actor": ib.IRAN_SIDE, "target": ib.US_SIDE, "country_iso": "JO"}
        assert ib.assign_section(ev) == ib.SECTION_FROM_IRAN


class TestPlaceHeadings:
    """Four bullets about the Strait of Hormuz shipped under the heading "Ürdün".

    14 of the 27 section-2 events on 10 Sep 2026 had target_country "unknown", and
    a narrator given a severity-ordered list with no place for half of it put them
    under whatever heading came before.
    """

    def test_section_two_reads_the_target_not_the_filing_country(self):
        """Pass C files "Iran strikes ships outside Hormuz" under IR because Iran
        is the dominant country in the text. Section 2 is about where it landed."""
        ev = {"country_iso": "IR", "target_country": "JO"}
        assert ib.bulletin_place(ev, ib.SECTION_FROM_IRAN) == "Ürdün"

    def test_section_two_falls_back_to_the_filing_country(self):
        ev = {"country_iso": "SA", "target_country": "unknown"}
        assert ib.bulletin_place(ev, ib.SECTION_FROM_IRAN) == "Suudi Arabistan"

    def test_a_placeless_section_two_event_gets_its_own_heading(self):
        """Iran filed, target unknown: the Hormuz shipping case. Iran is not the
        place a strike FROM Iran landed, so this is not "İran"."""
        ev = {"country_iso": "IR", "target_country": "unknown"}
        assert ib.bulletin_place(ev, ib.SECTION_FROM_IRAN) == ib.UNPLACED_LABEL

    def test_section_one_is_always_iran(self):
        ev = {"country_iso": "JO", "target_country": "unknown"}
        assert ib.bulletin_place(ev, ib.SECTION_ON_IRAN) == "İran"

    def test_places_group_together_and_the_placeless_sort_last(self):
        events = [
            {"country_iso": "IR", "target_country": "unknown",
             "actor": ib.IRAN_SIDE, "target": ib.US_SIDE, "severity": 100},
            {"country_iso": "IR", "target_country": "JO",
             "actor": ib.IRAN_SIDE, "target": ib.US_SIDE, "severity": 40},
            {"country_iso": "IR", "target_country": "SA",
             "actor": ib.IRAN_SIDE, "target": ib.US_SIDE, "severity": 90},
            {"country_iso": "IR", "target_country": "JO",
             "actor": ib.IRAN_SIDE, "target": ib.US_SIDE, "severity": 80},
        ]
        out = ib.group_into_sections(events)[ib.SECTION_FROM_IRAN]
        assert [e["place"] for e in out] == [
            "Suudi Arabistan", "Ürdün", "Ürdün", ib.UNPLACED_LABEL]
        assert [e["severity"] for e in out] == [90, 80, 40, 100]


class TestSectionCoverage:
    """Probed 2026-09-10: pushed to group its bullets by place, the narrator
    dropped all three section headings and listed countries instead — and the
    ALL-CAPS check passed anyway, because "YÖNETİCİ ÖZETİ" is ALL-CAPS too. The
    directional claim is the report; it cannot go missing quietly."""

    SECTIONS = {
        ib.SECTION_ON_IRAN: [{"standing": ib.STANDING_CONFIRMED}],
        ib.SECTION_FROM_IRAN: [{"standing": ib.STANDING_CLAIMED}],
        ib.SECTION_REGIONAL: [],
    }

    def test_a_missing_section_heading_is_rejected(self):
        text = ("YÖNETİCİ ÖZETİ\n\nÖzet.\n\nÜrdün\n- Bir olay — Durum: Doğrulandı")
        assert not ib.narrative_covers_sections(text, self.SECTIONS)

    def test_both_populated_sections_present_passes(self):
        text = "\n".join([ib.SECTION_TITLES[ib.SECTION_ON_IRAN],
                          ib.SECTION_TITLES[ib.SECTION_FROM_IRAN]])
        assert ib.narrative_covers_sections(text, self.SECTIONS)

    def test_an_empty_section_is_not_required(self):
        """The regional bucket is empty here; demanding its heading would ask the
        narrator to write a section with nothing in it."""
        text = "\n".join([ib.SECTION_TITLES[ib.SECTION_ON_IRAN],
                          ib.SECTION_TITLES[ib.SECTION_FROM_IRAN]])
        assert ib.SECTION_TITLES[ib.SECTION_REGIONAL] not in text
        assert ib.narrative_covers_sections(text, self.SECTIONS)


class TestSourcedNumbers:
    """"beş İran petrol tankeri vurulmuştur" and then "beş İran petrol tankeri
    DAHA vurulmuştur" — two filings of one strike, narrated as two strikes. Probed
    on the same payload the model summed them instead: "toplam 10 adet"."""

    PROMPT = 'VERİ: [{"baslik": "US strikes five tankers, 18 of 20 missiles"}]'

    def test_a_number_from_the_data_passes(self):
        assert ib.narrative_numbers_are_sourced(
            "- 5 tanker vuruldu", self.PROMPT)

    def test_an_english_number_word_counts_as_source(self):
        """The headlines are English and the report is Turkish: "five" in the data
        is where a "5" in the prose legitimately comes from."""
        assert ib.narrative_numbers_are_sourced("- 18 füze", self.PROMPT)

    def test_a_summed_total_is_rejected(self):
        assert not ib.narrative_numbers_are_sourced(
            "- toplam 10 adet tanker vuruldu", self.PROMPT)

    def test_one_is_exempt(self):
        """Turkish "bir" is the indefinite article; a rule that fires on "bir
        tanker" rejects every honest narrative."""
        assert ib.narrative_numbers_are_sourced("- bir tanker vuruldu", "VERİ: []")

    def test_a_thousand_separator_is_one_number(self):
        assert ib.narrative_numbers_are_sourced(
            "- 7.700 ihlal", 'VERİ: "violated the agreement 7,700 times"')


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

    def test_a_trailing_second_object_does_not_lose_the_batch(self):
        """gemini-3.5-flash-lite failed the direction probe twice on 2026-09-04
        with "Extra data", having answered every row correctly: it appended a
        second JSON object after the answer, and the first-brace-to-last-brace
        span then covered both. One trailing object was costing a whole batch."""
        body = ('{"items":[{"n":1,"actor":"iran","target":"us_coalition",'
                '"standing":"confirmed"}]}\n{"note":"analysis complete"}')
        assert ib._parse_extraction(body, 1)[0]["actor"] == ib.IRAN_SIDE

    def test_a_preamble_before_one_object_still_works(self):
        """The wide span is what handles this, so the fallback must not replace
        it — the two shapes need different readings of the same reply."""
        body = 'Analiz:\n{"items":[{"n":1,"actor":"us_coalition"}]}\nBitti.'
        assert ib._parse_extraction(body, 1)[0]["actor"] == ib.US_SIDE

    def test_an_invented_label_is_treated_as_absent(self):
        """A hallucinated actor would move a real strike into the wrong half of
        the war, so an unrecognised value must not be trusted."""
        body = json.dumps({"items": [{"n": 1, "actor": "russia_side",
                                      "standing": "probably"}]})
        out = ib._parse_extraction(body, 1)
        assert out[0] == {"actor": ib.UNATTRIBUTED, "target": ib.UNATTRIBUTED,
                          "target_country": ib.UNKNOWN_COUNTRY,
                          "standing": ib.STANDING_UNKNOWN, ib.WAR_RELATED: True,
                          ib.KINETIC: True}

    def test_a_target_country_is_read_as_an_iso_code_or_not_at_all(self):
        """It decides a section, so it is one of the values we asked for or absent."""
        body = json.dumps({"items": [
            {"n": 1, "actor": "iran", "target_country": "ye"},
            {"n": 2, "actor": "iran", "target_country": "Saudi Arabia"},
            {"n": 3, "actor": "iran", "target_country": "unknown"}]})
        out = ib._parse_extraction(body, 3)
        assert out[0]["target_country"] == "YE"
        assert out[1]["target_country"] == ib.UNKNOWN_COUNTRY
        assert out[2]["target_country"] == ib.UNKNOWN_COUNTRY

    def test_the_prompt_asks_for_the_target_country(self):
        prompt = ib._extraction_prompt([{"title": "x"}])
        assert "target_country" in prompt

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

    def test_the_head_is_the_slot_that_can_actually_serve(self):
        """Was qwen, on accuracy alone (actor 20/20). Reordered 6 Sep when it
        turned out it cannot serve this report at all: Groq's free tier caps
        OUTPUT at 1,000 tokens per minute and the bulletin sends ten batches of
        912 back to back. Accuracy you cannot reach is not accuracy."""
        from src.core.llm_router import BULLETIN_MEASURED_MODELS
        assert BULLETIN_MEASURED_MODELS[0] == "google/gemini-3.1-flash-lite"

    def test_router_only_ever_contains_measured_models(self, monkeypatch):
        from src.core import llm_router as lr

        class _Acct:
            def __init__(self, model):
                self.model = model
                self.bucket = None
                # The router de-duplicates on display_name now that it reads a
                # cascade which already contains the main one.
                self.display_name = f"fake/A/{model}"

        class _FakeRouter:
            accounts = [_Acct("openai/gpt-oss-20b"),
                        _Acct("nvidia/nemotron-3-super-120b-a12b:free"),
                        _Acct("gemini-3.5-flash-lite"),
                        _Acct("qwen/qwen3.8-27b"),
                        _Acct("google/gemini-3.1-flash-lite"),
                        _Acct("some/unmeasured-model")]

        # The bulletin router draws from the QUALITY cascade, which already ends
        # with the full main one — that is how the paid floor is reachable here.
        monkeypatch.setattr(lr, "build_quality_router", lambda: _FakeRouter())
        monkeypatch.setattr(lr, "build_llm_router", lambda: _FakeRouter())
        # The declared fallback rungs are not drawn from any cascade, so they are
        # not what this test is about; take them out and judge the filter alone.
        monkeypatch.setattr(lr, "_bulletin_fallback_slots", list)
        out = lr.build_bulletin_router()
        models = [a.model for a in out.accounts]
        # Every exclusion here is a measured failure. gpt-oss-20b and nemotron
        # assert a direction the text does not carry; gemini-3.5-flash-lite
        # (removed 5 Sep) returns short replies whose missing items silently
        # become "unattributed" — 51 of 73 events in one live bulletin.
        #
        # nemotron stays excluded HERE even though a Kilo-hosted nemotron is now a
        # declared fallback rung: this list is what may be drawn from the bulk
        # cascade, and a slug appearing in both places is not evidence that the
        # cascade's copy was ever measured on this task.
        assert models == ["google/gemini-3.1-flash-lite"]

    def test_an_empty_cascade_never_falls_back_to_the_full_one(self, monkeypatch):
        """An absent slot leaves events unattributed and the bulletin says so;
        falling back to the full cascade would silently reach the inverting one.

        Was "yields an empty router" until 2026-09-06. It cannot be empty any
        more — the keyless Kilo rung is always there — but the property that
        mattered was never emptiness, it was that nothing UNMEASURED gets in."""
        from src.core import llm_router as lr

        class _FakeRouter:
            accounts = []

        monkeypatch.setattr(lr, "build_quality_router", lambda: _FakeRouter())
        monkeypatch.setattr(lr, "build_llm_router", lambda: _FakeRouter())
        monkeypatch.delenv("AION_API_KEY", raising=False)
        accounts = lr.build_bulletin_router().accounts
        assert [a.provider for a in accounts] == ["kilo"]


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

    def test_no_internal_slug_reaches_the_narrator(self):
        """The 4 Sep bulletin printed "us_coalition tarafından gerçekleştirilen
        saldırılarda" eight times, because the actor went into the payload as the
        slug the sections are keyed on. Anything in that payload can end up in
        Turkish prose verbatim, so none of the slugs may be in it."""
        prompt = self._prompt()
        for slug in (ib.IRAN_SIDE, ib.US_SIDE, ib.OTHER_SIDE, ib.UNATTRIBUTED,
                     ib.STANDING_CONFIRMED, ib.STANDING_CLAIMED,
                     ib.STANDING_DENIED, ib.STANDING_UNKNOWN):
            assert slug not in prompt, slug

    def test_the_actor_reaches_the_model_by_name(self):
        prompt = self._prompt()
        assert '"fail": "ABD/koalisyon güçleri"' in prompt
        assert '"fail": "İran"' in prompt

    def test_an_unnamed_actor_is_labelled_rather_than_dropped(self):
        """A missing key would let the model pick an actor from the headline; the
        label plus its rule tells it not to attribute at all."""
        assert ib.ACTOR_LABELS[ib.UNATTRIBUTED] in self._prompt()
        assert "fail atfetme" in self._prompt()


class TestOffTopicEvents:
    """The theatre is a list of COUNTRIES, so everything that happens in twelve of
    them lands in the fetch. On 2026-09-04 the bulletin reported a Nazareth
    homicide, two sisters killed in Sharjah, an Iranian professors' pay dispute
    and an IndiGo diversion for a sick pilot — all correctly classified, none of
    them this report's subject."""

    def _build(self, monkeypatch, events):
        from datetime import datetime, timezone
        monkeypatch.setattr(ib, "fetch_theatre_events", lambda *a: events)
        monkeypatch.setattr(ib, "extract_direction",
                            lambda router, evs, **k: evs)
        monkeypatch.setattr(ib, "call_llm",
                            lambda **k: {"content": "YÖNETİCİ ÖZETİ\nX", "model": "m"})
        monkeypatch.setattr(ib, "log_llm_telemetry", lambda *a, **k: None)
        return ib.build_bulletin(None, None,
                                 datetime(2026, 9, 3, tzinfo=timezone.utc),
                                 datetime(2026, 9, 4, tzinfo=timezone.utc))

    def _ev(self, title, **kw):
        base = {"title": title, "country_iso": "IL", "actor": ib.OTHER_SIDE,
                "target": ib.OTHER_SIDE, "standing": ib.STANDING_CONFIRMED,
                "severity": 70, "domain": "x.com", "corroborating_sources": []}
        base.update(kw)
        return base

    def test_an_off_topic_event_leaves_the_report(self, monkeypatch):
        out = self._build(monkeypatch, [
            self._ev("Sharjah Police arrest Kazakh man after two sisters killed",
                     **{ib.WAR_RELATED: False}),
            self._ev("Israeli artillery shelling reported in southern Lebanon",
                     **{ib.WAR_RELATED: True}),
        ])
        titles = [e["title"] for e in out["events"]]
        assert titles == ["Israeli artillery shelling reported in southern Lebanon"]

    def test_a_belligerent_event_survives_the_field(self, monkeypatch):
        """A single mislabelled field must not be able to delete a strike: if the
        extractor put Iran or the coalition behind it, it is the war."""
        out = self._build(monkeypatch, [
            self._ev("US strike kills key broker in Houthi arms alliance",
                     actor=ib.US_SIDE, **{ib.WAR_RELATED: False}),
        ])
        assert len(out["events"]) == 1

    def test_a_missing_field_keeps_the_event(self, monkeypatch):
        """The default is fail-open everywhere: a parse failure is not evidence
        that an event is off-topic."""
        out = self._build(monkeypatch, [self._ev("Something we could not label")])
        assert len(out["events"]) == 1


class TestSafetyEventsExcluded:
    def test_the_fetch_excludes_accidental_occurrences(self):
        """An IndiGo diversion for a sick pilot was in the 4 Sep bulletin. The cut
        is deterministic and made in SQL, so it costs no extraction either."""

        class _Cur:
            def fetchall(self):
                return []

        seen = {}

        class _Conn:
            def execute(self, sql, params):
                seen["sql"], seen["params"] = sql, params
                return _Cur()

        from datetime import datetime, timezone
        ib.fetch_theatre_events(_Conn(), datetime(2026, 9, 3, tzinfo=timezone.utc),
                                datetime(2026, 9, 4, tzinfo=timezone.utc))
        assert "event_type" in seen["sql"]
        assert "emergency_landing" in seen["params"][-1]
        assert "bird_strike" in seen["params"][-1]

    def test_it_shares_the_sitrep_list_rather_than_copying_it(self):
        from src.services.sitrep_generator import SAFETY_ONLY_EVENT_TYPES
        assert ib.SAFETY_ONLY_EVENT_TYPES == SAFETY_ONLY_EVENT_TYPES


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


class TestNarrativeContract:
    """The 4 Sep bulletin printed "us_coalition" at the reader eight times.

    ACTOR_LABELS closed the hole that let it happen; this closes the class. Any
    payload field added later with a slug for a value gets caught here whether or
    not anyone remembers to write a label table for it.
    """

    HEADER = "İRAN TOPRAKLARINA YÖNELİK SALDIRILAR\n"

    def test_a_leaked_actor_slug_is_rejected(self):
        assert not ib.narrative_is_usable(
            self.HEADER + "us_coalition tarafından gerçekleştirilen saldırılarda...")

    def test_a_leaked_standing_slug_is_rejected(self):
        assert not ib.narrative_is_usable(
            self.HEADER + "Bu olayın durumu unknown olarak kaydedildi.")

    def test_proper_turkish_passes(self):
        assert ib.narrative_is_usable(
            self.HEADER + "ABD/koalisyon güçleri tarafından saldırı düzenlendi.")

    def test_a_narrative_with_no_section_line_is_rejected(self):
        """The renderer is shape-driven: with no ALL-CAPS line the whole report
        collapses into one undifferentiated block."""
        assert not ib.narrative_is_usable("Tek bir paragraf, hiç başlık yok.")

    def test_an_empty_narrative_is_rejected(self):
        assert not ib.narrative_is_usable("")
        assert not ib.narrative_is_usable("   \n  ")

    def test_a_slug_inside_a_longer_word_is_not_a_leak(self):
        """The match is on word boundaries: 'other' must not fire on a URL or a
        compound, or the check starts rejecting good narratives."""
        assert ib.narrative_is_usable(
            self.HEADER + "Kaynak: brotherhood-news sitesinde yayımlandı.")

    def test_the_token_list_covers_every_payload_value(self):
        """A field added to the payload without a label table is the bug this
        guards; the guard is only as good as this list."""
        for token in (ib.IRAN_SIDE, ib.US_SIDE, ib.OTHER_SIDE, ib.UNATTRIBUTED,
                      ib.STANDING_CONFIRMED, ib.STANDING_CLAIMED,
                      ib.STANDING_DENIED, ib.STANDING_UNKNOWN, ib.WAR_RELATED):
            assert token in ib._INTERNAL_TOKENS, token


class TestStandingContract:
    """The 10 Sep bulletin narrated 24 claims and 2 denials as confirmed fact.

    The standing was in the payload and the prompt already forbade the upgrade —
    the narrator collapsed seven near-duplicate Jordan filings into one bullet and
    kept the strongest label instead of the weakest. Advice the model can ignore
    becomes format it cannot: every bullet carries its standing, and the report
    cannot confirm more events than the extractor did.
    """

    SECTIONS = {
        ib.SECTION_ON_IRAN: [{"standing": ib.STANDING_CONFIRMED}],
        ib.SECTION_FROM_IRAN: [{"standing": ib.STANDING_CLAIMED},
                               {"standing": ib.STANDING_DENIED}],
        ib.SECTION_REGIONAL: [],
    }
    HEADER = "İRAN'DAN KOMŞU ÜLKELERE YÖNELİK SALDIRILAR\n"

    def test_a_tagged_narrative_passes(self):
        text = self.HEADER + "\n".join([
            "- Tankerlere saldırı düzenlendi — Durum: Doğrulandı",
            "- Üsse füze atıldığı öne sürüldü — Durum: Tek taraflı iddia",
            "- Muhrip iddiası yalanlandı — Durum: İddia edildi, yalanlandı",
        ])
        assert ib.narrative_standing_is_honest(text, self.SECTIONS)

    def test_an_untagged_bullet_is_rejected(self):
        """This is the 10 Sep shape: prose that ends in "doğrulanmıştır" and names
        no standing at all."""
        text = self.HEADER + "- Aramco rafinerisinde yangın çıktığı doğrulanmıştır."
        assert not ib.narrative_standing_is_honest(text, self.SECTIONS)

    def test_more_confirmations_than_confirmed_events_is_rejected(self):
        """The systemic case, and the only one arithmetic can catch: nineteen
        confirmations out of sixteen confirmed events."""
        text = self.HEADER + "\n".join([
            "- Birinci olay — Durum: Doğrulandı",
            "- İkinci olay — Durum: Doğrulandı",
        ])
        assert not ib.narrative_standing_is_honest(text, self.SECTIONS)

    def test_an_invented_label_is_rejected(self):
        """The tag has to be one of the four the extractor can produce; a fifth
        one the model made up is not a standing."""
        text = self.HEADER + "- Bir olay — Durum: Kısmen doğrulandı"
        assert not ib.narrative_standing_is_honest(text, self.SECTIONS)

    def test_a_narrative_with_no_bullets_is_rejected(self):
        assert not ib.narrative_standing_is_honest(
            self.HEADER + "Yalnızca düz paragraf.", self.SECTIONS)

    def test_the_prompt_carries_the_rule_the_check_enforces(self):
        """A checker the prompt never asked for fails every rung and ships nothing.
        These two have to move together."""
        from datetime import datetime, timezone

        prompt = ib._narrative_prompt(
            self.SECTIONS,
            datetime(2026, 9, 9, 8, tzinfo=timezone.utc),
            datetime(2026, 9, 10, 8, tzinfo=timezone.utc))
        assert "Durum: X" in prompt
        for label in ib.STANDING_LABELS.values():
            assert label in prompt


class TestShortReplyIsCounted:
    """A reply carrying fewer items than its batch is the failure that hid the
    5 Sep collapse: every missing item takes the unattributed default, the
    sections still render, and the report has quietly stopped saying which way
    anything was going. Fail open, but never fail silent."""

    def test_a_short_reply_bumps_the_counter(self):
        from src.core import counters

        counters.reset()
        body = json.dumps({"items": [{"n": 1, "actor": "iran"}]})
        out = ib._parse_extraction(body, 8)
        assert len(out) == 8
        assert out[7]["actor"] == ib.UNATTRIBUTED
        assert counters.snapshot().get(
            counters.BULLETIN_DIRECTION_SHORT_REPLY) == 7
        counters.reset()

    def test_a_complete_reply_counts_nothing(self):
        from src.core import counters

        counters.reset()
        body = json.dumps({"items": [
            {"n": i + 1, "actor": "iran", "target": "us_coalition",
             "standing": "confirmed"} for i in range(3)]})
        ib._parse_extraction(body, 3)
        assert counters.BULLETIN_DIRECTION_SHORT_REPLY not in counters.snapshot()
        counters.reset()

    def test_the_batch_fits_groqs_output_ceiling(self):
        """Groq's free tier rejects a request whose max_tokens exceeds 1,000
        OUTPUT tokens per minute before it runs. max_tokens is 50*batch + 512, so
        12 asked for 1,112 and the primary slot could not serve one batch."""
        assert 50 * ib.DIRECTION_BATCH_SIZE + 512 < 1000


def test_direction_has_more_than_one_slot(monkeypatch):
    """6 Sep: the list was narrowed to one model the day before, qwen hit its
    rate limit, every batch raised LLMAllThrottled and the bulletin failed
    outright. Below two slots there is no availability, whatever the accuracy.

    The invariant is about the ROUTER, not the measured-model list: since
    2026-09-06 the rungs below the floor are declared rather than filtered, so
    counting names in BULLETIN_MEASURED_MODELS stopped answering this question."""
    from src.core.llm_router import build_bulletin_router

    monkeypatch.setenv("OPENROUTER_API_KEY_A", "k")
    assert len(build_bulletin_router().accounts) >= 2


def test_the_direction_head_has_no_per_minute_output_ceiling(monkeypatch):
    """qwen is the most accurate slot measured and cannot serve this report:
    Groq's free tier caps OUTPUT at 1,000 tokens per MINUTE, the bulletin sends
    about ten batches back to back and each asks 912. Shrinking the batch makes
    it worse — fewer tokens per call, more calls, same per-minute budget.

    Kept as a fallback until 2026-09-06 on the grounds that it is the best of
    them when it answers. What that actually bought was one good batch in ten and
    nine 429s that fail open into unattributed events — the 5 and 6 Sep collapses.
    """
    from src.core.llm_router import build_bulletin_router

    monkeypatch.setenv("OPENROUTER_API_KEY_A", "k")
    accounts = build_bulletin_router().accounts
    assert accounts[0].model == "google/gemini-3.1-flash-lite"
    assert not any(a.provider == "groq" for a in accounts)


def test_the_slot_that_returned_nothing_is_gone():
    """gemini-3.5-flash-lite was restored on 6 Sep on a length theory and the
    next run said "recovered 0 of 8 items" ten times. It returns nothing
    parseable at any batch size; the probe agrees at actor 2/20."""
    from src.core.llm_router import BULLETIN_MEASURED_MODELS

    assert "gemini-3.5-flash-lite" not in BULLETIN_MEASURED_MODELS


class TestIranSideActorAtHome:
    """Section 2 needs the action to CROSS a border (9 Sep 2026).

    The Houthis are Iran-side by the actor table and are also one belligerent in
    Yemen's own civil war, so "2 children killed in Houthi shelling of displaced
    people in Yemen's Marib" was filed as an Iranian strike on a neighbour — 4-5
    rows a day across the 6-9 Sep bulletins, Hezbollah inside Lebanon included.

    The rule keys on the TARGET country because the filing country cannot separate
    the cases: the 9 Sep window filed both the Marib shelling and "Houthi strikes
    injure 73 in Saudi Arabia" under YE.
    """

    def test_the_headline_that_exposed_it(self):
        ev = {"country_iso": "YE", "actor": ib.IRAN_SIDE, "target": ib.OTHER_SIDE,
              "target_country": "YE"}
        assert ib.assign_section(ev) == ib.SECTION_REGIONAL

    def test_the_same_actor_reaching_across_the_border_is_still_section_two(self):
        """Filed under YE like the row above; only the target tells them apart."""
        ev = {"country_iso": "YE", "actor": ib.IRAN_SIDE, "target": ib.OTHER_SIDE,
              "target_country": "SA"}
        assert ib.assign_section(ev) == ib.SECTION_FROM_IRAN

    def test_hezbollah_inside_lebanon_is_regional(self):
        ev = {"country_iso": "LB", "actor": ib.IRAN_SIDE, "target": ib.OTHER_SIDE,
              "target_country": "LB"}
        assert ib.assign_section(ev) == ib.SECTION_REGIONAL

    def test_hezbollah_firing_into_israel_is_not(self):
        ev = {"country_iso": "LB", "actor": ib.IRAN_SIDE, "target": ib.OTHER_SIDE,
              "target_country": "IL"}
        assert ib.assign_section(ev) == ib.SECTION_FROM_IRAN

    def test_iraq_is_not_a_home_country(self):
        """Iran genuinely strikes Iraq — Erbil, the Surdash camp, the Kurdish
        opposition in the north. That is exactly what section 2 is for."""
        ev = {"country_iso": "IQ", "actor": ib.IRAN_SIDE, "target": ib.OTHER_SIDE,
              "target_country": "IQ"}
        assert ib.assign_section(ev) == ib.SECTION_FROM_IRAN

    def test_the_home_rule_never_moves_a_strike_ON_iran(self):
        ev = {"country_iso": "IR", "actor": ib.US_SIDE, "target": ib.IRAN_SIDE,
              "target_country": "YE"}
        assert ib.assign_section(ev) == ib.SECTION_ON_IRAN

    def test_a_missing_target_country_leaves_the_old_behaviour_alone(self):
        ev = {"country_iso": "SA", "actor": ib.IRAN_SIDE, "target": ib.OTHER_SIDE,
              "target_country": ib.UNKNOWN_COUNTRY}
        assert ib.assign_section(ev) == ib.SECTION_FROM_IRAN


class TestStorylineCollapse:
    """One row per story, not per outlet (9 Sep 2026).

    202 theatre events in that window carried 53 storylines, and the two biggest —
    the same Houthi attack on southern Saudi Arabia — were 105 of them. The
    bulletin was the one report in SIM reading raw event rows: the country SITREP
    has always collapsed (138 Saudi events into 22 clusters).
    """

    def _ev(self, sid, title, domain, **kw):
        ev = {"storyline_id": sid, "title": title, "domain": domain,
              "url": f"https://{domain}/x", "severity": kw.pop("severity", 90),
              "corroborating_sources": kw.pop("corroborating_sources", [])}
        ev.update(kw)
        return ev

    def test_one_story_becomes_one_row(self):
        events = [self._ev("s1", "Houthis attack Saudi cities", "a.com"),
                  self._ev("s1", "Houthi strikes hit Saudi oil sites", "b.com"),
                  self._ev("s1", "Saudi Arabia vows response", "c.com")]
        out = ib.collapse_by_storyline(events)
        assert len(out) == 1
        assert out[0]["outlet_count"] == 3

    def test_the_outlets_it_stands_for_are_kept(self):
        events = [self._ev("s1", "Houthis attack Saudi cities", "a.com"),
                  self._ev("s1", "Houthi strikes hit Saudi oil sites", "b.com")]
        out = ib.collapse_by_storyline(events)
        assert [s["name"] for s in out[0]["sibling_sources"]] == ["b.com"]

    def test_the_most_corroborated_filing_represents_the_story(self):
        """Severity saturates at 100, so it cannot pick. Corroboration can."""
        events = [self._ev("s1", "Houthi strikes hit Saudi oil sites", "a.com",
                           severity=100),
                  self._ev("s1", "Saudi energy sites hit", "b.com", severity=100,
                           corroborating_sources=[{"domain": "reuters.com"}])]
        out = ib.collapse_by_storyline(events)
        assert out[0]["domain"] == "b.com"

    def test_the_toll_breaks_a_tie_but_does_not_lead(self):
        """A storyline is a THREAD, not an incident: its biggest toll can belong
        to another strand of it. On 9 Sep the 79-filing Houthi/Saudi story led
        with "Yemen clashes kill 300" — a real toll, from the other front."""
        events = [self._ev("s1", "Saudi vows retaliation, Yemen clashes kill 300",
                           "wion.com"),
                  self._ev("s1", "Houthi strikes in Saudi Arabia wound 73",
                           "straitstimes.com",
                           corroborating_sources=[{"domain": "reuters.com"}])]
        assert ib.collapse_by_storyline(events)[0]["domain"] == "straitstimes.com"

        tied = [self._ev("s1", "Saudi energy sites hit", "a.com"),
                self._ev("s1", "Houthi attacks kill 12 in Saudi Arabia", "b.com")]
        assert ib.collapse_by_storyline(tied)[0]["domain"] == "b.com"

    def test_separate_stories_stay_separate(self):
        events = [self._ev("s1", "Houthis attack Saudi cities", "a.com"),
                  self._ev("s2", "Iran fires missiles at Jordan base", "b.com")]
        assert len(ib.collapse_by_storyline(events)) == 2

    def test_an_unlinked_event_is_its_own_story(self):
        """A NULL storyline means the linker never placed it, not that it
        belongs with every other unplaced event."""
        events = [self._ev(None, "Tanker struck near Kharg", "a.com"),
                  self._ev(None, "Blast heard off Jask", "b.com")]
        assert len(ib.collapse_by_storyline(events)) == 2

    def test_order_follows_the_fetch(self):
        events = [self._ev("s1", "First story", "a.com"),
                  self._ev("s2", "Second story", "b.com"),
                  self._ev("s1", "First story, refiled", "c.com")]
        out = ib.collapse_by_storyline(events)
        assert [e["title"] for e in out] == ["First story", "Second story"]

    def test_the_input_rows_are_not_mutated(self):
        """The representative is a copy: extract_direction writes in place, and
        the collapsed row must not reach back into the fetched list."""
        events = [self._ev("s1", "Houthis attack Saudi cities", "a.com")]
        ib.collapse_by_storyline(events)[0]["actor"] = ib.IRAN_SIDE
        assert "actor" not in events[0]

    def test_the_bulletin_collapses_before_it_extracts(self, monkeypatch):
        """Direction is a property of the story, not of the outlet that filed
        it — and collapsing first is what takes the LLM call count down with it."""
        seen = {}
        monkeypatch.setattr(ib, "fetch_theatre_events", lambda *a: [
            self._ev("s1", "Houthis attack Saudi cities", "a.com"),
            self._ev("s1", "Houthi strikes hit Saudi oil sites", "b.com"),
            self._ev("s2", "Iran fires missiles at Jordan base", "c.com")])

        def _extract(router, events, db_conn=None):
            seen["n"] = len(events)
            for ev in events:
                ev.update(actor=ib.IRAN_SIDE, standing=ib.STANDING_CONFIRMED)
            return events

        monkeypatch.setattr(ib, "extract_direction", _extract)
        monkeypatch.setattr(ib, "call_llm",
                            lambda **k: {"content": "YÖNETİCİ ÖZETİ\nOldu."})
        from datetime import datetime, timezone
        out = ib.build_bulletin(None, None,
                                datetime(2026, 9, 8, tzinfo=timezone.utc),
                                datetime(2026, 9, 9, tzinfo=timezone.utc))
        assert seen["n"] == 2
        assert len(out["events"]) == 2
