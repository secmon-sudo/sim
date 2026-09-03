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
    def _ev(self, country, actor, standing=ib.STANDING_CONFIRMED):
        return {"country_iso": country, "actor": actor, "standing": standing}

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
        assert out[0] == {"actor": ib.IRAN_SIDE, "standing": ib.STANDING_CLAIMED}
        assert out[1] == {"actor": ib.US_SIDE, "standing": ib.STANDING_CONFIRMED}

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
        assert out[0] == {"actor": ib.UNATTRIBUTED, "standing": ib.STANDING_UNKNOWN}

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
