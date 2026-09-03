"""Theatre countries are narrated first, 3 Sep 2026.

Measured that morning: five countries at 07:30 saturated Mistral, which returned
429 on the fifth. LLM7 then answered 502, and Iraq — a country at the centre of an
active war — was narrated by Pollinations, whose hidden reasoning consumes roughly
42% of whatever max_tokens it is handed. The last country in the order pays for
every country ahead of it, so the order is not cosmetic.

What this must NOT do is change coverage. The list reorders; it never admits a
country that failed selection and never drops one that passed.
"""

from src.services.sitrep_generator import PRIORITY_COUNTRIES, prioritise


class TestPrioritise:
    def test_a_theatre_country_moves_to_the_front(self):
        assert prioritise(["UA", "DE", "US", "IQ", "IR"])[:2] == ["IQ", "IR"]

    def test_nothing_is_added(self):
        """A priority country that did not qualify stays absent."""
        out = prioritise(["UA", "DE"])
        assert out == ["UA", "DE"]
        assert "IR" not in out

    def test_nothing_is_dropped(self):
        selected = ["UA", "DE", "US", "IQ", "IR"]
        assert sorted(prioritise(selected)) == sorted(selected)

    def test_the_query_ranking_survives_inside_each_group(self):
        """Selection already ranks by protected tier then volume, and that ranking
        is meaningful — reordering by the priority list would discard it."""
        assert prioritise(["UA", "IR", "DE", "IQ", "US"]) == \
            ["IR", "IQ", "UA", "DE", "US"]

    def test_an_all_theatre_run_is_left_alone(self):
        assert prioritise(["IR", "IQ", "JO"]) == ["IR", "IQ", "JO"]

    def test_an_empty_selection_is_not_a_special_case(self):
        assert prioritise([]) == []

    def test_an_empty_priority_list_is_a_no_op(self, monkeypatch):
        """Emptying the config must restore the previous behaviour exactly — this
        is how the list is retired when the war ends."""
        import src.services.sitrep_generator as sg
        monkeypatch.setattr(sg, "PRIORITY_COUNTRIES", [])
        assert sg.prioritise(["UA", "DE", "IR"]) == ["UA", "DE", "IR"]


class TestConfig:
    def test_the_theatre_is_configured(self):
        assert {"IR", "IQ", "KW", "JO"} <= set(PRIORITY_COUNTRIES)

    def test_entries_are_upper_case_iso2(self):
        for iso in PRIORITY_COUNTRIES:
            assert len(iso) == 2 and iso.isupper(), iso

    def test_it_matches_the_bulletin_theatre(self):
        """The two lists answer the same question — which countries this war is
        being fought in — so a country added to one belongs in the other."""
        from src.services.iran_bulletin import THEATRE_ISO
        assert set(PRIORITY_COUNTRIES) == set(THEATRE_ISO)
