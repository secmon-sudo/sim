"""airspace_closure — the aviation event the taxonomy had no code for.

Measured over 45 days before it existed, every one of these was ingested and then
dropped below every alert gate because it could only land in unclassified (base 20)
or other_aviation_related (base 20):

    "Stray Drones Trigger Finnish Airspace Closure on Russian Border"  sev 35, no alert
    "Kuwait Airport Congested After Airspace Closure"                  sev 65, no alert
    "Pakistan Airspace Closure Costs Air India $2.3 Billion"           sev 35, no alert

The first is exactly what the product exists to watch. The hard part is not detecting
closures — it is that a flypast closure and a drone-incursion closure are written in
the same words, so the classifier is asked for sub_type and the planned ones are
capped here.
"""

from src.pipeline.pass_d_score import (
    PLANNED_CLOSURE_SEVERITY_CAP,
    apply_planned_closure_downrank,
)
from src.pipeline.pass_c_classify import CLASSIFICATION_SYSTEM_PROMPT as PROMPT


class TestClassifierKnowsTheType:
    def test_type_is_offered(self):
        assert "airspace_closure" in PROMPT

    def test_planned_and_incident_are_both_named(self):
        """Without both labels the model has no vocabulary for the distinction."""
        assert '"planned"' in PROMPT
        assert '"incident"' in PROMPT

    def test_unexplained_closure_defaults_to_incident(self):
        """An unexplained closure is the case worth surfacing, so the default is
        the serious one — stated in the prompt, asserted here so a prompt edit
        cannot quietly flip it."""
        idx = PROMPT.index("USE airspace_closure")
        guidance = PROMPT[idx:idx + 1200]
        assert "does not say" in guidance
        assert '"incident"' in guidance


class TestPlannedClosureIsCapped:
    def test_planned_is_capped(self):
        assert apply_planned_closure_downrank(
            "airspace_closure", 70, {"sub_type": "planned"}) == PLANNED_CLOSURE_SEVERITY_CAP

    def test_cap_sits_below_every_alert_gate(self):
        from src.core.alerts import TIER_RULES
        assert PLANNED_CLOSURE_SEVERITY_CAP < min(
            r["severity_min"] for r in TIER_RULES.values())

    def test_incident_is_untouched(self):
        assert apply_planned_closure_downrank(
            "airspace_closure", 70, {"sub_type": "incident"}) == 70

    def test_missing_sub_type_is_untouched(self):
        """Absent reads as serious, matching the prompt's default."""
        assert apply_planned_closure_downrank("airspace_closure", 70, {}) == 70
        assert apply_planned_closure_downrank("airspace_closure", 70, None) == 70

    def test_case_and_whitespace_tolerated(self):
        for value in ("Planned", " PLANNED ", "planned"):
            assert apply_planned_closure_downrank(
                "airspace_closure", 70, {"sub_type": value}
            ) == PLANNED_CLOSURE_SEVERITY_CAP

    def test_only_applies_to_this_type(self):
        """A planned sub_type on any other type must not lower it — the ambiguity
        this cap exists for is specific to closure language."""
        assert apply_planned_closure_downrank(
            "missile_strike", 75, {"sub_type": "planned"}) == 75


class TestIngestCollectsThem:
    def test_queries_target_closures(self):
        from src.pipeline.ingest_queries import build_search_queries
        queries = " ".join(q["query"].lower() for q in build_search_queries(None))
        assert "airspace" in queries
        assert "notam" in queries

    def test_bare_closure_phrase_is_qualified(self):
        """Unqualified "airspace closed" is dominated by air shows and parades, so
        the ingest queries pair it with a cause rather than spending budget on them."""
        from src.pipeline.ingest_queries import build_search_queries
        bare = [q["query"] for q in build_search_queries(None)
                if q["query"].strip().lower() == '"airspace closed"']
        assert not bare
