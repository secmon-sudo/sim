"""Accidents and natural disasters must not page.

Measured over the 14 days to 2026-08-17: 85 of the 187 ALERT-tier mass_casualty_event
rows (45%) were accidents or natural disasters. Severity reads casualty counts, so the
Hungary coach crash scored 100 exactly like a suicide bombing does, and one crash
produced ~25 separate tiered rows. The Colombia earthquake produced ~30.

The gate is positive evidence only — cause-agnostic event_type + an accidental-cause
headline — because the classifier has no accident type to offer and because using a
hostile-act recall list as a veto is the mistake removed from Pass C on 2026-08-10.
"""

from src.core.alerts import accidental_cause_kind, evaluate_alert_tier_verbose


def _event(title, event_type="mass_casualty_event", **kw):
    """A fresh, well-located, high-severity event — one that pages without a gate."""
    base = {
        "severity_score": 100,
        "system_confidence": 0.45,
        "anchor_confidence": "LOW",
        "time_certainty": "same_day",
        "event_type": event_type,
        "latitude": 47.5,
        "source_title": title,
        "report_kind": "new_incident",
    }
    base.update(kw)
    return base


class TestAccidentalCauseDetection:
    def test_natural_disaster(self):
        assert accidental_cause_kind(
            "At least 111 killed in Colombia earthquake as national disaster declared",
            "mass_casualty_event") == "natural"

    def test_flood(self):
        assert accidental_cause_kind(
            "Indiana floods kill 5 people: where are warnings still in place?",
            "mass_casualty_event") == "natural"

    def test_stampede_named_separately(self):
        """A crowd crush is a venue-safety failure, not weather — worth its own name."""
        assert accidental_cause_kind("India temple stampede in Bihar kills 7",
                                     "mass_casualty_event") == "crowd_crush"

    def test_transport_accident_needs_both_terms(self):
        assert accidental_cause_kind(
            "Bus carrying Polish pilgrims crashes in Hungary, leaving 12 dead",
            "mass_casualty_event") == "transport_accident"
        # A vehicle noun alone is not an accident.
        assert accidental_cause_kind("Gunmen open fire on bus in Balochistan, 12 dead",
                                     "mass_casualty_event") is None

    def test_transport_accident_word_order_is_free(self):
        """Phrasing varies far too much for an adjacency rule."""
        for title in (
            "A Polish coach overturned on the M3 motorway in Hungary, causing 12 deaths",
            '"Driver fell asleep": 12 killed, 10 injured as bus ferrying 59 people crashes',
            "Truck crash in Egypt kills 18, many of them child labourers",
            "At least 44 dead after overcrowded ferry capsizes on Lake Kariba in Zimbabwe",
        ):
            assert accidental_cause_kind(title, "mass_casualty_event") == \
                "transport_accident", title

    def test_structural_and_industrial(self):
        assert accidental_cause_kind("India news: 7 dead, 3 trapped in Uttarakhand "
                                     "tunnel collapse", "mass_casualty_event") == \
            "structural_collapse"
        assert accidental_cause_kind("Gas leak at Bangladesh shipbreaking yard kills 8",
                                     "mass_casualty_event") == "industrial"


class TestExemptions:
    def test_hostile_type_is_never_gated(self):
        """The measured protection: a strike during a flood keeps its own event_type.

        "Myanmar Junta Airstrike Kills Two Displaced Brothers in Flood-Hit Ayeyarwady
        Region" matches the natural-cause pattern on "flood-hit" and is a real attack.
        """
        title = ("Myanmar Junta Airstrike Kills Two Displaced Brothers in Flood-Hit "
                 "Ayeyarwady Region")
        assert accidental_cause_kind(title, "missile_strike") is None
        # ...and even if it were filed under the cause-agnostic bucket, the hostile
        # context in the headline exempts it.
        assert accidental_cause_kind(title, "mass_casualty_event") is None

    def test_aviation_accident_still_pages(self):
        """Aviation is the priority domain — its accidents are the subject matter."""
        assert accidental_cause_kind(
            "13 killed as tourist plane bound for Nazca Lines crashes in Peru",
            "mass_casualty_event") is None
        assert accidental_cause_kind(
            "2 killed as US military helicopter crashes in Texas",
            "mass_casualty_event") is None

    def test_disaster_disrupting_an_airport_still_pages(self):
        assert accidental_cause_kind(
            "8 dead, 7,000 stranded at Narita airport as torrential rain batters Japan",
            "mass_casualty_event") is None


class TestTierIntegration:
    def test_accident_withholds_the_page(self):
        tier, veto = evaluate_alert_tier_verbose(
            _event("Five teenagers killed and four other people injured in "
                   "Ireland car crash"))
        assert tier is None
        assert veto == "accidental_transport_accident"

    def test_critical_is_not_exempt(self):
        """Unlike the article-shape gates: an accident is out of scope at any tier."""
        tier, veto = evaluate_alert_tier_verbose(
            _event("At least 132 killed, 570 injured in magnitude 7.4 earthquake",
                   system_confidence=0.7))
        assert tier is None
        assert veto == "accidental_natural"

    def test_attack_with_the_same_shape_still_pages(self):
        tier, veto = evaluate_alert_tier_verbose(
            _event("Suicide bomber strikes Pakistan peace rally, leaving 14 dead"))
        assert tier is not None
        assert veto is None

    def test_aviation_accident_reaches_a_tier(self):
        tier, _ = evaluate_alert_tier_verbose(
            _event("Tour flight crash near the Nazca Lines kills 13"))
        assert tier is not None
