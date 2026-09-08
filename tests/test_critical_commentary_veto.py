"""CRITICAL's exemption from the article-shape gates, narrowed to exclude commentary.

The exemption is written for a specific case, in its own words: "a roundup is
sometimes the only carrier of a genuinely major development, and missing that costs
more than the noise it lets through."

Measured 2026-09-08 over seven days: 74 of 427 CRITICAL-eligible events carried a
not-news report_kind — 52 followup, 19 commentary, 3 roundup. A sample splits about
evenly between the case the rule was written for and diplomatic reaction to an
incident already reported. commentary is where that split is worst AND where the
exemption's own argument does not apply: a commentary piece is by definition not
the carrier of a development, it is a reaction to one.

So commentary loses the exemption; followup and roundup keep it. Narrow on purpose —
19 events a week, about 4 cards once suppression has run.
"""

import pytest

from src.core.alerts import (
    CRITICAL_VETOED_REPORT_KINDS,
    REPORT_KIND_NOT_NEWS,
    evaluate_alert_tier_verbose,
)


def _critical(report_kind="new_incident", source_title="Missile strike on Kyiv kills 12"):
    """Signals that clear CRITICAL on the ladder: located, fresh, severe, confident."""
    return {
        "severity_score": 95,
        "system_confidence": 0.75,
        "anchor_confidence": "HIGH",
        "time_certainty": "same_day",
        "event_type": "missile_strike",
        "anchor_name_norm": "KBP",
        "latitude": 50.34,
        "source_title": source_title,
        "report_kind": report_kind,
    }


class TestCommentaryLosesTheExemption:
    def test_a_commentary_piece_does_not_page_at_critical(self):
        tier, veto = evaluate_alert_tier_verbose(_critical(report_kind="commentary"))
        assert tier is None
        assert veto == "report_kind_commentary"

    def test_the_same_piece_was_already_vetoed_below_critical(self):
        # Nothing changes for ALERT and WATCH; this is the tier that was exempt.
        ev = _critical(report_kind="commentary")
        ev["severity_score"] = 70
        ev["system_confidence"] = 0.55
        tier, veto = evaluate_alert_tier_verbose(ev)
        assert tier is None
        assert veto == "report_kind_commentary"


class TestFollowupAndRoundupKeepIt:
    @pytest.mark.parametrize("kind", ["followup", "roundup"])
    def test_they_still_page_at_critical(self, kind):
        """A followup is where a toll update or a state attribution arrives —
        "Germany says Russia was behind last month's attempted drone attack",
        "8 people killed in Kyiv as a result of Russian attack" — and the roundup
        case is the one the exemption was written for."""
        tier, veto = evaluate_alert_tier_verbose(_critical(report_kind=kind))
        assert tier == "CRITICAL"
        assert veto is None

    @pytest.mark.parametrize("kind", ["followup", "roundup"])
    def test_and_are_still_vetoed_below_critical(self, kind):
        ev = _critical(report_kind=kind)
        ev["severity_score"] = 70
        ev["system_confidence"] = 0.55
        tier, veto = evaluate_alert_tier_verbose(ev)
        assert tier is None
        assert veto == f"report_kind_{kind}"


class TestTheHeadlineGateIsUntouched:
    def test_critical_is_still_fully_exempt_from_the_desk_label_test(self):
        """The title gate reads a desk label rather than the article, and a desk
        label on a genuinely major development is the case the exemption exists
        for. Only the report_kind half was narrowed."""
        tier, _ = evaluate_alert_tier_verbose(
            _critical(source_title="Ukraine war latest: Russia makes slow gains"))
        assert tier == "CRITICAL"

    def test_the_title_gate_still_wins_the_attribution_below_critical(self):
        ev = _critical(report_kind="followup",
                       source_title="Ukraine war latest: Russia makes slow gains")
        ev["severity_score"] = 70
        ev["system_confidence"] = 0.55
        _, veto = evaluate_alert_tier_verbose(ev)
        assert veto == "aftermath_title"


class TestTheNarrowingIsNarrow:
    def test_only_commentary_is_taken_from_the_exemption(self):
        assert CRITICAL_VETOED_REPORT_KINDS == frozenset({"commentary"})
        assert CRITICAL_VETOED_REPORT_KINDS < REPORT_KIND_NOT_NEWS

    def test_an_unknown_report_kind_still_fails_open(self):
        """`new_incident` is what an absent or unparseable value resolves to, so a
        degraded classification lets alerts through rather than silencing them."""
        for kind in ("new_incident", None, "garbage"):
            tier, _ = evaluate_alert_tier_verbose(_critical(report_kind=kind))
            assert tier == "CRITICAL", kind
