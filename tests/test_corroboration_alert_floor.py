"""Independent corroboration as a path to ALERT.

Why it exists: the ladder's confidence floor is 0.50 and system_confidence has a
median of 0.45, so nearly every real incident reaches a tier through
SEVERITY_ALERT_FLOOR instead. Once the severity catalog stops saturating, that crutch
goes and the decision falls back on confidence — which cannot separate a real incident
from a roundup (measured 2026-08-11). Corroboration can.

Measured on the 6 days to 2026-08-17, a compressed severity catalog silences 25 tiered
events. Of the three carrying >= 2 independent domains, two are among the most
significant in the set (the mass drone attack on Moscow, the Benghazi car bombing that
killed Libya's military intelligence chief), and every piece of junk in the silenced
set carries zero.
"""

from src.core.alerts import (
    CORROBORATION_ALERT_MIN,
    TIER_RULES,
    corroboration_count,
    evaluate_alert_tier,
)

# Below the ladder's confidence floor and below SEVERITY_ALERT_FLOOR, so the only way
# out is the corroboration path. These are the real numbers off the silenced list.
_BASE = {
    "severity_score": 70,
    "system_confidence": 0.39,
    "anchor_confidence": "LOW",
    "time_certainty": "same_day",
    "event_type": "missile_strike",
    "latitude": 50.45,
    "source_title": "Ukraine targets Moscow in mass drone attack",
}


def _event(n_sources: int, **over):
    return {**_BASE, "corroborating_sources": [{"domain": f"pub{i}.com"}
                                               for i in range(n_sources)], **over}


class TestCorroborationCount:
    def test_counts_list(self):
        assert corroboration_count({"corroborating_sources": [{"domain": "a"}]}) == 1

    def test_parses_json_string(self):
        assert corroboration_count(
            {"corroborating_sources": '[{"domain": "a"}, {"domain": "b"}]'}) == 2

    def test_missing_is_zero(self):
        assert corroboration_count({}) == 0

    def test_malformed_is_zero(self):
        assert corroboration_count({"corroborating_sources": "not json"}) == 0
        assert corroboration_count({"corroborating_sources": 7}) == 0


class TestFloorRaises:
    def test_uncorroborated_does_not_page(self):
        assert evaluate_alert_tier(_event(0)) != "ALERT"

    def test_single_source_does_not_page(self):
        """One corroborating domain is a republish, not confirmation."""
        assert evaluate_alert_tier(_event(1)) != "ALERT"

    def test_two_independent_domains_page(self):
        assert evaluate_alert_tier(_event(CORROBORATION_ALERT_MIN)) == "ALERT"


class TestFloorIsNotANewWayIn:
    """The floor raises events that were already close. It admits nothing else."""

    def test_stale_event_stays_out(self):
        """Freshness is still required — corroboration says who, not when."""
        assert evaluate_alert_tier(_event(5, time_certainty="unknown")) != "ALERT"

    def test_below_watch_severity_stays_out(self):
        low = TIER_RULES["WATCH"]["severity_min"] - 5
        assert evaluate_alert_tier(_event(5, severity_score=low)) != "ALERT"

    def test_cannot_manufacture_critical(self):
        """Same ceiling as SEVERITY_ALERT_FLOOR: CRITICAL is earned, never granted."""
        assert evaluate_alert_tier(_event(9, severity_score=100)) != "CRITICAL"

    def test_does_not_lower_an_earned_tier(self):
        earned = _event(0, system_confidence=0.9, anchor_confidence="HIGH",
                        severity_score=100)
        before = evaluate_alert_tier(earned)
        after = evaluate_alert_tier({**earned,
                                     "corroborating_sources": [{"domain": "a"}] * 4})
        assert after == before
