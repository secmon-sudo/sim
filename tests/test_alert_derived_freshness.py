"""The publisher's own date stands in for freshness the classifier could not supply.

time_certainty is the CLASSIFIER's estimate of when the incident happened, and it answers
"unknown" for 80% of the corpus (4913 of 6164 scored events in the 7 days to 2026-08-13).
Both fresh paths — the ladder's require_location_or_fresh and SEVERITY_ALERT_FLOOR —
depend on it, so that single LLM field is the dominant alert gate: over those 7 days it
alone withheld 855 severity>=90 new_incident events across 641 distinct storylines, more
than the 836 events that reached any tier. It withheld them invisibly, too, since "never
qualified" records no veto reason.

The pipeline holds better evidence than the classifier's guess: a publication date the
PUBLISHER declared (date_verified, migration 021) timestamped inside the last day.

Measured cost of the relaxation over the same 7 days: 66 events / 56 storylines, ~8 pages
a day on ~100. Measured ceiling: 0 of the 66 also satisfied CRITICAL's confidence and
location requirements. The pages it recovers are the class the corpus was losing — the
Zawiya refinery and substation drone strikes (severity 100), "Ukraine Drone Hits Logistics
Hub", the Kurdish administration bombing — all of which reached the SITREP while never
paging.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.core.alerts import evaluate_alert_tier_verbose

NOW = datetime.now(timezone.utc)


def _event(**over):
    """A severity-100 incident with no location and no usable time_certainty — the exact
    shape the gate was dropping (Zawiya, run 20260813T090151)."""
    base = {
        "severity_score": 100,
        "system_confidence": 0.43,
        "anchor_confidence": "LOW",
        "time_certainty": "unknown",
        "date_verified": True,
        "published_at": NOW - timedelta(hours=3),
        "source_title": "Drone Attacks Strike Zawiya Oil Facilities",
        "report_kind": "new_incident",
        "anchor_name_norm": None,
        "latitude": None,
    }
    base.update(over)
    return base


class TestDerivedFreshnessRecoversPages:
    def test_verified_recent_publication_earns_a_tier(self):
        tier, veto = evaluate_alert_tier_verbose(_event())
        assert tier == "ALERT"
        assert veto is None

    def test_without_the_gate_the_same_event_pages_nothing(self):
        """The control: identical event, publication date just outside the window."""
        tier, _ = evaluate_alert_tier_verbose(_event(published_at=NOW - timedelta(days=3)))
        assert tier is None

    def test_naive_timestamps_are_read_as_utc(self):
        """The column is `timestamp without time zone`; psycopg hands back naive values."""
        naive = (NOW - timedelta(hours=2)).replace(tzinfo=None)
        assert evaluate_alert_tier_verbose(_event(published_at=naive))[0] == "ALERT"

    def test_iso_strings_are_accepted(self):
        iso = (NOW - timedelta(hours=2)).isoformat()
        assert evaluate_alert_tier_verbose(_event(published_at=iso))[0] == "ALERT"

    def test_day_precision_stamp_slightly_in_the_future_is_fresh(self):
        """Pass A stores day-precision dates as END of day (extract_date_from_url), so a
        story published this morning carries a stamp hours ahead of now."""
        assert evaluate_alert_tier_verbose(
            _event(published_at=NOW + timedelta(hours=8))
        )[0] == "ALERT"


class TestEvidenceIsRequired:
    def test_an_aggregator_crawl_stamp_is_not_evidence(self):
        """date_verified=False means the date is Google's crawl time. The whole point of
        migration 021 is that this cannot stand in for publication."""
        tier, _ = evaluate_alert_tier_verbose(_event(date_verified=False))
        assert tier is None

    def test_a_missing_publication_date_grants_nothing(self):
        assert evaluate_alert_tier_verbose(_event(published_at=None))[0] is None

    @pytest.mark.parametrize("bad", ["", "not-a-date", 1723526400, object()])
    def test_unreadable_dates_fail_closed(self, bad):
        """This path RAISES a tier, so anything unparseable must answer 'no'."""
        assert evaluate_alert_tier_verbose(_event(published_at=bad))[0] is None

    def test_a_known_time_certainty_is_never_overridden(self):
        """The gate fills a gap; it does not second-guess the classifier. 'this_week' is
        a real answer and stays outside FRESH_TIME_CERTAINTY."""
        tier, _ = evaluate_alert_tier_verbose(_event(time_certainty="this_week"))
        assert tier is None


class TestTheArticleShapeGatesStillRule:
    """What makes relaxing freshness safe: report_kind and the title patterns judge
    whether the ARTICLE reports something happening now — the job time_certainty was
    doing by proxy. A recovered tier must still pass both."""

    def test_followup_articles_are_still_vetoed(self):
        tier, veto = evaluate_alert_tier_verbose(_event(report_kind="followup"))
        assert tier is None
        assert veto == "report_kind_followup"

    def test_aftermath_headlines_are_still_vetoed(self):
        tier, veto = evaluate_alert_tier_verbose(
            _event(source_title="Autopsy conducted on Rawalpindi shooting suspect")
        )
        assert tier is None
        assert veto == "aftermath_title"

    def test_a_low_severity_event_is_not_promoted(self):
        """Freshness was never the only bar — the ladder's severity and confidence floors
        are untouched, so this cannot manufacture pages out of minor news."""
        assert evaluate_alert_tier_verbose(_event(severity_score=35))[0] is None


class TestAttribution:
    def test_the_grant_is_marked_for_telemetry(self):
        event = _event()
        evaluate_alert_tier_verbose(event)
        assert event["_derived_fresh_granted"] is True

    def test_no_credit_when_the_event_would_have_paged_anyway(self):
        """Same counterfactual discipline as the date_unverified veto: credited only when
        it CHANGED the outcome. This event is located and well-sourced, so it clears the
        ladder without any freshness help."""
        event = _event(system_confidence=0.66, latitude=32.7, anchor_name_norm="ZAWIYA")
        tier, _ = evaluate_alert_tier_verbose(event)
        assert tier is not None
        assert "_derived_fresh_granted" not in event

    def test_no_credit_when_the_gate_did_not_fire(self):
        event = _event(time_certainty="same_day")
        evaluate_alert_tier_verbose(event)
        assert "_derived_fresh_granted" not in event
