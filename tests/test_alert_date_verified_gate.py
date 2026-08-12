"""Tests for the date-provenance alert gate (migration 021).

Google News stamps a re-crawled archive page with the date IT saw the page, so an item
whose publisher declares no date enters the pipeline dated "now" and every downstream
freshness signal inherits that. Measured 2026-08-12: a 2026-03-25 Nepalnews explainer
about the September 2025 Kathmandu uprising was stored as published that day at 01:33,
classified time_certainty=same_day, and reached the daily report as a same-day event.

Heuristic extraction cannot repair it — htmldate and trafilatura both read that page as
2026-08-12, having picked up the sidebar's "August 12, 2026" links. So the pipeline keeps
the date and records where it came from: an unverified date may still be ingested,
classified and reported, but it may not be read as evidence that something is happening
now. What it must NOT do is silence events that never depended on freshness.
"""

from src.core.alerts import evaluate_alert_tier, evaluate_alert_tier_verbose


def _event(**over):
    """A fresh, well-located event that clears the ALERT ladder on its own merits."""
    ev = {
        "severity_score": 95,
        "system_confidence": 0.55,
        "anchor_confidence": "LOW",
        "time_certainty": "same_day",
        "event_type": "missile_strike",
        "anchor_name_norm": "KBP",
        "source_title": "Missile strike hits Kyiv district, three killed",
        "report_kind": "new_incident",
        "date_verified": True,
    }
    ev.update(over)
    return ev


class TestUnverifiedDatesCannotClaimFreshness:
    def test_unlocated_event_loses_its_page_without_a_verified_date(self):
        # Freshness was the only gate this event passed: no anchor, no coordinates.
        ev = _event(anchor_name_norm=None, latitude=None, date_verified=False)
        tier, veto = evaluate_alert_tier_verbose(ev)
        assert tier is None
        assert veto == "date_unverified"

    def test_same_event_pages_when_the_publisher_declared_the_date(self):
        ev = _event(anchor_name_norm=None, latitude=None, date_verified=True)
        assert evaluate_alert_tier(ev) == "ALERT"

    def test_located_event_is_untouched(self):
        # Location, not freshness, is carrying this one — the downgrade must not be
        # read as a blanket veto on unverified sources.
        ev = _event(date_verified=False)
        tier, veto = evaluate_alert_tier_verbose(ev)
        assert tier == "ALERT"
        assert veto is None

    def test_watch_tier_requires_a_verified_fresh_date(self):
        # WATCH keys on time_certainty alone (time_certainty_include), so an
        # unverified stamp takes the whole tier with it.
        ev = _event(severity_score=50, system_confidence=0.42,
                    anchor_name_norm=None, latitude=None, date_verified=False)
        assert evaluate_alert_tier(ev) is None


class TestTheGateStaysNarrow:
    def test_missing_field_reads_as_verified(self):
        # Every caller and fixture that predates the column must keep its behaviour;
        # a gate that failed closed here would silence alerts on an unrelated query
        # someone forgot to update.
        ev = _event(anchor_name_norm=None, latitude=None)
        ev.pop("date_verified")
        assert evaluate_alert_tier(ev) == "ALERT"

    def test_event_that_would_never_have_paged_is_not_attributed_to_this_gate(self):
        # First production run (2026-08-12): a severity-35 curfew-relaxation story was
        # counted as a date_unverified veto, though WATCH's severity bar alone would have
        # withheld it. A veto reason that fires on events the gate did not change makes
        # the telemetry useless for tuning it.
        ev = _event(severity_score=35, system_confidence=0.43,
                    anchor_name_norm=None, latitude=None, date_verified=False)
        tier, veto = evaluate_alert_tier_verbose(ev)
        assert tier is None
        assert veto is None

    def test_already_unknown_time_is_not_attributed_to_this_gate(self):
        # Nothing was downgraded: the event never claimed freshness, so whatever
        # withheld its page, it was not date provenance.
        ev = _event(time_certainty="unknown", anchor_name_norm=None, latitude=None,
                    date_verified=False)
        tier, veto = evaluate_alert_tier_verbose(ev)
        assert tier is None
        assert veto is None

    def test_travel_advisory_path_is_unaffected(self):
        # Advisories deliberately bypass the time gates — their "time" is the standing
        # advisory date, and severity alone decides. See evaluate_alert_tier_verbose.
        ev = _event(event_type="travel_advisory", severity_score=95,
                    date_verified=False)
        assert evaluate_alert_tier(ev) == "ALERT"


class TestTheReportSaysSo:
    """A withheld page is not enough — the event still appears in the daily report."""

    def test_sitrep_bullet_labels_an_unverified_date(self):
        from src.services.sitrep_generator import _event_date_label
        label = _event_date_label({
            "occurred_at_est": "2026-08-12 01:33:36",
            "time_certainty": "same_day",
            "date_verified": False,
        })
        assert "doğrulanmadı" in label
        # The classifier's "same_day" was itself derived from this timestamp, so the
        # label must not repeat it as if it were independent evidence.
        assert "same_day" not in label

    def test_verified_date_keeps_the_plain_label(self):
        from src.services.sitrep_generator import _event_date_label
        assert _event_date_label({
            "occurred_at_est": "2026-08-12 01:33:36",
            "time_certainty": "same_day",
            "date_verified": True,
        }) == "2026-08-12"

    def test_telegram_card_does_not_assert_an_unverified_stamp(self, monkeypatch):
        # A CRITICAL event still pages on location+severity with an unverified date, so
        # the card has to say which part of it is not evidence.
        from src.services import telegram_notifier as tn
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_ALERTS_CHAT_ID", "c")
        sent = {}

        class _Resp:
            status_code = 200
            def json(self):
                return {"ok": True, "result": {"message_id": 1}}

        monkeypatch.setattr(tn, "_post_telegram",
                            lambda url, payload: (sent.update(payload), _Resp())[1])
        tn.send_telegram_alert({
            "source_title": "Explosion reported at port",
            "event_type": "explosion",
            "severity_score": 80,
            "system_confidence": 0.5,
            "alert_tier": "ALERT",
            "occurred_at_est": "2026-08-12 01:33:36",
            "date_verified": False,
        })
        assert "date unverified" in sent["text"]
