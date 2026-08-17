"""The replay must ask the tier question the pipeline asks.

The script exists to decide alert-volume changes, so a gate it forgets to feed is not a
rounding error — it is a wrong answer. Two were missing until 2026-08-17: report_kind
(so every replayed event looked like new_incident to the article-shape veto) and
published_at (so the publisher-date freshness path, which grants ~8 pages a day where
time_certainty is 'unknown', never fired). Both silently biased the delta.

The --alert-floor override is tested here too, because it patches a module global and a
leak would corrupt every measurement taken after it in the same process.
"""

from src.core import alerts
from scripts.replay_severity import tier_for


def _ev(**kw):
    """A fresh, located, severity-95 event that pages on the ladder alone."""
    base = {
        "system_confidence": 0.55,
        "anchor_confidence": "MEDIUM",
        "time_certainty": "same_day",
        "date_verified": True,
        "event_type": "missile_strike",
        "anchor_name_norm": "KBP",
        "latitude": 50.34,
        "source_title": "Missile strike on Kyiv district kills seven",
        "corroboration": 0,
        "report_kind": "new_incident",
        "published_at": None,
        "llm_parsed": {},
    }
    base.update(kw)
    return base


class TestGatesAreFed:
    def test_report_kind_veto_reaches_the_replay(self):
        assert tier_for(_ev(), 95) is not None
        assert tier_for(_ev(report_kind="roundup"), 95) is None

    def test_publisher_date_freshness_reaches_the_replay(self):
        """time_certainty 'unknown' is 80% of the corpus; the publisher date is what
        rescues it in production, so the replay has to see it."""
        from datetime import datetime, timedelta, timezone
        stale = _ev(time_certainty="unknown", system_confidence=0.41)
        assert tier_for(stale, 95) is None
        fresh = dict(stale,
                     published_at=datetime.now(timezone.utc) - timedelta(hours=3))
        assert tier_for(fresh, 95) is not None


class TestAlertFloorOverride:
    def test_lower_floor_admits_a_compressed_score(self):
        """The case the flag exists for: confidence below the ladder's ALERT bar of
        0.50, severity compressed to just under the floor. Such an event is not
        silenced — it lands on WATCH — so what the floor decides is the BADGE."""
        ev = _ev(system_confidence=0.41, anchor_confidence="LOW",
                 anchor_name_norm=None, latitude=None)
        assert tier_for(ev, 86) == "WATCH"
        assert tier_for(ev, 86, alert_floor=86) == "ALERT"

    def test_override_does_not_leak(self):
        original = alerts.SEVERITY_ALERT_FLOOR
        tier_for(_ev(), 86, alert_floor=70)
        assert alerts.SEVERITY_ALERT_FLOOR == original

    def test_override_restored_after_an_exception(self):
        original = alerts.SEVERITY_ALERT_FLOOR
        try:
            tier_for(_ev(system_confidence=None), 86, alert_floor=70)
        except Exception:
            pass
        assert alerts.SEVERITY_ALERT_FLOOR == original

    def test_none_means_configured_floor(self):
        ev = _ev(system_confidence=0.41, anchor_confidence="LOW",
                 anchor_name_norm=None, latitude=None)
        assert tier_for(ev, alerts.SEVERITY_ALERT_FLOOR) == "ALERT"
        assert tier_for(ev, alerts.SEVERITY_ALERT_FLOOR - 1) == "WATCH"
