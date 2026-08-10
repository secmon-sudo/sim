"""Tests for the storyline-independent geo suppression safety net.

Covers build_geo_suppression_key (pure) and dispatch_alert's dual-key behaviour that
mutes duplicate alerts even when the storyline_id fragments across paraphrased sources.
"""

from unittest.mock import MagicMock

import src.pipeline.pass_d_score as pd
from src.core.alerts import (
    build_geo_suppression_key,
    build_suppression_key,
    suppression_blocks,
    tier_rank,
)


def _db_with_claim(tier: str | None):
    """A db_conn stub whose alert_suppression lookup returns `tier` (None = no claim)."""
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = (tier,) if tier else None
    return db


class TestSuppressionBlocks:
    def test_free_key_never_blocks(self):
        assert suppression_blocks(_db_with_claim(None), "K", "WATCH") is False

    def test_same_tier_blocks(self):
        """The 10 Aug 2026 double-page: both members were ALERT."""
        assert suppression_blocks(_db_with_claim("ALERT"), "K", "ALERT") is True

    def test_lower_tier_blocks(self):
        assert suppression_blocks(_db_with_claim("CRITICAL"), "K", "WATCH") is True

    def test_escalation_does_not_block(self):
        assert suppression_blocks(_db_with_claim("WATCH"), "K", "CRITICAL") is False


class TestGeoSuppressionKey:
    def test_prefers_iata_anchor(self):
        ev = {"anchor_name_norm": "KBP", "country_iso": "UA", "severity_score": 72}
        assert build_geo_suppression_key(ev) == "geofp|UA|KBP"

    def test_falls_back_to_geo_key(self):
        # No IATA anchor -> coarse geo_key from raw text; "Kiev" collapses to KYIV.
        ev = {"anchor_name_raw": "Kiev", "country_iso": "UA", "severity_score": 88}
        assert build_geo_suppression_key(ev) == "geofp|UA|KYIV"

    def test_paraphrased_location_shares_key(self):
        a = {"anchor_name_raw": "Kyiv", "country_iso": "UA", "severity_score": 90}
        b = {"anchor_name_raw": "Ukrainian capital", "country_iso": "UA", "severity_score": 91}
        assert build_geo_suppression_key(a) == build_geo_suppression_key(b)

    def test_severity_does_not_fragment_key(self):
        """Two reports of one incident that scored a bucket apart share the key."""
        a = {"anchor_name_raw": "Kyiv", "country_iso": "UA", "severity_score": 100}
        b = {"anchor_name_raw": "Kyiv", "country_iso": "UA", "severity_score": 90}
        assert build_geo_suppression_key(a) == build_geo_suppression_key(b)

    def test_none_when_no_location(self):
        assert build_geo_suppression_key({"country_iso": "UA", "severity_score": 90}) is None

    def test_none_when_unknown_anchor(self):
        ev = {"anchor_name_norm": "UNKNOWN", "country_iso": "UA", "severity_score": 90}
        assert build_geo_suppression_key(ev) is None

    def test_differs_from_primary_key(self):
        ev = {"storyline_id": "S1", "anchor_name_norm": "KBP",
              "country_iso": "UA", "severity_score": 72}
        assert build_geo_suppression_key(ev) != build_suppression_key(ev)


class TestPrimarySuppressionKey:
    def test_severity_does_not_fragment_key(self):
        """The 10 Aug 2026 double-page: one storyline, two members a bucket apart."""
        a = {"storyline_id": "S1", "anchor_name_norm": "KBP", "severity_score": 100}
        b = {"storyline_id": "S1", "anchor_name_norm": "KBP", "severity_score": 90}
        assert build_suppression_key(a) == build_suppression_key(b)

    def test_distinct_storylines_keep_distinct_keys(self):
        a = {"storyline_id": "S1", "anchor_name_norm": "KBP", "severity_score": 90}
        b = {"storyline_id": "S2", "anchor_name_norm": "KBP", "severity_score": 90}
        assert build_suppression_key(a) != build_suppression_key(b)

    def test_unclustered_events_never_share_a_key(self):
        """Without a storyline there is no evidence two events are the same incident,
        so the fallback must not collapse them into one shared slot."""
        a = {"id": "evt-a", "severity_score": 90}
        b = {"id": "evt-b", "severity_score": 90}
        assert build_suppression_key(a) != build_suppression_key(b)


def _fired_at(db, key, tier, recorded_tier):
    """Stand-in for suppression_blocks against a key that last fired at recorded_tier."""
    return tier_rank(tier) <= tier_rank(recorded_tier)


def _base_event(**over):
    ev = {
        "severity_score": 90,
        "alert_tier": "CRITICAL",
        "storyline_id": "S1",
        "anchor_name_raw": "Kyiv",
        "country_iso": "UA",
    }
    ev.update(over)
    return ev


class TestDispatchDualKey:
    def test_geo_net_suppresses_when_storyline_differs(self, monkeypatch):
        """A sibling event whose storyline fragmented (different storyline_id, so a
        different primary key) is still muted because the geo fingerprint already fired."""
        geo_key_str = build_geo_suppression_key(_base_event())
        # Only the geo key is 'already suppressed'; the primary key is not.
        monkeypatch.setattr(pd, "suppression_blocks",
                            lambda db, k, tier: k == geo_key_str)
        sent = MagicMock()
        monkeypatch.setattr(pd, "send_telegram_alert", sent)

        # Different storyline_id -> different primary key, but same geo fingerprint.
        ev = _base_event(storyline_id="S2")
        assert pd.dispatch_alert(MagicMock(), ev, "evt2") == "suppressed"
        sent.assert_not_called()

    def test_records_both_keys_on_send(self, monkeypatch):
        monkeypatch.setattr(pd, "suppression_blocks", lambda db, k, tier: False)
        monkeypatch.setattr(pd, "send_telegram_alert", lambda ev: True)
        recorded = []
        monkeypatch.setattr(pd, "record_suppression",
                            lambda db, k, *a, **kw: recorded.append(k))

        ev = _base_event()
        assert pd.dispatch_alert(MagicMock(), ev, "evt1") == "sent"
        assert build_suppression_key(ev) in recorded
        assert build_geo_suppression_key(ev) in recorded

    def test_escalation_still_pages(self, monkeypatch):
        """A storyline that already paged at WATCH must still page when it turns
        CRITICAL — the keys no longer carry severity, so the tier comparison in
        suppression_blocks is the only thing that lets a real escalation through."""
        monkeypatch.setattr(pd, "suppression_blocks",
                            lambda db, k, tier: _fired_at(db, k, tier, "WATCH"))
        monkeypatch.setattr(pd, "record_suppression", lambda *a, **kw: None)
        monkeypatch.setattr(pd, "register_alert", lambda *a, **kw: None)
        monkeypatch.setattr(pd, "get_peak_tier", lambda db, sid: "WATCH")
        sent = MagicMock(return_value=True)
        monkeypatch.setattr(pd, "send_telegram_alert", sent)

        ev = _base_event(alert_tier="CRITICAL")
        assert pd.dispatch_alert(MagicMock(), ev, "evt1") == "sent"
        assert ev["escalation_from"] == "WATCH"
        sent.assert_called_once()

    def test_same_tier_repeat_stays_suppressed(self, monkeypatch):
        """The counterpart, and the 10 Aug 2026 bug: a second ALERT-tier member of a
        storyline that already paged at ALERT is muted."""
        monkeypatch.setattr(pd, "suppression_blocks",
                            lambda db, k, tier: _fired_at(db, k, tier, "CRITICAL"))
        sent = MagicMock()
        monkeypatch.setattr(pd, "send_telegram_alert", sent)

        ev = _base_event(alert_tier="CRITICAL")
        assert pd.dispatch_alert(MagicMock(), ev, "evt1") == "suppressed"
        sent.assert_not_called()

    def test_release_deletes_both_keys_on_failure(self, monkeypatch):
        monkeypatch.setattr(pd, "suppression_blocks", lambda db, k, tier: False)
        monkeypatch.setattr(pd, "record_suppression", lambda *a, **kw: None)
        monkeypatch.setattr(pd, "send_telegram_alert", lambda ev: False)
        db = MagicMock()

        ev = _base_event()
        assert pd.dispatch_alert(db, ev, "evt1") == "failed"
        # The release DELETE is passed both keys as an array parameter.
        delete_calls = [c for c in db.execute.call_args_list if "DELETE" in c[0][0]]
        assert delete_calls
        passed_keys = delete_calls[0][0][1][0]
        assert build_suppression_key(ev) in passed_keys
        assert build_geo_suppression_key(ev) in passed_keys
