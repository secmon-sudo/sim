"""Tests for the storyline-independent geo suppression safety net.

Covers build_geo_suppression_key (pure) and dispatch_alert's dual-key behaviour that
mutes duplicate alerts even when the storyline_id fragments across paraphrased sources.
"""

from unittest.mock import MagicMock

import src.pipeline.pass_d_score as pd
from src.core.alerts import build_geo_suppression_key, build_suppression_key


class TestGeoSuppressionKey:
    def test_prefers_iata_anchor(self):
        ev = {"anchor_name_norm": "KBP", "country_iso": "UA", "severity_score": 72}
        assert build_geo_suppression_key(ev) == "geofp|UA|KBP|70"

    def test_falls_back_to_geo_key(self):
        # No IATA anchor -> coarse geo_key from raw text; "Kiev" collapses to KYIV.
        ev = {"anchor_name_raw": "Kiev", "country_iso": "UA", "severity_score": 88}
        assert build_geo_suppression_key(ev) == "geofp|UA|KYIV|80"

    def test_paraphrased_location_shares_key(self):
        a = {"anchor_name_raw": "Kyiv", "country_iso": "UA", "severity_score": 90}
        b = {"anchor_name_raw": "Ukrainian capital", "country_iso": "UA", "severity_score": 91}
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
        monkeypatch.setattr(pd, "is_suppressed",
                            lambda db, k: k == geo_key_str)
        sent = MagicMock()
        monkeypatch.setattr(pd, "send_telegram_alert", sent)

        # Different storyline_id -> different primary key, but same geo fingerprint.
        ev = _base_event(storyline_id="S2")
        assert pd.dispatch_alert(MagicMock(), ev, "evt2") == "suppressed"
        sent.assert_not_called()

    def test_records_both_keys_on_send(self, monkeypatch):
        monkeypatch.setattr(pd, "is_suppressed", lambda db, k: False)
        monkeypatch.setattr(pd, "send_telegram_alert", lambda ev: True)
        recorded = []
        monkeypatch.setattr(pd, "record_suppression",
                            lambda db, k, *a, **kw: recorded.append(k))

        ev = _base_event()
        assert pd.dispatch_alert(MagicMock(), ev, "evt1") == "sent"
        assert build_suppression_key(ev) in recorded
        assert build_geo_suppression_key(ev) in recorded

    def test_release_deletes_both_keys_on_failure(self, monkeypatch):
        monkeypatch.setattr(pd, "is_suppressed", lambda db, k: False)
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
