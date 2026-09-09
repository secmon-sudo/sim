"""The same event may not page twice, however its score is later revised.

Measured over the 14 hours to 2026-09-09 03:13: five of thirty-eight cards were one
event paging twice, ~90 seconds apart, every one with the same shape —

    00:11:09  ALERT     44813a56…|UNKNOWN          (and NO geo key at all)
    00:12:35  CRITICAL  44813a56…|KHE  +  geofp|UA|KHE

Pass D pages while the anchor is unresolved; Pass E resolves it, rescores, and the new
keys no longer collide with the claim the first card left. Both existing keys embed the
resolved location, so both move underneath themselves — and the run where that happens
is also the run with only one key protecting the card, because build_geo_suppression_key
returns None for an UNKNOWN location.
"""

from unittest.mock import MagicMock, patch

import pytest

import src.pipeline.pass_d_score as pd
from src.core.alerts import build_event_suppression_key

EVENT_ID = "44813a56-7a81-4982-a583-da9dfca89a8c"
STORYLINE = "1f0d2c9e-0000-4000-8000-00000000abcd"


class FakeConn:
    """Suppression table as a dict, so ON CONFLICT semantics stay out of the way."""

    def __init__(self):
        self.claims = {}      # suppression_key -> tier
        self.deleted = []

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        if s.startswith("SELECT alert_tier FROM alert_suppression"):
            tier = self.claims.get(params[0])
            return MagicMock(fetchone=lambda: (tier,) if tier else None)
        if s.startswith("INSERT INTO alert_suppression"):
            self.claims[params[0]] = params[1]
            return MagicMock(fetchone=lambda: None)
        if s.startswith("DELETE FROM alert_suppression"):
            self.deleted.extend(params[0])
            for k in params[0]:
                self.claims.pop(k, None)
            return MagicMock(fetchone=lambda: None)
        raise AssertionError(f"unexpected SQL: {s[:70]}")

    def commit(self):
        pass

    def rollback(self):
        pass


def _event(tier, anchor_norm, severity=90):
    """The same article as Pass D sees it, then as Pass E sees it after resolution."""
    return {
        "id": EVENT_ID,
        "alert_tier": tier,
        "severity_score": severity,
        "storyline_id": STORYLINE,
        "anchor_name_norm": anchor_norm,
        "anchor_name_raw": "Kherson region",
        "anchor_confidence": "HIGH" if anchor_norm else "LOW",
        "country_iso": "UA",
        "source_title": "Russian Drones Strike Kherson Region, Injuring Five People",
    }


@pytest.fixture
def sent():
    with patch.object(pd, "send_telegram_alert", return_value=True) as m, \
         patch.object(pd, "register_alert"), \
         patch.object(pd, "get_peak_tier", return_value=None):
        yield m


def test_the_overnight_duplicate_no_longer_pages(sent):
    conn = FakeConn()

    # Pass D: anchor unresolved, so the primary key ends in |UNKNOWN and the geo net
    # is not built at all — one key, and it is the one that will move.
    assert pd.dispatch_alert(conn, _event("ALERT", None), EVENT_ID) == "sent"

    # Pass E ~90s later: anchor resolved to KHE, tier rescored up. Both new keys are
    # different strings from anything claimed above.
    assert pd.dispatch_alert(conn, _event("CRITICAL", "KHE"), EVENT_ID) == "suppressed_rescore"
    assert sent.call_count == 1


def test_both_location_keys_move_under_the_first_card(sent):
    """Why nothing but an event-level claim could have caught this.

    BOTH existing keys carry the location, so both change when the anchor resolves —
    the primary from |UNKNOWN to |KHE, and the geo fingerprint from the coarse text
    key to the IATA one. Neither collides with what the first card claimed.
    """
    conn = FakeConn()
    pd.dispatch_alert(conn, _event("ALERT", None), EVENT_ID)

    from src.core.alerts import build_geo_suppression_key, build_suppression_key
    before, after = _event("ALERT", None), _event("CRITICAL", "KHE")
    assert build_suppression_key(before) != build_suppression_key(after)
    assert build_geo_suppression_key(before) != build_geo_suppression_key(after)
    assert build_event_suppression_key(EVENT_ID) in conn.claims


def test_the_first_card_may_have_no_geo_net_at_all(sent):
    """The production shape, and the reason the duplicates clustered where they did.

    build_geo_suppression_key returns None when no location is known, so an event that
    pages before its anchor resolves is protected by ONE key — and it is the key that
    is about to change. Every one of the five overnight duplicates had exactly this
    single-key first card.
    """
    from src.core.alerts import build_geo_suppression_key
    unlocated = _event("ALERT", None)
    unlocated["anchor_name_raw"] = None
    assert build_geo_suppression_key(unlocated) is None

    conn = FakeConn()
    assert pd.dispatch_alert(conn, unlocated, EVENT_ID) == "sent"
    assert len(conn.claims) == 2          # primary + event key, no geo net
    assert pd.dispatch_alert(conn, _event("CRITICAL", "KHE"), EVENT_ID) == "suppressed_rescore"


def test_a_downgrade_is_muted_too(sent):
    # WATCH after ALERT was never allowed by the tier ladder, but it reached the ladder
    # under a different key. Layer 0 does not consult the tier at all.
    conn = FakeConn()
    pd.dispatch_alert(conn, _event("ALERT", None), EVENT_ID)
    assert pd.dispatch_alert(conn, _event("WATCH", "KHE"), EVENT_ID) == "suppressed_rescore"
    assert sent.call_count == 1


def test_a_different_event_in_the_same_storyline_still_escalates(sent):
    """The escalation allowance is storyline-level and must survive: a genuinely worse
    storyline gets worse through a NEW report, which carries its own event key."""
    conn = FakeConn()
    other_id = "aaaaaaaa-0000-4000-8000-00000000beef"

    assert pd.dispatch_alert(conn, _event("ALERT", "KHE"), EVENT_ID) == "sent"
    newer = _event("CRITICAL", "KHE")
    newer["id"] = other_id
    assert pd.dispatch_alert(conn, newer, other_id) == "sent"
    assert sent.call_count == 2


def test_pass_e_still_pages_an_event_pass_d_never_sent(sent):
    """Pass E's real job — 17 CRITICALs had never paged at all — must not be broken:
    with no prior claim there is nothing to block."""
    conn = FakeConn()
    assert pd.dispatch_alert(conn, _event("CRITICAL", "KHE"), EVENT_ID) == "sent"


def test_a_failed_send_releases_the_event_claim_too(sent):
    """Otherwise one Telegram hiccup would mute the event permanently — the failure
    mode the release path exists to prevent, extended to the new key."""
    conn = FakeConn()
    with patch.object(pd, "send_telegram_alert", return_value=False):
        assert pd.dispatch_alert(conn, _event("ALERT", "KHE"), EVENT_ID) == "failed"
    assert build_event_suppression_key(EVENT_ID) not in conn.claims
    assert pd.dispatch_alert(conn, _event("ALERT", "KHE"), EVENT_ID) == "sent"
