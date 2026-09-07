"""Pass E reconciliation — Pass D parity after an anchor upgrade.

Pass E rewrites anchor, severity, confidence and alert_tier when a storyline's sibling
text finally resolves an anchor the original event missed. Everything it rewrites it
must rewrite with Pass D's own recipe, because the values it writes REPLACE Pass D's.
Two ways that failed, both measured over the 17 production runs to 2026-09-07:

  * the tier gates saw 8 of their 12 inputs, so report_kind never reached them and 13
    of 55 escalations were commentary/followup articles Pass D had already vetoed;
  * the 42 genuine escalations — 17 of them ALERT→CRITICAL — were logged and dropped,
    because the pass declined to dispatch at all.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import src.pipeline.pass_e_reconcile as pe


class _FakeConn:
    """Minimal psycopg-shaped stub: canned SELECTs, captured UPDATE."""

    def __init__(self, row, anchor_row=(False, 41.0, 29.0, "TR")):
        self._row = row
        self._anchor_row = anchor_row
        self.update_sql = None
        self.update_params = None

    def execute(self, sql, params=None):
        result = MagicMock()
        if sql.strip().upper().startswith("UPDATE"):
            self.update_sql = sql
            self.update_params = params
        elif "FROM anchor_master" in sql:
            result.fetchone.return_value = self._anchor_row
        elif "SELECT anchor_name_raw" in sql:
            result.fetchall.return_value = []
        else:
            result.fetchone.return_value = self._row
        return result

    @contextmanager
    def transaction(self):
        yield

    def commit(self):
        pass

    def rollback(self):
        pass


def _row(alert_tier, time_certainty="same_day",
         source_title="Missile strike reported near Istanbul airport",
         report_kind="new_incident", date_verified=True, corroborating_sources=None,
         event_type="missile_strike"):
    return (
        "11111111-1111-1111-1111-111111111111",   # 0  id
        event_type,                                # 1  event_type
        "somewhere near the airport",              # 2  anchor_name_raw
        None,                                      # 3  anchor_name_norm (unresolved)
        "LOW",                                     # 4  anchor_confidence
        None,                                      # 5  storyline_id
        None,                                      # 6  storyline_hint
        {"confidence": 0.9, "time_certainty": time_certainty,
         "report_kind": report_kind},              # 7  llm_parsed_output
        60,                                        # 8  severity_score
        0.41,                                      # 9  system_confidence
        alert_tier,                                # 10 alert_tier
        source_title,                              # 11 source_title (aftermath gate)
        "TR",                                      # 12 country_iso
        "https://example.com/a",                   # 13 source_url
        "example.com",                             # 14 source_domain
        None,                                      # 15 occurred_at_est
        None,                                      # 16 ingested_at
        None,                                      # 17 published_at
        date_verified,                             # 18 date_verified
        corroborating_sources,                     # 19 corroborating_sources
    )


def _upgrade(row):
    """Run reconcile with a LOW->HIGH anchor upgrade patched in.

    Returns (conn, dispatch_mock, dispatch_result).
    """
    conn = _FakeConn(row)
    with patch.object(pe, "normalize_anchor", return_value=("IST", 0.95)), \
         patch.object(pe, "get_anchor_confidence_level", return_value="HIGH"), \
         patch.object(pe, "compute_severity", return_value=100), \
         patch.object(pe, "apply_safety_downrank", side_effect=lambda t, s, p: (s, False)), \
         patch.object(pe, "compute_confidence", return_value=0.7), \
         patch.object(pe, "dispatch_alert", return_value="sent") as dispatch:
        ok, upgraded, dispatch_result = pe.reconcile_single_event(conn, "evt")
        # The second element is what makes anchor_upgrades observable — it read 0
        # on every run until 2026-08-17 because the function only returned success.
        assert ok is True
        assert upgraded is True
    return conn, dispatch, dispatch_result


def test_anchor_upgrade_rewrites_alert_tier():
    # Located (IST) + fresh + sev 100 + conf 0.7 clears every CRITICAL gate, so the
    # stale WATCH must not survive the upgrade.
    conn, _, _ = _upgrade(_row("WATCH"))
    assert "alert_tier" in conn.update_sql
    assert "CRITICAL" in conn.update_params


def test_upgrade_that_raises_the_tier_is_paged():
    # The whole point of the pass: WATCH→CRITICAL is a real escalation, and Pass D
    # has already finished with this event. Logging it was the old behaviour, and it
    # lost 17 ALERT→CRITICAL escalations in 2.5 days of production.
    _, dispatch, result = _upgrade(_row("WATCH"))
    assert dispatch.called
    assert dispatch.call_args[0][1]["alert_tier"] == "CRITICAL"
    assert result == "sent"


def test_upgrade_that_does_not_raise_the_tier_is_not_paged():
    # Already CRITICAL: the anchor got better, the story did not get worse.
    _, dispatch, result = _upgrade(_row("CRITICAL"))
    assert not dispatch.called
    assert result is None


def test_dispatch_receives_the_fields_the_card_and_the_ledger_need():
    # dispatch_alert keys suppression off storyline/geography and renders the card
    # from these; a missing country_iso silently disables the geo suppression net.
    _, dispatch, _ = _upgrade(_row("WATCH"))
    event = dispatch.call_args[0][1]
    for field in ("source_title", "source_url", "country_iso", "anchor_name_norm",
                  "severity_score", "system_confidence", "occurred_at_est"):
        assert field in event, field
    assert event["anchor_name_norm"] == "IST"


def test_report_kind_veto_survives_the_upgrade():
    # Pass D refuses to page commentary (core.alerts REPORT_KIND_NOT_NEWS). Pass E
    # re-evaluates from scratch, and without report_kind in its dict the gate fails
    # open — 13 of 55 production escalations were exactly this.
    # time_certainty='unknown' keeps this below CRITICAL, which is exempt by design.
    conn, dispatch, _ = _upgrade(_row(None, time_certainty="unknown",
                                      report_kind="commentary"))
    assert "ALERT" not in conn.update_params
    assert "CRITICAL" not in conn.update_params
    assert not dispatch.called


def test_upgrade_does_not_re_promote_an_aftermath_report():
    # The headline-shape half of the same contract, which Pass E already honoured.
    conn, _, _ = _upgrade(_row(None, time_certainty="unknown",
                               source_title="Ukraine war latest: Russia makes slow gains"))
    assert "ALERT" not in conn.update_params
    assert "CRITICAL" not in conn.update_params


def test_critical_upgrade_of_a_roundup_is_still_allowed():
    # The deliberate exemption: when a roundup is the only carrier of a major
    # development, withholding the page costs more than the noise it admits.
    conn, dispatch, _ = _upgrade(_row(None, source_title="Ukraine war latest: Russia makes slow gains",
                                      report_kind="roundup"))
    assert "CRITICAL" in conn.update_params
    assert dispatch.called


def test_unverified_aggregator_date_cannot_carry_the_upgrade():
    # date_verified=False means the timestamp is a crawl stamp, so 'same_day' is not
    # evidence of freshness — and CRITICAL needs a fresh time.
    conn, _, _ = _upgrade(_row("WATCH", date_verified=False))
    assert "CRITICAL" not in conn.update_params


def test_upgrade_without_fresh_time_does_not_reach_critical():
    # CRITICAL needs BOTH a resolved place and a fresh time_certainty; 86% of the
    # corpus carries 'unknown', so the location half alone must not be enough.
    conn, _, _ = _upgrade(_row("WATCH", time_certainty="unknown"))
    assert "CRITICAL" not in conn.update_params
    assert "ALERT" in conn.update_params


def test_severity_keeps_the_aviation_bonus_pass_d_applied():
    # compute_severity is patched to a bare 90 here, so the only thing that can lift
    # the stored value is the aviation nexus term Pass E used to drop entirely.
    conn = _FakeConn(_row("WATCH"))
    with patch.object(pe, "normalize_anchor", return_value=("IST", 0.95)), \
         patch.object(pe, "get_anchor_confidence_level", return_value="HIGH"), \
         patch.object(pe, "compute_severity", return_value=90), \
         patch.object(pe, "compute_confidence", return_value=0.7), \
         patch.object(pe, "dispatch_alert", return_value="sent"):
        pe.reconcile_single_event(conn, "evt")
    assert max(p for p in conn.update_params if isinstance(p, int)) > 90


def test_tier_rank_orders_tiers_and_floors_unknowns():
    from src.core.alerts import tier_rank

    assert tier_rank("CRITICAL") > tier_rank("ALERT") > tier_rank("WATCH")
    assert tier_rank(None) == 0
    assert tier_rank("nonsense") == 0
