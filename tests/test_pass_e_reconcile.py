"""Pass E reconciliation — alert tier consistency after an anchor upgrade.

Pass E rewrites anchor, severity and confidence when a storyline's concatenated text
finally resolves an anchor the original event missed. It used to leave alert_tier
holding the value derived from the PRE-upgrade inputs. That invariant break stayed
invisible because anchor_upgrades has been 0 on every observed production run, but it
matters now that resolving a location is itself a tier gate — an anchor upgrade is
precisely the event that turns an unlocated event into a located one.
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
        elif "SELECT anchor_name_raw, canonical_text" in sql:
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


def _row(alert_tier, time_certainty="same_day"):
    return (
        "11111111-1111-1111-1111-111111111111",   # id
        "missile_strike",                          # event_type
        "somewhere near the airport",              # anchor_name_raw
        None,                                      # anchor_name_norm (unresolved)
        "LOW",                                     # anchor_confidence
        None,                                      # storyline_id
        None,                                      # storyline_hint
        {"confidence": 0.9, "time_certainty": time_certainty},
        60,                                        # severity_score
        0.41,                                      # system_confidence
        alert_tier,                                # alert_tier
    )


def _upgrade(row):
    """Run reconcile with a LOW->HIGH anchor upgrade patched in."""
    conn = _FakeConn(row)
    with patch.object(pe, "normalize_anchor", return_value=("IST", 0.95)), \
         patch.object(pe, "get_anchor_confidence_level", return_value="HIGH"), \
         patch.object(pe, "compute_severity", return_value=100), \
         patch.object(pe, "apply_safety_downrank", side_effect=lambda t, s, p: (s, False)), \
         patch.object(pe, "compute_confidence", return_value=0.7):
        assert pe.reconcile_single_event(conn, "evt") is True
    return conn


def test_anchor_upgrade_rewrites_alert_tier():
    # Located (IST) + fresh + sev 100 + conf 0.7 clears every CRITICAL gate, so the
    # stale WATCH must not survive the upgrade.
    conn = _upgrade(_row("WATCH"))
    assert "alert_tier" in conn.update_sql
    assert "CRITICAL" in conn.update_params


def test_upgrade_that_raises_the_tier_is_logged_not_silently_paged():
    # Pass E must not dispatch (suppression/escalation state lives in Pass D), so a
    # tier that rises has to be visible in the log instead of decided silently.
    with patch.object(pe.logger, "warning") as warn:
        _upgrade(_row("WATCH"))
    assert warn.called
    msg = " ".join(str(a) for a in warn.call_args[0])
    assert "NOT paged" in msg


def test_upgrade_without_fresh_time_does_not_reach_critical():
    # CRITICAL needs BOTH a resolved place and a fresh time_certainty; 86% of the
    # corpus carries 'unknown', so the location half alone must not be enough.
    conn = _upgrade(_row("WATCH", time_certainty="unknown"))
    assert "CRITICAL" not in conn.update_params
    assert "ALERT" in conn.update_params


def test_tier_rank_orders_tiers_and_floors_unknowns():
    from src.core.alerts import tier_rank

    assert tier_rank("CRITICAL") > tier_rank("ALERT") > tier_rank("WATCH")
    assert tier_rank(None) == 0
    assert tier_rank("nonsense") == 0
