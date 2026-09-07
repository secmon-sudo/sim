"""The flash detector's daily shadow run.

Every one of the 86 triggers this detector has ever recorded fired on a Sunday,
because run_flash_detection is called only from the weekly forecast. Its window is
24 hours; its cadence was 168. Six days in seven were never examined, and a day not
examined is not recoverable later.

Turning the cadence up on its own is not the fix. Measured over 14 days, the two
triggers that read only the last 24 hours would fire for ~9 (convergence) and ~2
(high volume) countries a DAY — against ~11 alert cards a day from Pass D, that
roughly doubles this product's paging with cards describing an ordinary day in a
war. And no threshold can be chosen yet, because what an ordinary day looks like
per country was never recorded.

So: daily cadence for the RECORD, weekly cadence for the PAGE, until the record can
answer the question.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import src.pipeline.weekly_forecast as wf


def _events(n, country="UA", hours=0):
    base = datetime.utcnow() - timedelta(hours=1)
    return [
        {"id": f"e{i}", "country_iso": country, "event_type": "missile_strike" if i % 2 else "drone_attack",
         "occurred_at_est": base + timedelta(minutes=i * 10 + hours * 60),
         "anchor_name_norm": "KBP", "anchor_name_raw": "kyiv",
         "latitude": 50.3, "longitude": 30.9,
         "source_domain": "reuters.com", "system_confidence": 0.9}
        for i in range(n)
    ]


class _Conn:
    def __init__(self):
        self.inserts = []

    def execute(self, sql, params=None):
        if sql.strip().upper().startswith("INSERT"):
            self.inserts.append(params)
        return MagicMock()

    def commit(self):
        pass

    def rollback(self):
        pass


class TestShadowRunSendsNothing:
    def test_no_telegram_message_is_sent(self):
        conn = _Conn()
        with patch.object(wf, "send_flash_update_telegram") as send:
            out = wf.run_flash_detection(conn, _events(4), countries_data=[],
                                         dispatch=False)
        assert out, "the shadow run must still find triggers"
        assert not send.called

    def test_the_live_run_still_sends(self):
        conn = _Conn()
        with patch.object(wf, "send_flash_update_telegram", return_value="msg-1") as send:
            wf.run_flash_detection(conn, _events(4), countries_data=[], dispatch=True)
        assert send.called


class TestShadowRowsAreKeptApart:
    def _event_types(self, conn):
        return [p[0] for p in conn.inserts if p and isinstance(p[0], str)]

    def test_shadow_rows_use_their_own_telemetry_type(self):
        conn = _Conn()
        with patch.object(wf, "send_flash_update_telegram"):
            wf.run_flash_detection(conn, _events(4), countries_data=[], dispatch=False)
        assert "flash_trigger_shadow" in self._event_types(conn)
        assert "flash_trigger" not in self._event_types(conn)

    def test_live_rows_keep_the_name_the_history_is_written_under(self):
        conn = _Conn()
        with patch.object(wf, "send_flash_update_telegram", return_value="m"):
            wf.run_flash_detection(conn, _events(4), countries_data=[], dispatch=True)
        assert "flash_trigger" in self._event_types(conn)

    def test_the_row_carries_a_count_the_id_cap_would_hide(self):
        """event_ids is truncated at 20. A shadow row exists to be counted later,
        and a truncated list cannot be — this is the calibration input."""
        import json

        conn = _Conn()
        with patch.object(wf, "send_flash_update_telegram"):
            wf.run_flash_detection(conn, _events(30), countries_data=[], dispatch=False)
        payloads = [json.loads(p[1]) for p in conn.inserts
                    if p and p[0] == "flash_trigger_shadow"]
        assert payloads
        assert any(p["event_count"] > len(p["event_ids"]) for p in payloads)


class TestTheShadowRunHasNoZScorePath:
    def test_passing_no_countries_means_no_z_trigger(self):
        """The z-score is weekly tension-index history and does not exist daily.
        Measuring the two 24-hour triggers is the point; inventing a daily z-score
        would be measuring something nobody has agreed on yet."""
        conn = _Conn()
        with patch.object(wf, "send_flash_update_telegram"):
            out = wf.run_flash_detection(conn, _events(4), countries_data=[],
                                         dispatch=False)
        assert all(t["type"] != "Z-Score Exceeded" for t in out)
