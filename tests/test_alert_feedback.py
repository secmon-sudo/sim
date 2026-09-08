"""The feedback loop: two buttons on a card, and a poller that collects the presses.

The thing being protected here is not the happy path. It is that a press is either
recorded exactly once or left in Telegram's window for the next drain — never lost,
never double-counted, and never fabricated from an update that was not one of ours.
Every threshold argument in this pipeline is supposed to eventually rest on this table,
so a wrong row in it is more expensive than a missing one.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

import src.services.feedback_drain as fd
import src.services.telegram_notifier as tn

EVENT_ID = "3f1c8a52-9b41-4d0e-8f77-2a6b5c4d1e90"


# ------------------------------------------------------------------ the buttons

@patch("src.services.telegram_notifier._post_telegram")
@patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_ALERTS_CHAT_ID": "c"})
def test_alert_card_carries_two_feedback_buttons(mock_post):
    mock_post.return_value = MagicMock()
    assert tn.send_telegram_alert({
        "id": EVENT_ID, "alert_tier": "CRITICAL", "severity_score": 90,
        "event_type": "drone_attack", "anchor_name_norm": "KBP",
        "country_iso": "UA", "source_title": "Strike on airport",
    }) is True

    _, kwargs = mock_post.call_args
    keyboard = kwargs["payload"]["reply_markup"]["inline_keyboard"][0]
    assert [b["callback_data"] for b in keyboard] == [
        f"fb:u:C:{EVENT_ID}", f"fb:n:C:{EVENT_ID}"
    ]


def test_callback_data_fits_telegrams_64_byte_cap():
    """Telegram rejects the whole sendMessage if any callback_data exceeds 64 bytes.

    A card that fails to send is worse than a card without buttons, and the failure
    would look like a Telegram outage rather than a payload bug — so the budget is
    asserted rather than eyeballed.
    """
    for tier in ("CRITICAL", "ALERT", "WATCH", None):
        kb = tn.build_feedback_keyboard(EVENT_ID, tier)
        for button in kb["inline_keyboard"][0]:
            assert len(button["callback_data"].encode("utf-8")) <= 64


def test_no_keyboard_without_an_event_id():
    # A button that cannot name its event is a dead button; the card ships plain.
    assert tn.build_feedback_keyboard("", "ALERT") is None
    assert tn.build_feedback_keyboard(None, "ALERT") is None


@patch("src.services.telegram_notifier._post_telegram")
@patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_ALERTS_CHAT_ID": "c"})
def test_card_still_sends_when_it_cannot_be_keyed(mock_post):
    mock_post.return_value = MagicMock()
    assert tn.send_telegram_alert({"alert_tier": "WATCH", "source_title": "x"}) is True
    assert "reply_markup" not in mock_post.call_args[1]["payload"]


# ------------------------------------------------------------------ the parser

@pytest.mark.parametrize("data,expected", [
    (f"fb:u:C:{EVENT_ID}", ("useful", "CRITICAL")),
    (f"fb:n:A:{EVENT_ID}", ("noise", "ALERT")),
    (f"fb:u:W:{EVENT_ID}", ("useful", "WATCH")),
])
def test_parses_our_own_buttons(data, expected):
    parsed = fd.parse_feedback_callback(data)
    assert (parsed["verdict"], parsed["card_tier"]) == expected
    assert parsed["event_id"] == EVENT_ID


@pytest.mark.parametrize("data", [
    None, "", "fb:done", "hello",
    f"fb:x:C:{EVENT_ID}",            # unknown verdict code
    "fb:u:C:not-a-uuid",             # id that would not join to anything
    f"fb:u:{EVENT_ID}",              # old 3-field shape
    f"vote:u:C:{EVENT_ID}",          # someone else's button in the same chat
])
def test_refuses_everything_else(data):
    """A dropped press is recoverable. A fabricated one silently corrupts the only
    non-proxy dataset SIM has, so the parser fails closed on every field."""
    assert fd.parse_feedback_callback(data) is None


def test_unknown_tier_code_leaves_the_tier_unclaimed():
    # The card is still real and the verdict still counts; we just do not know which
    # tier it went out at, and inventing one would be worse than a NULL.
    assert fd.parse_feedback_callback(f"fb:u:Z:{EVENT_ID}")["card_tier"] is None


# ------------------------------------------------------------------ the drain

class FakeConn:
    """Enough psycopg surface for the drain: cursor read, snapshot, insert, cursor write."""

    def __init__(self, cursor=0, event_row=None, fail_insert=False):
        self.cursor_value = cursor
        self.event_row = event_row
        self.fail_insert = fail_insert
        self.inserts = []
        self.cursor_writes = []

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        if "FROM telegram_update_cursor" in s:
            return MagicMock(fetchone=lambda: (self.cursor_value,))
        if s.startswith("INSERT INTO telegram_update_cursor"):
            self.cursor_writes.append(params[0])
            return MagicMock(fetchone=lambda: None)
        if "FROM events" in s:
            return MagicMock(fetchone=lambda: self.event_row)
        if s.startswith("INSERT INTO alert_feedback"):
            if self.fail_insert:
                raise RuntimeError("db down")
            self.inserts.append(params)
            return MagicMock(fetchone=lambda: (params[0],))
        raise AssertionError(f"unexpected SQL: {s[:80]}")


def _press(update_id, data=f"fb:n:A:{EVENT_ID}", user_id=7, message_id=555):
    return {
        "update_id": update_id,
        "callback_query": {
            "id": f"cq{update_id}", "data": data,
            "from": {"id": user_id, "username": "analyst"},
            "message": {"message_id": message_id, "date": 1757308800,
                        "chat": {"id": -100123}},
        },
    }


def _drain(conn, updates, **kw):
    with patch.object(fd.httpx, "get") as get, patch.object(fd.httpx, "post") as post:
        get.return_value = MagicMock(
            raise_for_status=lambda: None,
            json=lambda: {"ok": True, "result": updates},
        )
        post.return_value = MagicMock()
        stats = fd.drain_feedback(conn, bot_token="t", **kw)
    return stats, get, post


def test_records_a_press_with_the_events_own_fields():
    conn = FakeConn(event_row=("drone_attack", "UA", 88, "reuters.com", "Strike"))
    stats, _, _ = _drain(conn, [_press(41)])

    assert (stats["fetched"], stats["recorded"], stats["ignored"]) == (1, 1, 0)
    params = conn.inserts[0]
    assert params[0] == 41                      # update_id
    assert params[1] == EVENT_ID
    assert params[2] == "noise"
    assert params[3] == "ALERT"                 # tier from the BUTTON, not the row
    assert params[4:9] == ("drone_attack", "UA", 88, "reuters.com", "Strike")
    assert conn.cursor_writes == [41]


def test_a_purged_event_still_records_a_verdict():
    """Pass F deletes events. A press on a card whose event is already gone is still
    the analyst telling us something; it just cannot be sliced afterwards."""
    conn = FakeConn(event_row=None)
    stats, _, _ = _drain(conn, [_press(42)])
    assert stats["recorded"] == 1
    assert conn.inserts[0][4:9] == (None, None, None, None, None)


def test_cursor_starts_after_the_stored_offset():
    conn = FakeConn(cursor=900)
    _, get, _ = _drain(conn, [])
    assert get.call_args[1]["params"]["offset"] == 901


def test_foreign_updates_are_counted_and_skipped():
    conn = FakeConn()
    stats, _, _ = _drain(conn, [
        {"update_id": 50, "message": {"text": "someone chatting"}},
        _press(51, data="fb:done"),
    ])
    assert (stats["recorded"], stats["ignored"]) == (0, 2)
    assert conn.inserts == []
    # The cursor still advances past them — otherwise a chatty group would make the
    # drain re-read the same window until Telegram expired it.
    assert conn.cursor_writes == [51]


def test_a_failed_write_leaves_the_press_in_the_window():
    """The cursor is the only thing standing between a DB hiccup and a lost press, so
    it must never advance past an update whose row did not commit."""
    conn = FakeConn(fail_insert=True)
    stats, _, _ = _drain(conn, [_press(60)])
    assert stats["recorded"] == 0
    assert conn.cursor_writes == [59]           # 60 will be re-delivered


def test_a_failed_write_keeps_the_presses_already_banked():
    class HalfFailing(FakeConn):
        def execute(self, sql, params=None):
            if " ".join(sql.split()).startswith("INSERT INTO alert_feedback") \
                    and params[0] == 71:
                raise RuntimeError("db down")
            return FakeConn.execute(self, sql, params)

    conn = HalfFailing()
    stats, _, _ = _drain(conn, [_press(70), _press(71, user_id=8)])
    assert stats["recorded"] == 1
    assert [p[0] for p in conn.inserts] == [70]
    assert conn.cursor_writes == [70]           # keeps 70, retries 71


def test_telegram_outage_is_not_a_pipeline_failure():
    conn = FakeConn()
    with patch.object(fd.httpx, "get", side_effect=RuntimeError("timeout")):
        stats = fd.drain_feedback(conn, bot_token="t")
    assert stats["recorded"] == 0 and stats["error"].startswith("RuntimeError")
    assert conn.cursor_writes == []


def test_missing_token_is_a_skip_not_a_crash():
    with patch.dict(os.environ, {}, clear=True):
        assert fd.drain_feedback(FakeConn())["error"] == "no_token"


def test_the_card_is_rewritten_to_show_what_was_recorded():
    """The toast expires within minutes and this poller is hours behind by design, so
    the keyboard edit is the acknowledgement the analyst actually sees."""
    conn = FakeConn()
    _, _, post = _drain(conn, [_press(80)])

    edits = [c for c in post.call_args_list if "editMessageReplyMarkup" in c.args[0]]
    assert len(edits) == 1
    markup = edits[0][1]["json"]["reply_markup"]["inline_keyboard"][0]
    assert len(markup) == 1 and "gürültü" in markup[0]["text"]


def test_an_expired_toast_does_not_stop_the_edit():
    conn = FakeConn()
    with patch.object(fd.httpx, "get") as get, patch.object(fd.httpx, "post") as post:
        get.return_value = MagicMock(raise_for_status=lambda: None,
                                     json=lambda: {"result": [_press(90)]})
        post.side_effect = [RuntimeError("query is too old"), MagicMock()]
        stats = fd.drain_feedback(conn, bot_token="t")
    assert stats["recorded"] == 1
    assert post.call_count == 2


# ------------------------------------------------------ transient vs permanent failure

@pytest.mark.parametrize("status,fatal", [
    (401, True),    # token revoked
    (404, True),    # bot deleted
    (409, True),    # a webhook has taken the update stream away from polling
    (429, False),   # rate limited — the window still holds the presses
    (502, False),   # Telegram having a moment
])
def test_only_permanent_failures_are_worth_paging_for(status, fatal):
    """A drain that goes red on every network blip is a drain whose red is ignored,
    and the 409 case is the one that otherwise looks exactly like a quiet week."""
    conn = FakeConn()
    response = MagicMock(status_code=status)
    err = fd.httpx.HTTPStatusError("boom", request=MagicMock(), response=response)
    with patch.object(fd.httpx, "get", side_effect=err):
        stats = fd.drain_feedback(conn, bot_token="t")
    assert stats["fatal"] is fatal
    assert stats["recorded"] == 0


def test_the_orchestrator_pages_only_on_the_permanent_class():
    from src.pipeline.orchestrator import _collect_degradations

    transient = _collect_degradations(
        {"feedback": {"error": "RuntimeError: timeout", "fatal": False}})
    assert transient == []

    permanent = _collect_degradations(
        {"feedback": {"error": "getUpdates HTTP 409", "fatal": True}})
    assert len(permanent) == 1 and "409" in permanent[0]
