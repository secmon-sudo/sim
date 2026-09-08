"""SIM — collect the analyst's verdict on the alert cards that were sent.

The cards carry two buttons (see telegram_notifier.build_feedback_keyboard). Pressing
one produces a Telegram `callback_query`, and something has to come and collect it.

There is no server. The pipeline is a cron'd GitHub Actions job, so the collector is a
POLLER: it asks getUpdates for everything since the cursor, writes what it finds, and
advances the cursor. That shape has two consequences worth stating out loud, because
both are deliberate:

  * The acknowledgement is late. answerCallbackQuery only works while the query is
    fresh, and by the time a drain runs the press may be hours old — so the toast is
    best-effort and the REAL confirmation is the card's keyboard being rewritten to say
    what was recorded. That edit works on a message of any age.

  * The window may be re-read. The cursor advances only after the rows are committed,
    so a crash in between re-delivers presses we already have. update_id is the primary
    key for exactly this reason; re-running a drain is a no-op, not a double count.
"""

import logging
import os
import uuid
from datetime import datetime, timezone

import httpx

from src.core import counters
from src.services.telegram_notifier import (
    FEEDBACK_PREFIX,
    TIER_CODES_INV,
    VERDICT_CODES,
)

logger = logging.getLogger(__name__)

# getUpdates long-polls when asked to; the drain deliberately does NOT. A run has other
# work to do and an empty inbox is the common case, so timeout=0 returns immediately.
_GET_UPDATES_TIMEOUT = 20.0
_MAX_UPDATES_PER_DRAIN = 100

# Statuses that will never fix themselves: 401 a revoked token, 404 a deleted bot, 409
# a webhook registered on the same bot (which takes the update stream away from
# getUpdates entirely and is the one failure that looks like "just quiet").
_FATAL_STATUSES = frozenset({401, 404, 409})

CONFIRMED_LABELS = {
    "useful": "✅ Kaydedildi: işe yaradı",
    "noise": "🗑️ Kaydedildi: gürültü",
}


def parse_feedback_callback(data: str) -> dict | None:
    """`fb:<verdict>:<tier>:<uuid>` → dict, or None if this is not one of our buttons.

    Strict on every field. The bot shares a chat with whatever else may be posted there,
    and an unparseable payload that got stored as a verdict would poison the one dataset
    that is supposed to settle threshold arguments — a dropped press is recoverable, a
    fabricated one is not.
    """
    parts = str(data or "").split(":")
    if len(parts) != 4 or parts[0] != FEEDBACK_PREFIX:
        return None
    _, verdict_code, tier_code, event_id = parts
    verdict = VERDICT_CODES.get(verdict_code)
    if not verdict:
        return None
    try:
        event_id = str(uuid.UUID(event_id))
    except (ValueError, AttributeError, TypeError):
        return None
    return {
        "verdict": verdict,
        "card_tier": TIER_CODES_INV.get(tier_code),
        "event_id": event_id,
    }


def _api(bot_token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{bot_token}/{method}"


def _read_cursor(db_conn) -> int:
    row = db_conn.execute(
        "SELECT last_update_id FROM telegram_update_cursor WHERE id = 1"
    ).fetchone()
    return int(row[0]) if row else 0


def _write_cursor(db_conn, last_update_id: int) -> None:
    db_conn.execute(
        """INSERT INTO telegram_update_cursor (id, last_update_id, updated_at)
                VALUES (1, %s, NOW())
           ON CONFLICT (id) DO UPDATE
                SET last_update_id = EXCLUDED.last_update_id,
                    updated_at     = NOW()
         WHERE telegram_update_cursor.last_update_id < EXCLUDED.last_update_id""",
        (int(last_update_id),),
    )


def _event_snapshot(db_conn, event_id: str) -> dict:
    """The event's own fields, copied into the feedback row.

    Pass F deletes events — archived noise at 30 days, reconciled rows in the archive
    batch — so a feedback row that only holds an id becomes an uninterpretable verdict
    about nothing within the month. An empty dict here is a real answer: the press
    still counts, it just cannot be sliced by type or country.
    """
    try:
        row = db_conn.execute(
            """SELECT event_type, country_iso, severity_score, source_domain, source_title
                 FROM events WHERE id = %s""",
            (event_id,),
        ).fetchone()
    except Exception:
        logger.exception("Feedback: event snapshot lookup failed for %s", event_id[:8])
        return {}
    if not row:
        return {}
    keys = ("event_type", "country_iso", "severity_score", "source_domain", "source_title")
    return dict(zip(keys, row))


def _record(db_conn, update_id: int, parsed: dict, cq: dict) -> bool:
    """Write one press. Returns True if a row was inserted or a verdict changed."""
    user = cq.get("from") or {}
    message = cq.get("message") or {}
    snap = _event_snapshot(db_conn, parsed["event_id"])

    pressed_at = None
    if message.get("date"):
        try:
            pressed_at = datetime.fromtimestamp(int(message["date"]), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            pressed_at = None

    # ON CONFLICT on the (event_id, tg_user_id) index, not the primary key: pressing the
    # other button is a CHANGE OF MIND, not a second vote, and the later press wins. The
    # update_id conflict is handled by the same statement's PK clause below only because
    # a re-read of the window must stay a no-op.
    row = db_conn.execute(
        """INSERT INTO alert_feedback (
               update_id, event_id, verdict, card_tier,
               event_type, country_iso, severity_score, source_domain, source_title,
               tg_user_id, tg_username, tg_message_id, pressed_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (event_id, tg_user_id) DO UPDATE
               SET verdict    = EXCLUDED.verdict,
                   update_id  = EXCLUDED.update_id,
                   card_tier  = EXCLUDED.card_tier,
                   pressed_at = EXCLUDED.pressed_at,
                   created_at = NOW()
             WHERE alert_feedback.update_id < EXCLUDED.update_id
           RETURNING update_id""",
        (
            int(update_id), parsed["event_id"], parsed["verdict"], parsed.get("card_tier"),
            snap.get("event_type"), snap.get("country_iso"), snap.get("severity_score"),
            snap.get("source_domain"), snap.get("source_title"),
            user.get("id"), (user.get("username") or user.get("first_name") or None),
            message.get("message_id"), pressed_at,
        ),
    ).fetchone()
    return row is not None


def _confirm(bot_token: str, cq: dict, verdict: str) -> None:
    """Tell the analyst the press landed. Never raises.

    Two attempts at the same thing. The toast is the one they would see immediately and
    is the one that usually fails — Telegram rejects a callback query that is more than
    a few minutes old, and this poller is minutes-to-hours behind by design. The
    keyboard rewrite has no such expiry, so it is the confirmation that actually
    arrives, and it doubles as the record of what was recorded.
    """
    message = cq.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    message_id = message.get("message_id")

    try:
        httpx.post(
            _api(bot_token, "answerCallbackQuery"),
            json={"callback_query_id": cq.get("id"),
                  "text": CONFIRMED_LABELS.get(verdict, "Kaydedildi")},
            timeout=10.0,
        )
    except Exception:
        logger.debug("Feedback: callback toast expired (expected for an old press)")

    if chat_id is None or message_id is None:
        return
    try:
        httpx.post(
            _api(bot_token, "editMessageReplyMarkup"),
            json={
                "chat_id": chat_id,
                "message_id": message_id,
                # A single inert button rather than an empty keyboard: removing the
                # markup would leave a card that looks identical to one never voted on,
                # and "did I already answer this" is the question the analyst is going
                # to have when scrolling back through a hundred cards.
                "reply_markup": {"inline_keyboard": [[
                    {"text": CONFIRMED_LABELS.get(verdict, "Kaydedildi"),
                     "callback_data": "fb:done"}
                ]]},
            },
            timeout=10.0,
        )
    except Exception:
        counters.bump("feedback_confirm_failed")
        logger.warning("Feedback: could not rewrite keyboard on message %s", message_id)


def drain_feedback(db_conn, bot_token: str | None = None) -> dict:
    """Collect pending button presses into alert_feedback. Returns stats.

    Isolated by contract: the caller is the pipeline, and no failure in here — not a
    Telegram outage, not a malformed update — may cost a run.

    Do not call this from a test or a scratch environment against the real bot token.
    getUpdates is a SHARED, single-consumer stream: a press collected here is written to
    whatever database this connection points at and its card is edited to say
    "recorded", so a run against a throwaway schema silently takes the press away from
    the only database anyone will ever query.
    """
    # `fatal` separates the two failure classes. A timeout or a 502 is a MISSED
    # collection: the presses sit in Telegram's 24h window and the next drain takes
    # them, so paging on it would train the ops channel to be ignored. A revoked token,
    # a deleted bot, or a webhook that has taken over the update stream (409) is
    # permanent — from then on every press is lost — and that is worth waking someone.
    stats = {"fetched": 0, "recorded": 0, "ignored": 0, "error": None, "fatal": False}

    bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.info("Feedback drain skipped: no TELEGRAM_BOT_TOKEN")
        stats["error"] = "no_token"
        stats["fatal"] = True
        return stats

    try:
        offset = _read_cursor(db_conn) + 1
        resp = httpx.get(
            _api(bot_token, "getUpdates"),
            params={
                "offset": offset,
                "limit": _MAX_UPDATES_PER_DRAIN,
                "timeout": 0,
                # Ask for callbacks only, so ordinary chatter in the alerts group
                # stops being buffered at all and cannot crowd presses out of a
                # single drain's limit.
                "allowed_updates": '["callback_query"]',
            },
            timeout=_GET_UPDATES_TIMEOUT,
        )
        resp.raise_for_status()
        updates = (resp.json() or {}).get("result") or []
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        counters.bump("feedback_drain_failed")
        stats["error"] = f"getUpdates HTTP {code}"
        stats["fatal"] = code in _FATAL_STATUSES
        log = logger.error if stats["fatal"] else logger.warning
        log("Feedback drain: getUpdates returned %s%s", code,
            " — presses are being LOST until this is fixed" if stats["fatal"] else "")
        return stats
    except Exception as e:
        counters.bump("feedback_drain_failed")
        logger.warning("Feedback drain: getUpdates failed: %s", e)
        stats["error"] = f"{type(e).__name__}: {e}"
        return stats

    stats["fetched"] = len(updates)
    highest = 0

    for upd in updates:
        try:
            update_id = int(upd.get("update_id"))
        except (TypeError, ValueError):
            continue
        highest = max(highest, update_id)

        cq = upd.get("callback_query")
        if not cq:
            stats["ignored"] += 1
            continue

        parsed = parse_feedback_callback(cq.get("data"))
        if not parsed:
            # Includes the inert "fb:done" button on an already-answered card.
            stats["ignored"] += 1
            counters.bump("feedback_callback_unparsed")
            continue

        try:
            if _record(db_conn, update_id, parsed, cq):
                stats["recorded"] += 1
                counters.bump("feedback_recorded")
        except Exception:
            counters.bump("feedback_write_failed")
            logger.exception("Feedback: failed to record press on %s",
                             parsed["event_id"][:8])
            # Do NOT advance past a press we could not store — leave it in the window
            # for the next drain rather than losing the one signal we have.
            highest = update_id - 1
            break

        _confirm(bot_token, cq, parsed["verdict"])

    # Cursor last, and only over rows that are already committed (autocommit pool).
    if highest > 0:
        try:
            _write_cursor(db_conn, highest)
        except Exception:
            counters.bump("feedback_cursor_failed")
            logger.exception("Feedback: cursor advance failed; window will be re-read")

    if stats["recorded"] or stats["fetched"]:
        logger.info("Feedback drain: %d updates, %d recorded, %d ignored",
                    stats["fetched"], stats["recorded"], stats["ignored"])
    return stats
