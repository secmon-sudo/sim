"""
SIM — Operational (health) notifier.

A deliberately dependency-light, best-effort channel for telling a human that the
PIPELINE ITSELF is in trouble — distinct from the event alert cards in
`telegram_notifier`. It is called on the failure paths (orchestrator caught an
exception, a pass returned an error stat) and by the standalone dead-man's-switch
check when the pipeline has not produced telemetry recently.

Design rules:
  - Never raises. The caller is usually already handling a failure; a broken ops
    ping must not mask the original problem.
  - No retry/backoff machinery. If the one POST fails, we log and move on — a
    health ping that hangs is worse than one that is occasionally missed.
  - Posts to TELEGRAM_OPS_CHAT_ID, falling back to the alert channel only when
    that is unset. This module used to say a separate ops channel "isn't worth
    the config", and that was true while ops pings meant "the pipeline crashed" —
    a few a month. On 2026-09-04 they became regular: hourly output-health checks
    and a weekly slot regression. The first health page duly landed in the
    channel real users read, in front of them, saying "minimax-m2.7" and
    "llm_contract_rejected=1". Engineering diagnostics and user-facing alerts are
    different audiences and now have different chats.
"""

import html
import logging
import os

import httpx

logger = logging.getLogger(__name__)


def send_ops_alert(text: str, *, title: str = "SIM PIPELINE HEALTH") -> bool:
    """Post a health/ops message to Telegram. Best-effort; returns success as bool.

    `text` is treated as plain text and HTML-escaped; `title` becomes a bold header.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    # The fallback is deliberate rather than lazy: losing a page about a dead
    # pipeline is worse than showing one to a reader. But it warns every time,
    # because the fallback IS the problem it is protecting against.
    chat_id = os.environ.get("TELEGRAM_OPS_CHAT_ID")
    if not chat_id:
        chat_id = os.environ.get("TELEGRAM_ALERTS_CHAT_ID")
        if chat_id:
            logger.warning(
                "TELEGRAM_OPS_CHAT_ID is not set — sending an engineering page to "
                "the USER-FACING alert channel. Set it to a private ops chat."
            )
    if not bot_token or not chat_id:
        logger.warning("Ops alert skipped: missing TELEGRAM_BOT_TOKEN or a chat id")
        return False

    message = f"🛠️ <b>{html.escape(title)}</b>\n" + html.escape(text)
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": message[:4000],  # Telegram hard-caps at 4096; leave headroom.
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        # Best-effort: log and swallow so we never mask the failure we're reporting.
        logger.error("Failed to send ops alert: %s", e)
        return False
