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
  - Posts to the alert channel. There is one chat, on purpose.

    This module carried a TELEGRAM_OPS_CHAT_ID for three days. It was added on
    2026-09-04, when ops pings went from "the pipeline crashed" a few times a
    month to hourly output-health checks and a weekly slot regression, and the
    first health page landed in the channel real users read saying "minimax-m2.7"
    and "llm_contract_rejected=1". The argument was that engineering diagnostics
    and user-facing alerts are different audiences.

    The secret was never set, so for those three days every page went to the alert
    channel anyway — with a warning saying it should not have. On 2026-09-07 the
    operator decided not to run a second chat. That makes the alert channel the
    ops channel by decision rather than by omission, and a warning about a settled
    decision is noise that teaches people to skip warnings.
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
    chat_id = os.environ.get("TELEGRAM_ALERTS_CHAT_ID")
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
