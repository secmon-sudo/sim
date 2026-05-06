"""
SIM — Telegram Alert Notifier
Blueprint V20.1 §5

Sends formatted alert cards to a Telegram group.
"""

import html
import logging
import os

import httpx

logger = logging.getLogger(__name__)

TIER_EMOJIS = {
    "CRITICAL": "🔴",
    "ALERT": "🟠",
    "WATCH": "🟡"
}


def send_telegram_alert(event: dict) -> bool:
    """
    Format and send an alert card to Telegram.
    Returns True if successful.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_ALERTS_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning("Telegram alert skipped: missing TELEGRAM_BOT_TOKEN or TELEGRAM_ALERTS_CHAT_ID")
        return False

    tier = event.get("alert_tier")
    if not tier:
        return False

    emoji = TIER_EMOJIS.get(tier, "⚠️")

    # Escape all values for Telegram HTML parse_mode
    safe_title = html.escape(str(event.get("source_title") or "Unknown"))
    safe_type = html.escape(str(event.get("event_type") or "Unknown"))
    safe_anchor = html.escape(str(event.get("anchor_name_norm") or "Unknown"))
    safe_country = html.escape(str(event.get("country_iso") or ""))
    safe_hint = html.escape(str(event.get("storyline_hint") or ""))
    safe_url = str(event.get("source_url") or "")

    location = f"{safe_anchor} ({safe_country})" if safe_country else safe_anchor
    severity = event.get("severity_score", 0)
    confidence = event.get("system_confidence", 0.0)

    # Format message with HTML
    message = f"<b>{emoji} {html.escape(tier)} ALERT</b>\n\n"
    message += f"<b>Title:</b> {safe_title}\n"
    message += f"<b>Type:</b> {safe_type}\n"
    message += f"<b>Location:</b> {location}\n"
    message += f"<b>Severity:</b> {severity}/100\n"
    message += f"<b>Confidence:</b> {confidence:.2f}\n"

    if safe_hint:
        message += f"\n<b>Hint:</b> <i>{safe_hint}</i>\n"

    if safe_url:
        message += f"\n🔗 <a href='{safe_url}'>Read Source</a>"

    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    try:
        resp = httpx.post(
            api_url,
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": False
            },
            timeout=10.0
        )
        resp.raise_for_status()
        logger.info("Sent Telegram alert for event %s", event.get("id", ""))
        return True
    except httpx.HTTPError as e:
        logger.error("Failed to send Telegram alert: %s", e)
        return False
    except Exception:
        logger.exception("Unexpected error sending Telegram alert")
        return False
