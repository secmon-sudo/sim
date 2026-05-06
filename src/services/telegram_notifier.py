"""
SIM — Telegram Alert Notifier
Blueprint V20.1 §5

Sends formatted alert cards to a Telegram group.
"""

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
    # Tries ALERTS_CHAT_ID first, fallback to ARCHIVE_CHAT_ID
    chat_id = os.environ.get("TELEGRAM_ALERTS_CHAT_ID") or os.environ.get("TELEGRAM_ARCHIVE_CHAT_ID")
    
    if not bot_token or not chat_id:
        logger.warning("Telegram alert skipped: missing credentials or chat ID")
        return False

    tier = event.get("alert_tier")
    if not tier:
        return False
        
    emoji = TIER_EMOJIS.get(tier, "⚠️")
    
    # Format message with HTML
    message = f"<b>{emoji} {tier} ALERT</b>\n\n"
    message += f"<b>Title:</b> {event.get('source_title', 'Unknown')}\n"
    message += f"<b>Type:</b> {event.get('event_type', 'Unknown')}\n"
    
    anchor = event.get('anchor_name_norm') or 'Unknown'
    country = event.get('country_iso') or ''
    location = f"{anchor} ({country})" if country else anchor
    message += f"<b>Location:</b> {location}\n"
    
    message += f"<b>Severity:</b> {event.get('severity_score', 0)}/100\n"
    message += f"<b>Confidence:</b> {event.get('system_confidence', 0.0):.2f}\n"
    
    storyline_hint = event.get("storyline_hint")
    if storyline_hint:
        message += f"\n<b>Hint:</b> <i>{storyline_hint}</i>\n"
        
    url = event.get('source_url')
    if url:
        message += f"\n🔗 <a href='{url}'>Read Source</a>"

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
        logger.info("Sent Telegram alert for event %s", event.get('id', ''))
        return True
    except httpx.HTTPError as e:
        logger.error("Failed to send Telegram alert: %s", e)
        return False
    except Exception:
        logger.exception("Unexpected error sending Telegram alert")
        return False
