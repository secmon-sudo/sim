"""
SIM — Telegram Alert Notifier
Blueprint V20.1 §5

Sends formatted alert cards to a Telegram group.
"""

import html
import logging
import os

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

logger = logging.getLogger(__name__)

def _is_retryable_http_error(exception) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code == 429 or exception.response.status_code >= 500
    if isinstance(exception, httpx.RequestError):
        return True
    return False

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=2, max=20),
    retry=retry_if_exception(_is_retryable_http_error),
    reraise=True
)
def _post_telegram(api_url: str, payload: dict) -> httpx.Response:
    resp = httpx.post(api_url, json=payload, timeout=15.0)
    resp.raise_for_status()
    return resp

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

    tier = event.get("alert_tier") or "ALERT"

    emoji = TIER_EMOJIS.get(tier, "⚠️")

    # Escape all values for Telegram HTML parse_mode
    safe_title = html.escape(str(event.get("source_title") or "Unknown"))
    safe_type = html.escape(str(event.get("event_type") or "Unknown"))
    safe_anchor = html.escape(str(event.get("anchor_name_norm") or "Unknown"))
    safe_country = html.escape(str(event.get("country_iso") or ""))
    safe_hint = html.escape(str(event.get("storyline_hint") or ""))
    safe_url = html.escape(str(event.get("source_url") or ""), quote=True)

    location = f"{safe_anchor} ({safe_country})" if safe_country else safe_anchor
    severity = event.get("severity_score", 0)
    confidence = event.get("system_confidence", 0.0)

    # Format occurred_at_est
    occurred_at = event.get("occurred_at_est")
    if occurred_at:
        if hasattr(occurred_at, "strftime"):
            safe_time = occurred_at.strftime("%Y-%m-%d %H:%M")
        else:
            safe_time = str(occurred_at)
    else:
        safe_time = "Unknown"

    # Check quiet hours (last 24 hours) for country/location flags
    header_suffix = ""
    headline_prefix = ""
    is_new_location = event.get("location_quiet_24h", False)
    is_new_country = event.get("country_quiet_24h", False)

    if is_new_location and is_new_country:
        header_suffix = " — 🚨 NEW LOCATION & COUNTRY"
        headline_prefix = "⚠️ <b>[NEW LOCATION & COUNTRY]</b> "
    elif is_new_location:
        header_suffix = " — 📍 NEW LOCATION"
        headline_prefix = "⚠️ <b>[NEW LOCATION]</b> "
    elif is_new_country:
        header_suffix = " — 🌍 NEW COUNTRY ACTIVITY"
        headline_prefix = "⚠️ <b>[NEW COUNTRY ACTIVITY]</b> "

    # Format message with HTML
    message = f"<b>{emoji} {html.escape(tier)} ALERT{header_suffix}</b>\n"
    message += "━━━━━━━━━━━━━━━━━━━━━\n"
    message += f"📰 <b>Headline:</b> {headline_prefix}{safe_title}\n\n"
    message += f"📍 <b>Location:</b> <code>{location}</code>\n"
    message += f"📂 <b>Event Type:</b> <code>{safe_type}</code>\n"
    message += f"⚡ <b>Severity:</b> <code>{severity}/100</code>\n"
    message += f"🛡️ <b>Confidence:</b> <code>{confidence:.2f}</code>\n"
    message += f"🕰️ <b>Incident Time:</b> <code>{safe_time} (EST)</code>\n"
    if safe_hint:
        message += f"🧵 <b>Storyline Hint:</b> <code>#{safe_hint}</code>\n"
    message += "━━━━━━━━━━━━━━━━━━━━━\n"
    if safe_url:
        message += f"🔗 <a href='{safe_url}'>Read Full Report</a>"

    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    try:
        _post_telegram(
            api_url,
            payload={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": False
            }
        )
        logger.info("Sent Telegram alert for event %s", event.get("id", ""))
        return True
    except httpx.HTTPError as e:
        logger.error("Failed to send Telegram alert: %s", e)
        return False
    except Exception:
        logger.exception("Unexpected error sending Telegram alert")
        return False
