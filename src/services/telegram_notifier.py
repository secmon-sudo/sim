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

# Human-facing tier labels — the raw tier name is an internal enum.
TIER_LABELS = {
    "CRITICAL": "CRITICAL",
    "ALERT": "ALERT",
    "WATCH": "WATCH",
}

_DIVIDER = "━━━━━━━━━━━━━━━━━━━━━"


def _humanize(slug: str) -> str:
    """snake_case event type → 'Title Case' words for display."""
    words = str(slug or "").replace("_", " ").replace("-", " ").split()
    return " ".join(w.capitalize() for w in words) or "Unknown"


def _severity_bar(sev: int, width: int = 10) -> str:
    """A 0–100 severity as a compact filled/empty block meter."""
    try:
        filled = round(max(0, min(100, int(sev))) / 100 * width)
    except (TypeError, ValueError):
        filled = 0
    return "█" * filled + "░" * (width - filled)


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
    tier_label = TIER_LABELS.get(tier, tier)

    # Escape all values for Telegram HTML parse_mode
    safe_title = html.escape(str(event.get("source_title") or "Unknown"))
    safe_type = html.escape(_humanize(event.get("event_type")))
    safe_anchor = html.escape(str(event.get("anchor_name_norm") or "Unknown"))
    safe_country = html.escape(str(event.get("country_iso") or ""))
    safe_hint = html.escape(str(event.get("storyline_hint") or "").lstrip("#"))
    safe_url = html.escape(str(event.get("source_url") or ""), quote=True)

    location = f"{safe_anchor} · {safe_country}" if safe_country else safe_anchor
    severity = event.get("severity_score", 0)
    confidence = event.get("system_confidence", 0.0)
    try:
        conf_pct = f"{float(confidence) * 100:.0f}%"
    except (TypeError, ValueError):
        conf_pct = "—"

    # Format occurred_at_est
    occurred_at = event.get("occurred_at_est")
    if occurred_at:
        stamp = (
            occurred_at.strftime("%Y-%m-%d %H:%M")
            if hasattr(occurred_at, "strftime")
            else str(occurred_at)
        )
        if event.get("occurred_at_is_fallback"):
            # The timestamp is when WE ingested the report, not when the incident
            # happened — label it so it doesn't read as a confirmed incident time.
            time_line = f"🕰️ Reported {stamp} EST · incident time unknown"
        else:
            time_line = f"🕰️ {stamp} EST"
    else:
        time_line = "🕰️ Incident time unknown"

    # Escalation chip — a storyline that already paged at a lower tier just crossed into
    # a higher one. Shown first because "this got worse" is the most decision-relevant cue.
    escalated_from = event.get("escalation_from")
    if escalated_from:
        badge = f"⬆️ <b>Escalated {html.escape(str(escalated_from))} → {html.escape(tier_label)}</b>\n"
    else:
        badge = ""

    # First-seen activity badge — one chip line under the header (no duplicate prefix).
    is_new_location = event.get("location_quiet_24h", False)
    is_new_country = event.get("country_quiet_24h", False)
    if is_new_location and is_new_country:
        badge += "🆕 <i>First activity at this location & country (24h)</i>\n"
    elif is_new_location:
        badge += "🆕 <i>First activity at this location (24h)</i>\n"
    elif is_new_country:
        badge += "🌍 <i>First activity in this country (24h)</i>\n"

    # Modern alert card: prominent headline, at-a-glance severity meter, compact metadata.
    message = f"{emoji} <b>{html.escape(tier_label)}</b> · {safe_type}\n"
    message += badge
    message += f"{_DIVIDER}\n"
    message += f"<b>{safe_title}</b>\n\n"
    message += f"📍 {location}\n"
    message += f"⚡ {_severity_bar(severity)}  {severity}/100  ·  🛡️ {conf_pct}\n"
    message += f"{time_line}\n"
    if safe_hint:
        message += f"🧵 <code>{safe_hint}</code>\n"
    message += _DIVIDER
    if safe_url:
        message += f"\n🔗 <a href='{safe_url}'>Open source report ↗</a>"

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


def send_storyline_closure(peak_tier: str, label: str, quiet_hours: float) -> bool:
    """Emit a single 'storyline quiet' note when an alerted storyline stops producing
    activity — the counterpart to the escalation cue, so a thread has a clear close."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_ALERTS_CHAT_ID")
    if not bot_token or not chat_id:
        logger.warning("Storyline closure skipped: missing Telegram credentials")
        return False

    safe_label = html.escape(str(label or "storyline"))
    safe_peak = html.escape(str(peak_tier or "—"))
    try:
        hours = f"{float(quiet_hours):.0f}"
    except (TypeError, ValueError):
        hours = str(quiet_hours)

    message = (
        "🟢 <b>STORYLINE QUIET</b>\n"
        f"{_DIVIDER}\n"
        f"<b>{safe_label}</b>\n\n"
        f"No new activity for {hours}h · peaked at {safe_peak}.\n"
        f"{_DIVIDER}"
    )
    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        _post_telegram(
            api_url,
            payload={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
        )
        return True
    except Exception as e:
        logger.error("Failed to send storyline closure: %s", e)
        return False
