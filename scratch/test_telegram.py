
import os
import logging
from src.services.telegram_notifier import send_telegram_alert

logging.basicConfig(level=logging.INFO)

def test_telegram():
    print("Testing Telegram Notification...")
    
    # Mock event data
    test_event = {
        "id": "test-uuid-123",
        "source_title": "TEST ALERT KEMAL: Aviation Security Incident Simulation",
        "event_type": "security_incident",
        "anchor_name_norm": "LHR",
        "country_iso": "GB",
        "severity_score": 85,
        "system_confidence": 0.92,
        "alert_tier": "CRITICAL",
        "storyline_hint": "Simulated security breach at London Heathrow",
        "source_url": "https://example.com/test-incident"
    }
    
    # Check for env vars
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_ALERTS_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_ALERTS_CHAT_ID not set in environment.")
        return

    print(f"Using Bot Token: {bot_token[:10]}...")
    print(f"Using Chat ID: {chat_id}")
    
    success = send_telegram_alert(test_event)
    
    if success:
        print("SUCCESS: Telegram alert sent!")
    else:
        print("FAILED: Check logs for errors.")

if __name__ == "__main__":
    test_telegram()
