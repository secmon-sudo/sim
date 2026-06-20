import pytest
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from src.services.telegram_notifier import send_telegram_alert
from src.pipeline.pass_a_ingest import (
    translate_to_english_if_needed,
    google_translate,
    build_search_queries,
    check_domain_penalty
)

# 1. Test translation helpers
def test_translate_to_english_if_needed_no_translation():
    text = "Airplane crash reported at airport."
    assert translate_to_english_if_needed(text) == text

@patch("src.pipeline.pass_a_ingest.google_translate")
def test_translate_to_english_if_needed_arabic(mock_translate):
    mock_translate.return_value = "Red Sea blockade"
    text = "حصار البحر الأحمر"
    res = translate_to_english_if_needed(text)
    assert res == "Red Sea blockade"
    mock_translate.assert_called_once_with(text, target="en")

@patch("src.pipeline.pass_a_ingest.google_translate")
def test_translate_to_english_if_needed_hebrew(mock_translate):
    mock_translate.return_value = "Attack at airport"
    text = "פיגוע בשדה התעופה"
    res = translate_to_english_if_needed(text)
    assert res == "Attack at airport"
    mock_translate.assert_called_once_with(text, target="en")


# 2. Test check_domain_penalty whitelist and minimum events
def test_check_domain_penalty_whitelist():
    # Whitelisted domain should return 0.0 penalty
    db = MagicMock()
    assert check_domain_penalty(db, "reuters.com") == 0.0
    # No SQL queries should have been run for whitelisted domain
    db.execute.assert_not_called()

def test_check_domain_penalty_under_5_events():
    db = MagicMock()
    # Mock return: penalty_score=0.9, total_events=4 (less than 5 threshold)
    db.execute().fetchone.return_value = (0.9, 4)
    
    assert check_domain_penalty(db, "unreliable-blog.com") == 0.0

def test_check_domain_penalty_over_5_events():
    db = MagicMock()
    # Mock return: penalty_score=0.9, total_events=5 (reaches threshold)
    db.execute().fetchone.return_value = (0.9, 5)
    
    assert check_domain_penalty(db, "unreliable-blog.com") == 0.9


# 3. Test build_search_queries with active storylines (sliding activity window)
def test_build_search_queries_with_active_storylines():
    db = MagicMock()
    now = datetime.now(timezone.utc)
    
    # Mock db return rows: (storyline_hint, last_update, max_severity)
    db.execute().fetchall.return_value = [
        # 1. Critical event (severity=85) updated 5 days ago (window=7d) -> INCLUDED
        ("Red Sea Strike Jun8", now - timedelta(days=5), 85),
        # 2. Alert event (severity=65) updated 2 days ago (window=3d) -> INCLUDED
        ("Tel Aviv Drone Jun9", now - timedelta(days=2), 65),
        # 3. Watch event (severity=45) updated 40 hours ago (window=36h) -> EXCLUDED
        ("Cairo Airport Riot Jun9", now - timedelta(hours=40), 45),
        # 4. Critical event updated 8 days ago -> EXCLUDED
        ("London Security Breach Jun1", now - timedelta(days=8), 90)
    ]
    
    queries = build_search_queries(db)
    query_texts = [q["query"] for q in queries]
    
    # Cleaned hints should be present: date suffix " JunX" stripped
    assert "Red Sea Strike" in query_texts
    assert "Tel Aviv Drone" in query_texts
    
    # Excluded ones should not be present
    assert "Cairo Airport Riot" not in query_texts
    assert "London Security Breach" not in query_texts


# 4. Test Telegram notifier formatting
@patch("src.services.telegram_notifier._post_telegram")
@patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "mock_token", "TELEGRAM_ALERTS_CHAT_ID": "mock_chat"})
def test_telegram_notifier_premium_formatting(mock_post):
    mock_post.return_value = MagicMock()
    
    event = {
        "id": "event_123",
        "severity_score": 85,
        "event_type": "mass_casualty",
        "alert_tier": "CRITICAL",
        "anchor_name_norm": "JFK Airport",
        "country_iso": "US",
        "source_title": "Active Shooter at JFK Airport Terminal 4",
        "source_url": "https://reuters.com/jfk-shooter",
        "occurred_at_est": datetime(2026, 6, 9, 14, 30, tzinfo=timezone.utc),
        "storyline_hint": "JFK Airport Shooting Jun9"
    }
    
    res = send_telegram_alert(event)
    assert res is True
    
    # Verify mock call parameters
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    payload = kwargs["payload"]
    text = payload["text"]
    
    # Check premium layout elements
    assert "CRITICAL" in text
    assert "JFK Airport" in text
    assert "mass_casualty" in text
    assert "2026-06-09 14:30" in text
    assert "━━━━━━━━━━━━━━━━━━━━━" in text
    assert "<code>" in text
    assert "🔴" in text


@patch("src.services.telegram_notifier._post_telegram")
@patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "mock_token", "TELEGRAM_ALERTS_CHAT_ID": "mock_chat"})
def test_telegram_notifier_quiet_hours_formatting(mock_post):
    mock_post.return_value = MagicMock()
    
    # Test 1: New Location only
    event = {
        "id": "event_123",
        "severity_score": 85,
        "event_type": "mass_casualty",
        "alert_tier": "CRITICAL",
        "anchor_name_norm": "JFK Airport",
        "country_iso": "US",
        "source_title": "Active Shooter at JFK Airport Terminal 4",
        "source_url": "https://reuters.com/jfk-shooter",
        "occurred_at_est": datetime(2026, 6, 9, 14, 30, tzinfo=timezone.utc),
        "location_quiet_24h": True,
        "country_quiet_24h": False
    }
    
    res = send_telegram_alert(event)
    assert res is True
    _, kwargs = mock_post.call_args
    text = kwargs["payload"]["text"]
    assert "CRITICAL ALERT — 📍 NEW LOCATION" in text
    assert "⚠️ <b>[NEW LOCATION]</b>" in text

    # Test 2: New Country only
    mock_post.reset_mock()
    event["location_quiet_24h"] = False
    event["country_quiet_24h"] = True
    res = send_telegram_alert(event)
    assert res is True
    _, kwargs = mock_post.call_args
    text = kwargs["payload"]["text"]
    assert "CRITICAL ALERT — 🌍 NEW COUNTRY ACTIVITY" in text
    assert "⚠️ <b>[NEW COUNTRY ACTIVITY]</b>" in text

    # Test 3: Both
    mock_post.reset_mock()
    event["location_quiet_24h"] = True
    event["country_quiet_24h"] = True
    res = send_telegram_alert(event)
    assert res is True
    _, kwargs = mock_post.call_args
    text = kwargs["payload"]["text"]
    assert "CRITICAL ALERT — 🚨 NEW LOCATION & COUNTRY" in text
    assert "⚠️ <b>[NEW LOCATION & COUNTRY]</b>" in text


@patch("src.pipeline.pass_d_score.resolve_anchor_for_event")
@patch("src.pipeline.pass_d_score.send_telegram_alert")
def test_score_single_event_quiet_hours(mock_send_tg, mock_resolve_anchor):
    from src.pipeline.pass_d_score import score_single_event
    
    mock_resolve_anchor.return_value = {
        "norm": "JFK",
        "confidence": 0.9,
        "level": "HIGH",
        "czib_flag": False,
        "latitude": 40.64,
        "longitude": -73.78,
        "country_iso": "US"
    }
    
    db_conn = MagicMock()
    # Mock event data query
    cursor_event = MagicMock()
    cursor_event.fetchone.return_value = (
        "event_123", # id
        "bomb_threat", # event_type
        "JFK Airport", # anchor_name_raw
        "US", # country_iso
        '{"confidence": 0.9, "time_certainty": "same_day"}', # llm_parsed_output
        "JFK Shooting", # storyline_hint
        datetime(2026, 6, 9, 14, 30, tzinfo=timezone.utc), # occurred_at_est
        "Active Shooter at JFK Airport Terminal 4", # source_title
        "https://reuters.com/jfk-shooter", # source_url
        datetime(2026, 6, 9, 14, 35, tzinfo=timezone.utc), # ingested_at
        "reuters.com" # source_domain
    )
    
    # New-activity query returns a single row (country_cnt, location_cnt); 0/0 means
    # this is the first genuine security event for both → both "new" flags set.
    cursor_counts = MagicMock()
    cursor_counts.fetchone.return_value = (0, 0)
    
    # Mock event_type_catalog query
    cursor_cat = MagicMock()
    cursor_cat.fetchone.return_value = (80,)

    # Mock diversity query
    cursor_div = MagicMock()
    cursor_div.fetchone.return_value = (1,)
    
    # Mock alert suppression query
    cursor_supp = MagicMock()
    cursor_supp.fetchone.return_value = None
    
    def side_effect(query, params=None):
        if "FROM events WHERE id =" in query:
            return cursor_event
        elif "FROM event_type_catalog" in query:
            return cursor_cat
        elif "COUNT(DISTINCT source_domain)" in query:
            return cursor_div
        elif "FROM alert_suppression" in query:
            return cursor_supp
        elif "COUNT(*)" in query:
            return cursor_counts
        return MagicMock()
        
    db_conn.execute.side_effect = side_effect
    
    # Call score_single_event
    res = score_single_event(db_conn, "event_123", [])
    
    # Verify send_telegram_alert was called with our flags set to True
    mock_send_tg.assert_called_once()
    called_event = mock_send_tg.call_args[0][0]
    assert called_event["country_quiet_24h"] is True
    assert called_event["location_quiet_24h"] is True


