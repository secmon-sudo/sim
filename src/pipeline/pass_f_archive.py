"""
SIM — Pass F: Cold Storage & Archive
Blueprint V20.1 §7

Archives events older than 90 days with no active storyline.
Saves as JSONL, uploads to Telegram, and deletes from DB on success.
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from io import BytesIO

import httpx

logger = logging.getLogger(__name__)

ARCHIVE_DAYS_THRESHOLD = 90
BATCH_SIZE = 500


def get_archivable_events(db_conn) -> list[dict]:
    """
    Selects events > 90 days old WHERE their storyline has NO recent events.
    """
    query = """
        SELECT id, source_url, source_title, canonical_text, event_type,
               alert_tier, severity_score, anchor_name_norm, country_iso,
               occurred_at_est, ingested_at, llm_parsed_output, storyline_id
        FROM events e
        WHERE e.status = 'reconciled'
          AND e.occurred_at_est < NOW() - INTERVAL '%s days'
          AND NOT EXISTS (
              -- Ensure no recent siblings in the same storyline
              SELECT 1 FROM events sibling
              WHERE sibling.storyline_id = e.storyline_id
                AND sibling.storyline_id IS NOT NULL
                AND sibling.occurred_at_est >= NOW() - INTERVAL '%s days'
          )
        ORDER BY e.occurred_at_est ASC
        LIMIT %s
    """
    try:
        rows = db_conn.execute(
            query,
            (ARCHIVE_DAYS_THRESHOLD, ARCHIVE_DAYS_THRESHOLD, BATCH_SIZE),
        ).fetchall()

        columns = [
            "id", "source_url", "source_title", "canonical_text", "event_type",
            "alert_tier", "severity_score", "anchor_name_norm", "country_iso",
            "occurred_at_est", "ingested_at", "llm_parsed_output", "storyline_id"
        ]
        
        events = []
        for row in rows:
            event = dict(zip(columns, row))
            # Serialize datetimes and jsonb for JSONL
            event["occurred_at_est"] = event["occurred_at_est"].isoformat() if event["occurred_at_est"] else None
            event["ingested_at"] = event["ingested_at"].isoformat() if event["ingested_at"] else None
            event["id"] = str(event["id"])
            if event["storyline_id"]:
                event["storyline_id"] = str(event["storyline_id"])
            if not isinstance(event["llm_parsed_output"], dict):
                try:
                    event["llm_parsed_output"] = json.loads(event["llm_parsed_output"] or "{}")
                except:
                    event["llm_parsed_output"] = {}
            events.append(event)
            
        return events
    except Exception:
        logger.exception("Failed to fetch archivable events")
        return []


def generate_jsonl_and_hash(events: list[dict]) -> tuple[bytes, str]:
    """Converts events to JSONL bytes and generates SHA-256 hash."""
    lines = [json.dumps(e, separators=(',', ':')) for e in events]
    content = "\n".join(lines).encode('utf-8')
    manifest_hash = hashlib.sha256(content).hexdigest()
    return content, manifest_hash


def upload_to_telegram(content: bytes, filename: str) -> dict | None:
    """Uploads file to Telegram via Bot API."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_ARCHIVE_CHAT_ID")
    
    if not bot_token or not chat_id:
        logger.warning("Telegram credentials missing, skipping upload")
        return None

    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    
    files = {
        'document': (filename, BytesIO(content), 'application/jsonl')
    }
    data = {
        'chat_id': chat_id,
        'caption': f"📦 SIM Archive Payload | {filename} | {len(content) // 1024} KB"
    }

    try:
        # httpx post with 60s timeout for large uploads
        response = httpx.post(url, data=data, files=files, timeout=60.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        logger.exception("Telegram upload failed")
        return None


def run_pass_f(db_conn) -> dict:
    """
    Execute Pass F: Cold Storage & Archive
    
    1. Select archivable events
    2. Convert to JSONL & Hash
    3. Upload to Telegram
    4. Delete from DB & save manifest
    """
    stats = {
        "events_archived": 0,
        "manifest_hash": None,
        "telegram_message_id": None,
        "error": None
    }

    # 1. Select
    events = get_archivable_events(db_conn)
    if not events:
        logger.info("Pass F: No events to archive.")
        return stats
        
    logger.info("Pass F: Found %d events to archive.", len(events))

    # 2. JSONL & Hash
    content, manifest_hash = generate_jsonl_and_hash(events)
    filename = f"sim_archive_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{len(events)}ev.jsonl"
    stats["manifest_hash"] = manifest_hash

    # 3. Upload to Telegram
    tg_response = upload_to_telegram(content, filename)
    
    if not tg_response or not tg_response.get("ok"):
        stats["error"] = "Telegram upload failed"
        logger.error("Pass F failed: %s", tg_response)
        return stats
        
    stats["telegram_message_id"] = tg_response.get("result", {}).get("message_id")
    logger.info("Pass F: Uploaded to Telegram message_id=%s", stats["telegram_message_id"])

    # 4. DELETE from DB & Save Manifest
    event_ids = [e["id"] for e in events]
    
    try:
        # Save manifest telemetry first
        db_conn.execute(
            "INSERT INTO system_telemetry(event_type, value_json) VALUES ('archive_manifest', %s)",
            (json.dumps({
                "manifest_hash": manifest_hash,
                "event_count": len(events),
                "filename": filename,
                "telegram_message_id": stats["telegram_message_id"],
                "archived_event_ids": event_ids
            }),),
        )
        
        # Then delete events
        db_conn.execute(
            "DELETE FROM events WHERE id = ANY(%s)",
            (event_ids,)
        )
        
        # 5. DB Maintenance: Delete old telemetry
        db_conn.execute(
            "DELETE FROM system_telemetry WHERE timestamp < NOW() - INTERVAL '90 days'"
        )
        
        db_conn.commit()
        stats["events_archived"] = len(events)
        logger.info("Pass F: Successfully archived and deleted %d events.", len(events))
        
    except Exception as e:
        db_conn.rollback()
        stats["error"] = f"DB Delete/Manifest error: {e}"
        logger.exception("Pass F DB Error")

    return stats
