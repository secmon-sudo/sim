"""
SIM — Pass F: Cold Storage & Archive
Blueprint V20.1 §7

Archives events older than 90 days with no active storyline,
OR events with NULL occurred_at_est older than archive_null_occurred_after_days.
Saves as JSONL, uploads to Telegram, and deletes from DB on success.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import httpx
import boto3
from botocore.config import Config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

logger = logging.getLogger(__name__)

def _is_retryable_http_error(exception) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code == 429 or exception.response.status_code >= 500
    if isinstance(exception, httpx.RequestError):
        return True
    return False

ARCHIVE_DAYS_THRESHOLD = 90
BATCH_SIZE = 500

# Column set shared by the cold-storage archive and the per-run snapshot export.
_EVENT_COLUMNS = [
    "id", "source_url", "source_title", "canonical_text", "event_type",
    "alert_tier", "severity_score", "anchor_name_norm", "country_iso",
    "occurred_at_est", "ingested_at", "llm_parsed_output", "storyline_id",
]


def _rows_to_event_dicts(rows) -> list[dict]:
    """Serialize DB rows (in _EVENT_COLUMNS order) into JSONL-ready dicts."""
    events = []
    for row in rows:
        event = dict(zip(_EVENT_COLUMNS, row))
        event["occurred_at_est"] = event["occurred_at_est"].isoformat() if event["occurred_at_est"] else None
        event["ingested_at"] = event["ingested_at"].isoformat() if event["ingested_at"] else None
        event["id"] = str(event["id"])
        if event["storyline_id"]:
            event["storyline_id"] = str(event["storyline_id"])
        if isinstance(event["llm_parsed_output"], str):
            try:
                event["llm_parsed_output"] = json.loads(event["llm_parsed_output"] or "{}")
            except Exception:
                event["llm_parsed_output"] = {}
        elif event["llm_parsed_output"] is None:
            event["llm_parsed_output"] = {}
        events.append(event)
    return events

# Load settings
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as f:
    _SETTINGS = json.load(f)

_ARCHIVE_NULL_AFTER_DAYS = _SETTINGS.get("cold_storage", {}).get("archive_null_occurred_after_days", 14)


def get_archivable_events(db_conn) -> list[dict]:
    """
    Selects events > 90 days old WHERE their storyline has NO recent events,
    OR events with NULL occurred_at_est older than archive_null_occurred_after_days.
    """
    query = """
        SELECT id, source_url, source_title, canonical_text, event_type,
               alert_tier, severity_score, anchor_name_norm, country_iso,
               occurred_at_est, ingested_at, llm_parsed_output, storyline_id
        FROM events e
        WHERE e.status = 'reconciled'
          AND (
              -- Case 1: Normal aging — occurred_at_est exists and is old
              (
                  e.occurred_at_est IS NOT NULL
                  AND e.occurred_at_est < NOW() - (%s * INTERVAL '1 day')
                  AND NOT EXISTS (
                      SELECT 1 FROM events sibling
                      WHERE sibling.storyline_id = e.storyline_id
                        AND sibling.storyline_id IS NOT NULL
                        AND sibling.occurred_at_est >= NOW() - (%s * INTERVAL '1 day')
                  )
              )
              OR
              -- Case 2: NULL occurred_at_est — archive after fallback days
              (
                  e.occurred_at_est IS NULL
                  AND e.ingested_at < NOW() - (%s * INTERVAL '1 day')
              )
          )
        ORDER BY e.occurred_at_est ASC NULLS LAST
        LIMIT %s
    """
    try:
        rows = db_conn.execute(
            query,
            (ARCHIVE_DAYS_THRESHOLD, ARCHIVE_DAYS_THRESHOLD, _ARCHIVE_NULL_AFTER_DAYS, BATCH_SIZE),
        ).fetchall()

        return _rows_to_event_dicts(rows)
    except Exception:
        logger.exception("Failed to fetch archivable events")
        return []


def generate_jsonl_and_hash(events: list[dict]) -> tuple[bytes, str]:
    """Converts events to JSONL bytes and generates SHA-256 hash."""
    lines = [json.dumps(e, separators=(',', ':')) for e in events]
    content = "\n".join(lines).encode('utf-8')
    manifest_hash = hashlib.sha256(content).hexdigest()
    return content, manifest_hash


def upload_to_cloudflare_r2(content: bytes, filename: str) -> bool:
    """Uploads file to Cloudflare R2 bucket via S3 compatible API."""
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME") or "sim-archive"

    if not all([account_id, access_key, secret_key]):
        logger.warning("Cloudflare R2 credentials missing, skipping R2 upload")
        return False

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                signature_version="s3v4",
                # Bound the upload so a stalled connection can't hang the whole
                # pipeline indefinitely (worst case ~3 * (10 + 60)s ≈ 3.5 min).
                connect_timeout=10,
                read_timeout=60,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
            region_name="auto",
        )
        
        s3.put_object(
            Bucket=bucket_name,
            Key=filename,
            Body=content,
            ContentType="application/jsonl"
        )
        return True
    except Exception:
        logger.exception("Cloudflare R2 upload failed")
        return False


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=30),
    retry=retry_if_exception(_is_retryable_http_error),
    reraise=True
)
def _post_telegram_document(url: str, data: dict, files: dict) -> httpx.Response:
    resp = httpx.post(url, data=data, files=files, timeout=60.0)
    resp.raise_for_status()
    return resp

def upload_to_telegram(content: bytes, filename: str) -> dict | None:
    """Uploads file to Telegram via Bot API."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_ARCHIVE_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning("Telegram credentials missing, skipping upload")
        return None

    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"

    files = {
        'document': (filename, content, 'application/jsonl')
    }
    data = {
        'chat_id': chat_id,
        'caption': f"📦 SIM Archive Payload | {filename} | {len(content) // 1024} KB"
    }

    try:
        response = _post_telegram_document(url, data=data, files=files)
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

    # 3. Upload to Archive Storages
    # Cloudflare R2 Upload
    r2_success = upload_to_cloudflare_r2(content, filename)
    if r2_success:
        logger.info("Pass F: Successfully uploaded %s to Cloudflare R2", filename)
        stats["r2_uploaded"] = True
    else:
        stats["r2_uploaded"] = False

    # Telegram Upload
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
                "r2_uploaded": stats.get("r2_uploaded", False),
                "telegram_message_id": stats.get("telegram_message_id"),
                "archived_event_ids": event_ids
            }),),
        )

        # Clear alert_suppression rows referencing these events (FK constraint)
        db_conn.execute(
            "DELETE FROM alert_suppression WHERE event_id = ANY(%s)",
            (event_ids,)
        )

        # Purge expired suppression entries (housekeeping)
        db_conn.execute(
            "DELETE FROM alert_suppression WHERE expires_at < NOW()"
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


def get_run_events(db_conn, run_started_at: datetime) -> list[dict]:
    """Select events that finished processing during this pipeline run.

    Keyed on updated_at (not ingested_at): when the Pass C backlog exceeds its
    per-run cap, events are classified 1-2 runs after ingestion, and a snapshot
    of "ingested this run" would export raw rows without storyline_id — which
    the storyboard worker silently drops. Requiring storyline_id also keeps
    prescreen-archived noise out of the export.
    """
    query = """
        SELECT id, source_url, source_title, canonical_text, event_type,
               alert_tier, severity_score, anchor_name_norm, country_iso,
               occurred_at_est, ingested_at, llm_parsed_output, storyline_id
        FROM events
        WHERE updated_at >= %s
          AND storyline_id IS NOT NULL
        ORDER BY severity_score DESC NULLS LAST, updated_at DESC
        LIMIT %s
    """
    try:
        rows = db_conn.execute(query, (run_started_at, BATCH_SIZE)).fetchall()
        return _rows_to_event_dicts(rows)
    except Exception:
        logger.exception("Failed to fetch run events for snapshot")
        return []


def run_run_snapshot(db_conn, run_started_at: datetime) -> dict:
    """Export this run's events as JSONL to R2 + Telegram (does NOT delete).

    Restores the per-run snapshot that ships alongside the alerts. Unlike Pass F
    (cold storage of >90-day events) this keeps the events in the DB.
    """
    stats = {"events": 0, "r2_uploaded": False, "telegram_message_id": None, "error": None}

    events = get_run_events(db_conn, run_started_at)
    if not events:
        logger.info("Run snapshot: no classified events this run, skipping.")
        return stats

    content, manifest_hash = generate_jsonl_and_hash(events)
    filename = f"sim_archive_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{len(events)}ev.jsonl"
    stats["events"] = len(events)
    stats["manifest_hash"] = manifest_hash

    stats["r2_uploaded"] = upload_to_cloudflare_r2(content, filename)
    if stats["r2_uploaded"]:
        logger.info("Run snapshot: uploaded %s to Cloudflare R2", filename)

    tg_response = upload_to_telegram(content, filename)
    if tg_response and tg_response.get("ok"):
        stats["telegram_message_id"] = tg_response.get("result", {}).get("message_id")
        logger.info("Run snapshot: uploaded to Telegram message_id=%s", stats["telegram_message_id"])
    elif tg_response is not None:
        stats["error"] = "Telegram snapshot upload failed"
        logger.error("Run snapshot Telegram upload failed: %s", tg_response)

    return stats
