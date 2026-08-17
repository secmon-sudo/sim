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
    # Provenance of published_at (migration 021). Exported so the archive records
    # WHICH dates were the publisher's and which were an aggregator's crawl stamp —
    # a distinction that cannot be recovered from the JSONL after the fact.
    "date_verified",
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

# Retention for status='archived' rows — the noise the classifier rejected and the
# prescreen's zero-signal drops. Nothing removed them before: get_archivable_events
# reads status='reconciled' only, so they were neither exported to cold storage nor
# deleted, and they were 26,484 of 55,226 rows (48%) on 2026-08-17 with the database
# at 367 MB of the 500 MB free tier and growing 4.4 MB/day — about 31 days of runway.
#
# 30 days is set against the widest consumer that reads archived rows, which is the
# weekly forecast at 7 days (weekly_forecast fetches on occurred_at_est with no status
# filter). Pass A's content dedup looks back max_article_age_days=2 and the SITREP
# window is 24 hours; storyline linking excludes archived entirely. Four times the
# margin of the widest reader, and it survives two missed weekly runs.
ARCHIVED_RETENTION_DAYS = _SETTINGS.get("cold_storage", {}).get(
    "archived_retention_days", 30)

# Retention for the per-call LLM telemetry rows. Measured 2026-08-17: system_telemetry
# was 65 MB, the second largest table in a 367 MB database on a 500 MB tier, and 50 MB
# of it was a single event_type — 44,680 'llm_call' rows kept since 9 May, one per LLM
# call, growing ~1.2 MB/day. Nothing reads them: log_llm_telemetry in core.llm_client
# is the only reference to the type in the whole repo, and it only writes.
#
# Every other telemetry type is an aggregate — pipeline_run, pass_a..pass_f,
# archive_manifest, geo_distribution — and all of them together are 3 MB, so they are
# deliberately kept forever and only this one type is aged out.
#
# 14 days because that is the window the model and quota investigations in this project
# actually use ("the 14 days to <date>"), and per-call latency is only ever read while
# chasing a live regression.
TELEMETRY_RETENTION_DAYS = _SETTINGS.get("cold_storage", {}).get(
    "telemetry_call_retention_days", 14)

# Larger than BATCH_SIZE: this table is not on the read path of any pass and both
# columns the delete keys on are indexed, so the lock it takes is short even at this
# size — and the first run has ~35,000 rows of backlog to work through, which at 500 a
# run would take a week and a half of runs.
TELEMETRY_BATCH_SIZE = 5000


def get_archivable_events(db_conn) -> list[dict]:
    """
    Selects events > 90 days old WHERE their storyline has NO recent events,
    OR events with NULL occurred_at_est older than archive_null_occurred_after_days.
    """
    query = """
        SELECT id, source_url, source_title, canonical_text, event_type,
               alert_tier, severity_score, anchor_name_norm, country_iso,
               occurred_at_est, ingested_at, llm_parsed_output, storyline_id,
               date_verified
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


def purge_expired_archived(db_conn) -> int:
    """Delete archived noise older than ARCHIVED_RETENTION_DAYS. Returns rows removed.

    Deliberately narrow on three axes:

      * status='archived' only. 'reconciled' rows are the corpus and leave through
        the cold-storage export on their own schedule.
      * alert_tier IS NULL. An archived row that once paged is the only surviving
        record that the card was sent, and the alert telemetry counts it.
      * ingested_at, not occurred_at_est. The latter is the classifier's estimate and
        is wrong often enough — measured 60% fabricated on 2026-08-13 — that ageing on
        it would delete rows that arrived yesterday.

    Batched so one run can never hold a long write lock on events; the remainder is
    picked up by the next run, which is the point of doing this every pass rather than
    as an occasional cleanup.
    """
    if not ARCHIVED_RETENTION_DAYS:
        return 0
    try:
        result = db_conn.execute(
            """DELETE FROM events
                WHERE id IN (
                    SELECT id FROM events
                     WHERE status = 'archived'
                       AND alert_tier IS NULL
                       AND ingested_at < NOW() - (%s * INTERVAL '1 day')
                     LIMIT %s
                )""",
            (ARCHIVED_RETENTION_DAYS, BATCH_SIZE),
        )
        removed = result.rowcount or 0
        db_conn.commit()
        if removed:
            logger.info("Pass F: purged %d archived events older than %d days",
                        removed, ARCHIVED_RETENTION_DAYS)
        return removed
    except Exception:
        try:
            db_conn.rollback()
        except Exception:
            pass
        # Retention is housekeeping — it must never be the reason a run reports failure.
        logger.exception("Archived-event purge failed")
        return 0


def purge_expired_telemetry(db_conn) -> int:
    """Delete per-call LLM telemetry older than TELEMETRY_RETENTION_DAYS.

    Narrow on event_type for the reason given at TELEMETRY_RETENTION_DAYS: the
    aggregates in this table are the run history and cost nothing to keep, while
    'llm_call' is a write-only debug trail that is 77% of the table.
    """
    if not TELEMETRY_RETENTION_DAYS:
        return 0
    try:
        result = db_conn.execute(
            """DELETE FROM system_telemetry
                WHERE id IN (
                    SELECT id FROM system_telemetry
                     WHERE event_type = 'llm_call'
                       AND timestamp < NOW() - (%s * INTERVAL '1 day')
                     LIMIT %s
                )""",
            (TELEMETRY_RETENTION_DAYS, TELEMETRY_BATCH_SIZE),
        )
        removed = result.rowcount or 0
        db_conn.commit()
        if removed:
            logger.info("Pass F: purged %d llm_call telemetry rows older than %d days",
                        removed, TELEMETRY_RETENTION_DAYS)
        return removed
    except Exception:
        try:
            db_conn.rollback()
        except Exception:
            pass
        # Housekeeping, like the archived purge — never a reason to fail a run.
        logger.exception("Telemetry purge failed")
        return 0


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
        "archived_purged": 0,
        "telemetry_purged": 0,
        "manifest_hash": None,
        "telegram_message_id": None,
        "error": None
    }

    # 0. Retention, BEFORE the early return below. The purges have nothing to do with
    #    whether there is anything to export, and putting them after that return would
    #    skip them on exactly the quiet runs where they are cheapest to do.
    stats["archived_purged"] = purge_expired_archived(db_conn)
    stats["telemetry_purged"] = purge_expired_telemetry(db_conn)

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
        # Manifest + deletes must land atomically (conn is autocommit): recording
        # the manifest without the deletes re-archives the same events next run;
        # deleting without the manifest loses the only pointer to the archive.
        with db_conn.transaction():
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
               occurred_at_est, ingested_at, llm_parsed_output, storyline_id,
               date_verified
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
