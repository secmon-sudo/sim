"""End-to-end pipeline smoke test against a REAL PostgreSQL database.

Every production failure of Jul 2026 (lock waits, idle-in-transaction reaps,
connection loss, transaction state) was invisible to the unit suite because it
mocks the DB. This test runs the full orchestrator — migrations, seed, Pass A→F,
narrator — against a disposable Postgres with only the network edges stubbed
(RSS, LLM, Telegram, R2, CZIB), so DB/transaction behavior is exercised for real.

Guarded twice so it can NEVER touch a production database:
  - skipped unless SIM_SMOKE_DATABASE_URL is set (CI sets it to a localhost
    service container; it is never read from .env), and
  - refuses to run against anything that isn't localhost/127.0.0.1.

Run locally with e.g.:
  docker run --rm -d -p 5433:5432 -e POSTGRES_PASSWORD=pg postgres:16
  SIM_SMOKE_DATABASE_URL=postgresql://postgres:pg@localhost:5433/postgres \
      python -m pytest tests/test_pipeline_smoke.py -q
"""

import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

SMOKE_URL = os.environ.get("SIM_SMOKE_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not SMOKE_URL, reason="SIM_SMOKE_DATABASE_URL not set (CI-only smoke test)"
)

if SMOKE_URL and not re.search(r"@(localhost|127\.0\.0\.1)[:/]", SMOKE_URL):
    raise RuntimeError("SIM_SMOKE_DATABASE_URL must point at localhost — refusing to run")


def _fixture_items():
    """Two fresh, clearly security-relevant reports from distinct domains."""
    now = datetime.now(timezone.utc) - timedelta(hours=1)
    return [
        {
            "title": "Explosion reported near Testville International Airport",
            "description": "A large explosion struck a cargo area; two people were killed "
                           "and the airport suspended departures, officials said.",
            "link": "https://news-alpha.example.com/testville-airport-explosion",
            "pub_dt": now,
        },
        {
            "title": "Drone incursion halts flights at Testville airfield",
            "description": "Military officials said an unidentified drone forced a 40-minute "
                           "ground stop; air defense units were placed on alert.",
            "link": "https://wire-beta.example.org/testville-drone-incursion",
            "pub_dt": now,
        },
    ]


def _fake_call_llm(router, prompt, system_prompt=None, max_tokens=1024, json_mode=True):
    """Canned LLM: classification batches get valid JSON, prose gets a sentence."""
    if not json_mode:
        return {"content": "The situation at Testville airport developed over one day.",
                "provider": "fake", "account": "T", "model": "fake-model",
                "latency_ms": 1, "finish_reason": "stop", "response": {}}
    reports = len(re.findall(r"^REPORT \d+:", prompt, flags=re.M)) or 1
    results = [
        {
            "report": i,
            "relevance_score": 85,
            "event_type": "security_incident",
            "sub_type": None,
            "anchor_name": None,
            "country_iso": "XK",
            "storyline_hint": "testville airport attack",
            "time_certainty": "day",
            "relevance_reasoning": "explosion at an airport with casualties",
        }
        for i in range(1, reports + 1)
    ]
    return {"content": json.dumps({"results": results}),
            "provider": "fake", "account": "T", "model": "fake-model",
            "latency_ms": 1, "finish_reason": "stop", "response": {}}


@pytest.fixture()
def smoke_db(monkeypatch):
    """Point the pool at the disposable Postgres, run migrations + anchor seed."""
    monkeypatch.setenv("DATABASE_URL", SMOKE_URL)
    monkeypatch.setenv("SIM_MATURATION_WINDOW_HOURS", "0")
    # Ensure no real provider/notifier secrets leak in from the environment.
    for var in ("GROQ_API_KEY_A", "GROQ_API_KEY_B", "OPENROUTER_API_KEY_A",
                "OPENROUTER_API_KEY_B", "GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN",
                "R2_ACCOUNT_ID"):
        monkeypatch.delenv(var, raising=False)

    import psycopg
    conn = psycopg.connect(SMOKE_URL, autocommit=True)
    for sql_file in sorted(glob.glob("db/migrations/*.sql")):
        try:
            conn.execute(open(sql_file).read())
        except Exception as e:  # same skip-on-error semantics as the workflow
            print(f"migration {sql_file} skipped: {e}")
    # Start from a clean events table so assertions are deterministic.
    conn.execute("TRUNCATE events, alert_suppression, system_telemetry CASCADE")
    conn.close()

    seed = subprocess.run(
        [sys.executable, "db/seed_anchors.py", "--file", "db/anchors.json"],
        env={**os.environ, "DATABASE_URL": SMOKE_URL},
        capture_output=True, text=True,
    )
    assert seed.returncode == 0, f"anchor seed failed: {seed.stderr[-800:]}"

    from src.services import supabase_client
    supabase_client.close_pool()  # force a fresh pool bound to SMOKE_URL
    yield SMOKE_URL
    supabase_client.close_pool()


def test_full_pipeline_run_against_real_postgres(smoke_db):
    from src.pipeline import orchestrator, pass_a_ingest, pass_c_classify
    from src.services import storyline_narrator

    fed = {"done": False}

    def fake_fetch_rss(query_info, is_direct_url=False, stats=None):
        if fed["done"]:
            return []
        fed["done"] = True
        return _fixture_items()

    fake_telegram = MagicMock(return_value=True)
    with patch.object(pass_a_ingest, "fetch_rss_feed", side_effect=fake_fetch_rss), \
         patch.object(pass_a_ingest, "fetch_nitter_feeds", return_value=[]), \
         patch.object(pass_a_ingest, "fetch_travel_advisories", return_value=[]), \
         patch.object(pass_a_ingest, "translate_to_english_if_needed", side_effect=lambda t: t), \
         patch.object(pass_c_classify, "call_llm", side_effect=_fake_call_llm), \
         patch.object(storyline_narrator, "call_llm", side_effect=_fake_call_llm), \
         patch.object(orchestrator, "sync_czib_to_db",
                      return_value={"fetched": 0, "inserted": 0, "updated": 0}), \
         patch.object(orchestrator, "run_run_snapshot", return_value={"skipped": "smoke"}), \
         patch("src.pipeline.pass_d_score.send_telegram_alert", fake_telegram), \
         patch("src.services.telegram_report_notifier.httpx", MagicMock()), \
         patch("src.services.ops_notifier.send_ops_alert", MagicMock(return_value=True)):
        results = orchestrator.run_pipeline()

    assert results["success"] is True, f"pipeline failed: {results.get('error')}"
    assert results["pass_a"]["events_inserted"] == 2
    assert results["pass_c"]["events_classified"] == 2
    assert results["pass_c"]["events_failed"] == 0
    assert results["pass_d"]["events_scored"] == 2

    # The events must have flowed through the whole state machine in the DB.
    import psycopg
    with psycopg.connect(smoke_db, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT status, count(*) FROM events GROUP BY status"
        ).fetchall()
        by_status = dict(rows)
        assert sum(by_status.values()) == 2
        assert set(by_status) <= {"scored", "reconciled", "alerted", "archived"}, by_status
        # Telemetry written by the run itself (not by our stubs).
        n_runs = conn.execute(
            "SELECT count(*) FROM system_telemetry WHERE event_type = 'pipeline_run'"
        ).fetchone()[0]
        assert n_runs == 1
        # No event may be left holding a classification lock.
        locked = conn.execute(
            "SELECT count(*) FROM events WHERE classification_lock = TRUE"
        ).fetchone()[0]
        assert locked == 0
