# Security Incident Monitor (SIM)

> Zero-cost, serverless Terrorism & Security OSINT platform with auditable cold storage.

**Blueprint Version:** V20.1 — Multi-Provider Production Fortress

**Mission focus:** terrorism, bombings/explosions, hotel & resort attacks, airport terror
attacks, mass-casualty events, and geopolitical escalation — with aviation security as a
supporting lens (airport/airspace impact of every event is assessed).

## Architecture

SIM is a multi-stage pipeline that collects, classifies, scores, and archives security incidents from global news sources. It runs headless as a GitHub Actions cron job; all outputs are delivered via Telegram (alert cards, flash updates, weekly HTML reports, JSONL archives) and Cloudflare R2.

```
┌──────────────────────────────────────────────────────────────────┐
│                    GitHub Actions (Every 2h)                     │
│                                                                  │
│  Pass A ──→ Pass B ──→ Pass C ──→ Pass D ──→ Pass E ──→ Pass F   │
│  Ingest     Dedup      LLM        Score      Reconcile  Archive  │
│                        Classify                                  │
└────────────────┬───────────────────────────────────┬─────────────┘
                 │                                   │
                 ▼                                   ▼
          ┌─────────────┐                   ┌────────────────┐
          │  Supabase    │                   │   Cloudflare   │
          │  PostgreSQL  │                   │   R2 + Telegram│
          │  (with RLS)  │                   │   Cold Storage │
          └──────┬──────┘                   └────────────────┘
                 │
                 ▼ (Weekly Run / On-Demand CLI)
          ┌───────────────────────────────────────────────┐
          │   Weekly Intelligence & Forecasting (Pass G)  │
          │  - Tension Index & Z-Score Trajectories       │
          │  - 3-Pass LLM (G1/G2/G3) Assessment Pipeline   │
          │  - Flash Detector Circuit Breaker             │
          │  - Telegram HTML Report & R2 Publish          │
          └───────────────────────────────────────────────┘
```

### Pipeline Passes

| Pass | Function | Key Features |
|------|----------|-------------|
| **A** | Ingest & Canonicalization | Google News RSS (terror/hotel/airport attack queries), 50+ curated news & terrorism-research feeds, GDELT 2.0, Nitter, Travel Advisories, content translation & dedup |
| **B** | Dedup & Distributed Locks | URL hash dedup, maturation window, stale lock cleanup with telemetry |
| **C** | LLM Classification | Multi-provider router (Groq + OpenRouter), heartbeat-protected locks |
| **D** | Scoring & Storyline | Anchor resolution, severity/confidence scoring, storyline linking, Telegram alerts |
| **E** | Targeted Reconciliation | Re-evaluate anchors from storyline text, no LLM calls |
| **F** | Cold Storage & Archive | JSONL export → Cloudflare R2 + Telegram, idempotent 5-step state machine |
| **G** | Weekly Intelligence (CLI) | Tension Index calculations, rolling Z-score trajectories, Watchlist/Emerging concern classification, 3-Pass LLM pipeline (G1/G2/G3), Cloudflare R2 backup, Telegram notifier with HTML report attachments. |

### LLM Provider Cascade

```
① OR-A    nemotron-3-super-120b:free  (primary — funded key, 1K RPD account-wide)
② OR-A    gpt-oss-120b:free           (secondary — shares ①'s account quota)
③ Groq-A  gpt-oss-120b                (smartest Groq slot)
④ Groq-A  qwen3.6-27b                 (quality backup)
⑤ Groq-B  gpt-oss-120b                (throughput)
⑥ Groq-B  qwen3.6-27b                 (burst)
⑦ OR-B    gpt-oss-120b:free           (cross-key mirror, 50 RPD unfunded)
⑧ Gemini  3.1-flash-lite / 3-flash    (third independent provider, 500+20 RPD)
⑨ Groq-A  gpt-oss-20b                 (last-resort; also the bulk router's model)
```

_(2026-06-17: Groq retired llama-3.3-70b-versatile, llama-4-scout, qwen3-32b and
llama-3.1-8b-instant on the free tier; no free chat model exceeds 1K RPD anymore.
2026-07-09: OpenRouter key A funded → Nemotron 3 Super became primary.)_

**Model capability profiles** (`src/core/model_profiles.py`): every provider/model quirk
lives here as declarative data — whether the provider accepts
`response_format:json_object` (OpenRouter free models 400 on it), which knob disables
hidden reasoning (qwen: `reasoning_effort:"none"`; gpt-oss: only `low` is valid; Nemotron
via OpenRouter needs `reasoning:{enabled:false}` and fails *silently* otherwise), and the
per-request token ceiling (Groq 413s above its 8K TPM window). `call_llm` consumes the
profile: it skips accounts the request can't fit **before** sending (no quota spend, no
cooldown), and treats an HTTP 413 as the *request's* fault rather than sidelining the
account. A request too large for every slot raises `LLMRequestTooLarge`, which callers
handle per-item (narrator skips that storyline, Pass C skips that chunk) instead of
aborting the stage. Adding a new model = answering the checklist at the top of
`model_profiles.py`; each rule is pinned by `tests/test_model_profiles.py`.

**Rate limiting:** The binding constraint on Groq's free tier is **TPM (8K)**, not RPM
(30) — a classification is ~2–3K tokens, so only ~3 fit per minute. The router models a
per-(key, model) TPM window (`TokenBucket.tpm_limit`) and charges each call's estimated
tokens against it, so a burst can't trip provider 429s and cascade the whole pool into
cooldown. When every slot is momentarily throttled, Pass C paces (waits for the next token
refill and retries) instead of aborting. OpenRouter `:free` limits are account-wide, so all
`:free` slots on one key share a single bucket; router instances that share a (key, model)
pair (main + bulk) also share one bucket to keep quota accounting truthful.

### Database Reliability

The pipeline runs 20+ minutes per cycle over WAN to Supabase, which produced a family of
production failures (Jul 2026: lock-wait hangs, idle-in-transaction reaps, silent
connection loss). Three layers now defend against it:

- **Autocommit connections** — single statements commit immediately, so the session never
  sits `idle in transaction` through long non-DB phases (RSS fetch, LLM calls). The
  genuinely multi-statement writes (Pass F manifest+deletes, weekly report+mappings,
  archive+domain-penalty pairs) use explicit `with conn.transaction():` blocks.
- **TCP keepalives + pool checks** (`supabase_client.py`) — a half-open connection is
  detected in ~1 minute instead of hanging until the server's 900s reaper fires; the pool
  never hands out a connection that died while idle.
- **Server-side session timeouts** (migration `015`) — `statement_timeout=120s`,
  `lock_timeout=30s`, `idle_in_transaction_session_timeout=900s` bound every failure mode
  that used to burn the whole workflow timeout.

## Source Coverage

- **Targeted Google News queries** (always-on static feeds): hotel attacks/bombings/sieges, airport attacks/bombings/explosions, suicide bombings & vehicle bombs, terror attacks with casualties, attacks on tourists — plus ~110 rotating tier queries (aviation security, transit attacks, protests, travel advisories).
- **Terrorism research:** Jamestown Foundation, The Soufan Center, CTC Sentinel (West Point), Counter Extremism Project, Long War Journal, HSToday.
- **Global & regional wires:** BBC (World/Middle East/Africa/Asia), Al Jazeera, Guardian, France24, NYT, WSJ, CNN, Fox, UN News and Israeli/Iranian/Russian/Ukrainian outlets for conflict-zone coverage.
- **Structured sources:** GDELT 2.0 (region-rotating), EASA CZIB conflict zones, US State Dept travel advisories, Nitter/X conflict trackers.

## Tech Stack

- **Pipeline:** Python 3.12, `httpx`, `tenacity`, `psycopg[binary]`, `trafilatura`
- **Database:** Supabase PostgreSQL (with `pg_trgm` and Row Level Security policies)
- **LLM Providers:** OpenRouter (free tier, 2 keys) + Groq (free tier, 2 accounts) + Google AI Studio (Gemini, free tier)
- **Delivery:** Telegram Bot API (alert cards, flash updates, weekly HTML reports, archives)
- **Cold Storage:** Cloudflare R2 + Telegram Bot API
- **CI/CD:** GitHub Actions (pipeline cron every 2 hours; lint + tests + real-Postgres e2e smoke on every push/PR)

---

## Weekly Geopolitical Intelligence & Forecasting (Pass G)

The weekly forecasting system evaluates Tension Index profiles, identifies rising country trajectories, and runs a structured 3-Pass LLM generation pipeline to produce HTML briefing reports.

### 1. Tension Index ($TI$) Formula
Calculated per country using:
- **Volume ($V$):** Normalized by weekly max country volume.
- **Diversity ($D$):** Log-scaled count of active storyline clusters $D = \log(1 + N_{\text{clusters}}) / \log(11)$.
- **Severity ($S$):** Weighting average and peak incident severity scores.
- **Recency Decay ($R$):** Exponential decay with a 24-hour half-life.
- **Cross-Domain ($X$):** Bonuses when physical, cyber, and airspace categories overlap.
- **Critical Modifier ($C$):** Severity triggers scaled by profile targets (PAX, CREW, DIPLOMAT, CARGO).
- **Z-Score Trajectory:** Evaluates deviation from a rolling 8-week history to classify country trends as **Tırmanıyor** (Escalating), **Stabil** (Stable), or **Azalıyor** (De-escalating).

### 2. 3-Pass LLM Pipeline
- **Pass G1 (Selection):** Dynamically filters top-8 countries with high $TI$ or Z-scores to identify the critical subset to assess.
- **Pass G2 (Country Assessment):** Produces structured Pydantic analysis for each selected country. Employs a cross-check validator that flags and retries (max 2) if LLM-selected `risk_direction` contradicts math-driven trajectories.
- **Pass G3 (Correlation & Spillover):** Evaluates regional spillover effects and synthesizes global trends, enforcing a default fallback if no spillovers are found.

### 3. Flash Detector Circuit Breaker
Automatically dispatches immediate **Flash Alerts** if:
- A country's Tension Index Z-score exceeds $+3.0$ within 24 hours.
- A location experiences $2+$ different event categories (e.g. `MILITARY` + `CYBER`) within a $6$-hour window.
- A country records $3+$ verified, high-confidence events in a $6$-hour window.

---

## Project Structure

```
sim/
├── .github/workflows/osint-pipeline.yml   # Regular 2-hour pipeline
├── .github/workflows/weekly-forecast.yml  # Weekly strategic forecast
├── .github/workflows/deadman.yml          # Hourly dead-man's switch (pipeline liveness)
├── .github/workflows/ci.yml               # Lint + unit suite + real-Postgres e2e smoke
├── config/
│   ├── keywords.json                      # Search queries & noise filters
│   └── settings.json                      # Pipeline configuration
├── db/
│   ├── migrations/                        # SQL migrations (001-015)
│   │   ├── 008_weekly_forecast.sql        # Weekly reports schema
│   │   ├── 009_rls_policies.sql           # Supabase RLS security policies
│   │   └── 015_session_timeouts.sql       # statement/lock/idle-in-tx timeouts
│   ├── anchors.json                       # Airport/location seed data (~80K)
│   └── seed_anchors.py                    # Seed script
├── src/
│   ├── core/                              # Core business logic
│   │   ├── alerts.py                      # 3-tier alert system (WATCH/ALERT/CRITICAL)
│   │   ├── anchor.py                      # IATA/ICAO normalization
│   │   ├── heartbeat.py                   # Thread-safe heartbeat worker
│   │   ├── llm_client.py                  # Unified LLM call wrapper + size guard
│   │   ├── llm_router.py                  # Multi-provider failover router
│   │   ├── model_profiles.py              # Declarative per-model quirks & limits
│   │   ├── storyline.py                   # Bigram Jaccard storyline linking
│   │   ├── storyline_clusterer.py         # Centrist greedy Jaccard clustering
│   │   ├── forecast_engine.py             # Tension Index & trajectory math
│   │   └── token_bucket.py                # Per-account rate limiter
│   ├── pipeline/                          # Pipeline passes
│   │   ├── orchestrator.py                # Main entry point (supports --weekly)
│   │   ├── weekly_forecast.py             # Weekly forecast pass coordinator
│   │   ├── pass_a_ingest.py               # Pass A orchestration + DB persistence
│   │   ├── ingest_sources.py              # All ingest network I/O (RSS/Nitter/advisories/GDELT)
│   │   ├── ingest_filters.py              # Pure text filters, canonicalization, dedup
│   │   ├── ingest_queries.py              # Search-query construction
│   │   ├── pass_b_dedup.py                # URL dedup & distributed locks
│   │   ├── pass_c_classify.py             # LLM classification
│   │   ├── pass_d_score.py                # Scoring + alerts + Telegram
│   │   ├── pass_e_reconcile.py            # Anchor reconciliation
│   │   └── pass_f_archive.py              # R2 + Telegram cold storage
│   └── services/                          # External integrations
│       ├── czib_client.py                 # EASA Conflict Zone parser
│       ├── forecast_generator.py          # 3-Pass LLM generation coordinator
│       ├── flash_detector.py              # Flash update event detector
│       ├── ops_notifier.py                # Pipeline health pings (Telegram)
│       ├── storyline_narrator.py          # Budgeted "story so far" prose (bulk router)
│       ├── telegram_report_notifier.py    # Weekly reports & HTML notifier
│       ├── supabase_client.py             # Autocommit pool + TCP keepalives
│       └── telegram_notifier.py           # Alert card sender
├── tests/                                 # pytest test suite (300+ tests)
│   └── test_pipeline_smoke.py             # Real-Postgres e2e smoke test (CI)
├── requirements.txt                       # Python dependencies
└── SIM_Blueprint_V20_EN.md                # Master implementation blueprint
```

## Setup

### Prerequisites

- Python 3.12+
- Supabase project (or any PostgreSQL 15+ with `pg_trgm`)
- At least one LLM API key (Groq or OpenRouter free tier)

### Local Development

```bash
# Clone and install
git clone https://github.com/secmon-sudo/sim.git
cd sim
pip install -r requirements.txt

# Configure secrets (.env file or exported environment variables)
# Set DATABASE_URL, LLM API keys, Telegram, and R2 credentials — see table below

# Run migrations
python -c "
import os, glob, psycopg
conn = psycopg.connect(os.environ['DATABASE_URL'], autocommit=True)
for f in sorted(glob.glob('db/migrations/*.sql')):
    conn.execute(open(f).read())
"

# Seed anchor data
python db/seed_anchors.py --file db/anchors.json

# Run standard 2-hour pipeline
python -m src.pipeline.orchestrator

# Run weekly forecast pipeline
python -m src.pipeline.orchestrator --weekly
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `GROQ_API_KEY_A` | Yes* | Groq organization A API key |
| `GROQ_API_KEY_B` | No | Groq organization B API key |
| `OPENROUTER_API_KEY_A` | No | OpenRouter account A key |
| `OPENROUTER_API_KEY_B` | No | OpenRouter account B key |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot for alerts and reports |
| `TELEGRAM_ALERTS_CHAT_ID` | No | Alert and weekly report channel chat ID |
| `TELEGRAM_ARCHIVE_CHAT_ID` | No | Archive channel chat ID |
| `DEADMAN_MAX_AGE_HOURS` | No | Dead-man's switch staleness threshold in hours (default `3`) |
| `R2_ACCOUNT_ID` | No | Cloudflare R2 account ID |
| `R2_ACCESS_KEY_ID` | No | Cloudflare R2 access key ID |
| `R2_SECRET_ACCESS_KEY` | No | Cloudflare R2 secret access key |
| `R2_BUCKET_NAME` | No | Cloudflare R2 bucket name (reports and archives) |
| `R2_PUBLIC_URL_BASE` | No | Public URL mapping to the R2 bucket |

\*At least one LLM API key is required.

## Tests & CI

```bash
# Run full unit tests
python -m pytest tests/ -v

# End-to-end smoke test against a real PostgreSQL (what CI runs).
# Only network edges are stubbed (RSS, LLM, Telegram, R2); migrations, seed
# and Pass A→F run for real. Refuses to run against non-localhost URLs.
docker run --rm -d -p 5433:5432 -e POSTGRES_PASSWORD=pg postgres:16
SIM_SMOKE_DATABASE_URL=postgresql://postgres:pg@localhost:5433/postgres \
    python -m pytest tests/test_pipeline_smoke.py -q
```

Every push/PR triggers the **SIM CI** workflow: ruff (F-rules) + vulture dead-code
lint, the unit suite, then the e2e smoke test against a `postgres:16` service
container — so DB/transaction regressions are caught before merge, not in the
2-hourly production run.

## License

Private — All rights reserved.
