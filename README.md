# Security Incident Monitor (SIM)

> Zero-cost, serverless Aviation & Geopolitical OSINT platform with auditable cold storage.

**Blueprint Version:** V20.1 — Multi-Provider Production Fortress

## Architecture

SIM is a multi-stage pipeline that collects, classifies, scores, and archives security incidents from global news sources. It runs as a GitHub Actions cron job and serves a Streamlit intelligence dashboard.

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
                 ├────────────────────────────────────────┐
                 ▼ (REST API)                             ▼
          ┌─────────────┐                          ┌──────────────┐
          │  Streamlit   │                          │  Cloudflare  │
          │  Dashboard   │                          │  Pages (Web) │
          └─────────────┘                          └──────────────┘
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
| **A** | Ingest & Canonicalization | Multi-region Google News RSS, GDELT 2.0, Nitter, Travel Advisories, content translation & dedup |
| **B** | Dedup & Distributed Locks | URL hash dedup, maturation window, stale lock cleanup with telemetry |
| **C** | LLM Classification | Multi-provider router (Groq + OpenRouter), heartbeat-protected locks |
| **D** | Scoring & Storyline | Anchor resolution, severity/confidence scoring, storyline linking, Telegram alerts |
| **E** | Targeted Reconciliation | Re-evaluate anchors from storyline text, no LLM calls |
| **F** | Cold Storage & Archive | JSONL export → Cloudflare R2 + Telegram, idempotent 5-step state machine |
| **G** | Weekly Intelligence (CLI) | Tension Index calculations, rolling Z-score trajectories, Watchlist/Emerging concern classification, 3-Pass LLM pipeline (G1/G2/G3), Cloudflare R2 backup, Telegram notifier with HTML report attachments. |

### LLM Provider Cascade

```
① Groq-A  gpt-oss-120b        (120B — primary)
② Groq-A  llama-3.3-70b       (70B — secondary)
③ Groq-B  llama-4-scout       (MoE — high throughput)
④ Groq-B  qwen3-32b           (32B — burst RPM)
⑤ OR-A    hermes-3-405b       (405B — emergency)
⑥ OR-B    gpt-oss-120b:free   (120B — cross-provider)
⑧ Groq    llama-3.1-8b-instant (8B — bulk fallback, 14.4K RPD)
```

**Total daily capacity:** ~33,200 RPD across 7 model slots.

## Tech Stack

- **Pipeline:** Python 3.12, `httpx`, `tenacity`, `psycopg[binary]`, `trafilatura`
- **Database:** Supabase PostgreSQL (with `pg_trgm` and Row Level Security policies)
- **LLM Providers:** Groq (free tier, 2 accounts) + OpenRouter (free tier, 2 accounts)
- **Dashboard:** Streamlit with PyDeck maps, NetworkX storyline graphs
- **Cold Storage:** Cloudflare R2 + Telegram Bot API
- **CI/CD:** GitHub Actions (cron every 2 hours)

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
├── .github/workflows/osint-pipeline.yml   # GitHub Actions pipeline
├── config/
│   ├── keywords.json                      # Search queries & noise filters
│   └── settings.json                      # Pipeline configuration
├── db/
│   ├── migrations/                        # SQL migrations (001-009)
│   │   ├── 008_weekly_forecast.sql        # Weekly reports schema
│   │   └── 009_rls_policies.sql           # Supabase RLS security policies
│   ├── anchors.json                       # Airport/location seed data (~80K)
│   └── seed_anchors.py                    # Seed script
├── src/
│   ├── core/                              # Core business logic
│   │   ├── alerts.py                      # 3-tier alert system (WATCH/ALERT/CRITICAL)
│   │   ├── anchor.py                      # IATA/ICAO normalization
│   │   ├── heartbeat.py                   # Thread-safe heartbeat worker
│   │   ├── llm_client.py                  # Unified LLM call wrapper
│   │   ├── llm_router.py                  # Multi-provider failover router
│   │   ├── storyline.py                   # Bigram Jaccard storyline linking
│   │   ├── storyline_clusterer.py         # [NEW] Centrist greedy Jaccard clustering
│   │   ├── forecast_engine.py             # [NEW] Tension Index & trajectory math
│   │   └── token_bucket.py                # Per-account rate limiter
│   ├── pipeline/                          # Pipeline passes
│   │   ├── orchestrator.py                # Main entry point (supports --weekly)
│   │   ├── weekly_forecast.py             # [NEW] Weekly forecast pass coordinator
│   │   ├── pass_a_ingest.py               # Ingestion & user-agent bypass
│   │   ├── pass_b_dedup.py                # URL dedup & distributed locks
│   │   ├── pass_c_classify.py             # LLM classification
│   │   ├── pass_d_score.py                # Scoring + alerts + Telegram
│   │   ├── pass_e_reconcile.py            # Anchor reconciliation
│   │   └── pass_f_archive.py              # R2 + Telegram cold storage
│   └── services/                          # External integrations
│       ├── czib_client.py                 # EASA Conflict Zone parser
│       ├── forecast_generator.py          # [NEW] 3-Pass LLM generation coordinator
│       ├── flash_detector.py              # [NEW] Flash update event detector
│       ├── telegram_report_notifier.py    # [NEW] Weekly reports & HTML notifier
│       ├── supabase_client.py             # Thread-safe connection pool
│       └── telegram_notifier.py           # Alert card sender
├── streamlit_app/                         # Dashboard UI
├── tests/                                 # pytest test suite
│   └── test_weekly_forecast.py            # [NEW] Weekly forecast test suite
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

# Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit with your DATABASE_URL, API keys, Telegram, and R2 credentials

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

# Launch dashboard
streamlit run streamlit_app/app.py
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
| `R2_ACCOUNT_ID` | No | Cloudflare R2 account ID |
| `R2_ACCESS_KEY_ID` | No | Cloudflare R2 access key ID |
| `R2_SECRET_ACCESS_KEY` | No | Cloudflare R2 secret access key |
| `R2_BUCKET_NAME` | No | Cloudflare R2 bucket name (reports and archives) |
| `R2_PUBLIC_URL_BASE` | No | Public URL mapping to the R2 bucket |

\*At least one LLM API key is required.

## Tests

```bash
# Run full unit tests
python -m pytest tests/ -v
```

## License

Private — All rights reserved.
