# Security Incident Monitor (SIM)

> Zero-cost, serverless Aviation & Geopolitical OSINT platform with auditable cold storage.

**Blueprint Version:** V20.1 — Multi-Provider Production Fortress

## Architecture

SIM is a multi-stage pipeline that collects, classifies, scores, and archives security incidents from global news sources. It runs as a GitHub Actions cron job and serves a Streamlit intelligence dashboard.

```
┌──────────────────────────────────────────────────────────────┐
│                    GitHub Actions (Every 2h)                  │
│                                                              │
│  Pass A ─→ Pass B ─→ Pass C ─→ Pass D ─→ Pass E ─→ Pass F   │
│  Ingest    Dedup     LLM       Score     Reconcile  Archive  │
│                     Classify                                  │
└──────────────┬───────────────────────────────────┬───────────┘
               │                                   │
               ▼                                   ▼
        ┌─────────────┐                   ┌───────────────┐
        │  Supabase    │                   │  Cloudflare   │
        │  PostgreSQL  │                   │  R2 + Telegram│
        └──────┬──────┘                   │  Cold Storage  │
               │                          └───────────────┘
               ▼
        ┌─────────────┐
        │  Streamlit   │
        │  Dashboard   │
        └─────────────┘
```

### Pipeline Passes

| Pass | Function | Key Features |
|------|----------|-------------|
| **A** | Ingest & Canonicalization | Multi-region Google News RSS, GDELT 2.0, Nitter, Travel Advisories, content dedup |
| **B** | Dedup & Distributed Locks | URL hash dedup, maturation window, stale lock cleanup with telemetry |
| **C** | LLM Classification | Multi-provider router (Groq + OpenRouter), heartbeat-protected locks |
| **D** | Scoring & Storyline | Anchor resolution, severity/confidence scoring, storyline linking, Telegram alerts |
| **E** | Targeted Reconciliation | Re-evaluate anchors from storyline text, no LLM calls |
| **F** | Cold Storage & Archive | JSONL export → Cloudflare R2 + Telegram, idempotent 5-step state machine |

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
- **Database:** Supabase PostgreSQL (with `pg_trgm` for fuzzy anchor matching)
- **LLM Providers:** Groq (free tier, 2 accounts) + OpenRouter (free tier, 2 accounts)
- **Dashboard:** Streamlit with PyDeck maps, NetworkX storyline graphs
- **Cold Storage:** Cloudflare R2 + Telegram Bot API
- **CI/CD:** GitHub Actions (cron every 2 hours)

## Project Structure

```
sim/
├── .github/workflows/osint-pipeline.yml   # GitHub Actions pipeline
├── config/
│   ├── keywords.json                      # Search queries & noise filters
│   └── settings.json                      # Pipeline configuration
├── db/
│   ├── migrations/                        # SQL schema migrations (001-006)
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
│   │   └── token_bucket.py               # Per-account rate limiter
│   ├── pipeline/                          # 6-pass pipeline
│   │   ├── orchestrator.py                # Main entry point
│   │   ├── pass_a_ingest.py               # RSS/GDELT/Nitter/Advisory ingestion
│   │   ├── pass_b_dedup.py                # URL dedup + distributed locks
│   │   ├── pass_c_classify.py             # LLM classification
│   │   ├── pass_d_score.py                # Scoring + alerts + Telegram
│   │   ├── pass_e_reconcile.py            # Anchor reconciliation
│   │   └── pass_f_archive.py              # R2 + Telegram cold storage
│   └── services/                          # External integrations
│       ├── czib_client.py                 # EASA Conflict Zone parser
│       ├── supabase_client.py             # Thread-safe connection pool
│       └── telegram_notifier.py           # Alert card sender
├── streamlit_app/                         # Dashboard UI
│   ├── app.py                             # Entry point with dark theme
│   ├── components/                        # UI components (map, table, alerts...)
│   ├── services/cache.py                  # @st.cache_data wrappers
│   └── config/ui_settings.json            # Theme & color palette
├── tests/                                 # pytest test suite
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
# Edit with your DATABASE_URL, API keys

# Run migrations
python -c "
import os, glob, psycopg
conn = psycopg.connect(os.environ['DATABASE_URL'], autocommit=True)
for f in sorted(glob.glob('db/migrations/*.sql')):
    conn.execute(open(f).read())
"

# Seed anchor data
python db/seed_anchors.py --file db/anchors.json

# Run pipeline manually
python -m src.pipeline.orchestrator

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
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot for alerts |
| `TELEGRAM_ALERTS_CHAT_ID` | No | Alert channel chat ID |
| `TELEGRAM_ARCHIVE_CHAT_ID` | No | Archive channel chat ID |
| `R2_ACCOUNT_ID` | No | Cloudflare R2 account |
| `R2_ACCESS_KEY_ID` | No | Cloudflare R2 access key |
| `R2_SECRET_ACCESS_KEY` | No | Cloudflare R2 secret key |

*At least one LLM API key is required.

## Tests

```bash
python -m pytest tests/ -v
```

## License

Private — All rights reserved.
