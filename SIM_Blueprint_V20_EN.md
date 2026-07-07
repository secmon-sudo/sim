# Security Incident Monitor (SIM) - Master Implementation Blueprint (V20.1 - Multi-Provider Production Fortress)

---

## Changelog (V19 → V20 → V20.1)

| Area | V19 Status | V20 Resolution |
|---|---|---|
| Heartbeat thread management | "MUST gracefully terminate" — implementation left open | `_HeartbeatWorker` class + full reference pseudocode |
| Storyline Jaccard | Tokenization strategy undefined | Bigram + stopword filter tokenization standard |
| ENUM rigidity | Migration burden ignored | `event_type_catalog` soft-enum table solution |
| Alert Gate (AND logic) | Single gate — silent critical alert risk | 3-tier system (WATCH / ALERT / CRITICAL) |
| Master DB definition | Missing entirely | `anchor_master` schema + seed sources |
| LLM Rate Limiting | Missing entirely | Token bucket + daily quota tracking |
| Streamlit UI | Single line mention | Full component inventory |
| QA Checklist | Yes/No items | Acceptance criteria + automation notes per item |

| Area | V20 Status | V20.1 Resolution |
|---|---|---|
| LLM Provider Strategy | `gpt-4o` hardcoded — zero-cost violation | Multi-provider router: Groq (2 accounts) + OpenRouter (2 accounts) |
| Model Selection | Undefined | Priority-tiered model inventory with real rate limit data |
| Rate Limiter | Single global bucket | Per-account `TokenBucket` pool + automatic failover |
| `_HeartbeatWorker` resilience | Infinite retry on DB failure | Consecutive error limit (5) with graceful stop |
| `normalize_anchor` safety | No input validation | Type guard + length limit |
| `classify_event` transactions | Implicit commit assumption | Explicit commit/rollback in `finally` block |
| `TokenBucket` day reset | `time.time()` drift risk | `datetime.date.today()` comparison |
| Telemetry dashboard | Hardcoded quota `1000` | Dynamic quota from `LLMRouter` state |
| Secrets management | GitHub Actions only | Added Streamlit Cloud `.streamlit/secrets.toml` |
| Dependency inventory | Missing `httpx` | Added `httpx`, `python-dotenv`; `psycopg2` → `psycopg[binary]` |
| QA Checklist | 8 items | QA-09: Multi-Provider LLM Failover |

---

## 1. Project Objective & Constraints

**Goal:** Zero-cost, serverless Aviation OSINT platform with auditable cold storage.

**Strict Directives (Unchanged):**
1. **Concurrency:** GitHub Actions MUST use `concurrency: group: osint-worker, cancel-in-progress: false`.
2. **Cold Storage Archiving (Idempotent & Safe):** `NOT EXISTS` guard, 5-step state machine.
3. **Resilience:** LLM calls use exponential backoff (`tenacity`). Distributed locking must use true parameterized `lock_owner`.

---

## 2. Configuration & Master Data

### 2.1 Dynamic Dictionaries (`config/keywords.json`, `config/settings.json`)
Unchanged. Multi-lingual emergency keywords and noise filtering. Dynamic radius mapping.

### 2.2 Master DB — `anchor_master` Table (NEW)

V19 referenced a Master DB for "Cairo Intl → CAI" normalization but never defined it. V20 provides the schema and seed sources.

#### Schema

```sql
CREATE TABLE anchor_master (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    iata_code      VARCHAR(4)    UNIQUE,        -- Primary normalized identifier (CAI, LHR...)
    icao_code      VARCHAR(4),                  -- Secondary (HECA, EGLL...)
    anchor_type    VARCHAR(20)   NOT NULL,       -- airport | hotel_chain | port | military_base
    canonical_name VARCHAR(200)  NOT NULL,       -- "Cairo International Airport"
    aliases        JSONB         DEFAULT '[]',   -- ["Cairo Intl", "القاهرة الدولي", "CAI Airport"]
    country_iso    CHAR(2)       NOT NULL,
    latitude       DOUBLE PRECISION,
    longitude      DOUBLE PRECISION,
    czib_flag      BOOLEAN       DEFAULT FALSE,  -- Conflict Zone Influence Buffer
    updated_at     TIMESTAMP     DEFAULT NOW()
);

-- GIN index for fast alias searches
CREATE INDEX idx_anchor_aliases ON anchor_master USING GIN(aliases);
CREATE INDEX idx_anchor_iata    ON anchor_master(iata_code);

-- Enable pg_trgm for fuzzy canonical_name matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_anchor_trgm ON anchor_master USING GIN(canonical_name gin_trgm_ops);
```

#### Normalization Function

```python
def normalize_anchor(raw_text: str, db_conn) -> tuple[str | None, float]:
    """
    Returns: (normalized_id, confidence)
    normalized_id: IATA code (preferred), ICAO, or None
    confidence: 1.0 (exact match), 0.8 (alias), 0.6 (fuzzy), 0.0 (not found)
    """
    # V20.1 — Input guard: reject non-string, empty, or excessively long input
    if not isinstance(raw_text, str) or len(raw_text) > 200:
        return None, 0.0
    raw_text = raw_text.strip()
    if not raw_text:
        return None, 0.0

    # 1. Direct IATA / ICAO exact match
    if re.match(r'^[A-Z]{3,4}$', raw_text):
        row = db_conn.fetchone(
            "SELECT iata_code FROM anchor_master WHERE iata_code=$1 OR icao_code=$1",
            raw_text
        )
        if row:
            return row['iata_code'], 1.0

    # 2. Case-insensitive alias JSONB search
    row = db_conn.fetchone(
        "SELECT iata_code FROM anchor_master WHERE aliases @> $1::jsonb",
        json.dumps([raw_text])
    )
    if row:
        return row['iata_code'], 0.8

    # 3. Trigram fuzzy match (pg_trgm)
    row = db_conn.fetchone(
        """SELECT iata_code, similarity(canonical_name, $1) AS sim
           FROM anchor_master
           WHERE similarity(canonical_name, $1) > 0.5
           ORDER BY sim DESC LIMIT 1""",
        raw_text
    )
    if row:
        return row['iata_code'], round(row['sim'] * 0.6, 2)

    return None, 0.0
```

#### Seed Sources

| Source | Content | Refresh Cadence |
|---|---|---|
| [OurAirports](https://ourairports.com/data/) | ~80,000 airports, IATA/ICAO | Monthly |
| IATA Airport Search API | Commercial + cargo airports | Quarterly |
| OSM Overpass (`amenity=hotel`) | Hotel chains + coordinates | Bi-annually |
| ACLED conflict zones | `czib_flag` seed data | Weekly |

---

## 3. Database Schema (Updated)

### 3.1 `events` Table

All V19 columns retained. Additions:

```sql
-- Replaced strict ENUM with soft-enum FK (see Section 3.3)
event_type     VARCHAR(60) NOT NULL REFERENCES event_type_catalog(code),
sub_type       VARCHAR(60)          REFERENCES event_type_catalog(code),

-- Alert tier column (new — see Section 5.2)
alert_tier     VARCHAR(10) CHECK (alert_tier IN ('WATCH', 'ALERT', 'CRITICAL')),
```

### 3.2 `domain_penalties` & `system_telemetry` Tables

Unchanged from V19.

### 3.3 `event_type_catalog` Table (NEW — Soft-ENUM Solution)

**Problem:** Altering a PostgreSQL ENUM requires `ALTER TYPE` — a risky production migration. Aviation OSINT continuously surfaces new event subtypes.

**Solution:** Drop ENUM, use `VARCHAR + FK` with a catalog table.

```sql
CREATE TABLE event_type_catalog (
    code          VARCHAR(60)  PRIMARY KEY,  -- e.g. 'runway_incursion', 'bomb_threat'
    label_en      VARCHAR(120),
    parent_code   VARCHAR(60)  REFERENCES event_type_catalog(code),
    severity_base INT          DEFAULT 30,   -- Base weight used in Pass D scoring
    active        BOOLEAN      DEFAULT TRUE,
    created_at    TIMESTAMP    DEFAULT NOW()
);

-- Example seed rows
INSERT INTO event_type_catalog VALUES
  ('security_incident',      'Security Incident',       NULL,                80, TRUE, NOW()),
  ('bomb_threat',            'Bomb Threat',             'security_incident', 80, TRUE, NOW()),
  ('runway_incursion',       'Runway Incursion',        NULL,                60, TRUE, NOW()),
  ('active_shooter',         'Active Shooter',          'security_incident', 90, TRUE, NOW()),
  ('emergency_landing',      'Emergency Landing',       NULL,                50, TRUE, NOW()),
  ('other_aviation_related', 'Other Aviation Related',  NULL,                20, TRUE, NOW());
```

**Migration strategy:** Adding a new type = single `INSERT`. Retiring a type = `active = FALSE`. Zero downtime.

---

## 4. Multi-Stage Pipeline (GitHub Actions — Every 30 Minutes)

### PASS A: Ingest & Canonicalization
Unchanged from V19, plus official travel-advisory ingestion (V20.2):

- **Sources:** US State Dept (RSS, "Level N") + UK FCDO per-country Atom feeds
  (`gov.uk/foreign-travel-advice/{country}.atom`) for a curated set of high-risk
  countries. `fetch_travel_advisories` parses both RSS `<item>` and Atom `<entry>`.
- **Filtering:** US/level feeds keep the level-increase / Level 3-4 gate via a
  multi-agency parser (`_parse_advisory_level` understands "Level N" AND phrase wording:
  "do not travel", "advise against all travel", "avoid all travel" → L4; "reconsider",
  "all but essential", "non-essential travel" → L3). UK feeds are curated (the high-risk
  country selection IS the filter) so every recent entry is ingested.
- **Re-ingestion:** advisory page URLs are stable, but the pipeline dedups permanently by
  URL hash, so each item's link is stamped with its update date (`#adv-YYYYMMDD`) — a
  genuine update becomes a new event/alert, repeated runs of the same update stay deduped.
- **Alerting:** advisories are country-level (no airport anchor), so `evaluate_alert_tier`
  routes `travel_advisory`/`travel_ban` on severity alone (bypassing the anchor/time gates)
  and they are excluded from the generic-umbrella incident gate.

---

### PASS B: Dedup, Maturation & Distributed Locks

Unchanged from V19. Additional clarification on stale lock telemetry payload:

```python
# This record MUST be committed BEFORE the lock is cleared (same transaction)
telemetry_payload = {
    "event_type":           "stale_lock_cleared",
    "event_id":             str(event_id),
    "lock_owner":           str(lock_owner),           # V19 requirement
    "lock_ts":              lock_ts.isoformat(),
    "last_heartbeat_at":    last_hb.isoformat(),       # V19 requirement
    "cleared_by_worker":    str(current_worker_uuid),
    "stale_duration_seconds": (datetime.utcnow() - last_hb).total_seconds()
}
```

---

### PASS C: LLM Classification (Cancellation-Safe Heartbeat — Full Implementation)

V19 stated "MUST gracefully terminate" without providing an implementation. V20 delivers a complete reference class.

#### `_HeartbeatWorker` Class

```python
import threading
import time
import logging

logger = logging.getLogger(__name__)


class _HeartbeatWorker:
    """
    Context-manager that runs a background heartbeat update thread.

    Usage:
        with _HeartbeatWorker(db, event_id, lock_owner, interval=60) as hb:
            result = call_llm_with_backoff(text)
        # On 'with' block exit, worker stops automatically — success OR exception.
    """

    def __init__(self, db_conn, event_id: str, lock_owner: str, interval: int = 60):
        self._db          = db_conn
        self._event_id    = event_id
        self._lock_owner  = lock_owner
        self._interval    = interval
        self._stop_event  = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True,
                                             name=f"hb-{event_id[:8]}")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()           # Signal the loop to exit
        self._thread.join(timeout=10)    # Wait at most 10 s
        if self._thread.is_alive():
            logger.warning(
                "Heartbeat thread %s did not terminate cleanly for event %s",
                self._thread.name, self._event_id
            )
        return False  # Always re-raise any exception from the caller

    def _run(self):
        """
        Writes a heartbeat every `interval` seconds.
        Exits immediately when _stop_event is set OR lock ownership is lost.
        V20.1: Stops after 5 consecutive DB errors to prevent infinite failure loops.
        """
        consecutive_errors = 0
        while not self._stop_event.wait(timeout=self._interval):
            try:
                rowcount = self._db.execute(
                    """UPDATE events
                       SET    last_heartbeat_at = NOW()
                       WHERE  id = %s AND lock_owner = %s""",
                    (self._event_id, self._lock_owner)
                ).rowcount

                if rowcount == 0:
                    # Lock was stolen or released externally — stop silently
                    logger.warning(
                        "Heartbeat: lock lost for event %s (owner %s). Stopping.",
                        self._event_id, self._lock_owner
                    )
                    return

                consecutive_errors = 0  # V20.1: Reset on success

            except Exception as exc:
                consecutive_errors += 1
                logger.error(
                    "Heartbeat DB error #%d for event %s: %s",
                    consecutive_errors, self._event_id, exc
                )
                # V20.1: Stop after 5 consecutive DB failures
                if consecutive_errors >= 5:
                    logger.critical(
                        "Heartbeat: %d consecutive DB failures — stopping for event %s",
                        consecutive_errors, self._event_id
                    )
                    return
```

#### Usage in the LLM Classification Call

```python
def classify_event(db_conn, event_id: str, lock_owner: str, text: str) -> dict:
    try:
        with _HeartbeatWorker(db_conn, event_id, lock_owner, interval=60):
            raw_response = call_llm_with_backoff(text)   # wrapped with tenacity
            parsed       = validate_and_parse(raw_response)
            # V20.1: Validate event_type against active catalog entries
            event_type = parsed.get('event_type', 'other_aviation_related')
            active_check = db_conn.fetchone(
                "SELECT code FROM event_type_catalog WHERE code = %s AND active = TRUE",
                (event_type,)
            )
            if not active_check:
                event_type = 'other_aviation_related'
            db_conn.execute(
                """UPDATE events
                   SET   llm_raw_output    = %s,
                         llm_parsed_output = %s,
                         event_type        = %s,
                         status            = 'classified'
                   WHERE id = %s AND lock_owner = %s""",
                (json.dumps(raw_response), json.dumps(parsed),
                 event_type, event_id, lock_owner)
            )
            db_conn.commit()
            return parsed

    except LLMParseError as e:
        db_conn.execute(
            """UPDATE events
               SET llm_parse_error = %s,
                   event_type      = 'other_aviation_related'
               WHERE id = %s""",
            (str(e), event_id)
        )
        db_conn.commit()
        return {}

    finally:
        # V20.1: Explicit commit/rollback for idempotent lock release
        try:
            result = db_conn.execute(
                """UPDATE events
                   SET classification_lock = FALSE,
                       lock_owner          = NULL
                   WHERE id = %s AND lock_owner = %s""",
                (event_id, lock_owner)
            )
            db_conn.commit()
            if result.rowcount == 0:
                logger.info(
                    "Lock release: 0 rows updated for event %s — already released.",
                    event_id
                )
        except Exception:
            db_conn.rollback()
            logger.exception("Lock release failed for event %s", event_id)
```

---

### PASS C: LLM Rate Limiting & Multi-Provider Strategy (V20.1 — REWRITTEN)

V19 contained no strategy for LLM API quota management. V20 added a single-provider `TokenBucket`. **V20.1 replaces this with a full multi-provider, multi-account architecture** using Groq and OpenRouter free tiers.

#### 4.5.1 Provider & Model Inventory

> **How the cascade works:** Her event için sadece **1 model** çağrılır. Router, listedeki ilk uygun modeli seçer. Eğer o model rate-limit'e takılırsa (HTTP 429), bir sonrakine düşer. Aşağıdaki tüm modeller aynı anda çalışmaz — yalnızca yedek zincir (waterfall/cascade) oluştururlar.

**Groq Free Tier (2 ayrı organizasyon hesabı):**

| Account | Model ID | Params | RPM | RPD | TPM | TPD | Role |
|---|---|---|---|---|---|---|---|
| Groq-A | `openai/gpt-oss-120b` | 120B | 30 | 1,000 | 8K | 200K | **① Primary** — en akıllı, ilk denenen model |
| Groq-A | `qwen/qwen3.6-27b` | 27B | 30 | 1,000 | 8K | 200K | **② Secondary** — Primary rate-limited olunca devralır (eski llama-3.3-70b-versatile) |
| Groq-B | `openai/gpt-oss-120b` | 120B | 30 | 1,000 | 8K | 200K | **③ Throughput** — Groq-A dolunca devralır (eski llama-4-scout) |
| Groq-B | `qwen/qwen3.6-27b` | 27B | 30 | 1,000 | 8K | 200K | **④ Burst** — model çeşitliliği (eski qwen3-32b) |
| Groq-A/B | `openai/gpt-oss-20b` | 20B | 30 | 1,000 | 8K | 200K | **⑧ Bulk fallback** — son çare (eski llama-3.1-8b-instant); A+B havuzlanınca ~2K RPD |

> **2026-06-17 Groq deprecation:** `llama-3.3-70b-versatile`, `llama-4-scout`, `qwen3-32b` ve `llama-3.1-8b-instant` ücretsiz katmandan kaldırıldı. Önerilen yerine geçenler: `openai/gpt-oss-120b` / `qwen/qwen3.6-27b` (orta/büyük slotlar) ve `openai/gpt-oss-20b` (bulk). Artık hiçbir ücretsiz sohbet modeli 1K RPD'yi aşmıyor — eski 14.4K'lık bulk kapasitesi, slotu iki Groq anahtarına havuzlayarak kısmen telafi edilir.

**OpenRouter Free Tier (2 ayrı hesap, para yüklemeyeceğiz):**

| Account | Model ID | Params | Context | RPM | RPD | Role |
|---|---|---|---|---|---|---|
| OR-A | `nousresearch/hermes-3-llama-3.1-405b:free` | 405B | 131K | 20 | 200 | **⑤ Emergency** — Groq tamamen dolunca, en zeki ücretsiz model |
| OR-B | `openai/gpt-oss-120b:free` | 120B | 131K | 20 | 200 | **⑥ Mirror** — Groq primary ile aynı model, farklı provider |

OpenRouter RPD = **200/gün** (unfunded hesap). Her hesap için tek model — 400 RPD kota, zincirdeki 5. ve 6. sırada.

**Toplam günlük kapasite:**

| Provider | Akıllı Model RPD | Bulk Fallback RPD | Toplam RPD |
|---|---|---|---|
| Groq (2 hesap) | 4,000 | 2,000 | 6,000 |
| OpenRouter (2 hesap) | 400 | — | 400 |
| **Toplam** | **4,400** | **2,000** | **6,400** |

Pipeline ihtiyacı: ~48 çalışma/gün × ~50 event = **~2,400 RPD**. Sadece akıllı modellerle bile ihtiyacın **~1.8× üstünde** kapasite var. (2026-06-17 sonrası bulk kapasitesi 28.8K → 2K RPD düştü; akıllı model kotası ihtiyacı hâlâ karşılıyor.)

#### 4.5.2 Routing Priority Chain

```
① Groq-A  gpt-oss-120b        (120B — en akıllı, ilk denenir)
   ↓ rate-limited veya günlük kota doldu
② Groq-A  qwen3.6-27b         (27B — kanıtlanmış kalite yedeği)
   ↓
③ Groq-B  gpt-oss-120b        (120B — throughput)
   ↓
④ Groq-B  qwen3.6-27b         (27B — model çeşitliliği)
   ↓
⑤ OR-A    hermes-3-405b       (405B — en büyük ücretsiz model)
   ↓
⑥ OR-B    gpt-oss-120b:free   (120B — cross-provider yedek)
   ↓
⑧ Groq    gpt-oss-20b         (20B — 1K RPD, son çare)
```

#### 4.5.3 TokenBucket (V20.1 — Fixed Day Reset)

```python
import threading
import time
import datetime
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """
    Thread-safe sliding-window token bucket rate limiter.
    V20.1: Fixed daily reset using datetime.date instead of time.time() drift.

    rate_per_minute : maximum requests allowed per minute
    daily_limit     : hard daily cap (None = unlimited)
    """
    rate_per_minute: float
    daily_limit:     int | None = None

    _tokens:      float          = field(init=False)
    _last_refill: float          = field(default_factory=time.monotonic, init=False)
    _daily_used:  int            = field(default=0, init=False)
    _current_day: datetime.date  = field(default_factory=datetime.date.today, init=False)
    _lock:        threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self._tokens = self.rate_per_minute

    def acquire(self, timeout: float = 300.0) -> bool:
        """Block until a token is available or timeout is reached."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                self._refill()
                if self._daily_used >= (self.daily_limit or float('inf')):
                    raise RuntimeError(
                        "Daily LLM quota exhausted. Pipeline will resume tomorrow."
                    )
                if self._tokens >= 1:
                    self._tokens    -= 1
                    self._daily_used += 1
                    return True
            time.sleep(1.0)
        raise TimeoutError(f"Rate limiter: could not acquire token within {timeout}s")

    def _refill(self):
        now     = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.rate_per_minute,
            self._tokens + elapsed * (self.rate_per_minute / 60.0)
        )
        self._last_refill = now
        # V20.1: Reliable day-boundary reset using date comparison
        today = datetime.date.today()
        if self._current_day != today:
            self._daily_used = 0
            self._current_day = today
```

#### 4.5.4 Multi-Provider Router

```python
import os, time, threading, logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

class ProviderStatus(Enum):
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    ERROR = "error"

@dataclass
class LLMAccount:
    provider: str            # "groq" | "openrouter"
    account_id: str          # "A" | "B"
    model: str               # e.g. "openai/gpt-oss-120b"
    api_key: str
    rpm: int
    rpd: int
    bucket: TokenBucket
    status: ProviderStatus = ProviderStatus.ACTIVE
    cooldown_until: float = 0.0
    daily_errors: int = 0


class LLMRouter:
    """
    Priority-ordered failover across multiple provider accounts.
    Automatically rotates to next account on rate-limit or error.
    """

    def __init__(self, accounts: list[LLMAccount]):
        self._accounts = accounts
        self._lock = threading.Lock()

    @property
    def total_daily_quota(self) -> int:
        """Sum of all account RPD limits (for telemetry dashboard)."""
        return sum(a.rpd for a in self._accounts)

    @property
    def total_daily_used(self) -> int:
        return sum(a.bucket._daily_used for a in self._accounts)

    def get_available_account(self) -> Optional[LLMAccount]:
        with self._lock:
            now = time.monotonic()
            for acct in self._accounts:
                if acct.status in (ProviderStatus.ACTIVE,) and acct.cooldown_until <= now:
                    try:
                        acct.bucket.acquire(timeout=0)
                        return acct
                    except TimeoutError:
                        acct.status = ProviderStatus.RATE_LIMITED
                        acct.cooldown_until = now + 60
                    except RuntimeError:
                        acct.status = ProviderStatus.QUOTA_EXHAUSTED
                # Auto-recover rate-limited accounts after cooldown
                elif acct.status == ProviderStatus.RATE_LIMITED and acct.cooldown_until <= now:
                    acct.status = ProviderStatus.ACTIVE
                    try:
                        acct.bucket.acquire(timeout=0)
                        return acct
                    except (TimeoutError, RuntimeError):
                        acct.cooldown_until = now + 120
        return None

    def report_success(self, acct: LLMAccount):
        with self._lock:
            acct.status = ProviderStatus.ACTIVE
            acct.daily_errors = 0

    def report_failure(self, acct: LLMAccount, is_rate_limit: bool = False):
        with self._lock:
            if is_rate_limit:
                acct.status = ProviderStatus.RATE_LIMITED
                acct.cooldown_until = time.monotonic() + 120
            else:
                acct.daily_errors += 1
                if acct.daily_errors >= 10:
                    acct.status = ProviderStatus.ERROR
                    acct.cooldown_until = time.monotonic() + 600

    def get_status_snapshot(self) -> dict:
        """Returns serializable status for telemetry logging."""
        return {
            f"{a.provider}/{a.account_id}/{a.model}": {
                "status": a.status.value,
                "daily_used": a.bucket._daily_used,
                "daily_limit": a.rpd,
            }
            for a in self._accounts
        }
```

#### 4.5.5 Account Initialization

```python
def build_llm_router() -> LLMRouter:
    """Initialize all accounts from environment variables."""
    accounts = [
        # === GROQ Account A === (Primary: smartest models)
        LLMAccount(
            provider="groq", account_id="A",
            model="openai/gpt-oss-120b",
            api_key=os.environ["GROQ_API_KEY_A"],
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000),
        ),
        LLMAccount(
            provider="groq", account_id="A",
            model="qwen/qwen3.6-27b",
            api_key=os.environ["GROQ_API_KEY_A"],
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000),
        ),
        # === GROQ Account B === (High throughput)
        LLMAccount(
            provider="groq", account_id="B",
            model="openai/gpt-oss-120b",
            api_key=os.environ["GROQ_API_KEY_B"],
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000),
        ),
        LLMAccount(
            provider="groq", account_id="B",
            model="qwen/qwen3.6-27b",
            api_key=os.environ["GROQ_API_KEY_B"],
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000),
        ),
        # === OPENROUTER Account A === (Emergency — en zeki ücretsiz model)
        LLMAccount(
            provider="openrouter", account_id="A",
            model="nousresearch/hermes-3-llama-3.1-405b:free",
            api_key=os.environ["OPENROUTER_API_KEY_A"],
            rpm=20, rpd=200,
            bucket=TokenBucket(rate_per_minute=20, daily_limit=200),
        ),
        # === OPENROUTER Account B === (Cross-provider mirror)
        LLMAccount(
            provider="openrouter", account_id="B",
            model="openai/gpt-oss-120b:free",
            api_key=os.environ["OPENROUTER_API_KEY_B"],
            rpm=20, rpd=200,
            bucket=TokenBucket(rate_per_minute=20, daily_limit=200),
        ),
        # === GROQ Bulk Fallback === (son çare, 1K RPD; A+B havuzlanınca ~2K)
        LLMAccount(
            provider="groq", account_id="A",
            model="openai/gpt-oss-20b",
            api_key=os.environ["GROQ_API_KEY_A"],
            rpm=30, rpd=1000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=1000),
        ),
    ]
    return LLMRouter(accounts)
```

#### 4.5.6 Unified LLM Call Wrapper

```python
import httpx
import tenacity

PROVIDER_ENDPOINTS = {
    "groq":       "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}

@tenacity.retry(
    retry=tenacity.retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
    stop=tenacity.stop_after_attempt(3),
    before_sleep=lambda rs: logger.warning(
        "LLM connection retry #%d: %s", rs.attempt_number, rs.outcome.exception()
    )
)
def _send_request(acct: LLMAccount, prompt: str) -> httpx.Response:
    """Single request to a specific account. Retries on connection errors only."""
    headers = {
        "Authorization": f"Bearer {acct.api_key}",
        "Content-Type": "application/json",
    }
    if acct.provider == "openrouter":
        headers["HTTP-Referer"] = "https://sim-osint.app"
        headers["X-Title"] = "SIM-OSINT-Pipeline"
    response = httpx.post(
        PROVIDER_ENDPOINTS[acct.provider],
        headers=headers,
        json={
            "model": acct.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response


def call_llm_with_backoff(router: LLMRouter, prompt: str) -> dict:
    """
    Try all available accounts in priority order.
    Returns: {"response": ..., "provider": ..., "account": ..., "model": ...}
    """
    last_error = None
    for _ in range(len(router._accounts)):
        acct = router.get_available_account()
        if acct is None:
            break
        try:
            t0 = time.monotonic()
            resp = _send_request(acct, prompt)
            latency_ms = int((time.monotonic() - t0) * 1000)
            router.report_success(acct)
            return {
                "response": resp.json(),
                "provider": acct.provider,
                "account":  acct.account_id,
                "model":    acct.model,
                "latency_ms": latency_ms,
            }
        except httpx.HTTPStatusError as e:
            is_429 = e.response.status_code == 429
            router.report_failure(acct, is_rate_limit=is_429)
            last_error = e
            logger.warning(
                "LLM %s/%s/%s failed (HTTP %d), rotating...",
                acct.provider, acct.account_id, acct.model, e.response.status_code
            )
        except Exception as e:
            router.report_failure(acct)
            last_error = e

    raise RuntimeError(f"All LLM accounts exhausted. Last error: {last_error}")
```

#### 4.5.7 Secrets Configuration

```yaml
# .github/workflows/osint-pipeline.yml
env:
  GROQ_API_KEY_A:       ${{ secrets.GROQ_API_KEY_A }}
  GROQ_API_KEY_B:       ${{ secrets.GROQ_API_KEY_B }}
  OPENROUTER_API_KEY_A: ${{ secrets.OPENROUTER_API_KEY_A }}
  OPENROUTER_API_KEY_B: ${{ secrets.OPENROUTER_API_KEY_B }}
```

```toml
# .streamlit/secrets.toml (for Streamlit Cloud deployment)
[supabase]
url = "https://xxx.supabase.co"
key = "service_role_key_here"

[llm]
groq_keys = ["gsk_..._A", "gsk_..._B"]
openrouter_keys = ["sk-or-..._A", "sk-or-..._B"]
```

#### 4.5.8 Telemetry Logging per LLM Call (V20.1 — Multi-Provider)

```python
# Insert after every successful or failed LLM call
def log_llm_telemetry(db, result: dict, router: LLMRouter, success: bool):
    db.execute(
        "INSERT INTO system_telemetry(event_type, value_json) VALUES ('llm_call', %s)",
        json.dumps({
            "provider":     result.get("provider", "unknown"),
            "account":      result.get("account", "unknown"),
            "model":        result.get("model", "unknown"),
            "tokens_used":  result.get("response", {}).get("usage", {}).get("total_tokens", 0),
            "latency_ms":   result.get("latency_ms", 0),
            "success":      success,
            "daily_used":   router.total_daily_used,
            "daily_quota":  router.total_daily_quota,
            "accounts":     router.get_status_snapshot(),
        })
    )
```

---

### PASS D: Resolution, Storyline, Spatial & Scoring

#### Storyline Matching — Tokenization Standard (NEW)

V19 specified `Jaccard > 0.4` but left the tokenization strategy undefined, creating both false-positive and false-negative risks.

```python
import re
from typing import Set

# Context-independent words that dilute Jaccard signal in aviation text
AVIATION_STOPWORDS = {
    "the", "a", "an", "at", "in", "on", "of", "to", "and", "or",
    "airport", "terminal", "flight", "gate", "apron"
}

def tokenize_storyline_hint(text: str) -> Set[str]:
    """
    Bigram-enhanced tokenization.
    Example: "runway incursion CAI" → {"runway", "incursion", "cai",
                                        "runway incursion", "incursion cai"}
    """
    clean  = re.sub(r'[^\w\s]', '', text.lower())
    tokens = [t for t in clean.split() if t not in AVIATION_STOPWORDS]
    unigrams = set(tokens)
    bigrams  = {f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)}
    return unigrams | bigrams


def jaccard_similarity(hint_a: str, hint_b: str) -> float:
    set_a = tokenize_storyline_hint(hint_a)
    set_b = tokenize_storyline_hint(hint_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def should_link_storyline(event_a: dict, event_b: dict) -> bool:
    """True only when ALL three conditions hold."""
    similarity   = jaccard_similarity(
        event_a.get('storyline_hint') or '',
        event_b.get('storyline_hint') or ''
    )
    same_country = event_a['country_iso'] == event_b['country_iso']
    within_window = abs(
        (event_a['occurred_at_est'] - event_b['occurred_at_est']).days
    ) <= 14

    return similarity > 0.4 and same_country and within_window
```

#### Hybrid Storyline Dedup (V20.2 — geo-assist + LLM adjudicator + alert fingerprint)

Pure lexical Jaccard on `storyline_hint` fails for the dominant real-world case: one
incident (e.g. a Kyiv strike) reported by ~20 sources whose paraphrased hints share
almost no tokens ("Kyiv Russia drone strike" vs "Ukrainian capital missile attack").
Each fragment spawned a new `storyline_id`, and because the Telegram suppression key was
keyed on `storyline_id`, every source paged separately. Three layers fix this:

- **Layer 1 — coarse geo-assist** (`src/core/geo.py`). The airport `anchor_master`
  gazetteer is IATA-centric, so city events never resolve to an anchor. `geo_key()` is a
  DB-free, paraphrase-stable coarse key (`Kyiv`/`Kiev`/`Ukraine capital` → `KYIV`), wired
  into `should_link_storyline` as a fallback "anchor". It requires a 0.2 lexical floor —
  never a pure-time auto-link — so two DISTINCT same-city incidents are not merged on
  geography alone.
- **Layer 2 — LLM adjudicator** (`src/core/storyline_adjudicator.py`). When deterministic
  linking finds no match, same-country + same-geo + near-time candidates (the ambiguous,
  near-zero-overlap residue) are judged by a single LLM call: "SAME real-world incident,
  or NEW?". Runs on the **bulk router (gpt-oss-20b)** so it never competes with Pass C
  classification quota; fails safe to NEW (can only ever merge, never lose an event).
  Config-gated via `storyline.llm_adjudication_enabled`.
- **Layer 3 — storyline-independent alert fingerprint** (`build_geo_suppression_key`).
  Alongside the primary suppression key, `dispatch_alert` also checks/records a
  `geofp|country|geo|severity_bucket` key, so duplicate alerts collapse within the TTL
  even if the storyline still fragments.

Pass C is also nudged to emit canonical hints (city name over descriptors like
"capital", consistent `LOCATION → ACTOR → ACTION` order) so paraphrases converge upstream.

#### Scoring (V20.2 — umbrella rebalance + incident gate)

- **Severity:** `Base_Type_Weight + Proximity_Bonus (+30) + CZIB_Bonus (+20) + Casualty_Bonus`. Max 100.
- **Confidence:** `Max(0.0, Min(1.0, (llm_confidence * 0.4) + (anchor_score_val * 0.3) + (diversity_score * 0.3)))`
- **Umbrella rebalance** (migration `012`): generic parent types the LLM over-uses as
  catch-alls were dropped — `geopolitical_conflict` 85→45, `political_event` 60→35. The
  SPECIFIC incident children (missile_strike 100, military_action 95, war_escalation 90,
  civilian_casualties 92, …) are unchanged and still carry the real severity.
- **Incident gate** (defense-in-depth, `compute_severity`): a generic umbrella type with
  NO located anchor (no proximity bonus) AND NO reported casualties is commentary/analysis,
  not an actionable incident, and is capped at `scoring.incident_gate_cap` (50). This makes
  an LLM mislabel — e.g. an inflation/CPI survey or a corporate "companies remaining in
  Russia" story tagged `geopolitical_conflict` — impossible to escalate to near-critical,
  regardless of the type's catalog base.

---

### PASS E: Targeted Reconciliation

Unchanged. Strictly NO LLM. Re-evaluate anchors on concatenated text, clear Top-10 arrays on anchor upgrade, recalculate scores.

---

## 5. UI & Alerting (Streamlit — Fully Specified)

### 5.1 Component Inventory (NEW)

V19 mentioned Streamlit in a single heading. V20 provides a full file tree and reference snippets.

```
streamlit_app/
├── app.py                       # Entry point — st.set_page_config, routing
├── components/
│   ├── map_view.py              # PyDeck ScatterplotLayer + tooltip
│   ├── event_table.py           # Filterable/sortable event dataframe
│   ├── alert_feed.py            # Real-time alert stream (st.empty loop)
│   ├── storyline_graph.py       # NetworkX + streamlit-agraph
│   ├── telemetry_dashboard.py   # LLM quota, lock metrics, pipeline health
│   └── anchor_lookup.py         # anchor_master search + manual alias add UI
├── services/
│   ├── supabase_client.py       # Connection pool singleton
│   ├── cache.py                 # @st.cache_data wrappers (TTL 60 s)
│   └── alert_engine.py          # Tier evaluation + suppression key logic
└── config/
    └── ui_settings.json         # Color palette, refresh intervals, tier colors
```

#### `map_view.py` — Core Logic

```python
import pydeck as pdk
import streamlit as st

TIER_COLORS = {
    'CRITICAL': [220, 38,  38,  200],
    'ALERT':    [234, 88,  12,  180],
    'WATCH':    [202, 138, 4,   160],
    None:       [100, 116, 139, 100],
}

def render_map(events_df):
    events_df['color'] = events_df['alert_tier'].map(
        lambda t: TIER_COLORS.get(t, TIER_COLORS[None])
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=events_df,
        get_position='[longitude, latitude]',
        get_color='color',
        get_radius='severity_score * 200',
        pickable=True,
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        tooltip={"text": "{anchor_name_norm}\n{event_type}\nSeverity: {severity_score}"}
    ))
```

#### `telemetry_dashboard.py` — Key Metrics

```python
def render_telemetry(db_conn, llm_router=None):
    col1, col2, col3, col4 = st.columns(4)

    llm_stats = db_conn.fetchone(
        """SELECT COUNT(*) AS calls,
                  COALESCE(SUM((value_json->>'tokens_used')::int), 0) AS tokens
           FROM   system_telemetry
           WHERE  event_type = 'llm_call'
             AND  timestamp  > NOW() - INTERVAL '24h'"""
    )
    stale = db_conn.fetchone(
        """SELECT COUNT(*) AS n FROM system_telemetry
           WHERE event_type = 'stale_lock_cleared'
             AND timestamp  > NOW() - INTERVAL '1h'"""
    )

    # V20.1: Dynamic quota from LLMRouter instead of hardcoded 1000
    total_quota = llm_router.total_daily_quota if llm_router else 1000
    total_used  = llm_router.total_daily_used  if llm_router else llm_stats['calls']

    col1.metric("LLM Calls (24h)",      llm_stats['calls'])
    col2.metric("Tokens Used (24h)",    f"{llm_stats['tokens']:,}")
    col3.metric("Stale Locks (1h)",     stale['n'])
    col4.metric("Daily Quota Remaining",
                f"{total_quota - total_used} / {total_quota}")

    # V20.1: Per-account status breakdown
    if llm_router:
        st.subheader("Provider Account Status")
        status_data = []
        for acct in llm_router._accounts:
            status_data.append({
                "Provider": f"{acct.provider}/{acct.account_id}",
                "Model": acct.model,
                "Status": acct.status.value,
                "Used/Limit": f"{acct.bucket._daily_used}/{acct.rpd}",
                "RPM": acct.rpm,
            })
        st.dataframe(status_data, width="stretch")
```

---

### 5.2 Alert Gate — 3-Tier System (NEW)

**Problem:** V19's triple-AND gate with high thresholds could silently miss critical early-stage events where not all three criteria are simultaneously met.

**Solution:** Tiered gate. V19's original gate is preserved as the CRITICAL tier. Two lower tiers are added beneath it.

```python
from dataclasses import dataclass

@dataclass
class AlertTier:
    name:             str
    color:            str
    notify_channels:  list[str]

TIERS = {
    'CRITICAL': AlertTier('CRITICAL', '#DC2626', ['telegram', 'email', 'sms']),
    'ALERT':    AlertTier('ALERT',    '#EA580C', ['telegram', 'email']),
    'WATCH':    AlertTier('WATCH',    '#CA8A04', ['telegram']),
}

def evaluate_alert_tier(event: dict) -> str | None:
    sev   = event['severity_score']
    conf  = event['system_confidence']
    anc   = event['anchor_confidence']
    time_ = event['time_certainty']

    # CRITICAL — original V19 gate, unchanged
    if (sev >= 80 and conf >= 0.8
            and anc == 'HIGH' and time_ != 'unknown'):
        return 'CRITICAL'

    # ALERT — mid tier, actionable but not highest urgency
    if (sev >= 65 and conf >= 0.65
            and anc in ('HIGH', 'MEDIUM') and time_ != 'unknown'):
        return 'ALERT'

    # WATCH — early signal, situational awareness only
    if (sev >= 45 and conf >= 0.5
            and time_ in ('same_day', 'previous_day')):
        return 'WATCH'

    return None
```

### 5.3 Alert Suppression (Unchanged — `anchor_name_norm` clarified)

Composite suppression key: `storyline_id + anchor_name_norm + floor(severity_score / 10)`

`anchor_name_norm` MUST be the IATA/ICAO code (e.g. `CAI`), never the raw string. Using raw text causes key fragmentation ("Cairo Intl" vs "Cairo International" = two distinct keys → double alerts).

```python
def build_suppression_key(event: dict) -> str:
    return "|".join([
        str(event.get('storyline_id') or 'no_storyline'),
        event.get('anchor_name_norm') or 'UNKNOWN',   # IATA code
        str(int(event['severity_score'] // 10) * 10)  # e.g. 85 → 80
    ])

# Mute identical keys for 4 hours (Redis TTL or DB timestamp check)
```

---

## 6. Comprehensive QA Checklist (Testable Acceptance Criteria)

V19 left each item as a yes/no checkbox. V20 assigns concrete acceptance criteria, test pseudocode, and an automation note to every item.

---

### QA-01: Heartbeat Loop Termination Flag

**Check:** Does `_HeartbeatWorker` stop within 10 seconds after the LLM call succeeds, fails, or crashes?

**Acceptance Criteria:**
- `_stop_event.is_set()` is `True` immediately after the `with` block exits.
- `_thread.join(timeout=10)` completes; `_thread.is_alive()` → `False`.
- No further DB writes occur after the `with` block exits.

```python
def test_heartbeat_stops_after_llm():
    mock_db = MagicMock()
    hb = _HeartbeatWorker(mock_db, "event-1", "owner-1", interval=1)
    with hb:
        time.sleep(0.1)   # Simulated LLM call
    time.sleep(2)
    call_count_before = mock_db.execute.call_count
    time.sleep(2)
    assert mock_db.execute.call_count == call_count_before  # no new writes
```

**Automation:** pytest + `unittest.mock`.

---

### QA-02: Stale Lock Telemetry Written Before Deletion

**Check:** Are `lock_owner` and `last_heartbeat_at` committed to `system_telemetry` BEFORE the lock is cleared?

**Acceptance Criteria:**
- Telemetry INSERT's transaction commit timestamp precedes the `UPDATE events SET classification_lock=false`.
- `value_json` contains `lock_owner`, `lock_ts`, `last_heartbeat_at`, and `stale_duration_seconds`.
- If the telemetry write fails, the lock deletion is rolled back (same transaction).

```python
def test_stale_lock_telemetry_before_delete():
    ops = []
    mock_db.on_execute = lambda sql, *a: ops.append(
        'telemetry' if 'system_telemetry' in sql else 'lock'
    )
    clear_stale_lock(mock_db, stale_event)
    assert ops.index('telemetry') < ops.index('lock')
```

**Automation:** pytest + ordered mock call tracking.

---

### QA-03: `anchor_name_norm` Generation and Usage

**Check:** Does `normalize_anchor()` run for every event, and is `anchor_name_norm` used as the suppression key instead of `anchor_name_raw`?

**Acceptance Criteria:**
- `"Cairo International Airport"` → `anchor_name_norm = 'CAI'`, confidence ≥ 0.6.
- Arabic alias `"القاهرة الدولي"` → `anchor_name_norm = 'CAI'`, confidence ≥ 0.8.
- Unrecognized anchor → `anchor_name_norm = NULL`, `anchor_confidence = 'LOW'`.
- Suppression key falls back to `'UNKNOWN'` when `anchor_name_norm IS NULL`.

```python
@pytest.mark.parametrize("raw,expected_norm,min_conf", [
    ("CAI",                         "CAI", 1.0),
    ("Cairo Intl",                  "CAI", 0.8),
    ("القاهرة الدولي",             "CAI", 0.8),
    ("Cairo International Airport", "CAI", 0.6),
    ("Unknown Airspace",            None,  0.0),
])
def test_normalize_anchor(raw, expected_norm, min_conf, seeded_db):
    norm, conf = normalize_anchor(raw, seeded_db)
    assert norm == expected_norm
    assert conf >= min_conf
```

**Automation:** pytest + seeded test database fixture.

---

### QA-04: Idempotent Lock Release

**Check:** Does a `finally`-block UPDATE that affects 0 rows complete without raising an exception?

**Acceptance Criteria:**
- When mock DB returns `rowcount = 0`, `classify_event()` returns normally.
- An `INFO`-level log entry is written.
- No exception propagates to the caller.

```python
def test_lock_release_zero_rows_no_exception():
    mock_db.execute = MagicMock(return_value=MockResult(rowcount=0))
    try:
        classify_event(mock_db, "event-1", "owner-1", "test payload")
    except Exception as e:
        pytest.fail(f"Unexpected exception on zero-row lock release: {e}")
```

**Automation:** pytest.

---

### QA-05: Storyline Jaccard Tokenization

**Check:** Does bigram tokenization produce correct similarity scores and avoid cross-airport false positives?

**Acceptance Criteria:**
- Same event type at two airports in different countries does NOT get linked (country mismatch guard).
- Two differently worded descriptions of the same event at the same airport DO get linked.
- Empty or single-token hints produce similarity 0.0, not a division error.

```python
def test_jaccard_no_false_positive():
    lhr = {'storyline_hint': 'terminal bomb threat LHR', 'country_iso': 'GB',
           'occurred_at_est': datetime(2025, 1, 1)}
    cdg = {'storyline_hint': 'terminal bomb threat CDG', 'country_iso': 'FR',
           'occurred_at_est': datetime(2025, 1, 1)}
    assert not should_link_storyline(lhr, cdg)

def test_jaccard_connects_same_event():
    e1 = {'storyline_hint': 'runway incursion CAI', 'country_iso': 'EG',
          'occurred_at_est': datetime(2025, 1, 1)}
    e2 = {'storyline_hint': 'CAI runway incident', 'country_iso': 'EG',
          'occurred_at_est': datetime(2025, 1, 2)}
    assert should_link_storyline(e1, e2)

def test_jaccard_empty_hint():
    assert jaccard_similarity('', 'runway incursion') == 0.0
```

**Automation:** pytest.

---

### QA-06: Alert Tier Logic (NEW)

**Check:** Does `evaluate_alert_tier()` assign the correct tier for all boundary conditions?

**Acceptance Criteria:**
- CRITICAL threshold met → `'CRITICAL'`, SMS channel included.
- Only ALERT threshold met → `'ALERT'`, SMS channel NOT included.
- Only WATCH threshold met → `'WATCH'`.
- No threshold met → `None`.

```python
@pytest.mark.parametrize("sev,conf,anc,time_,expected", [
    (85, 0.85, 'HIGH',   'same_day',     'CRITICAL'),
    (70, 0.70, 'MEDIUM', 'same_day',     'ALERT'),
    (50, 0.55, 'LOW',    'previous_day', 'WATCH'),
    (30, 0.30, 'LOW',    'unknown',       None),
    # Boundary: CRITICAL fails on anchor_confidence
    (85, 0.85, 'MEDIUM', 'same_day',     'ALERT'),
])
def test_alert_tier(sev, conf, anc, time_, expected):
    event = dict(severity_score=sev, system_confidence=conf,
                 anchor_confidence=anc, time_certainty=time_)
    assert evaluate_alert_tier(event) == expected
```

**Automation:** pytest parametrize.

---

### QA-07: LLM Rate Limiter (NEW)

**Check:** Does the token bucket correctly enforce per-minute and daily limits?

**Acceptance Criteria:**
- Acquiring 21 tokens at `rate_per_minute=20` with `timeout=0` raises `TimeoutError`.
- Reaching `daily_limit=1000` raises `RuntimeError("Daily LLM quota exhausted")`.
- `system_telemetry` receives one record per LLM call containing `tokens_used` and `daily_used`.

```python
def test_rate_limiter_per_minute():
    bucket = TokenBucket(rate_per_minute=20, daily_limit=None)
    for _ in range(20):
        bucket.acquire(timeout=0)
    with pytest.raises(TimeoutError):
        bucket.acquire(timeout=0)

def test_rate_limiter_daily_quota():
    bucket = TokenBucket(rate_per_minute=10000, daily_limit=5)
    for _ in range(5):
        bucket.acquire(timeout=0)
    with pytest.raises(RuntimeError, match="Daily LLM quota exhausted"):
        bucket.acquire(timeout=0)
```

**Automation:** pytest.

---

### QA-08: Cold Storage Idempotency (Carried over from V19)

**Check:** Is the 5-step state machine enforced end-to-end?

**Acceptance Criteria:**
- Pushing the same JSONL twice results in exactly one manifest record (`NOT EXISTS` guard).
- Manifest is NOT written before `200 OK` is received from the archive endpoint.
- `DELETE` does NOT execute before manifest is committed.
- A simulated HTTP 5xx causes the run to abort before DELETE; next run retries safely.

**Automation:** pytest + `responses` library to mock HTTP endpoint.

---

### QA-09: Multi-Provider LLM Failover (NEW — V20.1)

**Check:** Does `LLMRouter` correctly rotate through accounts on rate-limit and exhaust scenarios?

**Acceptance Criteria:**
- Groq-A rate-limited → automatic rotation to Groq-A secondary model, <2s added latency.
- All Groq accounts exhausted → seamless failover to OpenRouter-A.
- All smart accounts exhausted → fallback to `openai/gpt-oss-20b` (1K RPD bulk slot).
- All accounts exhausted → `RuntimeError` raised, pipeline defers to next cycle, telemetry record written.
- Correct `provider`, `account`, `model` fields written to `system_telemetry` after each rotation.
- Rate-limited accounts auto-recover after cooldown period (60s RPM, 120s 429).

```python
def test_router_failover_on_rate_limit():
    accounts = [
        make_account("groq", "A", "openai/gpt-oss-120b", rpd=1),
        make_account("groq", "A", "qwen/qwen3.6-27b", rpd=1000),
    ]
    router = LLMRouter(accounts)
    # Exhaust first account
    acct1 = router.get_available_account()
    assert acct1.model == "openai/gpt-oss-120b"
    router.report_failure(acct1, is_rate_limit=True)
    # Should rotate to second
    acct2 = router.get_available_account()
    assert acct2.model == "qwen/qwen3.6-27b"

def test_router_all_exhausted():
    accounts = [make_account("groq", "A", "test-model", rpd=0)]
    router = LLMRouter(accounts)
    assert router.get_available_account() is None

def test_router_cooldown_recovery():
    accounts = [make_account("groq", "A", "test-model", rpd=1000)]
    router = LLMRouter(accounts)
    acct = router.get_available_account()
    router.report_failure(acct, is_rate_limit=True)
    # Immediately after → should be None (in cooldown)
    assert router.get_available_account() is None
    # After cooldown → should recover
    acct.cooldown_until = time.monotonic() - 1
    recovered = router.get_available_account()
    assert recovered is not None
```

**Automation:** pytest + `unittest.mock`.

---

## 7. Error Recovery Matrix (V20.1 — Updated)

| Scenario | Detection | Automatic Recovery | Manual Intervention |
|---|---|---|---|
| LLM API HTTP 429 | `tenacity` + `LLMRouter.report_failure(is_rate_limit=True)` | Auto-rotate to next account in priority chain | No — automatic |
| LLM JSON parse failure | `llm_parse_error` column populated | Fallback to `other_aviation_related` | No |
| Heartbeat DB connection drop | Exception caught, consecutive error counter | Retry on next iteration; stop after 5 consecutive failures | No |
| Stale lock > 15 min | Pass B startup check | Auto-clear + telemetry record | No |
| `anchor_master` trigram 0 results | `normalize_anchor` returns `(None, 0.0)` | NULL anchor, `anchor_confidence = LOW` | Periodic seed refresh |
| PostGIS query timeout | `statement_timeout` setting | Skip Pass E, preserve existing score | No |
| GitHub Actions concurrency cancel | `cancel-in-progress: false` | No cancellation by design | By design |
| Cold storage endpoint HTTP 5xx | `requests` retry adapter (3 attempts) | Retry in next 30-min cycle | Check archive endpoint logs |
| All LLM accounts exhausted | `LLMRouter.get_available_account()` returns `None` | Pipeline skips LLM pass, logs telemetry, retries next cycle | Check account statuses in dashboard |
| Groq account rate-limited | HTTP 429 + `x-ratelimit-*` headers | Auto-rotate to next account (60s cooldown) | No |
| OpenRouter free model removed | HTTP 404 on model endpoint | Mark account as ERROR, failover to next | Update model config |
| API key revoked/expired | HTTP 401 | Mark account as ERROR, continue with remaining accounts | Regenerate key, update secret |
| Groq org rate limit (not per-key) | 429 on all models under same org | Switch to second Groq org (Account B) | No |

---

## 8. Security Notes (V20.1 — Updated)

- **Supabase RLS:** Row Level Security MUST be enabled on the `events` table. Streamlit MUST use the service role key, never the anon key.
- **`anchor_master` write protection:** A dedicated DB role with `INSERT`-only on `anchor_master` for the seed job; no direct writes from the pipeline worker.
- **`system_telemetry` retention:** Partition by `RANGE(timestamp)` with 90-day auto-drop to prevent unbounded growth.
- **LLM prompt injection:** Pipeline ingests OSINT text rather than end-user input, so risk is low. However, patterns such as `[INST]`, `<|system|>`, and `IGNORE PREVIOUS INSTRUCTIONS` MUST be stripped or escaped during Pass A canonicalization before text reaches the LLM prompt.
- **Secrets management (GitHub Actions):** All API keys stored as GitHub Actions encrypted secrets; never hardcoded in `config/` files. Four LLM keys required: `GROQ_API_KEY_A`, `GROQ_API_KEY_B`, `OPENROUTER_API_KEY_A`, `OPENROUTER_API_KEY_B`.
- **Secrets management (Streamlit Cloud):** Use `.streamlit/secrets.toml` for Supabase and LLM keys. File MUST be in `.gitignore`. See Section 4.5.7 for format.
- **Multi-account isolation:** Groq accounts MUST be separate organizations (separate email signups) to ensure independent rate limits. OpenRouter accounts MUST be separate user registrations.

---

## 9. Dependency Inventory (V20.1 — Updated)

| Component | Package | Min Version | Notes |
|---|---|---|---|
| HTTP client | `httpx` | ≥ 0.27 | V20.1: Required for multi-provider LLM calls |
| LLM retry/backoff | `tenacity` | ≥ 8.2 | |
| PostgreSQL driver | `psycopg[binary]` | ≥ 3.1 | V20.1: Upgraded from `psycopg2-binary` — async support |
| Supabase client | `supabase-py` | ≥ 2.0 | |
| Streamlit | `streamlit` | ≥ 1.35 | |
| Map rendering | `pydeck` | ≥ 0.9 | |
| Storyline graph | `streamlit-agraph` | ≥ 0.0.45 | |
| eTLD+1 normalization | `tldextract` | ≥ 5.1 | |
| Env management | `python-dotenv` | ≥ 1.0 | V20.1: Local dev secret loading |
| HTTP mock in tests | `responses` | ≥ 0.25 | |
| Test framework | `pytest`, `pytest-mock` | ≥ 8.x | |

---

*SIM Blueprint V20.1 — Multi-Provider Production Fortress. V20 gaps closed: heartbeat implementation, soft-ENUM, anchor_master schema, 3-tier alert system, LLM rate limiting, Streamlit component inventory, testable QA criteria, error recovery matrix, security notes. V20.1 additions: Groq + OpenRouter multi-account LLM strategy (9 model slots, ~33K RPD capacity), provider-aware failover router, fixed TokenBucket day reset, HeartbeatWorker resilience, classify_event transaction safety, event_type_catalog active guard, dynamic telemetry dashboard, QA-09 failover tests.*
