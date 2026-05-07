-- ============================================================
-- SIM (Security Incident Monitor) — Database Schema
-- Blueprint V20.1 — Full schema migration
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- 1. event_type_catalog — Soft-ENUM replacement
-- ============================================================
CREATE TABLE IF NOT EXISTS event_type_catalog (
    code          VARCHAR(60)  PRIMARY KEY,
    label_en      VARCHAR(120),
    parent_code   VARCHAR(60)  REFERENCES event_type_catalog(code),
    severity_base INT          DEFAULT 30,
    active        BOOLEAN      DEFAULT TRUE,
    created_at    TIMESTAMP    DEFAULT NOW()
);

INSERT INTO event_type_catalog (code, label_en, parent_code, severity_base, active, created_at) VALUES
    ('security_incident',      'Security Incident',       NULL,                80, TRUE, NOW()),
    ('bomb_threat',            'Bomb Threat',             'security_incident', 80, TRUE, NOW()),
    ('active_shooter',         'Active Shooter',          'security_incident', 90, TRUE, NOW()),
    ('hijacking',              'Hijacking',               'security_incident', 95, TRUE, NOW()),
    ('runway_incursion',       'Runway Incursion',        NULL,                60, TRUE, NOW()),
    ('emergency_landing',      'Emergency Landing',       NULL,                50, TRUE, NOW()),
    ('bird_strike',            'Bird Strike',             NULL,                30, TRUE, NOW()),
    ('engine_failure',         'Engine Failure',          NULL,                55, TRUE, NOW()),
    ('fire_on_board',          'Fire on Board',           NULL,                70, TRUE, NOW()),
    ('depressurization',       'Cabin Depressurization',  NULL,                65, TRUE, NOW()),
    ('unruly_passenger',       'Unruly Passenger',        NULL,                25, TRUE, NOW()),
    ('drone_incursion',        'Drone Incursion',         NULL,                45, TRUE, NOW()),
    ('laser_attack',           'Laser Attack',            NULL,                40, TRUE, NOW()),
    ('suspicious_package',     'Suspicious Package',      'security_incident', 70, TRUE, NOW()),
    ('evacuation',             'Evacuation',              NULL,                60, TRUE, NOW()),
    ('other_aviation_related', 'Other Aviation Related',  NULL,                20, TRUE, NOW())
ON CONFLICT (code) DO NOTHING;

-- ============================================================
-- 2. anchor_master — Airport/Location normalization
-- ============================================================
CREATE TABLE IF NOT EXISTS anchor_master (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    iata_code      VARCHAR(4)    UNIQUE,
    icao_code      VARCHAR(4),
    anchor_type    VARCHAR(20)   NOT NULL,
    canonical_name VARCHAR(200)  NOT NULL,
    aliases        JSONB         DEFAULT '[]',
    country_iso    CHAR(2)       NOT NULL,
    latitude       DOUBLE PRECISION,
    longitude      DOUBLE PRECISION,
    czib_flag      BOOLEAN       DEFAULT FALSE,
    updated_at     TIMESTAMP     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_anchor_aliases ON anchor_master USING GIN(aliases);
CREATE INDEX IF NOT EXISTS idx_anchor_iata    ON anchor_master(iata_code);
CREATE INDEX IF NOT EXISTS idx_anchor_trgm    ON anchor_master USING GIN(canonical_name gin_trgm_ops);

-- ============================================================
-- 3. domain_penalties — Source reliability tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS domain_penalties (
    domain         VARCHAR(255) PRIMARY KEY,
    penalty_score  FLOAT        DEFAULT 0.0,
    total_events   INT          DEFAULT 0,
    false_positives INT         DEFAULT 0,
    last_seen      TIMESTAMP    DEFAULT NOW(),
    created_at     TIMESTAMP    DEFAULT NOW()
);

-- ============================================================
-- 4. events — Main event table
-- ============================================================
CREATE TABLE IF NOT EXISTS events (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source fields
    source_url            TEXT        NOT NULL,
    source_url_hash       VARCHAR(64) NOT NULL UNIQUE,
    source_domain         VARCHAR(255),
    source_title          TEXT,
    raw_text              TEXT,
    canonical_text        TEXT,
    published_at          TIMESTAMP,
    ingested_at           TIMESTAMP   DEFAULT NOW(),

    -- Classification
    event_type            VARCHAR(60) REFERENCES event_type_catalog(code) DEFAULT 'other_aviation_related',
    sub_type              VARCHAR(60) REFERENCES event_type_catalog(code),
    alert_tier            VARCHAR(10) CHECK (alert_tier IN ('WATCH', 'ALERT', 'CRITICAL')),

    -- LLM outputs
    llm_raw_output        JSONB,
    llm_parsed_output     JSONB,
    llm_parse_error       TEXT,
    llm_provider          VARCHAR(30),
    llm_model             VARCHAR(100),

    -- Anchor / Location
    anchor_name_raw       TEXT,
    anchor_name_norm      VARCHAR(10),
    anchor_confidence     VARCHAR(10) CHECK (anchor_confidence IN ('HIGH', 'MEDIUM', 'LOW')),
    country_iso           CHAR(2),
    latitude              DOUBLE PRECISION,
    longitude             DOUBLE PRECISION,

    -- Temporal
    occurred_at_est       TIMESTAMP,
    time_certainty        VARCHAR(20) DEFAULT 'unknown',

    -- Scoring
    severity_score        INT         DEFAULT 0,
    system_confidence     FLOAT       DEFAULT 0.0,

    -- Storyline
    storyline_id          UUID,
    storyline_hint        TEXT,

    -- Pipeline state
    status                VARCHAR(20) DEFAULT 'raw'
                          CHECK (status IN ('raw', 'deduped', 'locked', 'classified', 'scored', 'reconciled', 'archived')),
    classification_lock   BOOLEAN     DEFAULT FALSE,
    lock_owner            UUID,
    last_heartbeat_at     TIMESTAMP,

    -- Metadata
    created_at            TIMESTAMP   DEFAULT NOW(),
    updated_at            TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_status      ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_url_hash    ON events(source_url_hash);
CREATE INDEX IF NOT EXISTS idx_events_storyline   ON events(storyline_id);
CREATE INDEX IF NOT EXISTS idx_events_anchor_norm ON events(anchor_name_norm);
CREATE INDEX IF NOT EXISTS idx_events_occurred    ON events(occurred_at_est);
CREATE INDEX IF NOT EXISTS idx_events_alert_tier  ON events(alert_tier);
CREATE INDEX IF NOT EXISTS idx_events_ingested    ON events(ingested_at DESC);

-- ============================================================
-- 5. system_telemetry — Pipeline health & LLM tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS system_telemetry (
    id          UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type  VARCHAR(60) NOT NULL,
    value_json  JSONB       NOT NULL DEFAULT '{}',
    timestamp   TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_telemetry_type ON system_telemetry(event_type);
CREATE INDEX IF NOT EXISTS idx_telemetry_ts   ON system_telemetry(timestamp);

-- ============================================================
-- 6. alert_suppression — Prevent duplicate alerts
-- ============================================================
CREATE TABLE IF NOT EXISTS alert_suppression (
    suppression_key VARCHAR(255) PRIMARY KEY,
    first_fired_at  TIMESTAMP    DEFAULT NOW(),
    expires_at      TIMESTAMP    NOT NULL,
    alert_tier      VARCHAR(10),
    event_id        UUID REFERENCES events(id)
);

CREATE INDEX IF NOT EXISTS idx_suppression_expires ON alert_suppression(expires_at);
