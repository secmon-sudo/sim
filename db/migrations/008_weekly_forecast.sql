-- ============================================================
-- SIM (Security Incident Monitor) — Weekly Forecast Schema
-- Blueprint V20.1 — Migration 008
-- ============================================================

-- 1. ti_weight_configs
CREATE TABLE IF NOT EXISTS ti_weight_configs (
    config_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version        VARCHAR(50) NOT NULL UNIQUE,
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    is_active      BOOLEAN DEFAULT FALSE,
    w_volume       NUMERIC NOT NULL DEFAULT 0.15,
    w_diversity    NUMERIC NOT NULL DEFAULT 0.10,
    w_severity_avg NUMERIC NOT NULL DEFAULT 0.20,
    w_severity_max NUMERIC NOT NULL DEFAULT 0.15,
    w_delta        NUMERIC NOT NULL DEFAULT 0.15,
    w_recency      NUMERIC NOT NULL DEFAULT 0.10,
    w_quality      NUMERIC NOT NULL DEFAULT 0.05,
    w_cross_domain NUMERIC NOT NULL DEFAULT 0.05,
    w_critical     NUMERIC NOT NULL DEFAULT 0.05,
    notes          TEXT
);

-- Insert default active config
INSERT INTO ti_weight_configs (version, is_active, notes)
VALUES ('default_v1.0', TRUE, 'Default weights from spec')
ON CONFLICT (version) DO NOTHING;

-- 2. weekly_reports
CREATE TABLE IF NOT EXISTS weekly_reports (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_date         DATE NOT NULL,
    week_start          DATE NOT NULL,
    week_end            DATE NOT NULL,
    generated_at        TIMESTAMPTZ DEFAULT NOW(),
    is_flash            BOOLEAN DEFAULT FALSE,
    top_countries       TEXT[],
    deteriorating       TEXT[],
    watchlist           TEXT[],
    scores_json         JSONB NOT NULL DEFAULT '{}',
    llm_assessment_json JSONB NOT NULL DEFAULT '{}',
    html_payload        TEXT,
    r2_url              TEXT,
    telegram_message_id TEXT,
    model_version       VARCHAR(100) NOT NULL,
    prompt_version      VARCHAR(50) NOT NULL,
    config_id           UUID REFERENCES ti_weight_configs(config_id)
);

CREATE INDEX IF NOT EXISTS idx_weekly_reports_date ON weekly_reports(report_date DESC);
CREATE INDEX IF NOT EXISTS idx_weekly_reports_flash ON weekly_reports(is_flash);

-- 3. report_event_mapping
CREATE TABLE IF NOT EXISTS report_event_mapping (
    report_id UUID REFERENCES weekly_reports(id) ON DELETE CASCADE,
    event_id  UUID REFERENCES events(id) ON DELETE CASCADE,
    PRIMARY KEY (report_id, event_id)
);

-- 4. report_feedback
CREATE TABLE IF NOT EXISTS report_feedback (
    feedback_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id      UUID REFERENCES weekly_reports(id) ON DELETE CASCADE,
    analyst_id     VARCHAR(100),
    country        VARCHAR(10),
    agree_risk     BOOLEAN,
    agree_forecast BOOLEAN,
    comment        TEXT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- 5. report_validations
CREATE TABLE IF NOT EXISTS report_validations (
    validation_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id         UUID REFERENCES weekly_reports(id) ON DELETE CASCADE,
    validated_at      TIMESTAMPTZ DEFAULT NOW(),
    direction_correct BOOLEAN,
    escalation_recall NUMERIC,
    fp_rate           NUMERIC,
    analyst_notes     TEXT,
    used_for_tuning   BOOLEAN DEFAULT FALSE
);
