-- ============================================================
-- SIM (Security Incident Monitor) — Daily Country SITREP storage
-- Migration 016
-- ============================================================

CREATE TABLE IF NOT EXISTS sitreps (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_iso   CHAR(2)   NOT NULL,
    window_start  TIMESTAMP NOT NULL,
    window_end    TIMESTAMP NOT NULL,
    report_text   TEXT,                 -- Turkish SITREP body
    events_json   JSONB,                -- structured event clusters incl. verification labels (audit + rendering)
    event_count   INT       DEFAULT 0,
    status        VARCHAR(20) DEFAULT 'completed'
                  CHECK (status IN ('completed', 'failed', 'empty')),
    llm_provider  VARCHAR(30),
    llm_model     VARCHAR(100),
    r2_url        TEXT,
    error_message TEXT,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sitreps_country ON sitreps(country_iso, created_at DESC);

ALTER TABLE sitreps ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow public read access on sitreps" ON sitreps;
CREATE POLICY "Allow public read access on sitreps" ON sitreps FOR SELECT USING (true);
