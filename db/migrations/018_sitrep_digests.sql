-- ============================================================
-- SIM (Security Incident Monitor) — Daily cross-country SITREP digest
-- Migration 018
--
-- One row per daily run: the short executive briefing synthesised from that
-- run's country SITREPs. Separate from `sitreps` because it is not country
-- scoped — it is the run-level artifact.
-- ============================================================

CREATE TABLE IF NOT EXISTS sitrep_digests (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    window_start  TIMESTAMP NOT NULL,
    window_end    TIMESTAMP NOT NULL,
    country_isos  TEXT[],               -- countries covered by this digest
    digest_text   TEXT,                 -- raw LLM output (audit)
    digest_json   JSONB,                -- parsed sections used for rendering
    status        VARCHAR(20) DEFAULT 'completed'
                  CHECK (status IN ('completed', 'failed', 'skipped')),
    llm_provider  VARCHAR(30),
    llm_model     VARCHAR(100),
    r2_url        TEXT,
    error_message TEXT,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sitrep_digests_created ON sitrep_digests(created_at DESC);

ALTER TABLE sitrep_digests ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow public read access on sitrep_digests" ON sitrep_digests;
CREATE POLICY "Allow public read access on sitrep_digests" ON sitrep_digests FOR SELECT USING (true);
