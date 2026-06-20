-- ============================================================
-- SIM (Security Incident Monitor) — Storyline Narratives (cache)
-- Blueprint V20.1 — Migration 010
-- ============================================================
-- Caches the LLM-generated "story so far" prose per storyline. The `signature`
-- column lets the narrator skip regeneration when a storyline is unchanged, so the
-- LLM is only called when a storyline gains new events (token-frugal).

CREATE TABLE IF NOT EXISTS storyline_narratives (
    storyline_id   UUID PRIMARY KEY,
    narrative      TEXT,
    summary_json   JSONB,
    signature      TEXT,
    event_count    INT          DEFAULT 0,
    peak_severity  INT          DEFAULT 0,
    severity_trend VARCHAR(20),
    llm_provider   VARCHAR(40),
    llm_model      VARCHAR(80),
    updated_at     TIMESTAMP    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_storyline_narratives_updated
    ON storyline_narratives(updated_at DESC);

-- Mirror the RLS posture of the other tables (migration 009).
ALTER TABLE storyline_narratives ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow public read access on storyline_narratives" ON storyline_narratives;
CREATE POLICY "Allow public read access on storyline_narratives"
    ON storyline_narratives FOR SELECT USING (true);
