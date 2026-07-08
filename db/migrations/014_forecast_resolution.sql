-- ============================================================
-- SIM — Forecast Resolution (automated forecast verification)
-- Blueprint V20.1 — Migration 014
--
-- report_validations was created in 008 for analyst use but never
-- populated. This migration extends it so the weekly pipeline can
-- auto-resolve last week's forecasts against this week's computed
-- TI/delta/Z scores (zero-LLM, pure math).
-- ============================================================

ALTER TABLE report_validations
    ADD COLUMN IF NOT EXISTS resolution_kind VARCHAR(20),          -- 'auto' for pipeline-generated rows; NULL for analyst rows
    ADD COLUMN IF NOT EXISTS accuracy        NUMERIC,              -- fraction of per-country direction forecasts that verified
    ADD COLUMN IF NOT EXISTS brier           NUMERIC,              -- mean (confidence_prob - outcome)^2 over resolved countries
    ADD COLUMN IF NOT EXISTS details_json    JSONB DEFAULT '{}';   -- per-country forecast-vs-actual records

-- One automatic resolution per report (analyst rows are unconstrained).
CREATE UNIQUE INDEX IF NOT EXISTS idx_report_validations_auto
    ON report_validations(report_id) WHERE resolution_kind = 'auto';
