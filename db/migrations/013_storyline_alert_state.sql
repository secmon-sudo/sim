-- ============================================================
-- SIM (Security Incident Monitor) — Storyline Alert State
-- Blueprint V20.1 — Migration 013
-- ============================================================
-- Tracks the paging history of each storyline so we can (a) annotate an alert as an
-- ESCALATION when a storyline that already paged at a lower tier crosses into a higher
-- one, and (b) emit a single "storyline quiet" closure note when an alerted storyline
-- stops producing activity. Without this, a WATCH that later becomes CRITICAL just
-- fires as an unrelated fresh card with no "this escalated" context.

CREATE TABLE IF NOT EXISTS storyline_alert_state (
    storyline_id    UUID PRIMARY KEY,
    last_tier       VARCHAR(10),
    last_severity   INT          DEFAULT 0,
    peak_tier       VARCHAR(10),                 -- highest tier ever paged for this storyline
    label           TEXT,                        -- short human context (title · location)
    last_alerted_at TIMESTAMP    DEFAULT NOW(),
    closed          BOOLEAN      DEFAULT FALSE,   -- a quiet-closure note has been sent
    closed_at       TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_storyline_alert_state_open
    ON storyline_alert_state(closed, last_alerted_at);

-- Mirror the RLS posture of the other operational tables (migration 009): deny public.
ALTER TABLE storyline_alert_state ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Deny public access on storyline_alert_state" ON storyline_alert_state;
CREATE POLICY "Deny public access on storyline_alert_state"
    ON storyline_alert_state FOR ALL USING (false);
