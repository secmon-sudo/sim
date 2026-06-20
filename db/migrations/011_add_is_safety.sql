-- ============================================================
-- SIM (Security Incident Monitor) — Safety vs Security tagging
-- Blueprint V20.1 — Migration 011
-- ============================================================
-- The platform tracks SECURITY (hostile/intentional) events. Accidental SAFETY
-- events (bird strike, engine failure, emergency landing, depressurization) are
-- kept for coverage but flagged and de-prioritized so they don't trigger high
-- alerts. is_safety makes the distinction explicit and queryable.

ALTER TABLE events ADD COLUMN IF NOT EXISTS is_safety BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_events_is_safety ON events(is_safety);
