-- ============================================================
-- Migration 006: Add ON DELETE CASCADE to alert_suppression.event_id
-- Fixes ForeignKeyViolation when Pass F archives old events
-- ============================================================

-- Drop existing FK constraint and re-add with CASCADE
ALTER TABLE alert_suppression
    DROP CONSTRAINT IF EXISTS alert_suppression_event_id_fkey;

ALTER TABLE alert_suppression
    ADD CONSTRAINT alert_suppression_event_id_fkey
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE;

-- Cleanup: remove expired suppression entries
DELETE FROM alert_suppression WHERE expires_at < NOW();
