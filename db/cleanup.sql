-- ============================================================
-- SIM — Database Cleanup Script
-- Run this in Supabase SQL Editor to clear ALL old data
-- WARNING: This permanently deletes events, telemetry, and suppressions
-- ============================================================

-- 1. Disable triggers and locks temporarily (if any)
-- 2. Delete all events
DELETE FROM events;

-- 3. Delete all telemetry
DELETE FROM system_telemetry;

-- 4. Delete all alert suppressions
DELETE FROM alert_suppression;

-- 5. Reset domain penalties (optional — keep if you want historical source reliability)
-- DELETE FROM domain_penalties;

-- 6. Verify counts are zero
SELECT 'events' AS table_name, COUNT(*) AS row_count FROM events
UNION ALL
SELECT 'system_telemetry', COUNT(*) FROM system_telemetry
UNION ALL
SELECT 'alert_suppression', COUNT(*) FROM alert_suppression;
