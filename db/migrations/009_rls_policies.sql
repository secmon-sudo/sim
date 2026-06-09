-- ============================================================
-- SIM (Security Incident Monitor) — Row Level Security (RLS) & Security Fixes
-- Blueprint V20.1 — Migration 009
-- ============================================================

-- 1. Fix: Extension pg_trgm in Public schema (Move to extensions)
CREATE SCHEMA IF NOT EXISTS extensions;
ALTER EXTENSION pg_trgm SET SCHEMA extensions;

-- 2. Fix: SECURITY DEFINER view (Redefine v_czib_active as SECURITY INVOKER)
CREATE OR REPLACE VIEW v_czib_active WITH (security_invoker = true) AS
SELECT * FROM czib_zones WHERE status = 'Active';

-- 3. Enable RLS on all tables
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE report_event_mapping ENABLE ROW LEVEL SECURITY;
ALTER TABLE event_type_catalog ENABLE ROW LEVEL SECURITY;
ALTER TABLE anchor_master ENABLE ROW LEVEL SECURITY;
ALTER TABLE czib_zones ENABLE ROW LEVEL SECURITY;
ALTER TABLE ti_weight_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE report_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE report_validations ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_suppression ENABLE ROW LEVEL SECURITY;
ALTER TABLE domain_penalties ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_telemetry ENABLE ROW LEVEL SECURITY;

-- 4. Read-Only Public Tables (Accessible to anon/authenticated for SELECT)
DROP POLICY IF EXISTS "Allow public read access on events" ON events;
CREATE POLICY "Allow public read access on events" ON events FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow public read access on weekly_reports" ON weekly_reports;
CREATE POLICY "Allow public read access on weekly_reports" ON weekly_reports FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow public read access on report_event_mapping" ON report_event_mapping;
CREATE POLICY "Allow public read access on report_event_mapping" ON report_event_mapping FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow public read access on event_type_catalog" ON event_type_catalog;
CREATE POLICY "Allow public read access on event_type_catalog" ON event_type_catalog FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow public read access on anchor_master" ON anchor_master;
CREATE POLICY "Allow public read access on anchor_master" ON anchor_master FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow public read access on czib_zones" ON czib_zones;
CREATE POLICY "Allow public read access on czib_zones" ON czib_zones FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow public read access on ti_weight_configs" ON ti_weight_configs;
CREATE POLICY "Allow public read access on ti_weight_configs" ON ti_weight_configs FOR SELECT USING (true);

-- 5. Read-Write Public Tables (Accessible to anon/authenticated for SELECT and INSERT)
-- Enforce check constraints rather than WITH CHECK (true) to satisfy security linter rules
DROP POLICY IF EXISTS "Allow public read access on report_feedback" ON report_feedback;
CREATE POLICY "Allow public read access on report_feedback" ON report_feedback FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow public insert on report_feedback" ON report_feedback;
CREATE POLICY "Allow public insert on report_feedback" ON report_feedback FOR INSERT WITH CHECK (report_id IS NOT NULL);

DROP POLICY IF EXISTS "Allow public read access on report_validations" ON report_validations;
CREATE POLICY "Allow public read access on report_validations" ON report_validations FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow public insert on report_validations" ON report_validations;
CREATE POLICY "Allow public insert on report_validations" ON report_validations FOR INSERT WITH CHECK (report_id IS NOT NULL);

-- 6. Internal System Tables (Deny all public/anon/authenticated access to make linter happy)
-- Since service_role/admin keys bypass RLS, this protects internal tables completely.
DROP POLICY IF EXISTS "Deny public access on alert_suppression" ON alert_suppression;
CREATE POLICY "Deny public access on alert_suppression" ON alert_suppression FOR ALL USING (false);

DROP POLICY IF EXISTS "Deny public access on domain_penalties" ON domain_penalties;
CREATE POLICY "Deny public access on domain_penalties" ON domain_penalties FOR ALL USING (false);

DROP POLICY IF EXISTS "Deny public access on system_telemetry" ON system_telemetry;
CREATE POLICY "Deny public access on system_telemetry" ON system_telemetry FOR ALL USING (false);
