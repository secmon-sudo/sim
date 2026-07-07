-- 012_rebalance_umbrella_severity.sql
-- Rebalance generic "umbrella" event types down to sane bases.
--
-- Problem: the LLM reaches for the broad parent `geopolitical_conflict` as a catch-all
-- for any geopolitics-flavoured story (e.g. an inflation survey mentioning a "U.S.-Iran
-- deal", or corporate ESG activism about companies remaining in Russia). That umbrella
-- carried severity_base = 85, so a single mislabel became a near-CRITICAL event with no
-- location and no casualties. The SPECIFIC incident children (missile_strike 100,
-- military_action 95, war_escalation 90, civilian_casualties 92, ...) are unchanged —
-- they carry the real severity. Only the generic parents are lowered here.
--
-- Idempotent: pure UPDATEs, safe to re-run.

UPDATE event_type_catalog SET severity_base = 45 WHERE code = 'geopolitical_conflict';
UPDATE event_type_catalog SET severity_base = 35 WHERE code = 'political_event';
