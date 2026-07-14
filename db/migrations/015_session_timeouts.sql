-- 015: Role-level session timeouts.
--
-- Incident 2026-07-13: Pass E sat in a silent lock wait for 14 minutes, three
-- consecutive runs burned the whole GitHub Actions budget and were killed
-- before run telemetry could be written, so the dead-man's switch paged for a
-- pipeline that was actually running. Nothing in the stack bounded how long a
-- statement could block.
--
-- Role-level settings apply at session start on the SERVER, so they survive
-- Supavisor transaction-mode pooling where per-connection SET does not.
-- CURRENT_USER = whatever role DATABASE_URL logs in with.
--
-- statement_timeout: no pipeline statement legitimately runs > 2 minutes
--   (Pass E reconciles ~1 event/second; archive batches are 500 rows).
-- lock_timeout: fail a blocked statement after 30s instead of waiting forever;
--   the pipeline already logs + skips per-event errors and retries next run.
-- idle_in_transaction_session_timeout: kills forgotten open transactions
--   (e.g. a SQL-editor tab) — the likely lock holder in the Jul 13 incident.
--   15 min, NOT lower: psycopg auto-begins a transaction on the first read and
--   Pass C legitimately idles inside it for minutes during LLM batch calls.

ALTER ROLE CURRENT_USER SET statement_timeout = '120s';
ALTER ROLE CURRENT_USER SET lock_timeout = '30s';
ALTER ROLE CURRENT_USER SET idle_in_transaction_session_timeout = '900s';
