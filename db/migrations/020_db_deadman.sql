-- 020: Database-resident dead-man's switch.
--
-- Why this exists. The dead-man's switch ran only as a GitHub Actions workflow, i.e.
-- on the same infrastructure as the pipeline it watches. On 6 Aug 2026 that
-- assumption failed exactly the way it always would: a GitHub-side incident killed
-- pipeline runs #1374/#1375 AND deadman runs #318/#319, so the pipeline went silent
-- for 8h35m (15:26 -> 00:01) with nobody paging. A watchdog that shares fate with
-- its subject is not a watchdog.
--
-- Secondary problem: GitHub drops scheduled cron firings under load. The hourly
-- deadman actually fired 7-12 times a day, not 24, so its real resolution was 2-3h.
-- pg_cron fires on the database's own clock and does not skip.
--
-- The Actions workflow is deliberately KEPT as a second, independent watchdog: two
-- watchdogs in two failure domains beat one in either.
--
-- MANUAL STEP REQUIRED (once, and only the project owner can do it — it needs the
-- real bot token). Until these two vault secrets exist the function is a no-op and
-- pages nothing:
--
--   select vault.create_secret('<bot-token>',      'telegram_bot_token');
--   select vault.create_secret('<alerts-chat-id>', 'telegram_alerts_chat_id');
--
-- Everything below is idempotent: this file is re-executed on every pipeline run.

-- pg_net is already present; pg_cron may need enabling and may be refused over a
-- pooled connection. Neither must abort the migration, so both are guarded.
DO $ext$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_net;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_net unavailable: %', SQLERRM;
END
$ext$;

DO $ext$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_cron;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_cron unavailable: %', SQLERRM;
END
$ext$;


CREATE OR REPLACE FUNCTION public.sim_deadman_check(max_age_hours numeric DEFAULT 3.0)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, vault, net
AS $fn$
DECLARE
    v_ts      timestamp;
    v_age     numeric;
    v_success text;
    v_token   text;
    v_chat    text;
    v_msg     text;
BEGIN
    SELECT timestamp,
           EXTRACT(EPOCH FROM (NOW() - timestamp)) / 3600.0,
           value_json ->> 'success'
      INTO v_ts, v_age, v_success
      FROM public.system_telemetry
     WHERE event_type = 'pipeline_run'
     ORDER BY timestamp DESC
     LIMIT 1;

    -- Healthy: a run landed inside the window.
    IF v_ts IS NOT NULL AND v_age <= max_age_hours THEN
        RETURN;
    END IF;

    -- Re-page at most once every 2h so an outage produces a heartbeat, not a flood.
    IF EXISTS (
        SELECT 1 FROM public.system_telemetry
         WHERE event_type = 'deadman_page'
           AND timestamp > NOW() - INTERVAL '2 hours'
    ) THEN
        RETURN;
    END IF;

    BEGIN
        SELECT decrypted_secret INTO v_token
          FROM vault.decrypted_secrets WHERE name = 'telegram_bot_token';
        SELECT decrypted_secret INTO v_chat
          FROM vault.decrypted_secrets WHERE name = 'telegram_alerts_chat_id';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'vault unreadable: %', SQLERRM;
        RETURN;
    END;

    -- Not configured yet (see MANUAL STEP above) — stay silent rather than error.
    IF v_token IS NULL OR v_chat IS NULL THEN
        RETURN;
    END IF;

    IF v_ts IS NULL THEN
        v_msg := '🚨 DEAD-MAN (db): no pipeline run has ever been recorded.';
    ELSE
        v_msg := format(
            '🚨 DEAD-MAN (db): no pipeline run in %sh (threshold %sh). '
            || 'Last run %s UTC, success=%s. Paged from Postgres, so GitHub Actions '
            || 'itself may be down.',
            round(v_age, 1), round(max_age_hours, 0), v_ts, coalesce(v_success, '?')
        );
    END IF;

    PERFORM net.http_post(
        url     := 'https://api.telegram.org/bot' || v_token || '/sendMessage',
        body    := jsonb_build_object('chat_id', v_chat, 'text', v_msg),
        headers := '{"Content-Type": "application/json"}'::jsonb
    );

    INSERT INTO public.system_telemetry (event_type, value_json)
    VALUES ('deadman_page', jsonb_build_object(
        'source', 'db_pg_cron',
        'age_hours', round(v_age, 2),
        'last_run', v_ts
    ));
END
$fn$;

-- SECURITY DEFINER + vault access: keep it off the API roles. Guarded because a
-- non-Supabase target (local dev, CI) has no anon/authenticated roles.
REVOKE ALL ON FUNCTION public.sim_deadman_check(numeric) FROM PUBLIC;

-- Separate block: a missing anon/authenticated role (local dev, CI) must not roll
-- back the REVOKE above, which is the one that actually matters.
DO $grants$
BEGIN
    REVOKE ALL ON FUNCTION public.sim_deadman_check(numeric) FROM anon, authenticated;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'no anon/authenticated roles to revoke from: %', SQLERRM;
END
$grants$;


-- Every 30 minutes: half the 3h staleness threshold gives the switch a real chance
-- to fire twice inside a window rather than riding on a single firing.
DO $sched$
BEGIN
    PERFORM cron.schedule(
        'sim-deadman',
        '*/30 * * * *',
        $job$SELECT public.sim_deadman_check()$job$
    );
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'could not schedule sim-deadman: %', SQLERRM;
END
$sched$;
