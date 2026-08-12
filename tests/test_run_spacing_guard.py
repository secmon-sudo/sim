"""Double-trigger run-spacing guard (orchestrator) — pure logic, no DB."""
class TestRunSpacingGuard:
    def _fake_conn(self, age_minutes):
        class R:
            def __init__(self, row): self._row = row
            def fetchone(self): return self._row
        class Conn:
            def execute(self, sql, params=None):
                self.sql = sql
                return R((age_minutes,) if age_minutes is not None else None)
        return Conn()

    def test_age_is_measured_from_run_start(self):
        """Spacing must key off started_at, not the completion timestamp.

        Measuring from completion made a long run plus GitHub cron drift look like a
        duplicate trigger: on 2026-08-12 run #1488 fired 114 min after the previous
        run STARTED but only 84 min after it FINISHED, was skipped as a duplicate, and
        the resulting 3h05m telemetry silence paged the dead-man's switch.
        """
        from src.pipeline.orchestrator import _last_successful_run_age_minutes
        conn = self._fake_conn(114.0)
        _last_successful_run_age_minutes(conn)
        assert "started_at" in conn.sql

    def test_real_next_slot_run_is_not_absorbed(self):
        """The #1488 timings must survive the guard once measured from start."""
        from src.pipeline.orchestrator import MIN_RUN_SPACING_MINUTES
        minutes_since_previous_start = 114.0
        assert minutes_since_previous_start >= MIN_RUN_SPACING_MINUTES

    def test_back_to_back_double_trigger_still_absorbed(self):
        """Genuine duplicates land 2-5 min apart and must still be skipped."""
        from src.pipeline.orchestrator import MIN_RUN_SPACING_MINUTES
        assert 4.0 < MIN_RUN_SPACING_MINUTES

    def test_recent_success_reports_age(self):
        from src.pipeline.orchestrator import _last_successful_run_age_minutes
        assert _last_successful_run_age_minutes(self._fake_conn(42.0)) == 42.0

    def test_no_prior_run_returns_none(self):
        from src.pipeline.orchestrator import _last_successful_run_age_minutes
        assert _last_successful_run_age_minutes(self._fake_conn(None)) is None

    def test_query_error_never_blocks(self):
        from src.pipeline.orchestrator import _last_successful_run_age_minutes
        class BrokenConn:
            def execute(self, *a): raise RuntimeError("db down")
        assert _last_successful_run_age_minutes(BrokenConn()) is None
