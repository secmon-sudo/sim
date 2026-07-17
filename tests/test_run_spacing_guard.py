"""Double-trigger run-spacing guard (orchestrator) — pure logic, no DB."""
class TestRunSpacingGuard:
    def _fake_conn(self, age_minutes):
        class R:
            def __init__(self, row): self._row = row
            def fetchone(self): return self._row
        class Conn:
            def execute(self, sql, params=None):
                return R((age_minutes,) if age_minutes is not None else None)
        return Conn()

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
