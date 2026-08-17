"""The dead-man's switch must also notice a dead BACKSTOP.

The pipeline has two triggers: GitHub's schedule (best-effort, drops firings) and a
cron-job.org workflow_dispatch job that exists to cover for it. Measured 2026-08-17:
the dispatch trigger stopped on 12 Aug at 06:56Z — 8 runs a day, then 4, then zero,
with cron-job.org getting 422 from GitHub — and nothing paged for five days, because
the staleness check only asks whether the pipeline ran and 12 scheduled runs a day said
yes. So the redundancy was gone while every monitor read green.

These tests pin the properties that make this check trustworthy rather than noisy: it
pages when the backstop is absent, it stays silent about its own failures, it cannot
spam an hourly cron, and it never interferes with the primary staleness page.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from scripts import deadman_check as dm


def _api_response(created_at=None, status=200, empty=False):
    body = {"workflow_runs": []}
    if not empty and created_at is not None:
        body = {"workflow_runs": [{"created_at": created_at}]}
    return httpx.Response(status, json=body, request=httpx.Request("GET", "https://x"))


def _iso(hours_ago):
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _conn(cooldown_age_hours=None):
    """A psycopg-shaped connection whose cooldown lookup returns the given age."""
    conn = MagicMock()
    conn.__enter__ = lambda s: s
    conn.__exit__ = lambda s, *a: False
    row = None if cooldown_age_hours is None else (cooldown_age_hours,)
    conn.execute.return_value.fetchone.return_value = row
    return conn


class TestAgeReading:
    def test_reads_age_of_newest_dispatch_run(self):
        with patch.object(dm.httpx, "get", return_value=_api_response(_iso(4.0))):
            age = dm.latest_dispatch_age_hours("secmon-sudo/sim")
        assert 3.9 < age < 4.2

    def test_no_dispatch_run_at_all_is_none_not_an_error(self):
        with patch.object(dm.httpx, "get", return_value=_api_response(empty=True)):
            assert dm.latest_dispatch_age_hours("secmon-sudo/sim") is None

    def test_queries_only_dispatch_triggered_runs(self):
        """Scheduled runs are the thing this check must NOT be reassured by."""
        with patch.object(dm.httpx, "get", return_value=_api_response(_iso(1))) as get:
            dm.latest_dispatch_age_hours("secmon-sudo/sim")
        assert get.call_args.kwargs["params"]["event"] == "workflow_dispatch"
        assert dm.DISPATCH_WORKFLOW_FILE in get.call_args.args[0]

    def test_falls_back_to_anonymous_when_the_token_is_rejected(self):
        """A token without actions:read would otherwise disable the monitor forever."""
        responses = [_api_response(status=403), _api_response(_iso(2))]
        with patch.object(dm.httpx, "get", side_effect=responses) as get:
            age = dm.latest_dispatch_age_hours("secmon-sudo/sim", token="bad")
        assert 1.9 < age < 2.2
        assert "Authorization" not in get.call_args.kwargs["headers"]


class TestPaging:
    def test_pages_when_the_backstop_is_stale(self):
        with patch.object(dm, "latest_dispatch_age_hours", return_value=30.0), \
             patch.object(dm.psycopg, "connect", return_value=_conn()), \
             patch.object(dm, "send_ops_alert") as alert:
            assert dm.check_dispatch_backstop("postgres://x", 6.0) is False
        assert alert.called
        assert "30.0h" in alert.call_args.args[0]

    def test_pages_when_no_dispatch_has_ever_fired(self):
        with patch.object(dm, "latest_dispatch_age_hours", return_value=None), \
             patch.object(dm.psycopg, "connect", return_value=_conn()), \
             patch.object(dm, "send_ops_alert") as alert:
            assert dm.check_dispatch_backstop("postgres://x", 6.0) is False
        assert "never" in alert.call_args.args[0]

    def test_silent_when_healthy(self):
        with patch.object(dm, "latest_dispatch_age_hours", return_value=2.5), \
             patch.object(dm.psycopg, "connect") as connect, \
             patch.object(dm, "send_ops_alert") as alert:
            assert dm.check_dispatch_backstop("postgres://x", 6.0) is True
        alert.assert_not_called()
        connect.assert_not_called()  # A healthy check must not even touch the DB.

    def test_says_scheduled_runs_are_unaffected(self):
        """The page must not read as an outage: the pipeline is still running."""
        with patch.object(dm, "latest_dispatch_age_hours", return_value=99.0), \
             patch.object(dm.psycopg, "connect", return_value=_conn()), \
             patch.object(dm, "send_ops_alert") as alert:
            dm.check_dispatch_backstop("postgres://x", 6.0)
        assert "Scheduled runs are unaffected" in alert.call_args.args[0]


class TestNoiseControl:
    def test_cooldown_suppresses_the_repeat(self):
        """Hourly cron + a fault that takes a human to fix = 24 identical cards a day."""
        with patch.object(dm, "latest_dispatch_age_hours", return_value=99.0), \
             patch.object(dm.psycopg, "connect", return_value=_conn(cooldown_age_hours=3.0)), \
             patch.object(dm, "send_ops_alert") as alert:
            assert dm.check_dispatch_backstop("postgres://x", 6.0, cooldown_hours=12.0) is False
        alert.assert_not_called()

    def test_pages_again_once_the_cooldown_expires(self):
        with patch.object(dm, "latest_dispatch_age_hours", return_value=99.0), \
             patch.object(dm.psycopg, "connect", return_value=_conn(cooldown_age_hours=20.0)), \
             patch.object(dm, "send_ops_alert") as alert:
            dm.check_dispatch_backstop("postgres://x", 6.0, cooldown_hours=12.0)
        assert alert.called

    def test_marker_row_is_recorded(self):
        conn = _conn()
        with patch.object(dm, "latest_dispatch_age_hours", return_value=99.0), \
             patch.object(dm.psycopg, "connect", return_value=conn), \
             patch.object(dm, "send_ops_alert"):
            dm.check_dispatch_backstop("postgres://x", 6.0)
        written = " ".join(str(c.args[0]) for c in conn.execute.call_args_list if c.args)
        assert "INSERT INTO system_telemetry" in written
        assert conn.commit.called

    def test_disabled_by_zero(self):
        with patch.object(dm, "latest_dispatch_age_hours") as api, \
             patch.object(dm, "send_ops_alert") as alert:
            assert dm.check_dispatch_backstop("postgres://x", 0) is True
        api.assert_not_called()
        alert.assert_not_called()


class TestFailureIsolation:
    @pytest.mark.parametrize("boom", [
        httpx.ConnectError("dns"),
        httpx.HTTPStatusError("rate limited", request=MagicMock(), response=MagicMock()),
        ValueError("garbage json"),
    ])
    def test_api_trouble_never_pages(self, boom):
        """This watches a redundancy — it must not invent an alert about a healthy system."""
        with patch.object(dm, "latest_dispatch_age_hours", side_effect=boom), \
             patch.object(dm, "send_ops_alert") as alert:
            assert dm.check_dispatch_backstop("postgres://x", 6.0) is True
        alert.assert_not_called()

    def test_backstop_crash_does_not_cost_the_staleness_page(self):
        """The primary check runs first and its page is the one that matters."""
        with patch.dict("os.environ", {"DATABASE_URL": "postgres://x"}, clear=False), \
             patch.object(dm, "check", return_value=False) as primary, \
             patch.object(dm, "check_dispatch_backstop",
                          side_effect=RuntimeError("boom")) as backstop:
            assert dm.main() == 0
        primary.assert_called_once()
        backstop.assert_called_once()
