"""Tests for HeartbeatWorker — the lock keep-alive during long LLM calls.

The worker runs on its own thread with its own pooled connection while Pass C
waits on a model. Three things must hold or the pipeline leaks: the thread
always stops when the `with` block exits (success or exception), the connection
always returns to the pool, and the worker gives up rather than spinning when
the lock is gone or the database keeps failing.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.heartbeat import MAX_CONSECUTIVE_ERRORS, HeartbeatWorker

INTERVAL = 0.01


def _cursor(rowcount=1):
    c = MagicMock()
    c.rowcount = rowcount
    return c


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestLifecycle:
    def test_thread_stops_after_the_with_block(self):
        conn = MagicMock()
        conn.execute.return_value = _cursor()
        with patch("src.core.heartbeat.get_connection", return_value=conn), \
             patch("src.core.heartbeat.put_connection"):
            with HeartbeatWorker("evt-1", "owner-1", interval=INTERVAL) as hb:
                assert hb._thread.is_alive()
            assert not hb._thread.is_alive()

    def test_thread_stops_when_the_body_raises(self):
        # An LLM timeout inside the block must not leave a thread behind.
        conn = MagicMock()
        conn.execute.return_value = _cursor()
        with patch("src.core.heartbeat.get_connection", return_value=conn), \
             patch("src.core.heartbeat.put_connection"):
            worker = HeartbeatWorker("evt-2", "owner-1", interval=INTERVAL)
            with pytest.raises(ValueError):
                with worker:
                    raise ValueError("llm exploded")
            assert not worker._thread.is_alive()

    def test_exception_from_the_body_is_re_raised(self):
        conn = MagicMock()
        conn.execute.return_value = _cursor()
        with patch("src.core.heartbeat.get_connection", return_value=conn), \
             patch("src.core.heartbeat.put_connection"):
            with pytest.raises(RuntimeError, match="propagate me"):
                with HeartbeatWorker("evt-3", "owner-1", interval=INTERVAL):
                    raise RuntimeError("propagate me")

    def test_connection_is_returned_to_the_pool(self):
        conn = MagicMock()
        conn.execute.return_value = _cursor()
        with patch("src.core.heartbeat.get_connection", return_value=conn), \
             patch("src.core.heartbeat.put_connection") as put:
            with HeartbeatWorker("evt-4", "owner-1", interval=INTERVAL):
                pass
        assert _wait_until(lambda: put.call_count == 1)
        assert put.call_args.args[0] is conn

    def test_connection_returned_even_when_the_loop_dies(self):
        # Pool exhaustion is how a leaked connection shows up hours later.
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("db gone")
        with patch("src.core.heartbeat.get_connection", return_value=conn), \
             patch("src.core.heartbeat.put_connection") as put:
            with HeartbeatWorker("evt-5", "owner-1", interval=INTERVAL):
                time.sleep(INTERVAL * (MAX_CONSECUTIVE_ERRORS + 3))
        assert _wait_until(lambda: put.call_count == 1)

    def test_pool_checkout_failure_does_not_kill_the_caller(self, recwarn):
        # The worker thread is a daemon helper; if it cannot get a connection
        # the LLM call in the main thread must still proceed — and the failure
        # must be logged, not escape as an unhandled-thread traceback.
        with patch("src.core.heartbeat.get_connection", side_effect=RuntimeError("pool empty")), \
             patch("src.core.heartbeat.put_connection") as put:
            worker = HeartbeatWorker("evt-6", "owner-1", interval=INTERVAL)
            with worker:
                pass
            assert _wait_until(lambda: not worker._thread.is_alive())
        put.assert_not_called()

    def test_pool_checkout_failure_is_logged_not_raised(self, caplog):
        import logging as _logging
        with patch("src.core.heartbeat.get_connection", side_effect=RuntimeError("pool empty")), \
             patch("src.core.heartbeat.put_connection"), \
             caplog.at_level(_logging.ERROR, logger="src.core.heartbeat"):
            worker = HeartbeatWorker("evt-6b", "owner-1", interval=INTERVAL)
            with worker:
                pass
            assert _wait_until(lambda: not worker._thread.is_alive())
        assert any("could not acquire a DB connection" in r.message for r in caplog.records)

    def test_thread_is_a_daemon(self):
        # A non-daemon heartbeat would keep the process alive after a crash.
        worker = HeartbeatWorker("evt-7", "owner-1", interval=INTERVAL)
        assert worker._thread.daemon is True


class TestHeartbeatWrites:
    def test_writes_scoped_to_event_and_owner(self):
        conn = MagicMock()
        conn.execute.return_value = _cursor()
        with patch("src.core.heartbeat.get_connection", return_value=conn), \
             patch("src.core.heartbeat.put_connection"):
            with HeartbeatWorker("evt-8", "owner-9", interval=INTERVAL):
                assert _wait_until(lambda: conn.execute.called)
        sql, params = conn.execute.call_args.args
        assert "last_heartbeat_at" in sql
        assert "lock_owner" in sql
        assert params == ("evt-8", "owner-9")

    def test_no_write_before_the_first_interval(self):
        # The loop waits first, so a fast block never touches the DB at all.
        conn = MagicMock()
        conn.execute.return_value = _cursor()
        with patch("src.core.heartbeat.get_connection", return_value=conn), \
             patch("src.core.heartbeat.put_connection"):
            with HeartbeatWorker("evt-9", "owner-1", interval=30):
                pass
        conn.execute.assert_not_called()


class TestStopConditions:
    def test_lost_lock_stops_the_worker(self):
        # rowcount 0 means another worker stole the lock; continuing would let
        # two workers process the same event.
        conn = MagicMock()
        conn.execute.return_value = _cursor(rowcount=0)
        with patch("src.core.heartbeat.get_connection", return_value=conn), \
             patch("src.core.heartbeat.put_connection"):
            worker = HeartbeatWorker("evt-10", "owner-1", interval=INTERVAL)
            worker.__enter__()
            assert _wait_until(lambda: not worker._thread.is_alive())
            worker.__exit__(None, None, None)
        assert conn.execute.call_count == 1

    def test_repeated_db_errors_stop_the_worker(self):
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("connection reset")
        with patch("src.core.heartbeat.get_connection", return_value=conn), \
             patch("src.core.heartbeat.put_connection"):
            worker = HeartbeatWorker("evt-11", "owner-1", interval=INTERVAL)
            worker.__enter__()
            assert _wait_until(lambda: not worker._thread.is_alive(), timeout=3.0)
            worker.__exit__(None, None, None)
        assert conn.execute.call_count == MAX_CONSECUTIVE_ERRORS

    def test_transient_error_does_not_stop_the_worker(self):
        # A single blip must not end the keep-alive — the counter resets on
        # the next success, otherwise a long LLM call loses its lock.
        conn = MagicMock()
        conn.execute.side_effect = [RuntimeError("blip")] + [_cursor() for _ in range(50)]
        with patch("src.core.heartbeat.get_connection", return_value=conn), \
             patch("src.core.heartbeat.put_connection"):
            worker = HeartbeatWorker("evt-12", "owner-1", interval=INTERVAL)
            worker.__enter__()
            assert _wait_until(lambda: conn.execute.call_count > MAX_CONSECUTIVE_ERRORS + 2)
            assert worker._thread.is_alive()
            worker.__exit__(None, None, None)


def test_no_thread_leak_across_many_workers():
    conn = MagicMock()
    conn.execute.return_value = _cursor()
    before = threading.active_count()
    with patch("src.core.heartbeat.get_connection", return_value=conn), \
         patch("src.core.heartbeat.put_connection"):
        for i in range(10):
            with HeartbeatWorker(f"evt-{i}", "owner-1", interval=INTERVAL):
                pass
    assert _wait_until(lambda: threading.active_count() <= before)
