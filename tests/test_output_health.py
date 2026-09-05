"""Output-health checks: the reports arrived, are they hollow? (4 Sep 2026)

The dead-man's switch answers "did the pipeline run". On 2026-09-04 it answered
yes five times while five SITREPs shipped with every one of their 108 citation
URLs blanked, and a person reading a report found it. These tests pin the
queries that would have paged that morning, and — just as important — pin the
cases where they must stay silent, because a check that fires every day is a
check nobody reads.

The connection is faked rather than mocked at the driver: each check is one
query and the interesting behaviour is what it does with the ROWS, so the fake
returns canned rows per call and the SQL itself is exercised against the live
database separately.
"""

import pytest

from src.core import output_health as oh


class _Cur:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        self._conn.seen.append((sql, params))
        self._rows = self._conn.queue.pop(0) if self._conn.queue else []

    def fetchall(self):
        return self._rows


class _Conn:
    """Returns the queued row-sets in order, one per execute()."""

    def __init__(self, *row_sets):
        self.queue = list(row_sets)
        self.seen = []

    def cursor(self):
        return _Cur(self)


class TestCitationCollapse:
    def test_it_names_every_report_and_the_model_that_wrote_it(self):
        conn = _Conn([("IR", "minimax-m2.7"), ("UA", "minimax-m2.7")])
        out = oh.check_sitrep_citations(conn, 30.0)
        assert len(out) == 1
        assert out[0].key == "sitrep_no_citations"
        assert "2 SITREP" in out[0].message
        # The model belongs in the page: on 4 Sep it was the whole diagnosis.
        assert "minimax-m2.7" in out[0].detail
        assert "IR" in out[0].detail and "UA" in out[0].detail

    def test_a_healthy_day_says_nothing(self):
        assert oh.check_sitrep_citations(_Conn([]), 30.0) == []

    def test_it_looks_only_at_the_latest_run(self):
        """The window is 30 hours and a SITREP runs daily, so a plain window
        query spans two runs. On 5 Sep this reported the previous morning's five
        broken reports — twice, hours after the cause had been fixed and removed
        — while that morning's five were perfect. A check that keeps announcing a
        solved problem is the fastest way to teach someone to ignore it."""
        conn = _Conn([])
        oh.check_sitrep_citations(conn, 30.0)
        sql = conn.seen[0][0]
        assert "max(window_end)" in sql, sql

    def test_truncation_is_scoped_the_same_way(self):
        conn = _Conn([])
        oh.check_sitrep_truncation(conn, 30.0)
        assert "max(window_end)" in conn.seen[0][0]


class TestTruncation:
    def test_it_reports_reports_cut_off_at_the_ceiling(self):
        out = oh.check_sitrep_truncation(_Conn([("RU",), ("PK",)]), 30.0)
        assert out and "token ceiling" in out[0].message

    def test_silent_when_nothing_was_truncated(self):
        assert oh.check_sitrep_truncation(_Conn([]), 30.0) == []


class TestNarratorChange:
    """Written wrong first, which is why the abstention rule is pinned."""

    def test_a_new_narrator_is_flagged_with_the_usual_ones_named(self):
        conn = _Conn(
            [("2026-09-04 07:31",)],                       # latest run
            [("mistral-medium-latest", 20), ("gemini-3.5-flash-lite", 10)],
            [("minimax-m2.7",), ("mistral-medium-latest",)],
        )
        out = oh.check_narrator_changed(conn, 30.0)
        assert len(out) == 1
        assert "minimax-m2.7" in out[0].detail
        # The familiar slot must NOT be reported as new.
        assert out[0].detail.count("minimax-m2.7") == 1
        assert "usual:" in out[0].detail

    def test_it_abstains_on_a_thin_baseline(self):
        """The first version flagged mistral-medium and laguna — the two most
        ordinary slots in the cascade — because there was nothing behind them to
        compare with. No baseline means no finding."""
        conn = _Conn(
            [("2026-09-04 07:31",)],
            [("mistral-medium-latest", 2)],               # under the minimum
            [("minimax-m2.7",)],
        )
        assert oh.check_narrator_changed(conn, 30.0) == []

    def test_no_run_at_all_is_not_a_finding(self):
        """Pipeline silence is the other checks' job; this one has nothing to say."""
        assert oh.check_narrator_changed(_Conn([(None,)]), 30.0) == []

    def test_the_same_models_as_last_week_say_nothing(self):
        conn = _Conn(
            [("2026-09-04 07:31",)],
            [("mistral-medium-latest", 30)],
            [("mistral-medium-latest",)],
        )
        assert oh.check_narrator_changed(conn, 30.0) == []


class TestBulletinAttribution:
    def test_a_collapse_is_reported(self):
        out = oh.check_bulletin_attribution(_Conn([("2026-09-04", 0.91, 76)]), 30.0)
        assert out and "91%" in out[0].message

    def test_the_observed_range_does_not_fire(self):
        """13.2%, 13.7% and 19.7% are every bulletin the report has produced. The
        ceiling has to sit clear of all of them or it is a daily alarm."""
        for share in (0.132, 0.137, 0.197):
            assert oh.check_bulletin_attribution(
                _Conn([("2026-09-04", share, 76)]), 30.0) == []

    def test_a_tiny_bulletin_cannot_trip_it(self):
        """Four events, three unattributed, is 75% and means nothing."""
        assert oh.check_bulletin_attribution(_Conn([("2026-09-04", 0.75, 4)]), 30.0) == []


class TestDegradationCounters:
    def test_counters_are_summed_across_runs(self):
        conn = _Conn([
            ("pipeline_run", {"llm_contract_rejected": 2}),
            ("sitrep_run", {"llm_contract_rejected": 3, "llm_unusable_200": 1}),
        ])
        out = oh.check_degradation_counters(conn, 30.0)
        assert out and "llm_contract_rejected=5" in out[0].detail
        # Below its own threshold, so it must not ride along on another
        # counter's finding — that is how a page fills up with non-events.
        assert "llm_unusable_200" not in out[0].detail

    def test_routine_provider_weather_is_not_a_page(self):
        """Measured 5 Sep over six consecutive pipeline runs: 1, 1, 1, 1, 4, 1 —
        every one completing successfully. The counter records the router
        rotating past an empty 200, which is the system healing itself, and the
        daily total sits near 8-12 simply because there are 8-12 runs."""
        conn = _Conn([("pipeline_run", {"llm_unusable_200": 12})])
        assert oh.check_degradation_counters(conn, 30.0) == []

    def test_the_guard_working_once_is_not_a_page(self):
        """A single llm_contract_rejected means the citation guard caught a bad
        slot and rotated past it — the system working as designed. The first
        version of this check paged about exactly that, in the channel real
        users read, and taught them the channel reports non-events."""
        conn = _Conn([("sitrep_run", {"llm_contract_rejected": 1,
                                      "llm_unusable_200": 1})])
        assert oh.check_degradation_counters(conn, 30.0) == []

    def test_a_silent_fail_open_pages_at_one(self):
        """bulletin_direction_batch_failed has no threshold to hide behind: it
        fails OPEN, so the report renders perfectly while having stopped saying
        which way anything was going."""
        conn = _Conn([("pipeline_run", {"bulletin_direction_batch_failed": 1})])
        out = oh.check_degradation_counters(conn, 30.0)
        assert out and "bulletin_direction_batch_failed=1" in out[0].detail

    def test_no_counters_is_silence(self):
        assert oh.check_degradation_counters(_Conn([]), 30.0) == []

    def test_a_junk_counter_value_does_not_crash_the_check(self):
        conn = _Conn([("pipeline_run", {"llm_contract_rejected": "many"})])
        assert oh.check_degradation_counters(conn, 30.0) == []


class TestOpenRouterCredit:
    """OpenRouter is prepaid, so the balance is a hard ceiling and nothing can
    overspend it. The risk is not a surprise bill — it is a surprise SILENCE:
    the credit runs out, the paid floor drops away, and the free rungs beneath
    quietly take over the reports. That is the exact failure the paid slot was
    added to end, so a floor that can vanish without saying so is not a floor."""

    def _patch(self, monkeypatch, payload, key="k"):
        import httpx

        monkeypatch.setenv("OPENROUTER_API_KEY_A", key)

        class _Resp:
            def json(self):
                return payload

        monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())

    def test_a_low_balance_is_reported_in_days(self, monkeypatch):
        # $0.20 left ÷ $0.044/day ≈ 5 days
        self._patch(monkeypatch, {"data": {"limit": 10.0, "usage": 9.8}})
        out = oh.check_openrouter_credit(None, 30.0)
        assert out and out[0].key == "openrouter_credit_low"
        assert "5 gün" in out[0].message
        assert "$0.20" in out[0].detail

    def test_a_healthy_balance_says_nothing(self, monkeypatch):
        self._patch(monkeypatch, {"data": {"limit": 10.0, "usage": 1.0}})
        assert oh.check_openrouter_credit(None, 30.0) == []

    def test_an_uncapped_key_is_not_an_alarm(self, monkeypatch):
        """limit=None means no ceiling; inventing one would fire every day."""
        self._patch(monkeypatch, {"data": {"limit": None, "usage": 3.0}})
        assert oh.check_openrouter_credit(None, 30.0) == []

    def test_no_key_means_no_check(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY_A", raising=False)
        assert oh.check_openrouter_credit(None, 30.0) == []

    def test_a_network_failure_is_not_a_finding(self, monkeypatch):
        """An unreachable billing endpoint is not evidence of anything, and a
        raise here would be reported as a check_error instead."""
        import httpx

        monkeypatch.setenv("OPENROUTER_API_KEY_A", "k")

        def _boom(*a, **k):
            raise httpx.ConnectError("down")

        monkeypatch.setattr(httpx, "get", _boom)
        assert oh.check_openrouter_credit(None, 30.0) == []

    def test_junk_values_do_not_crash_the_check(self, monkeypatch):
        self._patch(monkeypatch, {"data": {"limit": "ten", "usage": 1.0}})
        assert oh.check_openrouter_credit(None, 30.0) == []


class TestRunChecks:
    def test_one_broken_check_does_not_silence_the_others(self):
        """A health check that swallows its own errors stops checking silently,
        which is the exact failure this module exists to end."""

        def _boom(conn, window_hours):
            raise RuntimeError("column disappeared")

        def _finds(conn, window_hours):
            return [oh.Finding("k", "something is wrong")]

        original = oh.CHECKS
        oh.CHECKS = (_boom, _finds)
        try:
            out = oh.run_checks(object(), 30.0)
        finally:
            oh.CHECKS = original
        keys = [f.key for f in out]
        assert "k" in keys
        assert any(k.startswith("check_error:") for k in keys)


class TestFormatting:
    def test_a_healthy_run_produces_no_message(self):
        assert oh.format_report([]) is None

    def test_findings_render_with_their_detail(self):
        text = oh.format_report([oh.Finding("k", "başlık", "ayrıntı")])
        assert "başlık" in text and "ayrıntı" in text


@pytest.mark.parametrize("check", oh.CHECKS)
def test_every_check_is_registered_and_callable(check):
    assert callable(check)
    assert check.__doc__, f"{check.__name__} must say what it is for"


class TestPageDeduplication:
    """The dead-man runs several times a day and the health window is 30 hours,
    so without this one bad SITREP is reported five or six times — same words,
    same run, until the window rolls past it. An alarm that repeats itself is one
    people learn to swipe away, and then it is worth less than no alarm."""

    def _conn(self, previous_keys=None):
        """deadman_check calls conn.execute(...).fetchone() directly (psycopg3
        style), which is a different shape from the cursor fake above."""

        class _Res:
            def fetchone(self):
                if previous_keys is None:
                    return None
                return ({"keys": previous_keys},)

        class _C:
            def execute(self, *a, **k):
                return _Res()

        return _C()

    def test_a_finding_already_paged_today_is_suppressed(self):
        import scripts.deadman_check as dm

        conn = self._conn(["sitrep_no_citations"])
        fresh = dm._unreported_findings(
            conn, [oh.Finding("sitrep_no_citations", "x")], 20.0)
        assert fresh == []

    def test_a_new_finding_still_pages_alongside_an_old_one(self):
        import scripts.deadman_check as dm

        conn = self._conn(["sitrep_no_citations"])
        fresh = dm._unreported_findings(conn, [
            oh.Finding("sitrep_no_citations", "eski"),
            oh.Finding("sitrep_truncated", "yeni"),
        ], 20.0)
        assert [f.key for f in fresh] == ["sitrep_truncated"]

    def test_the_count_changing_is_not_a_new_incident(self):
        """"3 SITREP(s)" becoming "4 SITREP(s)" an hour later is the same
        incident; keying on the text rather than the key would defeat this."""
        import scripts.deadman_check as dm

        conn = self._conn(["sitrep_no_citations"])
        fresh = dm._unreported_findings(
            conn, [oh.Finding("sitrep_no_citations", "4 SITREP(s) ...")], 20.0)
        assert fresh == []

    def test_nothing_reported_yet_means_everything_is_fresh(self):
        import scripts.deadman_check as dm

        fresh = dm._unreported_findings(
            self._conn(), [oh.Finding("k", "x"), oh.Finding("j", "y")], 20.0)
        assert len(fresh) == 2

    def test_no_findings_needs_no_query(self):
        import scripts.deadman_check as dm

        assert dm._unreported_findings(None, [], 20.0) == []


class TestOpsChannelSeparation:
    """The first health page landed in the channel real users read, in front of
    them, saying "minimax-m2.7" and "llm_contract_rejected=1". Engineering
    diagnostics and user-facing alerts are different audiences."""

    def _sent(self, monkeypatch):
        import httpx

        from src.services import ops_notifier

        captured = {}

        class _R:
            status_code = 200

            def json(self):
                return {"ok": True}

        def _post(url, json=None, timeout=None, **k):
            captured.update(json or {})
            return _R()

        monkeypatch.setattr(httpx, "post", _post)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        return ops_notifier, captured

    def test_the_ops_chat_is_preferred(self, monkeypatch):
        notifier, sent = self._sent(monkeypatch)
        monkeypatch.setenv("TELEGRAM_OPS_CHAT_ID", "-100OPS")
        monkeypatch.setenv("TELEGRAM_ALERTS_CHAT_ID", "-100USERS")
        notifier.send_ops_alert("bir şey")
        assert sent["chat_id"] == "-100OPS"

    def test_it_falls_back_rather_than_losing_a_page(self, monkeypatch):
        """Losing a page about a dead pipeline is worse than showing one to a
        reader — but the fallback warns every time, because it IS the problem."""
        notifier, sent = self._sent(monkeypatch)
        monkeypatch.delenv("TELEGRAM_OPS_CHAT_ID", raising=False)
        monkeypatch.setenv("TELEGRAM_ALERTS_CHAT_ID", "-100USERS")
        notifier.send_ops_alert("bir şey")
        assert sent["chat_id"] == "-100USERS"

    def test_no_chat_at_all_sends_nothing(self, monkeypatch):
        notifier, sent = self._sent(monkeypatch)
        monkeypatch.delenv("TELEGRAM_OPS_CHAT_ID", raising=False)
        monkeypatch.delenv("TELEGRAM_ALERTS_CHAT_ID", raising=False)
        assert notifier.send_ops_alert("bir şey") is False
        assert not sent
