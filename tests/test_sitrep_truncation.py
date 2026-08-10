"""
SITREP narrative truncation — detect it, mark it, and give it room not to happen.

Measured on 2026-08-10: 17 of the 69 SITREPs written in the preceding two weeks
ended mid-sentence or mid-URL because the narrative had a hard-coded 4000-token
budget and an active-conflict country spends most of its output on per-bullet
citation lists (one UA bullet carried 14 URLs). Nothing noticed — the row saved as
'completed', the HTML rendered, and half a report shipped to Telegram. The
guardrails in validate_sitrep cannot catch it either: a cut-off report still has
its YÖNETİCİ ÖZETİ header and still cites only allowlisted URLs.
"""

import pytest

from src.services.sitrep_generator import (
    NARRATIVE_MAX_TOKENS,
    TRUNCATION_NOTICE,
    is_truncated,
    validate_sitrep,
)

_HDR = "YÖNETİCİ ÖZETİ\nGünün özeti.\n"


class TestIsTruncated:
    @pytest.mark.parametrize("reason", ["length", "max_tokens", "LENGTH", " length "])
    def test_length_stops_are_truncation(self, reason):
        assert is_truncated({"finish_reason": reason}) is True

    @pytest.mark.parametrize("reason", ["stop", "", None, "tool_calls"])
    def test_everything_else_is_a_clean_finish(self, reason):
        assert is_truncated({"finish_reason": reason}) is False

    def test_missing_key_is_not_truncation(self):
        # Providers that omit finish_reason must not mark every report.
        assert is_truncated({"content": "x"}) is False


class TestBudget:
    def test_budget_is_configured_not_hard_coded(self):
        # The 4000 that produced the incident is gone; the knob is in
        # config/settings.json -> sitrep.narrative_max_tokens.
        assert NARRATIVE_MAX_TOKENS >= 6000

    def test_run_sitrep_llm_spends_the_configured_budget(self):
        import src.services.sitrep_generator as gen

        captured = {}

        def fake_call_llm(router, prompt, system_prompt, **kw):
            captured.update(kw)
            return {"content": _HDR, "finish_reason": "stop"}

        original = gen.call_llm
        gen.call_llm = fake_call_llm
        try:
            gen.run_sitrep_llm(None, "UA", "Ukrayna", _T0, _T1, [], [], [])
        finally:
            gen.call_llm = original

        assert captured["max_tokens"] == NARRATIVE_MAX_TOKENS
        # Prose, not JSON — a reasoning/JSON-mode narrator is a separate old wound.
        assert captured["json_mode"] is False


class TestTruncatedReportIsMarked:
    """The end-to-end behaviour: a cut-off narrative still ships, but marked."""

    def test_truncated_report_gets_the_notice_and_still_saves_completed(self, monkeypatch):
        saved = _run_country(monkeypatch, finish_reason="length")
        assert saved["status"] == "completed"
        assert saved["report_text"].endswith(TRUNCATION_NOTICE)

    def test_clean_report_is_left_alone(self, monkeypatch):
        saved = _run_country(monkeypatch, finish_reason="stop")
        assert saved["status"] == "completed"
        assert TRUNCATION_NOTICE not in saved["report_text"]

    def test_notice_survives_the_guardrails(self):
        # The notice is appended AFTER validate_sitrep, but it must not be the kind
        # of text that would trip it if the order ever changed.
        out = validate_sitrep(_HDR + TRUNCATION_NOTICE, [])
        assert "künye" in out


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

from datetime import datetime  # noqa: E402

_T0 = datetime(2026, 8, 9, 8, 0)
_T1 = datetime(2026, 8, 10, 8, 0)

_CLUSTER = {
    "location": "Kiev", "event_type": "security_incident", "date": "2026-08-09",
    "verification": "Onaylandı (Çoklu kaynak)", "severity": 100,
    "snippet": "Saldırı.", "sources": [{"url": "https://ex.example/a", "domain": "ex.example"}],
    "country_iso": "UA", "latitude": None, "longitude": None,
}


def _run_country(monkeypatch, finish_reason: str) -> dict:
    """Drive run_country_sitrep with every collaborator stubbed; return the saved row."""
    import src.pipeline.daily_sitrep as ds

    saved: dict = {}

    def fake_save(db_conn, country_iso, window_start, window_end, *, status,
                  report_text, clusters, **kw):
        saved.update(status=status, report_text=report_text, clusters=clusters)

    monkeypatch.setattr(ds, "_save_sitrep", fake_save)
    monkeypatch.setattr(ds, "get_country_name", lambda db, iso: "Ukrayna")
    monkeypatch.setattr(ds, "fetch_sitrep_events", lambda *a, **k: [{"id": "1"}])
    monkeypatch.setattr(ds, "fetch_penalized_domains", lambda db: [])
    monkeypatch.setattr(ds, "build_sitrep_clusters", lambda ev, pen: [dict(_CLUSTER)])
    monkeypatch.setattr(ds, "drop_safety_clusters", lambda c: c)
    monkeypatch.setattr(ds, "split_strategic", lambda c: (c, []))
    monkeypatch.setattr(ds, "fetch_spillover_events", lambda *a, **k: [])
    monkeypatch.setattr(ds, "fetch_aviation_spillover_events", lambda *a, **k: [])
    monkeypatch.setattr(ds, "fetch_active_czib_by_country", lambda db: {})
    monkeypatch.setattr(ds, "build_airspace_assessment", lambda *a, **k: None)
    monkeypatch.setattr(ds, "resolve_cluster_urls", lambda c: None)
    monkeypatch.setattr(ds, "relabel_cluster", lambda c, pen: None)
    monkeypatch.setattr(ds, "render_sitrep_html", lambda *a, **k: "<html></html>")
    monkeypatch.setattr(ds, "upload_report_to_r2", lambda *a, **k: None)
    monkeypatch.setattr(ds, "send_sitrep_telegram", lambda **k: None)
    monkeypatch.setattr(
        ds, "run_sitrep_llm",
        lambda *a, **k: {"content": _HDR + "— Olay. Kaynak: X (https://ex.example/a)",
                         "finish_reason": finish_reason,
                         "provider": "mistral", "model": "mistral-large-2512"},
    )

    ds.run_country_sitrep(object(), object(), "UA", window_end=_T1)
    return saved
