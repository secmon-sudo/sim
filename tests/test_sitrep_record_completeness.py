"""
The SITREP record is the whole day; only the narrative prompt is capped.

Measured on 2026-08-10: build_sitrep_clusters ended with
`ranked[:MAX_CLUSTERS_IN_PROMPT]`, and the single list it returned was BOTH the
prompt payload and the stored record. Ukraine ran 72 events / 48 storylines in the
2026-08-09 window; its report, its events_json and its HTML appendix all stopped at
25. The US ran 30 storylines and also stopped at 25. Half the day was unrecorded,
not merely un-narrated — which defeats the whole point of a deterministic appendix
(the guarantee restored on 2026-07-18 after four narration-fidelity failures).

The prompt still has to be paid for per cluster, so the cap survives — at the
prompt boundary, where the constant's name says it lives.
"""

from datetime import datetime, timezone

from src.services.sitrep_generator import (
    MAX_CLUSTERS_IN_PROMPT,
    SAFETY_ONLY_EVENT_TYPES,
    build_sitrep_clusters,
    cap_for_prompt,
)

_OVER_CAP = MAX_CLUSTERS_IN_PROMPT + 12


def _event(idx: int, severity: int = 100):
    """One event per distinct place, so each becomes its own cluster."""
    return {
        "source_title": f"Strike on Kasaba{idx}",
        "source_url": f"https://ex{idx}.example/a",
        "source_domain": f"ex{idx}.example",
        "event_type": "missile_strike",
        "occurred_at_est": datetime(2026, 8, 9, 12, tzinfo=timezone.utc),
        "published_at": datetime(2026, 8, 9, 12, tzinfo=timezone.utc),
        "time_certainty": "same_day",
        "anchor_name_raw": f"Kasaba{idx}",
        "anchor_name_norm": None,
        "country_iso": "UA",
        "severity_score": severity,
        "storyline_id": f"story-{idx}",
        "canonical_text": f"Strike on Kasaba{idx}",
        "corroborating_sources": [],
        "latitude": None,
        "longitude": None,
    }


def _busy_day():
    # Descending severity so the expected ranking is unambiguous.
    return [_event(i, severity=100 - i) for i in range(_OVER_CAP)]


class TestBuilderKeepsTheWholeDay:
    def test_builder_no_longer_caps(self):
        clusters = build_sitrep_clusters(_busy_day(), [])
        assert len(clusters) == _OVER_CAP > MAX_CLUSTERS_IN_PROMPT

    def test_builder_still_ranks_so_the_cap_takes_the_top(self):
        clusters = build_sitrep_clusters(_busy_day(), [])
        severities = [c["severity"] for c in clusters]
        assert severities == sorted(severities, reverse=True)

    def test_quiet_day_is_untouched(self):
        clusters = build_sitrep_clusters([_event(0), _event(1)], [])
        assert len(clusters) == 2


class TestCapForPrompt:
    def test_caps_at_the_configured_ceiling(self):
        assert len(cap_for_prompt(build_sitrep_clusters(_busy_day(), []))) \
            == MAX_CLUSTERS_IN_PROMPT

    def test_takes_the_highest_ranked(self):
        narrated = cap_for_prompt(build_sitrep_clusters(_busy_day(), []))
        assert narrated[0]["severity"] == 100
        assert narrated[-1]["severity"] == 100 - (MAX_CLUSTERS_IN_PROMPT - 1)

    def test_short_list_passes_through_unchanged(self):
        clusters = build_sitrep_clusters([_event(0)], [])
        assert cap_for_prompt(clusters) == clusters


class TestRecordVsNarrative:
    """The end-to-end split: the row stores everything, the prompt sees the top N."""

    def test_stored_record_is_whole_while_prompt_is_capped(self, monkeypatch):
        saved, prompt_field = _run_country(monkeypatch, _busy_day())
        assert len(saved["clusters"]) == _OVER_CAP, \
            "events_json/appendix must carry the full day"
        assert len(prompt_field) == MAX_CLUSTERS_IN_PROMPT, \
            "the narrative prompt is still paid for per cluster"

    def test_safety_clusters_no_longer_spend_narrative_slots(self, monkeypatch):
        # Safety clusters are dropped BEFORE the cap now, so a day padded with them
        # still gets a full-strength narrative — and still records them.
        day = _busy_day()
        safety = [dict(_event(900 + i), event_type="bird_strike") for i in range(10)]
        saved, prompt_field = _run_country(monkeypatch, safety + day)
        assert len(saved["clusters"]) == _OVER_CAP + 10
        assert len(prompt_field) == MAX_CLUSTERS_IN_PROMPT
        assert all(c["event_type"] not in SAFETY_ONLY_EVENT_TYPES for c in prompt_field)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

_T1 = datetime(2026, 8, 10, 8, 0)
_HDR = "YÖNETİCİ ÖZETİ\nGünün özeti.\n"


def _run_country(monkeypatch, events) -> tuple:
    """Drive run_country_sitrep over real clustering; return (saved row, prompt field)."""
    import src.pipeline.daily_sitrep as ds

    saved: dict = {}
    captured: dict = {}

    def fake_save(db_conn, country_iso, window_start, window_end, *, status,
                  report_text, clusters, **kw):
        saved.update(status=status, report_text=report_text, clusters=clusters)

    def fake_llm(router, iso, name, ws, we, field, strategic, spill, **kw):
        captured["field"] = field
        return {"content": _HDR, "finish_reason": "stop",
                "provider": "mistral", "model": "mistral-large-2512"}

    monkeypatch.setattr(ds, "_save_sitrep", fake_save)
    monkeypatch.setattr(ds, "run_sitrep_llm", fake_llm)
    monkeypatch.setattr(ds, "get_country_name", lambda db, iso: "Ukrayna")
    monkeypatch.setattr(ds, "fetch_sitrep_events", lambda *a, **k: events)
    monkeypatch.setattr(ds, "fetch_penalized_domains", lambda db: [])
    monkeypatch.setattr(ds, "fetch_spillover_events", lambda *a, **k: [])
    monkeypatch.setattr(ds, "fetch_aviation_spillover_events", lambda *a, **k: [])
    monkeypatch.setattr(ds, "fetch_active_czib_by_country", lambda db: {})
    monkeypatch.setattr(ds, "build_airspace_assessment", lambda *a, **k: None)
    monkeypatch.setattr(ds, "resolve_cluster_urls", lambda c: None)
    monkeypatch.setattr(ds, "render_sitrep_html", lambda *a, **k: "<html></html>")
    monkeypatch.setattr(ds, "upload_report_to_r2", lambda *a, **k: None)
    monkeypatch.setattr(ds, "send_sitrep_telegram", lambda **k: None)

    ds.run_country_sitrep(object(), object(), "UA", window_end=_T1)
    return saved, captured["field"]
