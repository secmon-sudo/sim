"""Iran bulletin run: dispatch, persistence, and what happens when they fail.

The report is dispatched before it is stored, so the tests that matter are the
ones about partial failure: a Telegram outage must not lose the record, and a
storage failure must not lose a report already in someone's hands.
"""

import pathlib

import pytest

from src.pipeline import iran_bulletin_run as run
from src.services import iran_bulletin as ib

REPO = pathlib.Path(__file__).resolve().parent.parent


class _Conn:
    def __init__(self, fail_on_insert=False):
        self.inserted = []
        self._fail = fail_on_insert

    def transaction(self):
        conn = self

        class _Tx:
            def __enter__(self_inner):
                return conn

            def __exit__(self_inner, *exc):
                return False

        return _Tx()

    def execute(self, sql, params=None):
        if self._fail:
            raise RuntimeError("db down")
        self.inserted.append((sql, params))
        return self


def _result(n_on=1, n_from=1, n_reg=0):
    def _ev(country, actor, i):
        return {"title": f"headline {i}", "country_iso": country, "actor": actor,
                "standing": ib.STANDING_CONFIRMED, "severity": 90 - i,
                "domain": f"outlet{i}.com", "url": f"https://outlet{i}.com/{i}",
                "corroborating_sources": [{}] if i % 2 else []}

    events = ([_ev("IR", ib.US_SIDE, i) for i in range(n_on)]
              + [_ev("JO", ib.IRAN_SIDE, 10 + i) for i in range(n_from)]
              + [_ev("OM", ib.UNATTRIBUTED, 20 + i) for i in range(n_reg)])
    return {"events": events, "sections": ib.group_into_sections(events),
            "narrative": "YÖNETİCİ ÖZETİ\nDurum böyle.", "status": "ok",
            "model": "qwen/qwen3.8-27b"}


@pytest.fixture
def _quiet(monkeypatch):
    monkeypatch.setattr(run, "upload_report_to_r2", lambda *a, **k: "https://r2/x")
    monkeypatch.setattr(run, "send_sitrep_telegram", lambda **k: "msg-1")
    monkeypatch.setattr(run, "build_bulletin_router", lambda: object())


class TestRun:
    def test_an_empty_window_dispatches_nothing(self, monkeypatch, _quiet):
        monkeypatch.setattr(run, "build_bulletin", lambda *a: {
            "status": "empty", "events": [], "sections": ib.group_into_sections([]),
            "narrative": ""})

        def _must_not_send(**k):
            raise AssertionError("an empty window must not page anyone")

        monkeypatch.setattr(run, "send_sitrep_telegram", _must_not_send)
        conn = _Conn()
        out = run.run_iran_bulletin(conn)
        assert out == {"success": True, "status": "empty", "events": 0}
        assert conn.inserted, "an empty run is still a run and must be recorded"

    def test_a_completed_run_reports_its_section_split(self, monkeypatch, _quiet):
        monkeypatch.setattr(run, "build_bulletin", lambda *a: _result(2, 3, 1))
        out = run.run_iran_bulletin(_Conn())
        assert out["events"] == 6
        assert out["sections"][ib.SECTION_TITLES[ib.SECTION_ON_IRAN]] == 2
        assert out["sections"][ib.SECTION_TITLES[ib.SECTION_FROM_IRAN]] == 3
        assert out["sections"][ib.SECTION_TITLES[ib.SECTION_REGIONAL]] == 1

    def test_a_telegram_outage_does_not_fail_the_run(self, monkeypatch, _quiet):
        monkeypatch.setattr(run, "build_bulletin", lambda *a: _result())

        def _boom(**k):
            raise RuntimeError("telegram down")

        monkeypatch.setattr(run, "send_sitrep_telegram", _boom)
        conn = _Conn()
        out = run.run_iran_bulletin(conn)
        assert out["success"] is True
        assert conn.inserted, "the report still has to be stored"

    def test_an_r2_outage_still_dispatches(self, monkeypatch, _quiet):
        sent = {}
        monkeypatch.setattr(run, "build_bulletin", lambda *a: _result())

        def _boom(*a, **k):
            raise RuntimeError("r2 down")

        monkeypatch.setattr(run, "upload_report_to_r2", _boom)
        monkeypatch.setattr(run, "send_sitrep_telegram",
                            lambda **k: sent.update(k) or "msg-1")
        out = run.run_iran_bulletin(_Conn())
        assert out["success"] is True
        assert sent["r2_url"] is None
        assert sent["html_doc"], "the document is the report; it must still go"

    def test_a_storage_outage_does_not_lose_a_dispatched_report(
            self, monkeypatch, _quiet):
        """_save swallows: the report is already in someone's hands."""
        monkeypatch.setattr(run, "build_bulletin", lambda *a: _result())
        out = run.run_iran_bulletin(_Conn(fail_on_insert=True))
        assert out["success"] is True

    def test_extraction_failure_is_recorded_not_raised(self, monkeypatch, _quiet):
        def _boom(*a):
            raise RuntimeError("all measured slots exhausted")

        monkeypatch.setattr(run, "build_bulletin", _boom)
        conn = _Conn()
        out = run.run_iran_bulletin(conn)
        assert out["success"] is False
        assert conn.inserted


class TestRenderAdapter:
    def test_corroborated_events_carry_the_multi_source_label(self):
        from src.core.sitrep_verify import LABEL_MULTI, LABEL_SINGLE
        clusters = run._clusters_for_render(_result(2, 2, 0))
        labels = {c["verification"] for c in clusters}
        assert labels <= {LABEL_MULTI, LABEL_SINGLE}
        assert len(clusters) == 4

    def test_every_section_reaches_the_renderer(self):
        clusters = run._clusters_for_render(_result(1, 1, 1))
        assert len(clusters) == 3


class TestWiring:
    def test_the_workflow_invokes_the_orchestrator_flag(self):
        wf = (REPO / ".github/workflows/iran-bulletin.yml").read_text()
        assert "--iran-bulletin" in wf

    def test_the_orchestrator_knows_the_flag(self):
        src = (REPO / "src/pipeline/orchestrator.py").read_text()
        assert '"--iran-bulletin" in sys.argv' in src

    def test_the_migration_creates_the_table(self):
        sql = (REPO / "db/migrations/023_iran_bulletins.sql").read_text()
        assert "CREATE TABLE IF NOT EXISTS iran_bulletins" in sql

    def test_the_bulletin_is_not_filed_under_the_sitrep_table(self):
        """Every SITREP consumer keys off country_iso; a bulletin has no single
        country, and a synthetic ISO would put it inside those queries."""
        src = (REPO / "src/pipeline/iran_bulletin_run.py").read_text()
        assert "INSERT INTO iran_bulletins" in src
        assert "INSERT INTO sitreps" not in src


class TestAppendixRowIsNotEmpty:
    """The first real bulletin drew 182 separator rules with nothing between them.

    _appendix_row reads seven fields — location, snippet, date, event_type,
    severity, verification and each source's `name` — and the first adapter
    supplied four. Every row rendered a bold em-dash, an empty meta line, an empty
    snippet and a chip labelled "kaynak". This pins the whole contract, because the
    failure was silent: the HTML was well-formed and 271KB of it was blank.
    """

    def _row(self):
        return run._clusters_for_render(_result(1, 0, 0))[0]

    def test_every_field_the_renderer_reads_is_supplied(self):
        row = self._row()
        for field in ("location", "snippet", "date", "event_type", "severity",
                      "verification", "sources"):
            assert field in row, field

    def test_location_is_not_the_placeholder_dash(self):
        assert self._row()["location"] not in ("", "—", None)

    def test_location_names_the_country_in_turkish_and_the_direction(self):
        row = self._row()
        assert "İran" in row["location"]
        assert "yönelik" in row["location"]

    def test_the_headline_reaches_the_snippet(self):
        assert self._row()["snippet"] == "headline 0"

    def test_sources_carry_a_name_so_the_chip_is_not_generic(self):
        source = self._row()["sources"][0]
        assert source["name"] and source["name"] != "kaynak"
        assert source["url"]

    def test_standing_rides_in_the_meta_line(self):
        """The badge already says how many outlets; standing says whether anyone
        stands behind it, and those are different claims."""
        assert self._row()["date"] == ib.STANDING_LABELS[ib.STANDING_CONFIRMED]

    def test_every_theatre_country_has_a_turkish_name(self):
        for iso in ib.THEATRE_ISO:
            assert ib.THEATRE_NAMES.get(iso), iso


class TestReportIdentity:
    def test_the_attachment_does_not_collide_with_the_iran_sitrep(self, monkeypatch,
                                                                  _quiet):
        """Iran gets its own SITREP the same morning into the same chat; the
        second file to arrive would overwrite the first on the reader's phone."""
        sent = {}
        monkeypatch.setattr(run, "build_bulletin", lambda *a: _result())
        monkeypatch.setattr(run, "send_sitrep_telegram",
                            lambda **k: sent.update(k) or "m")
        run.run_iran_bulletin(_Conn())
        assert sent["filename_stem"] == "iran_bulletin"
        assert sent["heading"] == run.REPORT_TITLE

    def test_the_title_is_not_jargon(self):
        assert "Tiyatro" not in run.REPORT_TITLE
        assert run.REPORT_TITLE.isupper()


class TestGroupedAppendix:
    """The full log is split into the same three blocks as the narrative.

    Without it the bulletin's organising idea — direction — survives only in the
    prose, and the first real run put 182 undivided rows underneath it.
    """

    def test_every_row_declares_its_section(self):
        for row in run._clusters_for_render(_result(2, 2, 1)):
            assert row["group"] in ib.SECTION_TITLES.values()

    def test_the_log_renders_one_block_per_section(self):
        # Titles are compared ESCAPED: "İRAN'DAN" reaches the page as
        # "İRAN&#x27;DAN", which is the renderer being correct, not a mismatch.
        from src.services.sitrep_html import _appendix_groups, _esc
        html = _appendix_groups(run._clusters_for_render(_result(2, 3, 1)))
        for title in ib.SECTION_TITLES.values():
            assert _esc(title) in html
        # counts are printed beside each block heading
        assert "(2)" in html and "(3)" in html and "(1)" in html

    def test_an_empty_section_prints_no_block(self):
        from src.services.sitrep_html import _appendix_groups
        html = _appendix_groups(run._clusters_for_render(_result(1, 0, 0)))
        from src.services.sitrep_html import _esc
        assert _esc(ib.SECTION_TITLES[ib.SECTION_ON_IRAN]) in html
        assert _esc(ib.SECTION_TITLES[ib.SECTION_FROM_IRAN]) not in html

    def test_a_sitrep_without_groups_is_unchanged(self):
        """The SITREP is already one country and has nothing to group by."""
        from src.services.sitrep_html import _appendix_groups, _appendix_row
        clusters = [{"location": "Kyiv", "snippet": "x", "severity": 50},
                    {"location": "Lviv", "snippet": "y", "severity": 40}]
        assert _appendix_groups(clusters) == "".join(
            _appendix_row(c) for c in clusters)

    def test_block_order_follows_the_narrative_not_the_alphabet(self):
        from src.services.sitrep_html import _appendix_groups, _esc
        html = _appendix_groups(run._clusters_for_render(_result(1, 1, 1)))
        positions = [html.index(_esc(ib.SECTION_TITLES[k])) for k in
                     (ib.SECTION_ON_IRAN, ib.SECTION_FROM_IRAN, ib.SECTION_REGIONAL)]
        assert positions == sorted(positions)
