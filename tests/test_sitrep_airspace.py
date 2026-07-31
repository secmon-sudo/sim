"""
Airspace analysis wired into the SITREP — the plumbing, not the geometry.

The geometry lives in test_airspace.py. What matters here is that the computed
exposure actually reaches every consumer, and that it cannot be silently lost:
the HTML block is rendered deterministically (not by the narrative), and the
digest line is re-appended after parsing (not left to the model to repeat).
"""

from datetime import datetime, timezone

from src.core.airspace import build_airspace_assessment
from src.services.czib_client import fetch_active_czib_by_country
from src.services.sitrep_digest import build_digest_inputs, run_digest_llm
from src.services.sitrep_generator import _EVENT_COLUMNS, build_sitrep_clusters
from src.services.sitrep_html import render_sitrep_html

T0 = datetime(2026, 7, 30, 7, 30, tzinfo=timezone.utc)
T1 = datetime(2026, 7, 31, 7, 30, tzinfo=timezone.utc)

CZIB = {
    "UA": [{"czib_id": "1", "name": "UKRAINE — CZIB-2022-01", "valid_until": "2026-12-31"}],
    "BY": [{"czib_id": "2", "name": "BELARUS — CZIB-2023-04", "valid_until": "2026-10-01"}],
}

DISCLAIMER = "coğrafi yakınlık analizidir"


def _event(**over):
    d = {c: None for c in _EVENT_COLUMNS}
    d.update(id="e1", source_title="Drone hits site", source_domain="reuters.com",
             source_url="https://reuters.com/a", event_type="drone_attack_critical_infra",
             country_iso="PL", severity_score=78, anchor_name_raw="Lublin",
             occurred_at_est=T0, canonical_text="A drone struck a facility near Lublin.",
             corroborating_sources=[])
    d.update(over)
    return d


class _FakeConn:
    """Minimal db_conn stand-in returning one fixed result set."""

    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        rows = self._rows

        class R:
            def fetchall(self_inner):
                return rows

        return R()


class _FailingConn:
    def execute(self, sql, params=None):
        raise RuntimeError("relation v_czib_active does not exist")


class TestClusterCarriesLocation:
    def test_coordinates_and_country_reach_the_cluster(self):
        clusters = build_sitrep_clusters([_event(latitude=51.25, longitude=22.57)], [])
        assert clusters[0]["latitude"] == 51.25
        assert clusters[0]["longitude"] == 22.57
        assert clusters[0]["country_iso"] == "PL"

    def test_any_member_supplies_the_coordinate(self):
        """Clustered events are the same incident, so the representative event
        having no coordinate must not cost the cluster its location."""
        members = [
            _event(id="a", storyline_id="s1", severity_score=90,
                   source_domain="gov.pl"),  # official → sorts first, no coords
            _event(id="b", storyline_id="s1", severity_score=40,
                   latitude=51.25, longitude=22.57),
        ]
        clusters = build_sitrep_clusters(members, [])
        assert len(clusters) == 1
        assert clusters[0]["latitude"] == 51.25

    def test_missing_coordinates_stay_none(self):
        clusters = build_sitrep_clusters([_event()], [])
        assert clusters[0]["latitude"] is None


class TestCzibFetch:
    def test_indexes_zones_by_every_covered_state(self):
        rows = [("42", "UKRAINE — CZIB", ["UA", "MD"], "2026-12-31", "")]
        out = fetch_active_czib_by_country(_FakeConn(rows))
        assert set(out) == {"UA", "MD"}
        assert out["UA"][0]["name"] == "UKRAINE — CZIB"
        assert out["UA"][0]["valid_until"] == "2026-12-31"

    def test_falls_back_to_the_free_text_validity(self):
        rows = [("42", "X", ["UA"], None, "until further notice")]
        assert fetch_active_czib_by_country(_FakeConn(rows))["UA"][0][
            "valid_until"] == "until further notice"

    def test_unreadable_table_costs_the_annotation_not_the_report(self):
        assert fetch_active_czib_by_country(_FailingConn()) == {}


class TestHtmlSection:
    def _render(self, airspace):
        clusters = build_sitrep_clusters([_event()], [])
        return render_sitrep_html(
            "Polonya", "PL", "2026-07-30 07:30", "2026-07-31 07:30",
            "YÖNETİCİ ÖZETİ\nDurum değerlendirmesi.\n", clusters, [], airspace)

    def test_renders_fir_airports_and_restrictions(self):
        clusters = build_sitrep_clusters([_event()], [])
        html = self._render(build_airspace_assessment(clusters, "PL", CZIB))
        assert "HAVA SAHASI ETKİ ANALİZİ" in html
        assert "EPWW" in html and "Varşova FIR" in html
        assert "UKLV" in html and "EASA CZIB" in html
        assert "LUZ" in html and "km" in html

    def test_carries_the_proximity_disclaimer(self):
        """The block must never read as a confirmed closure notice."""
        clusters = build_sitrep_clusters([_event()], [])
        html = self._render(build_airspace_assessment(clusters, "PL", CZIB))
        assert DISCLAIMER in html
        assert "DEĞİLDİR" in html

    def test_absent_when_there_is_no_assessment(self):
        html = self._render(None)
        assert "HAVA SAHASI ETKİ ANALİZİ" not in html
        assert DISCLAIMER not in html

    def test_multi_fir_country_card_lists_them_all(self):
        """The card must not present one guessed FIR as the country's airspace."""
        clusters = build_sitrep_clusters(
            [_event(anchor_name_raw=None, country_iso="IN",
                    event_type="military_action")], [])
        html = self._render(build_airspace_assessment(clusters, "IN", CZIB))
        assert "Ülkenin hava sahaları (4)" in html
        for icao in ("VIDF", "VABF", "VOMF", "VECF"):
            assert icao in html
        assert "tek bir FIR belirtilmemiştir" in html

    def test_country_scope_card_shows_no_distances(self):
        clusters = build_sitrep_clusters(
            [_event(anchor_name_raw=None, event_type="military_action")], [])
        html = self._render(build_airspace_assessment(clusters, "PL", CZIB))
        assert "Ülkenin başlıca ticari havalimanları" in html
        assert "yarıçapındaki ticari havalimanları" not in html

    def test_stays_optional_for_existing_callers(self):
        clusters = build_sitrep_clusters([_event()], [])
        html = render_sitrep_html("Polonya", "PL", "a", "b",
                                  "YÖNETİCİ ÖZETİ\nX\n", clusters)
        assert "HAVA SAHASI ETKİ ANALİZİ" not in html


class TestSitrepPrompt:
    def test_airspace_reaches_the_prompt_in_compact_form(self):
        import src.services.sitrep_generator as gen
        captured = {}

        def fake_call_llm(router, prompt, system_prompt, **kw):
            captured["prompt"] = prompt
            return {"content": "YÖNETİCİ ÖZETİ\nX"}

        clusters = build_sitrep_clusters([_event()], [])
        airspace = build_airspace_assessment(clusters, "PL", CZIB)
        original = gen.call_llm
        gen.call_llm = fake_call_llm
        try:
            gen.run_sitrep_llm(None, "PL", "Polonya", T0, T1, clusters, [], [],
                               airspace=airspace)
        finally:
            gen.call_llm = original

        prompt = captured["prompt"]
        assert "EPWW" in prompt and "LUZ" in prompt
        assert "kisitlamali_komsu_firlar" in prompt
        # The rich HTML-only shape must not be what we pay tokens for.
        assert "neighbor_firs" not in prompt

    def test_prompt_forbids_inventing_airspace_facts(self):
        from src.services.sitrep_generator import _SYSTEM_PROMPT
        assert "UYDURMA" in _SYSTEM_PROMPT
        assert "MARUZİYET/YAKINLIK" in _SYSTEM_PROMPT


class TestDigestIntegration:
    def _results(self):
        clusters = build_sitrep_clusters([_event()], [])
        return [
            {"country_iso": "PL", "country_name": "Polonya", "status": "completed",
             "report_text": "YÖNETİCİ ÖZETİ\nDurum.", "clusters": clusters,
             "airspace": build_airspace_assessment(clusters, "PL", CZIB)},
            {"country_iso": "UA", "country_name": "Ukrayna", "status": "completed",
             "report_text": "YÖNETİCİ ÖZETİ\nDurum.", "clusters": clusters,
             "airspace": None},
        ]

    def test_summary_lands_on_the_digest_input_row(self):
        rows = build_digest_inputs(self._results())
        by_iso = {r["iso"]: r for r in rows}
        assert "EPWW" in by_iso["PL"]["airspace_summary"]
        assert by_iso["UA"]["airspace_summary"] == ""

    def test_summary_reaches_the_prompt_as_labelled_system_data(self):
        captured = {}

        def fake_call_llm(router, prompt, system_prompt, **kw):
            captured["prompt"] = prompt
            return {"content": "x"}

        import src.services.sitrep_digest as digest_mod
        original = digest_mod.call_llm
        digest_mod.call_llm = fake_call_llm
        try:
            run_digest_llm(None, build_digest_inputs(self._results()), "a", "b")
        finally:
            digest_mod.call_llm = original

        assert "[HAVA SAHASI (sistem hesabı)]" in captured["prompt"]
        assert "EPWW" in captured["prompt"]

    def _build(self, model_output):
        import src.services.sitrep_digest as digest_mod
        original = digest_mod.call_llm
        digest_mod.call_llm = lambda *a, **kw: {
            "content": model_output, "provider": "p", "model": "m"}
        try:
            return digest_mod.build_digest(None, self._results(), "a", "b")
        finally:
            digest_mod.call_llm = original

    def test_system_line_survives_a_narrative_that_omits_aviation(self):
        """The whole point of appending after parsing: a model that wrote 'YOK'
        must not be able to drop the computed exposure from the briefing."""
        digest = self._build(
            "GENEL DURUM DEĞERLENDİRMESİ\nBölgesel durum gergin.\n\n"
            "ÜLKE DEĞERLENDİRMELERİ\n- PL | Sınır bölgesinde İHA ihlali.\n"
            "- UA | Saldırılar sürdü.\n\n"
            "HAVACILIK OPERASYONLARINA ETKİ\nYOK\n"
        )
        aviation = digest["aviation"]
        assert len(aviation) == 1
        assert aviation[0].startswith("[hava sahası · sistem hesabı] Polonya:")
        assert "EPWW" in aviation[0]

    def test_system_line_appends_after_reported_disruptions(self):
        digest = self._build(
            "GENEL DURUM DEĞERLENDİRMESİ\nBölgesel durum gergin.\n\n"
            "ÜLKE DEĞERLENDİRMELERİ\n- PL | İHA ihlali.\n- UA | Saldırılar sürdü.\n\n"
            "HAVACILIK OPERASYONLARINA ETKİ\n- LOT, Rzeszów seferlerini durdurdu.\n"
        )
        assert digest["aviation"][0].startswith("LOT")
        assert digest["aviation"][-1].startswith("[hava sahası · sistem hesabı]")
