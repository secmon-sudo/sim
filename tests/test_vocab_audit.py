"""Cover for the weekly vocabulary audit (scripts/vocab_audit.py).

The audit exists because three vocabulary gaps in a row were found by accident
(prescreen noun-phrase blindness, the missing "kill" verb, the missing
"diverted"/"delayed" aviation verbs). Its own failure mode would be to report a
comfortable number, so these tests pin the parts that decide the number: where
each gate is applied, what happens to an unanswered sample, and when it pages.
"""

from unittest import mock

from scripts import vocab_audit as va


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows=()):
        self._rows = list(rows)
        self.writes = []

    def execute(self, sql, params=None):
        if sql.strip().upper().startswith("INSERT"):
            self.writes.append((sql, params))
            return _FakeCursor([])
        return _FakeCursor(self._rows)


def _judge(mapping):
    """A judge that answers from a title -> in-scope map, silent for unknowns."""
    def _call(router, prompt, system, **kwargs):
        verdicts = []
        for line in prompt.splitlines():
            idx, _, title = line.partition(". ")
            if title.strip() in mapping:
                verdicts.append({"i": int(idx), "kapsam": mapping[title.strip()]})
        return {"content": str(verdicts).replace("'", '"').replace("True", "true")
                .replace("False", "false")}
    return _call


class TestVerdictParsing:
    def test_reads_the_array(self):
        out = va._parse_verdicts('önsöz [{"i":0,"kapsam":true},{"i":1,"kapsam":false}]', 2)
        assert out == {0: True, 1: False}

    def test_garbage_yields_nothing(self):
        assert va._parse_verdicts("model bugün konuşmuyor", 3) == {}

    def test_out_of_range_index_is_dropped(self):
        assert va._parse_verdicts('[{"i":9,"kapsam":true}]', 2) == {}


class TestSampling:
    def test_is_deterministic_per_week(self):
        items = [{"title": str(i)} for i in range(100)]
        assert va.sample(items, 10, "2026-W35") == va.sample(items, 10, "2026-W35")

    def test_different_weeks_draw_differently(self):
        items = [{"title": str(i)} for i in range(100)]
        assert va.sample(items, 10, "2026-W35") != va.sample(items, 10, "2026-W36")

    def test_small_population_is_taken_whole(self):
        items = [{"title": "a"}, {"title": "b"}]
        assert va.sample(items, 40, "x") == items


class TestAuditGate:
    ITEMS = [
        {"title": "Drones attacked the airport in Kyiv", "url": "u1"},
        {"title": "Council debates parking fees", "url": "u2"},
        {"title": "Gunmen kill 12 at a market", "url": "u3"},
    ]

    def test_counts_only_judged_samples(self):
        judge = _judge({"Drones attacked the airport in Kyiv": True,
                        "Council debates parking fees": False})
        with mock.patch.object(va, "call_llm", judge):
            out = va.audit_gate(None, va.GATE_NOISE, self.ITEMS, 40, "w")
        # The third headline got no verdict, so it is missing data, not a pass.
        assert out["sampled"] == 3
        assert out["judged"] == 2
        assert out["misses"] == 1
        assert out["miss_rate"] == 0.5

    def test_unanswered_batch_leaves_no_rate(self):
        with mock.patch.object(va, "call_llm", lambda *a, **k: {"content": "???"}):
            out = va.audit_gate(None, va.GATE_NOISE, self.ITEMS, 40, "w")
        assert out["judged"] == 0
        assert out["miss_rate"] is None

    def test_judge_failure_does_not_kill_the_audit(self):
        def _boom(*a, **k):
            raise RuntimeError("provider down")
        with mock.patch.object(va, "call_llm", _boom):
            out = va.audit_gate(None, va.GATE_NOISE, self.ITEMS, 40, "w")
        assert out["misses"] == 0 and out["miss_rate"] is None

    def test_examples_carry_the_missed_headlines(self):
        judge = _judge({"Gunmen kill 12 at a market": True})
        with mock.patch.object(va, "call_llm", judge):
            out = va.audit_gate(None, va.GATE_PRESCREEN, self.ITEMS, 40, "w")
        assert out["examples"] == ["Gunmen kill 12 at a market"]
        assert out["example_urls"] == ["u3"]


class TestIngestCollection:
    """The keyword filter guards the configured feeds only. Auditing it over query
    results would invent misses production never makes — a query feed is already
    filtered by the query itself."""

    QUERY_ITEMS = [{"title": "Airport shooting kills three", "description": "",
                    "link": "q1"}]
    FEED_ITEMS = [{"title": "Quarterly earnings beat forecasts", "description": "",
                   "link": "f1"},
                  {"title": "Drone attack on refinery", "description": "", "link": "f2"}]

    def _collect(self, settings):
        def _fetch(target, is_direct_url=False, stats=None):
            return list(self.FEED_ITEMS) if is_direct_url else list(self.QUERY_ITEMS)
        with mock.patch.object(va, "build_search_queries", return_value=[{"q": 1}]), \
             mock.patch.object(va, "fetch_rss_feed", _fetch), \
             mock.patch.dict(va.SETTINGS, settings, clear=False):
            return va.collect_ingest_rejections(None)

    def test_keyword_gate_only_sees_configured_feeds(self):
        out = self._collect({"sources": {"publisher_feeds": ["http://f"],
                                         "news_queries": []}})
        rejected = [r["title"] for r in out[va.GATE_KEYWORD]]
        assert "Quarterly earnings beat forecasts" in rejected
        assert "Airport shooting kills three" not in rejected

    def test_no_configured_feeds_means_no_keyword_rejections(self):
        out = self._collect({"sources": {"publisher_feeds": [], "news_queries": []}})
        assert out[va.GATE_KEYWORD] == []


class TestPrescreenCollection:
    def test_only_rows_the_prescreen_scores_zero_are_kept(self):
        rows = [
            ("Gunmen kill 12 at a market in Kaduna", "u1", "Gunmen killed 12 people."),
            ("Council debates parking fees", "u2", "The council met on Tuesday."),
        ]
        out = va.collect_prescreen_rejections(_FakeConn(rows))
        titles = [o["title"] for o in out]
        assert "Council debates parking fees" in titles
        # A row archived for some other reason must not be reported as a gap.
        assert "Gunmen kill 12 at a market in Kaduna" not in titles


class TestReporting:
    REPORT = {"week": "2026-W35", "days": 7, "samples": 40, "gates": [
        {"gate": "noise_filter", "rejected_total": 900, "sampled": 40, "judged": 38,
         "misses": 6, "miss_rate": 0.158, "examples": ["Drone attack on airport"],
         "example_urls": ["u1"]},
        {"gate": "prescreen", "rejected_total": 120, "sampled": 40, "judged": 40,
         "misses": 1, "miss_rate": 0.025, "examples": [], "example_urls": []},
    ]}

    def test_only_breaching_gates_page(self):
        breached = va.alerting_gates(self.REPORT, 0.10)
        assert [g["gate"] for g in breached] == ["noise_filter"]

    def test_missing_rate_never_pages(self):
        report = {"gates": [{"gate": "noise_filter", "miss_rate": None}]}
        assert va.alerting_gates(report, 0.0) == []

    def test_report_text_names_gate_rate_and_example(self):
        text = va.format_report(self.REPORT)
        assert "noise_filter" in text and "%16" in text
        assert "Drone attack on airport" in text

    def test_telemetry_row_is_written_once(self):
        conn = _FakeConn()
        va.record_audit(conn, self.REPORT)
        assert len(conn.writes) == 1
        assert "vocab_audit" in conn.writes[0][0]
