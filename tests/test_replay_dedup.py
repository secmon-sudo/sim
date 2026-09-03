"""Dedup replay harness, 3 Sep 2026.

content_dedup_cpu is Pass A's largest phase (132.1s of 284.1s, 46%) and the only
one whose optimisation can change the PRODUCT: a faster fetch returns the same page
or it does not, while a faster matcher can quietly decide two stories are no longer
the same story — changing which events exist, which get a corroboration credit, and
which reach a report as a second card for something already sent.

The previous dedup optimisation was verified exactly this way and the verification
was discarded; test_dedup_cost.py records "480K comparisons ... zero differing
verdicts" as prose, so the next change had to rebuild it from nothing.

The property that matters is not that the harness runs. It is that the harness
NOTICES: a tool that reports IDENTICAL after a real change is worse than no tool,
because it converts an open question into false confidence.
"""

import json

import pytest

from scripts import replay_dedup


def _corpus(n=120):
    stories = [
        "Russian drone strike hits {} apartment block killing {}",
        "Iranian missiles target coalition airbase near {} overnight",
        "Wildfire forces evacuations across {} county",
        "Explosion at {} cargo terminal injures {} staff",
    ]
    places = ["Kyiv", "Erbil", "Reno", "Leipzig"]
    rows = []
    for i in range(n):
        title = stories[i % len(stories)].format(places[i % len(places)], 2 + i % 30)
        rows.append({
            "id": f"e{i}", "domain": f"outlet{i % 9}.com", "title": title,
            "canonical": (title + " Officials confirmed it. " * 6).lower(),
            "anchor": places[i % len(places)],
        })
    return rows


@pytest.fixture
def corpus_file(tmp_path):
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps({"days": 14, "rows": _corpus()}), encoding="utf-8")
    return p


class TestVerdicts:
    def test_a_verdict_names_the_matched_event_not_a_position(self):
        """An index is a place in a list a change may legitimately reorder; the
        identity of the story a candidate merged into is what must not move."""
        rows = _corpus(40)
        verdicts, _ = replay_dedup._verdicts(rows[:20], rows[-10:])
        for value in verdicts.values():
            assert value is None or value.startswith("e")

    def test_every_candidate_gets_a_verdict(self):
        rows = _corpus(40)
        verdicts, _ = replay_dedup._verdicts(rows[:20], rows[-10:])
        assert set(verdicts) == {r["id"] for r in rows[-10:]}

    def test_a_no_match_is_recorded_rather_than_dropped(self):
        stored = [{"id": "s1", "title": "Wildfire near Reno",
                   "canonical": "wildfire near reno " * 8, "anchor": "Reno"}]
        cand = [{"id": "c1", "title": "Central bank holds interest rates steady",
                 "canonical": "central bank holds interest rates steady " * 8,
                 "anchor": ""}]
        verdicts, _ = replay_dedup._verdicts(stored, cand)
        assert verdicts == {"c1": None}


class TestReplayCommand:
    def test_an_unchanged_matcher_reports_identical(self, corpus_file, tmp_path):
        base = tmp_path / "base.json"
        assert replay_dedup.replay(str(corpus_file), str(base), None, 60, 40) == 0
        assert replay_dedup.replay(str(corpus_file), None, str(base), 60, 40) == 0

    def test_a_changed_matcher_is_NOTICED(self, corpus_file, tmp_path, monkeypatch):
        """The whole point. A harness that misses a change is worse than none."""
        base = tmp_path / "base.json"
        replay_dedup.replay(str(corpus_file), str(base), None, 60, 40)

        import src.pipeline.ingest_filters as f
        monkeypatch.setattr(f, "_TITLE_SIM_THRESHOLD", 0.999)
        monkeypatch.setattr(f, "_TITLE_TOKEN_THRESHOLD", 0.999)
        monkeypatch.setattr(f, "_CONTENT_SHINGLE_THRESHOLD", 0.999)

        assert replay_dedup.replay(str(corpus_file), None, str(base), 60, 40) == 1

    def test_a_baseline_is_written_where_asked(self, corpus_file, tmp_path):
        base = tmp_path / "base.json"
        replay_dedup.replay(str(corpus_file), str(base), None, 60, 40)
        saved = json.loads(base.read_text(encoding="utf-8"))
        assert len(saved) == 40

    def test_candidates_and_stored_are_taken_from_opposite_ends(self, corpus_file):
        """They must overlap the way a run's do — new items against an older
        window — not let a candidate match its own row."""
        data = json.loads(corpus_file.read_text(encoding="utf-8"))
        rows = data["rows"]
        stored_ids = {r["id"] for r in rows[:60]}
        candidate_ids = {r["id"] for r in rows[-40:]}
        assert not (stored_ids & candidate_ids)


class TestDumpNeedsCredentials:
    def test_it_refuses_rather_than_crashing_without_a_database(self, monkeypatch,
                                                               tmp_path):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert replay_dedup.dump(str(tmp_path / "x.json"), 100, 14) == 2


class TestPairsMode:
    """The mode with a right answer.

    events holds only what was INSERTED — a duplicate was dropped and never became
    a row — so replaying events against events cannot exercise matching at all.
    Measured before this mode existed: 800 real candidates against 600 real stored
    events found 2 duplicates, against production's ~32%. It ran the rejection path
    almost exclusively and would have reported IDENTICAL while proving nothing
    about the decision the harness exists to protect.

    corroborating_sources is where the discarded side survives: 5,902 recorded
    duplicate-to-survivor pairs over 14 days, each one a headline production
    actually merged into a named event.
    """

    def _rows_and_pairs(self):
        rows = _corpus(40)
        # Reworded, not copied. A real duplicate is a second outlet's phrasing of
        # the same story, which is what puts it NEAR the threshold — an identical
        # string matches at ratio 1.0 and would survive any tightening, so a
        # fixture built from copies proves the harness works when it does not.
        pairs = [
            {"survivor": "e0",
             "title": rows[0]["title"].replace("hits", "strikes")
                                      .replace("killing", "leaving") + " dead"},
            {"survivor": "e1",
             "title": rows[1]["title"].replace("target", "hit")
                                      .replace("overnight", "in overnight raid")},
        ]
        return rows, pairs

    def test_a_recorded_duplicate_still_reaches_its_survivor(self):
        rows, pairs = self._rows_and_pairs()
        verdicts, _, n = replay_dedup._replay_pairs(rows, pairs, 100)
        assert n == 2
        assert verdicts["e0#0"] == "e0"

    def test_a_pair_whose_survivor_aged_out_is_skipped_not_failed(self):
        """A missing row would otherwise read as a regression it is not."""
        rows, _ = self._rows_and_pairs()
        pairs = [{"survivor": "gone-from-corpus", "title": rows[0]["title"]}]
        verdicts, _, n = replay_dedup._replay_pairs(rows, pairs, 100)
        assert n == 0 and verdicts == {}

    def test_the_limit_is_honoured(self):
        rows = _corpus(40)
        pairs = [{"survivor": f"e{i}", "title": rows[i]["title"]} for i in range(20)]
        _, _, n = replay_dedup._replay_pairs(rows, pairs, 5)
        assert n == 5

    def test_pairs_mode_notices_a_changed_matcher(self, tmp_path, monkeypatch):
        rows, pairs = self._rows_and_pairs()
        corpus = tmp_path / "c.json"
        corpus.write_text(json.dumps({"days": 14, "rows": rows, "pairs": pairs}),
                          encoding="utf-8")
        base = tmp_path / "b.json"
        assert replay_dedup.replay(str(corpus), str(base), None, 40, 100,
                                   mode="pairs") == 0

        import src.pipeline.ingest_filters as f
        monkeypatch.setattr(f, "_TITLE_SIM_THRESHOLD", 0.999)
        monkeypatch.setattr(f, "_TITLE_TOKEN_THRESHOLD", 0.999)
        monkeypatch.setattr(f, "_CONTENT_SHINGLE_THRESHOLD", 0.999)
        assert replay_dedup.replay(str(corpus), None, str(base), 40, 100,
                                   mode="pairs") == 1

    def test_a_corpus_without_pairs_does_not_crash(self, corpus_file):
        """Dumps taken before this mode existed carry no pairs key."""
        assert replay_dedup.replay(str(corpus_file), None, None, 60, 40,
                                   mode="pairs") == 0
