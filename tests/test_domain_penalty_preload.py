"""Domain penalty snapshot, added 2026-08-25.

The penalty gate cost 210 s/run — 25% of Pass A — measured over 11 production runs
once phase timing landed. It was not the data: the eligible table is ~700 rows. It
was one round trip per candidate, wrapped in an explicit transaction, so a single
SELECT cost BEGIN + SELECT + COMMIT against a pooler in another region, ~1000 times
a run. These tests pin the two things that make the snapshot safe to substitute:
identical verdicts, and a failure that falls back rather than silently clearing
every penalty.
"""

from unittest.mock import MagicMock

from src.pipeline.pass_a_ingest import check_domain_penalty, load_domain_penalties


class TestLoadDomainPenalties:
    def test_returns_domain_to_score_map(self):
        db = MagicMock()
        db.execute.return_value.fetchall.return_value = [
            ("unreliable-blog.com", 0.9), ("borderline.net", 0.4),
        ]
        assert load_domain_penalties(db) == {"unreliable-blog.com": 0.9, "borderline.net": 0.4}

    def test_null_score_becomes_zero_not_none(self):
        """penalty_score is nullable; a None in the map would raise at `penalty > 0.8`."""
        db = MagicMock()
        db.execute.return_value.fetchall.return_value = [("quiet.example", 0.0)]
        assert load_domain_penalties(db)["quiet.example"] == 0.0
        assert "COALESCE" in db.execute.call_args[0][0]

    def test_query_applies_the_five_event_floor(self):
        """The floor moves into SQL; a domain under it must never enter the map,
        because a snapshot miss is indistinguishable from a domain with no row."""
        db = MagicMock()
        db.execute.return_value.fetchall.return_value = []
        load_domain_penalties(db)
        sql = db.execute.call_args[0][0]
        assert "total_events >= 5" in sql

    def test_read_failure_returns_none_not_empty(self):
        """None sends callers back to per-item queries. An empty dict would read as
        'nothing is penalized' and open the gate for the whole run."""
        db = MagicMock()
        db.execute.side_effect = RuntimeError("pooler dropped the connection")
        assert load_domain_penalties(db) is None


class TestCheckWithSnapshot:
    def test_snapshot_hit_matches_the_query_path(self):
        db = MagicMock()
        assert check_domain_penalty(db, "unreliable-blog.com", {"unreliable-blog.com": 0.9}) == 0.9

    def test_snapshot_miss_is_zero(self):
        db = MagicMock()
        assert check_domain_penalty(db, "never-seen.com", {"unreliable-blog.com": 0.9}) == 0.0

    def test_snapshot_makes_no_database_call(self):
        """The whole point: ~1000 lookups per run, zero round trips."""
        db = MagicMock()
        check_domain_penalty(db, "unreliable-blog.com", {"unreliable-blog.com": 0.9})
        db.execute.assert_not_called()

    def test_whitelist_still_wins_over_the_snapshot(self):
        """A trusted outlet that somehow earned a score is still trusted — the
        whitelist is checked first on both paths."""
        db = MagicMock()
        assert check_domain_penalty(db, "reuters.com", {"reuters.com": 0.95}) == 0.0

    def test_none_snapshot_falls_back_to_querying(self):
        db = MagicMock()
        db.execute().fetchone.return_value = (0.9, 5)
        assert check_domain_penalty(db, "unreliable-blog.com", None) == 0.9
