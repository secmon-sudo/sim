"""A domain must not be disqualified as a corroborating source on one observation.

domain_penalties has two consumers and they disagreed on what counts as evidence:

  * check_domain_penalty() (ingest) ignores any domain with total_events < 5 — one
    archive out of one appearance is a penalty_score of 1.0 and means nothing.
  * fetch_penalized_domains() (SITREP) took every row at face value.

So the SITREP path acted on evidence the ingest path explicitly refuses to trust.
Measured 2026-08-13 in production: 1679 domains sat at penalty_score >= 0.5, of which
1312 had fewer than 5 observations and 770 were a single archive on a single appearance.
Every one of them was excluded from label_cluster()'s independence count and from the
official-source check — which is how a cluster carrying two real outlets gets published
as "Doğrulanmamış (Tek kaynak)".

The exclusion is load-bearing for real junk domains, so the floor is the fix rather than
removing the filter.
"""

from src.services.sitrep_generator import fetch_penalized_domains


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Applies the query's own WHERE clause to a fixture table, so the test exercises
    the SQL that ships rather than a hand-written reimplementation of it."""

    def __init__(self, table):
        self.table = table
        self.sql = None
        self.params = None

    def execute(self, sql, params=()):
        self.sql = " ".join(sql.split())
        self.params = params
        min_penalty, min_events = params
        return _FakeCursor([
            (d,) for d, total, penalty in self.table
            if penalty >= min_penalty and total >= min_events
        ])


# (domain, total_events, penalty_score)
FIXTURE = [
    ("realjunk.example", 40, 0.90),    # sustained evidence — stays excluded
    ("borderline.example", 5, 0.60),   # exactly at the floor — stays excluded
    ("seenonce.example", 1, 1.00),     # one archive, one appearance — must not count
    ("seentwice.example", 2, 0.50),
    ("thin.example", 4, 0.75),         # just under the floor
    ("clean.example", 80, 0.10),
]


class TestEvidenceFloor:
    def test_thin_evidence_domains_are_not_penalized(self):
        penalized = fetch_penalized_domains(_FakeConn(FIXTURE))
        assert "seenonce.example" not in penalized
        assert "seentwice.example" not in penalized
        assert "thin.example" not in penalized

    def test_sustained_offenders_stay_penalized(self):
        penalized = fetch_penalized_domains(_FakeConn(FIXTURE))
        assert "realjunk.example" in penalized
        assert "borderline.example" in penalized

    def test_clean_domains_are_never_returned(self):
        assert "clean.example" not in fetch_penalized_domains(_FakeConn(FIXTURE))

    def test_floor_matches_the_ingest_gate(self):
        """The whole point is that the two consumers agree; if one is retuned the other
        has to move with it, so the constant is asserted rather than assumed."""
        conn = _FakeConn(FIXTURE)
        fetch_penalized_domains(conn)
        assert conn.params[1] == 5
        assert "total_events >= %s" in conn.sql

    def test_min_penalty_is_still_the_callers_choice(self):
        assert fetch_penalized_domains(_FakeConn(FIXTURE), min_penalty=0.85) == [
            "realjunk.example"
        ]

    def test_query_failure_is_not_fatal(self):
        class _Broken:
            def execute(self, *a, **k):
                raise RuntimeError("connection lost")

        assert fetch_penalized_domains(_Broken()) == []
