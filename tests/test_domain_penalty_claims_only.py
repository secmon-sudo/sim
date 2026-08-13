"""domain_penalties must measure credibility, not topicality.

penalty_score is read as "can this outlet be believed": fetch_penalized_domains() bars a
domain from label_cluster()'s independence count and from the official-source check, and
check_domain_penalty() drops its items at ingest. But every archive used to charge the
domain a false_positive, and most archives are simply off-topic — so the score was
ranking outlets by how much non-security news they publish.

Measured 2026-08-13: 1679 domains sat at >= 0.5, among them reuters.com, bloomberg.com,
cnbc.com, ft.com, thetimes.com, washingtonpost.com, economist.com, politico.eu,
nhk.or.jp and aa.com.tr — all disqualified as corroborating sources for the crime of
writing about finance. Separately, the prescreen charged 1538 archives across 426 domains
for headlines its own vocabulary could not parse, which then withheld those outlets'
corroboration from the events they had reported.

The contract now: only an item that CLAIMED a security event is an observation.
  * hostile-act headline + classifier finds no incident  -> charge (clickbait)
  * claim held up, event classified                      -> credit
  * never claimed anything (off-topic, prescreen)        -> not an observation at all
"""

import json

import pytest

from src.pipeline import pass_c_classify as pc


class _FakeTransaction:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    """Records penalty writes; everything else is a no-op with a plausible return."""

    def __init__(self, catalog_hit=True):
        self.penalties = []          # [(domain, is_noise)]
        self.catalog_hit = catalog_hit

    def transaction(self):
        return _FakeTransaction()

    def execute(self, sql, params=()):
        if "domain_penalties" in sql:
            self.penalties.append((params[0], params[1]))
        self._last = sql
        return self

    def fetchone(self):
        return ("terrorism",) if self.catalog_hit else None

    def commit(self):
        pass


@pytest.fixture
def conn():
    return _FakeConn()


def _event(title, domain="outlet.example"):
    return {
        "id": "11111111-1111-1111-1111-111111111111",
        "source_title": title,
        "source_domain": domain,
        "canonical_text": "",
    }


class TestPrescreenChargesNothing:
    def test_off_topic_archive_is_not_an_observation(self, conn):
        det = pc.deterministic_relevance("Grocery retailer reports Q3 profit down", "")
        assert det["score"] < pc.PRESCREEN_SKIP_FLOOR
        assert pc._try_prescreen_archive(conn, _event("Grocery retailer reports Q3"), det) is True
        assert conn.penalties == []

    def test_the_archive_itself_still_happens(self, conn):
        det = pc.deterministic_relevance("Airport lounge review: the new terminal", "")
        assert pc._try_prescreen_archive(conn, _event("Airport lounge review"), det) is True
        assert "status = 'archived'" in conn._last

    def test_a_hostile_act_headline_cannot_reach_the_prescreen_archive(self):
        """The clickbait class is judged on the LLM path, where the charge is made. This
        is what makes charging nothing here safe rather than merely lenient."""
        det = pc.deterministic_relevance(
            "Iran Launches Missile Attack On Bahrain As U.S. Announces Strikes", ""
        )
        assert det["has_hostile_act"] is True
        assert det["score"] >= pc.PRESCREEN_SKIP_FLOOR


class TestLLMArchiveChargesOnlyEmptyClaims:
    def _archive(self, conn, title, relevance=10, llm_type="noise"):
        parsed = {"event_type": llm_type, "relevance_score": relevance,
                  "relevance_reasoning": "reason"}
        return pc._apply_llm_classification(
            conn, None, _event(title),
            pc.deterministic_relevance(title, ""),
            parsed,
            {"response": {}, "provider": "p", "model": "m"},
            worker_id=None,
            log_telemetry=False,
        )

    def test_empty_hostile_claim_is_charged(self, conn):
        out = self._archive(conn, "Iran Launches Missile Attack On Bahrain, reports say")
        assert conn.penalties == [("outlet.example", 1)]
        assert out.get("_high_signal_archived") is True

    def test_off_topic_archive_is_not_charged(self, conn):
        self._archive(conn, "Semiconductor supply chain and corporate evaluations")
        assert conn.penalties == []

    def test_commentary_without_a_claim_is_not_charged(self, conn):
        self._archive(conn, "Opinion: why the region's economic future matters")
        assert conn.penalties == []

    def test_charge_and_counter_read_the_same_signal(self, conn):
        """high_signal_archived is the inflow metric for this score; if they ever
        disagree, one of the two is lying about the same event."""
        out = self._archive(conn, "Drone attack on the refinery, officials deny")
        assert bool(out.get("_high_signal_archived")) is (len(conn.penalties) == 1)


class TestSuccessPathStillCredits:
    def test_classified_event_credits_the_domain(self, conn):
        parsed = {"event_type": "terrorism", "relevance_score": 80}
        title = "Suicide bomber kills 14 at market"
        pc._apply_llm_classification(
            conn, None, _event(title), pc.deterministic_relevance(title, ""),
            parsed, {"response": {}, "provider": "p", "model": "m"},
            worker_id=None, log_telemetry=False,
        )
        assert ("outlet.example", 0) in conn.penalties

    def test_credit_side_is_deliberately_left_broad(self, conn):
        """Only the CHARGE side was retuned. Crediting every classified event — including
        a low-relevance `unclassified` fallback — errs toward leniency, and keeping it
        unchanged means a shift in the score distribution is attributable to the charge
        change alone."""
        parsed = {"event_type": "noise", "relevance_score": 45}
        title = "Officials discuss regional security cooperation"
        pc._apply_llm_classification(
            conn, None, _event(title), pc.deterministic_relevance(title, ""),
            parsed, {"response": {}, "provider": "p", "model": "m"},
            worker_id=None, log_telemetry=False,
        )
        assert ("outlet.example", 0) in conn.penalties


class TestWriteShape:
    def test_penalty_write_is_unchanged_for_real_observations(self, conn):
        pc.update_domain_penalty(conn, "outlet.example", 1)
        assert conn.penalties == [("outlet.example", 1)]

    def test_unknown_domain_is_never_recorded(self, conn):
        pc.update_domain_penalty(conn, "unknown", 1)
        pc.update_domain_penalty(conn, "", 1)
        assert conn.penalties == []

    def test_db_failure_is_swallowed(self):
        class _Broken:
            def execute(self, *a, **k):
                raise RuntimeError("gone")

        pc.update_domain_penalty(_Broken(), "outlet.example", 1)  # must not raise


def test_prescreen_payload_still_records_why(conn):
    """The stored payload is what the repair script re-scores, so it has to keep naming
    the reason and carrying the prescreen signals."""
    captured = {}

    def _execute(sql, params=()):
        if "llm_parsed_output" in sql and "archived" in sql:
            captured["payload"] = json.loads(params[0])
        conn._last = sql
        return conn

    conn.execute = _execute
    det = pc.deterministic_relevance("Local community cleanup effort", "")
    pc._try_prescreen_archive(conn, _event("Local community cleanup effort"), det)
    assert captured["payload"]["archived_reason"] == "deterministic_prescreen"
    assert captured["payload"]["prescreen"]["score"] == det["score"]
