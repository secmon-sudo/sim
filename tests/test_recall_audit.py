"""Tests for the monthly recall audit against UCDP candidate data.

The audit exists to say what SIM never saw, so its failure mode is flattering
itself: a dropped reference event, a mis-parsed month or an unmapped country name
all shrink the denominator and make recall look better than it is. These pin the
places that can happen.
"""

import csv
import io
from datetime import date

import pytest

from scripts import recall_audit as ra


def _csv(rows):
    """A UCDP candidate CSV with only the columns the audit reads."""
    fields = ["best", "date_start", "country", "where_coordinates"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


class TestReleaseAddressing:
    def test_a_monthly_version_names_its_file(self):
        assert ra.candidate_url("26.0.7").endswith("GEDEvent_v26_0_7.csv")

    def test_a_quarterly_version_is_refused(self):
        """26.01.26.06 is a quarterly release and a different file shape; taking
        it here would audit six months against one month's corpus window."""
        with pytest.raises(ValueError):
            ra.candidate_url("26.01.26.06")

    def test_a_yearly_version_is_refused(self):
        with pytest.raises(ValueError):
            ra.candidate_url("26.1")

    def test_the_month_comes_from_the_version(self):
        assert ra._release_month("26.0.7") == (2026, 7)

    def test_discovery_walks_back_to_the_newest_release(self):
        """UCDP publishes a month behind, so the current month is normally
        missing. The offset is asked, never assumed."""
        seen = []

        def fetch(url):
            seen.append(url)
            return url.endswith("v26_0_7.csv")

        assert ra.discover_latest_version(date(2026, 9, 11), fetch=fetch) == "26.0.7"
        assert seen[0].endswith("v26_0_9.csv")

    def test_discovery_crosses_the_year_boundary(self):
        def fetch(url):
            return url.endswith("v25_0_12.csv")

        assert ra.discover_latest_version(date(2026, 1, 20), fetch=fetch) == "25.0.12"

    def test_discovery_gives_up_rather_than_guessing(self):
        assert ra.discover_latest_version(date(2026, 9, 11),
                                          fetch=lambda _u: False) is None


class TestReferenceSet:
    def test_the_threshold_and_month_both_bind(self):
        text = _csv([
            {"best": "12", "date_start": "2026-07-04 00:00:00.000",
             "country": "Mali", "where_coordinates": "Di village"},
            {"best": "3", "date_start": "2026-07-05 00:00:00.000",
             "country": "Mali", "where_coordinates": "x"},
            {"best": "40", "date_start": "2026-05-05 00:00:00.000",
             "country": "Mali", "where_coordinates": "y"},
        ])
        records, unmapped = ra.reference_events(text, 2026, 7, 10)
        assert [(r["iso"], r["day"], r["deaths"]) for r in records] == [
            ("ML", "2026-07-04", 12)]
        assert not unmapped

    def test_revisions_to_earlier_months_are_not_rescored(self):
        """The July file carries corrections reaching back to March. Those months
        were already audited against the same corpus; counting them again scores
        one miss twice and drifts the trend."""
        text = _csv([{"best": "50", "date_start": "2026-03-02 00:00:00.000",
                      "country": "Sudan", "where_coordinates": "x"}])
        records, _ = ra.reference_events(text, 2026, 7, 10)
        assert records == []

    def test_an_unmapped_country_is_counted_not_dropped_silently(self):
        """A name with no ISO leaves the reference set. That makes SIM look
        better, so it has to arrive as a number the report prints."""
        text = _csv([{"best": "20", "date_start": "2026-07-09 00:00:00.000",
                      "country": "Atlantis", "where_coordinates": "x"}])
        records, unmapped = ra.reference_events(text, 2026, 7, 10)
        assert records == []
        assert unmapped == {"Atlantis": 1}

    def test_ucdp_historical_names_map(self):
        """UCDP writes "DR Congo (Zaire)" and "Yemen (North Yemen)"; events
        stores CD and YE."""
        assert ra.UCDP_COUNTRY_ISO["DR Congo (Zaire)"] == "CD"
        assert ra.UCDP_COUNTRY_ISO["Yemen (North Yemen)"] == "YE"
        assert ra.UCDP_COUNTRY_ISO["Myanmar (Burma)"] == "MM"

    def test_a_missing_death_count_does_not_crash_the_parse(self):
        text = _csv([{"best": "", "date_start": "2026-07-09 00:00:00.000",
                      "country": "Mali", "where_coordinates": "x"}])
        assert ra.reference_events(text, 2026, 7, 10) == ([], ra.Counter())


class _Conn:
    """Answers (in_corpus, tiered) per country-day from a dict."""

    def __init__(self, answers):
        self.answers = answers
        self.queries = []

    def execute(self, _sql, params=None):
        iso, day = params[0], params[1]
        self.queries.append((iso, day))
        answer = self.answers.get((iso, day), (0, 0))

        class _R:
            def fetchone(self_inner):
                return answer
        return _R()


class TestMeasurement:
    RECORDS = [
        {"iso": "ET", "day": "2026-07-03", "deaths": 40},
        {"iso": "ET", "day": "2026-07-11", "deaths": 20},
        {"iso": "UA", "day": "2026-07-06", "deaths": 15},
    ]

    def test_blind_covered_and_paged_are_counted_apart(self):
        """Three states, not two: a country-day SIM ingested but never paged is
        neither a miss nor a success."""
        conn = _Conn({("UA", "2026-07-06"): (12, 3),
                      ("ET", "2026-07-11"): (2, 0)})
        result = ra.measure(conn, self.RECORDS, None)
        assert result["covered"] == 2
        assert result["paged"] == 1
        assert result["blind"] == 1
        assert result["per_country"]["ET"] == {"days": 2, "blind": 1}

    def test_days_before_the_corpus_are_not_misses(self):
        """SIM's corpus starts where it starts. Scoring an event from before it
        would invent a miss the pipeline never had a chance at."""
        conn = _Conn({})
        result = ra.measure(conn, self.RECORDS, date(2026, 7, 10))
        assert result["skipped_before_corpus"] == 2
        assert result["scored_days"] == 1
        assert result["blind"] == 1

    def test_one_query_per_distinct_country_day(self):
        """Two reference events on the same country-day are one question."""
        conn = _Conn({})
        records = self.RECORDS + [{"iso": "ET", "day": "2026-07-03", "deaths": 11}]
        ra.measure(conn, records, None)
        assert len(conn.queries) == 3

    def test_the_window_reaches_further_forward_than_back(self):
        """UCDP dates the violence, SIM dates the ingest, and the report of a
        remote massacre lands days later — never earlier by much."""
        assert ra._WINDOW_FORWARD_DAYS > ra._WINDOW_BACK_DAYS
