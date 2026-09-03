"""Corroborations are written once per run, not once per duplicate.

Measured 2026-09-03 over 30 runs, this write was 90 s/run — 20% of Pass A and 11%
of the whole pipeline — spent as ~730 separate UPDATEs against a remote pooler. The
round trips were the cost, not the work; the same finding, and the same fix, as
load_domain_penalties (210 s/run as a per-item query).

What must not change in the move: the refusals (an outlet republishing itself, a
carrier that is not a witness) still run per duplicate, and the statements still
execute one per duplicate in order, because the idempotency guard only works if the
previous append is already visible.
"""

from unittest.mock import MagicMock

from src.pipeline.pass_a_ingest import (
    _corroboration_params,
    _flush_corroborations,
    _headline_fingerprint,
)


def _cursor(rowcounts):
    """A cursor whose executemany yields one result set per rowcount given."""
    counts = list(rowcounts)
    cur = MagicMock()
    state = {"i": 0}

    def _rowcount():
        return counts[state["i"]] if state["i"] < len(counts) else -1

    def _nextset():
        state["i"] += 1
        return True if state["i"] < len(counts) else None

    type(cur).rowcount = property(lambda s: _rowcount())
    cur.nextset.side_effect = _nextset
    return cur


def _db(cur):
    db = MagicMock()
    db.transaction.return_value.__enter__ = lambda s: None
    db.transaction.return_value.__exit__ = lambda s, *a: False
    db.cursor.return_value.__enter__ = lambda s: cur
    db.cursor.return_value.__exit__ = lambda s, *a: False
    return db


class TestParamsApplyTheSameRefusals:
    def test_valid_duplicate_produces_params(self):
        params = _corroboration_params("evt-1", "reuters.com", "bbc.co.uk",
                                       "https://bbc.co.uk/x", "Strike reported")
        assert params is not None
        assert params[1] == "evt-1"

    def test_self_republish_refused(self):
        """An outlet republishing itself proves nothing — it never reaches the batch."""
        assert _corroboration_params("evt-1", "www.reuters.com", "reuters.com",
                                     "https://reuters.com/x", "Strike") is None

    def test_carrier_refused(self):
        """A carrier is not a witness: Yahoo redistributes one newsroom's filing."""
        assert _corroboration_params("evt-1", "reuters.com", "news.yahoo.com",
                                     "https://news.yahoo.com/x", "Strike") is None

    def test_missing_event_or_domain_refused(self):
        assert _corroboration_params(None, "reuters.com", "bbc.co.uk",
                                     "https://bbc.co.uk/x", "Strike") is None
        assert _corroboration_params("evt-1", "reuters.com", "",
                                     "https://bbc.co.uk/x", "Strike") is None


class TestFlush:
    def test_empty_batch_touches_no_connection(self):
        db = MagicMock()
        assert _flush_corroborations(db, []) == 0
        db.cursor.assert_not_called()

    def test_one_round_trip_for_the_whole_run(self):
        """The point of the change: N duplicates, ONE executemany."""
        cur = _cursor([1, 1, 1])
        db = _db(cur)
        pending = [("e", "evt-1", 8, "p")] * 3
        _flush_corroborations(db, pending)
        cur.executemany.assert_called_once()
        assert cur.executemany.call_args.args[1] == pending

    def test_counts_only_rows_actually_appended(self):
        """A duplicate whose domain is already credited updates nothing (rowcount 0),
        and must not be counted as a new corroboration."""
        cur = _cursor([1, 0, 1])
        assert _flush_corroborations(_db(cur), [("e", "evt-1", 8, "p")] * 3) == 2

    def test_statements_run_in_order_one_per_duplicate(self):
        """Ordering is load-bearing: the `NOT @> probe` guard makes the second
        duplicate from a credited domain a no-op only if the first is visible. A
        single UPDATE ... FROM (VALUES ...) would be one round trip too, and would
        silently drop every duplicate after the first for the same event."""
        cur = _cursor([1, 0])
        pending = [("first", "evt-1", 8, "p"), ("second", "evt-1", 8, "p")]
        _flush_corroborations(_db(cur), pending)
        sent = cur.executemany.call_args.args[1]
        assert [p[0] for p in sent] == ["first", "second"]

    def test_failure_never_breaks_the_run(self):
        """Pre-migration DBs lack the column; corroboration is a bonus signal."""
        db = MagicMock()
        db.transaction.side_effect = RuntimeError("no such column")
        assert _flush_corroborations(db, [("e", "evt-1", 8, "p")]) == 0


class TestInRunDedupTelemetry:
    """Sizing counters for the article_fetch parallelisation (3 Sep 2026).

    `recent_events.insert(0, ...)` puts this run's own inserts at the HEAD of the
    dedup corpus, so they are compared first and win the match. A design that
    defers an insert while its article fetch is in flight would hide those entries
    from the next K candidates and change dedup outcomes. These counters measure
    the exposure instead of assuming it is small.
    """

    def test_dup_index_below_insert_count_is_an_in_run_match(self):
        """dup_idx < inserted is the whole discriminator — pin its arithmetic."""
        # 3 inserts this run, so corpus indices 0..2 are in-run, 3+ preloaded.
        inserted = 3
        assert 0 < inserted and 2 < inserted  # in-run
        assert not 3 < inserted  # first preloaded entry
        assert not 9 < inserted

    def test_windows_are_nested(self):
        """within_4 ⊆ within_8 ⊆ within_16 ⊆ matched_in_run, by construction."""
        inserted = 100
        counts = {"in_run": 0, 4: 0, 8: 0, 16: 0}
        for dup_idx in (0, 3, 4, 7, 8, 15, 16, 40):
            if dup_idx < inserted:
                counts["in_run"] += 1
                for window in (4, 8, 16):
                    if dup_idx < window:
                        counts[window] += 1
        assert counts[4] == 2  # 0, 3
        assert counts[8] == 4  # + 4, 7
        assert counts[16] == 6  # + 8, 15
        assert counts["in_run"] == 8
        assert counts[4] <= counts[8] <= counts[16] <= counts["in_run"]


class TestSyndicatedFilingRefusal:
    """One newsroom's filing under a second masthead is not corroboration.

    Measured 3 Sep 2026 over 14 days: 865 of 6009 corroboration records (14.4%)
    were byte-identical headlines after the publisher suffix was stripped, and 12
    of 12 sampled were real syndication. 413 events were carrying a "Çoklu kaynak"
    label on that evidence, 62 of them ALERT or CRITICAL cards. Validated against
    600 production pairs: 600/600 agreement, no divergence in either direction.
    """

    def _params(self, event_title, dup_title, event_domain="reuters.com",
                dup_domain="bbc.co.uk"):
        return _corroboration_params("evt-1", event_domain, dup_domain,
                                     "https://x/1", dup_title, event_title)

    def test_same_headline_under_two_mastheads_is_refused(self):
        assert self._params(
            "Russia strike blows up arms depot near Kyiv, killing 37 - Yahoo News",
            "Russia strike blows up arms depot near Kyiv, killing 37 - euractiv.com",
        ) is None

    def test_identical_headline_with_no_suffix_at_all_is_refused(self):
        assert self._params(
            "Two injured in blast at train station in southern Germany",
            "Two injured in blast at train station in southern Germany",
        ) is None

    def test_own_cctld_edition_is_refused(self):
        """bbc.com corroborating bbc.co.uk — the registrable-domain guard cannot
        see this, because the two ARE different registrable domains."""
        from src.core.sitrep_verify import registrable_domain
        assert registrable_domain("bbc.com") != registrable_domain("bbc.co.uk")
        assert self._params(
            "Airport warns passengers ahead of busy weekend - BBC",
            "Airport warns passengers ahead of busy weekend - BBC",
            event_domain="bbc.com", dup_domain="bbc.co.uk",
        ) is None

    def test_independent_reporting_of_one_event_is_still_recorded(self):
        """The signal has to be exact: near-identical is what content dedup already
        selected for, so anything looser would refuse the corroboration worth having."""
        assert self._params(
            "Russian missile strikes kill 12 in Kyiv overnight - Reuters",
            "Death toll from Kyiv missile attack rises to 12, officials say - BBC",
        ) is not None

    def test_short_headline_match_is_a_coincidence_not_syndication(self):
        """Under the floor two newsrooms could reach the same words on their own."""
        assert self._params("Explosions heard in Kyiv",
                            "Explosions heard in Kyiv") is not None

    def test_case_and_spacing_do_not_defeat_the_match(self):
        assert self._params(
            "Drone  attack  kills two children in Russia's Krasnodar - Reuters",
            "DRONE ATTACK KILLS TWO CHILDREN IN RUSSIA'S KRASNODAR - Devdiscourse",
        ) is None

    def test_en_and_em_dash_suffixes_are_stripped_too(self):
        assert self._params(
            "Iran strikes US base in Erbil, Kurdish officials say – Rudaw",
            "Iran strikes US base in Erbil, Kurdish officials say — Kurdistan24",
        ) is None

    def test_missing_event_title_keeps_the_old_behaviour(self):
        """The parameter is optional: callers that never learned the survivor's
        headline must still record corroboration exactly as before."""
        assert _corroboration_params(
            "evt-1", "reuters.com", "bbc.co.uk", "https://x/1",
            "Russian missile strikes kill 12 in Kyiv overnight - BBC",
        ) is not None

    def test_fingerprint_leaves_a_dashless_headline_intact(self):
        assert _headline_fingerprint("Explosions rock Kyiv city centre") == \
            "explosions rock kyiv city centre"

    def test_fingerprint_does_not_eat_a_hyphenated_headline_tail(self):
        """The suffix pattern needs spaces around the dash, so 'Kryvyi Rih' style
        hyphenation and dashless titles survive."""
        assert _headline_fingerprint("US-Iran tensions escalate over Hormuz") == \
            "us-iran tensions escalate over hormuz"
