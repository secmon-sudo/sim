"""Parallel article fetch must not move a single decision (3 Sep 2026).

article_fetch was Pass A's largest phase — 145-216s over recent runs for ~120
sequential fetches, most paying two round trips because the Google News handle
resolves before the page loads. It is pure network wait, so it parallelises; what
does NOT parallelise is the loop's ordering. This run's own inserts are PREPENDED
to recent_events and find_content_duplicate returns the FIRST match, so an insert
still in flight is an insert the next candidates cannot see, and the order they
land in decides which event a later duplicate corroborates.

Measured before the change (run 33721288075): of 896 content duplicates, 11 matched
an event inserted earlier in the same run and 6 of those within the last 8 inserts.
A window of 8 that ignored this would have silently changed 6 dedup decisions in one
run — roughly 66 a day, each a duplicate event that should have been merged.

So the test that matters is not "does it go faster", it is "is the output identical
to the sequential loop". These run the real loop over the same items with the window
closed (1) and open (8) and compare what reached the database.
"""

import json

from src.pipeline import pass_a_ingest as pa


def _decisions(updates):
    """Corroboration writes minus seen_at, which is wall-clock and cannot match.

    What has to be identical is the DECISION: which outlet was credited, for which
    headline, to which surviving event.
    """
    out = []
    for entry_json, event_id, _cap, _probe in updates:
        entry = json.loads(entry_json)[0]
        out.append((entry["domain"], entry["title"], event_id))
    return out


class _RecordingConn:
    """Enough psycopg surface for the insert path, recording what was written."""

    def __init__(self):
        self.inserted = []          # (url, domain, title) in insert order
        self.updates = []           # corroboration UPDATE parameter tuples
        self._next_id = 0
        self._seen_hashes = set()
        self._last = None

    # -- context manager used as db_conn.transaction() --
    def transaction(self):
        conn = self

        class _Tx:
            def __enter__(self_inner):
                return conn

            def __exit__(self_inner, *exc):
                return False

        return _Tx()

    def execute(self, sql, params=None):
        self._last = None
        if "INSERT INTO events" in sql:
            url, url_hash, domain, title = params[0], params[1], params[2], params[3]
            if url_hash in self._seen_hashes:
                self._last = None            # NOT EXISTS guard rejected it
            else:
                self._seen_hashes.add(url_hash)
                self._next_id += 1
                self.inserted.append((url, domain, title))
                self._last = (f"evt-{self._next_id}",)
        elif "UPDATE events" in sql:
            self.updates.append(params)
        return self

    def fetchone(self):
        return self._last

    def fetchall(self):
        return []

    def commit(self):
        pass

    # -- cursor()/executemany(), the path _flush_corroborations takes --
    def cursor(self):
        conn = self

        class _Cur:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def executemany(self_inner, sql, seq, returning=False):
                conn.updates.extend(seq)
                self_inner._left = len(seq)

            @property
            def rowcount(self_inner):
                return 1

            def fetchone(self_inner):
                return ("x",)

            def nextset(self_inner):
                self_inner._left -= 1
                return self_inner._left > 0

        return _Cur()

    @property
    def rowcount(self):
        return 1


_STORIES = [
    "Russian drone strike hits Kyiv apartment block leaving twelve dead",
    "Wildfire forces mass evacuations across northern Nevada counties",
    "Iranian missiles target coalition airbase near Erbil overnight",
    "Explosion at Leipzig airport cargo terminal injures four staff",
    "Houthi ballistic missile strikes tanker off the Yanbu coast",
    "Nigerian troops repel militant assault on Kukawa garrison town",
    "Coup attempt in Niamey collapses after presidential guard mutiny",
    "Kherson thermal power plant loses all equipment in repeat strike",
    "Strait of Hormuz convoy escorted by eighteen warships this week",
    "Bomb threat grounds departures at Labrador regional airport",
    "Ukrainian drones destroy oil refinery hub in Krasnodar region",
    "Germany summons Russian ambassador over airport drone incident",
]


def _items(n=30):
    """Distinct stories, with planted duplicates that land INSIDE the window.

    Every fourth item repeats the story three places back, so the repeat is close
    enough behind its original to still be in flight when a window of 8 is open —
    the only configuration where a naive deferral would answer wrongly. The repeat
    is reworded rather than reprinted so it survives the syndication guard and
    produces a real corroboration to compare. Both details matter: without the
    spacing the window is never consulted, without the rewording nothing is
    recorded, and the test passes vacuously either way.
    """
    out = []
    for i in range(n):
        repeat = i % 4 == 3 and i >= 3
        idx = (i - 3) if repeat else i
        story = _STORIES[idx % len(_STORIES)]
        # A repeat is REWORDED, not reprinted: content dedup still catches it, but
        # the syndication guard must not, or the corroboration comparison below
        # would be asserting 0 == 0.
        if repeat:
            story = "Report: " + story[0].lower() + story[1:]
        out.append({
            "title": story,
            "description": f"{story}. Officials confirmed the account on the record.",
            "link": f"https://outlet{i % 9}news.example{i % 9}.com/story-{i}",
            "source": "rss",
            "_priority": 1,
            "pub_dt": None,
        })
    return out


def _fake_fetch_article(url):
    """Deterministic, and reprints a slice of the corpus the way production does."""
    return {"url": url, "text": f"Body text for {url}. " * 8,
            "published_at": None, "fetch_ok": True}


def _run(monkeypatch, window, items):
    conn = _RecordingConn()
    monkeypatch.setattr(pa, "_ARTICLE_FETCH_WINDOW", window)
    monkeypatch.setattr(pa, "_ARTICLE_FETCH_WORKERS", 4)
    monkeypatch.setattr(pa, "fetch_article", _fake_fetch_article)
    monkeypatch.setattr(pa, "translate_to_english_if_needed", lambda t: t)
    monkeypatch.setattr(pa, "_fetch_recent_events_for_dedup", lambda c: ([], []))
    monkeypatch.setattr(pa, "load_domain_penalties", lambda c: {})
    # One synthetic query feed carries every item; everything else returns nothing.
    monkeypatch.setattr(pa, "build_search_queries",
                        lambda c: [{"query": "test", "dynamic": False}])
    calls = {"n": 0}

    def _feed(src, is_direct_url=False, stats=None):
        calls["n"] += 1
        return [dict(it) for it in items] if calls["n"] == 1 else []

    monkeypatch.setattr(pa, "fetch_rss_feed", _feed)
    monkeypatch.setattr(pa, "fetch_travel_advisories", lambda **k: [])
    stats = pa.run_pass_a(conn, max_events=12)
    return conn, stats


class TestWindowEquivalence:
    def test_same_inserts_in_the_same_order(self, monkeypatch):
        items = _items()
        seq_conn, seq_stats = _run(monkeypatch, 1, items)
        par_conn, par_stats = _run(monkeypatch, 8, items)
        assert par_conn.inserted == seq_conn.inserted

    def test_same_corroborations(self, monkeypatch):
        items = _items()
        seq_conn, _ = _run(monkeypatch, 1, items)
        par_conn, _ = _run(monkeypatch, 8, items)
        assert _decisions(par_conn.updates) == _decisions(seq_conn.updates)
        assert _decisions(seq_conn.updates), "fixture recorded no corroboration"

    def test_the_window_was_actually_used(self, monkeypatch):
        """Guards the three assertions above from passing by going sequential.

        If the fixture ever stops putting a duplicate inside an open window, the
        equivalence tests would still pass while proving nothing.
        """
        items = _items()
        _, seq = _run(monkeypatch, 1, items)
        _, par = _run(monkeypatch, 8, items)
        assert seq["dedup_window_stalls"] == 0
        assert par["dedup_window_stalls"] > 0

    def test_same_counters(self, monkeypatch):
        items = _items()
        _, seq = _run(monkeypatch, 1, items)
        _, par = _run(monkeypatch, 8, items)
        for key in ("events_inserted", "content_duplicates_skipped",
                    "duplicates_skipped", "full_text_attempted",
                    "corroborations_recorded"):
            assert par[key] == seq[key], key


class TestPendingMatch:
    def test_an_in_flight_headline_is_seen(self):
        pending = [{"title": "Russian drone strike hits Kyiv apartment block",
                    "canonical": "russian drone strike hits kyiv apartment block " * 4}]
        assert pa._pending_matches(
            pending, "Russian drone strike hits Kyiv apartment block",
            "russian drone strike hits kyiv apartment block " * 4)

    def test_an_unrelated_headline_is_not(self):
        pending = [{"title": "Russian drone strike hits Kyiv apartment block",
                    "canonical": "russian drone strike hits kyiv apartment block " * 4}]
        assert not pa._pending_matches(
            pending, "Wildfire forces evacuations near Reno Nevada",
            "wildfire forces evacuations near reno nevada " * 4)

    def test_empty_window_matches_nothing(self):
        assert not pa._pending_matches([], "anything at all", "anything at all")


class TestDrainAttribution:
    """Why the window was drained, counted per reason.

    The first parallel production run (33736464695) cut article_fetch from 145.5s
    to 88.1s but recorded ZERO stalls against 21 in-run duplicate matches — so the
    window was usually empty when those arrived. It is draining more often than
    correctness requires, and the remaining time sits behind whichever reason
    dominates. Guessing which would repeat the mistake this whole change avoided.
    """

    def _drains(self, stats):
        return {k: v for k, v in stats.items() if k.startswith("window_drain_")}

    def test_all_four_reasons_are_counted_separately(self, monkeypatch):
        items = _items()
        _, stats = _run(monkeypatch, 8, items)
        assert set(self._drains(stats)) == {
            "window_drain_full", "window_drain_same_domain",
            "window_drain_body_grows", "window_drain_no_fetch",
            "window_drain_stall",
        }

    def test_a_closed_window_never_reports_a_full_drain(self, monkeypatch):
        """With a window of 1 every item settles immediately, so 'full' is the
        only reason that cannot be the interesting one."""
        _, stats = _run(monkeypatch, 1, _items())
        assert stats["window_drain_stall"] == 0

    def test_the_stall_drain_tracks_the_stall_counter(self, monkeypatch):
        """Two names for one event; if they ever disagree the attribution is
        reading a different code path than the correctness counter."""
        for window in (1, 8):
            _, stats = _run(monkeypatch, window, _items())
            assert stats["window_drain_stall"] == stats["dedup_window_stalls"]

    def test_an_open_window_drains_for_at_least_one_reason(self, monkeypatch):
        _, stats = _run(monkeypatch, 8, _items())
        assert sum(self._drains(stats).values()) > 0
