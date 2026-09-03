"""
SIM — Pass A: Ingest & Canonicalization
Blueprint V20.1 §4 PASS A

Orchestrates ingest: builds the query set, fans out to the source fetchers,
applies noise/age/dedup filtering and inserts raw events. The heavy lifting
lives in focused modules (split from this 1.9K-line monolith on 2026-07-16):

  - ingest_queries   — search-query construction (static tiers + storyline queries)
  - ingest_sources   — all network I/O (RSS, advisories, translate)
  - ingest_filters   — pure text filters, canonicalization, similarity dedup

This module keeps only the DB-touching pieces and run_pass_a itself. The
re-exports below preserve the historical import surface of pass_a_ingest.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Re-exported: historical import surface of this module (consumers: orchestrator,
# pass_c_classify, tests). Keep these names importable from pass_a_ingest.
# sitrep_verify owns the question "does this domain count as an outlet of its own",
# so ingest and the report layer cannot disagree about it. Safe to import at module
# level: sitrep_verify pulls ingest_filters lazily, inside registrable_domain.
from src.core.sitrep_verify import is_independent_publisher

from src.pipeline.ingest_filters import (  # noqa: F401
    _HIGH_SIGNAL_TERMS,
    _SECURITY_KEYWORD_PATTERN,
    KEYWORDS_CONFIG,
    NOISE_PATTERNS,
    PROMPT_INJECTION_PATTERNS,
    _matches_security_keywords,
    canonicalize_text,
    check_content_duplicate,
    compute_url_hash,
    extract_domain,
    is_social_platform,
    social_publisher_domain,
    find_content_duplicate,
    is_content_farm,
    is_noise,
    normalize_title,
    priority_score,
    title_similarity,
    title_token_similarity,
)
from src.pipeline.ingest_queries import (  # noqa: F401
    MAX_DYNAMIC_QUERIES,
    build_search_queries,
)
from src.pipeline.ingest_sources import (  # noqa: F401
    GOOGLE_NEWS_RSS,
    fetch_article,
    fetch_full_text,
    fetch_rss_feed,
    fetch_travel_advisories,
    google_translate,
    reset_translation_counter,
    translate_to_english_if_needed,
    translation_call_count,
    translation_failure_count,
)

logger = logging.getLogger(__name__)

# Load configuration
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as f:
    SETTINGS = json.load(f)

# Settings lookups
_INGESTION = SETTINGS.get("ingestion", {})
_MAX_ARTICLE_AGE_DAYS = _INGESTION.get("max_article_age_days", 4)
_FETCH_FULL_TEXT = _INGESTION.get("fetch_full_text", True)
# Full-text extraction costs ~1s of wall clock per article (measured 2026-07-23:
# 0.5s to resolve the Google News redirect, 0.5s for trafilatura), and roughly
# two in five URLs return nothing — paywalls and JS-rendered pages. Running it on
# every candidate would add minutes to a run for little gain, since most items
# carry a one-line RSS description that scores 1-2 on the priority triage.
# Gate on that score and cap the total: the depth goes where it changes a report.
_FULL_TEXT_MIN_PRIORITY = _INGESTION.get("full_text_min_priority", 3)
# Trust the article page's own publication date over the feed's. Google News
# stamps re-crawled archive pages with the crawl date — measured 2026-08-05, a
# Yeni Safak story published 2016-10-25 was served as "Tue, 04 Aug 2026 18:50"
# and cleared the max_article_age_days gate, which reads that same field. Three
# 2016-2021 reprints reached ALERT tier that way. The page's own
# schema.org/OpenGraph date is the one field the aggregator cannot rewrite.
_VERIFY_PUBLISH_DATE = _INGESTION.get("verify_publish_date", True)
# Ceiling on article fetches per run. Unlike the full-text gate this is NOT
# priority-scoped: the Aden reprint scored 1 on the ingest triage and would have
# slipped straight past a priority-gated check.
_ARTICLE_FETCH_MAX_PER_RUN = _INGESTION.get("article_fetch_max_per_run", 120)
# Article fetches are the largest phase in Pass A — 145-216s across recent runs for
# ~120 sequential fetches, most paying TWO round trips because the Google News handle
# has to resolve before the page loads. They are pure network wait, so they overlap;
# what does not overlap is the loop's own ordering (see _settle_pending_inserts).
_ARTICLE_FETCH_WORKERS = _INGESTION.get("article_fetch_workers", 8)
# How many fetched-but-not-yet-inserted items may be in flight. Every one of them is
# an insert the dedup corpus cannot see yet, which is why the window is drained the
# moment that could matter rather than sized generously.
_ARTICLE_FETCH_WINDOW = _INGESTION.get("article_fetch_window", 8)
# A wall-clock bound beside the per-request timeout, for the reason resolve_cluster_urls
# already records: per-request timeouts do not bound total time when the slow path is
# many requests each landing just under the limit.
_ARTICLE_FETCH_DEADLINE_S = _INGESTION.get("article_fetch_deadline_s", 240.0)
_MAX_EVENT_FUTURE_DAYS = _INGESTION.get("max_event_future_days", 1)
_MAX_EVENTS_PER_DOMAIN = _INGESTION.get("max_events_per_domain", 8)
# Per-domain overrides of the cap above (eTLD+1 → cap). For high-volume,
# single-source rapid-relay feeds (e.g. OSINT aggregator accounts) that would
# otherwise claim a disproportionate share of every run's insert budget.
_PER_DOMAIN_CAPS = {
    k.lower(): int(v) for k, v in _INGESTION.get("per_domain_caps", {}).items()
}

# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

# Pass A is ~78% of run wall-clock (measured 2026-08-24 over 43 runs) and had no
# phase timing at all, so the only honest answer to "why" was inference. These
# accumulate seconds per phase into the existing pass_a telemetry blob; the loop
# phases are accumulators because the cost is spread over ~1000 iterations rather
# than spent in one block.
@contextmanager
def _timed(acc: dict, key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        acc[key] = acc.get(key, 0.0) + (time.perf_counter() - start)


def _fetch_recent_events_for_dedup(db_conn) -> tuple[list[tuple[str, str]], list[tuple]]:
    """Fetch recent events once to avoid O(N) database queries during ingestion.

    Returns (texts, meta) as two INDEX-ALIGNED lists: texts feeds the similarity
    matcher (title, canonical_text, anchor); meta carries (event_id, source_domain)
    so a detected duplicate can be credited back to the surviving event as
    corroboration.

    The anchor rides along as place evidence for the place-disagreement veto in
    find_content_duplicate: a wire headline routinely omits the town its own body
    names, and the anchor is the only place the classifier's answer is recorded.
    """
    try:
        rows = db_conn.execute(
            """SELECT id, source_domain, source_title, canonical_text, anchor_name_raw
               FROM events
               WHERE ingested_at > NOW() - (%s * INTERVAL '1 day')
               ORDER BY ingested_at DESC
               LIMIT 2000""",
            (_MAX_ARTICLE_AGE_DAYS,),
        ).fetchall()
        texts = [(row[2] or "", row[3] or "", row[4] or "") for row in rows]
        meta = [(row[0], row[1] or "") for row in rows]
        return texts, meta
    except Exception:
        logger.exception("Failed to fetch recent events for dedup")
        return [], []


# Max corroborating sources kept per event — enough for a Çoklu Kaynak/Resmî
# upgrade; beyond that more entries add bytes, not information.
_MAX_CORROBORATING_SOURCES = 5

# A cut URL is not a shorter link, it is a broken one. Google News RSS links run
# past a thousand characters (max measured 1,054 over four days) and the old
# blanket dup_url[:500] sliced straight through the base64 article id, so the
# resolver could not decode it and the SITREP appendix rendered a dead citation.
# Measured 2026-08-19: 19 of the 51 Google News URLs across four days of SITREPs
# were mangled this way, while events.source_url — which is never sliced — held
# all 533 of them intact.
#
# The ceiling stays, because corroborating_sources is inline JSONB on a table
# already at 284 MB and an unbounded string there is a row-size risk. But over it
# we store NO url rather than a broken one: the corroboration SIGNAL is the
# domain (that is what CORROBORATION_ALERT_MIN counts), and every consumer of
# these entries already skips a missing url.
_MAX_SOURCE_URL_CHARS = 2048


def _citable_url(url: str | None) -> str | None:
    """The URL if it survives whole, else None — never a truncated one."""
    url = (url or "").strip()
    return url if url and len(url) <= _MAX_SOURCE_URL_CHARS else None



# A publisher suffix — " - Reuters", " – The Eastern Herald", " — BBC" — is the only
# part of a syndicated headline that changes as one filing moves between mastheads.
# Stripped before comparison so the wire copy underneath can be recognised.
_PUBLISHER_SUFFIX_RE = re.compile(r"\s+[-\u2013\u2014]\s+[^-\u2013\u2014]{2,40}$")

# Below this many characters an identical headline stops being evidence of
# syndication and starts being a coincidence two newsrooms could reach on their own
# ("Explosions heard in Kyiv"). Measured over 14 days of production corroborations:
# the shortest genuine match ran 21 characters and exactly ONE of 865 fell under 25,
# so the floor costs 0.1% of the signal and removes the whole coincidence class.
_SYNDICATION_MIN_HEADLINE_CHARS = 25


def _headline_fingerprint(title: str | None) -> str:
    """The headline with its publisher suffix removed, lowercased, spaces collapsed."""
    if not title:
        return ""
    return " ".join(_PUBLISHER_SUFFIX_RE.sub("", title.strip()).lower().split())


def _is_syndicated_filing(event_title: str | None, dup_title: str | None) -> bool:
    """True when these two headlines are one newsroom's filing under two mastheads.

    Corroboration is supposed to mean a second newsroom looked at the same event.
    An identical headline means the opposite: the wire copy was redistributed, and
    counting it feeds both the "Onaylandı (Çoklu kaynak)" label and the
    corroboration ALERT floor with evidence that does not exist.

    Measured over 14 days (3 Sep 2026): 865 of 6009 corroboration records — 14.4% —
    were byte-identical after suffix stripping, and 12 of 12 sampled were real
    syndication. The signal catches three classes at once that no domain list
    covers: agency copy under many mastheads (economictimes/ottumwacourier),
    station groups under one owner (abc45/katu, wowt/foxcarolina), and an outlet's
    own ccTLD editions — bbc.com corroborating bbc.co.uk, which the
    same-registrable-domain guard above cannot see.

    Deliberately exact, not fuzzy: near-identical headlines are what content dedup
    already selected for, so anything looser would refuse independent reporting of
    the same event, which is precisely the signal worth keeping.
    """
    a = _headline_fingerprint(event_title)
    if len(a) < _SYNDICATION_MIN_HEADLINE_CHARS:
        return False
    return a == _headline_fingerprint(dup_title)


def _record_corroboration(db_conn, event_id, event_domain: str,
                          dup_domain: str, dup_url: str, dup_title: str,
                          event_title: str = "") -> bool:
    """Append a dropped duplicate's source to the surviving event's
    corroborating_sources. Same-registrable-domain duplicates are NOT recorded —
    an outlet republishing itself proves nothing. Idempotent per domain.

    Each entry carries seen_at, the moment this pipeline observed the duplicate.
    The count alone already proved to be the signal confidence is not: measured
    2026-08-17, every silenced event carrying >= 2 independent domains was real
    (the mass drone attack on Moscow, the Benghazi car bombing) and every piece of
    junk carried zero, which is why CORROBORATION_ALERT_MIN exists. The timestamp
    turns that count into a rate — how fast a story spread across independent
    outlets — which is strictly more information for the same write.

    It is OBSERVATION time, not publication time, and the two are far apart here:
    the median gap between a publisher's own date and our ingest is 228 minutes.
    So seen_at measures how quickly SIM saw the spread, bounded below by the run
    cadence — useful for ranking within a window, not for claiming a story broke
    at a particular minute.
    """
    from src.core.sitrep_verify import registrable_domain
    if event_id is None or not dup_domain:
        return False
    if registrable_domain(dup_domain) == registrable_domain(event_domain or ""):
        return False
    # A carrier is not a witness: Yahoo/AOL syndication and Reddit crossposts
    # redistribute one newsroom's filing, and recording them here is what feeds both
    # the "Onaylandı (Çoklu kaynak)" label and the corroboration ALERT floor. Refused
    # at the source so the column stops accumulating them; the two readers filter as
    # well, because rows written before 2026-08-25 keep theirs for the retention window.
    if not is_independent_publisher(dup_domain):
        logger.debug("Corroboration from carrier %s not recorded", dup_domain)
        return False

    params = _corroboration_params(event_id, event_domain, dup_domain, dup_url,
                                   dup_title, event_title)
    if params is None:
        return False
    try:
        with db_conn.transaction():
            result = db_conn.execute(_CORROBORATION_SQL, params)
            return result.rowcount > 0
    except Exception:
        # Pre-migration DBs lack the column — corroboration is a bonus signal,
        # never worth failing an ingest run over.
        logger.debug("Corroboration record failed for event %s", event_id)
        return False


_CORROBORATION_SQL = """UPDATE events
   SET corroborating_sources = corroborating_sources || %s::jsonb
   WHERE id = %s
     AND jsonb_array_length(corroborating_sources) < %s
     AND NOT corroborating_sources @> %s::jsonb"""


def _corroboration_params(event_id, event_domain: str, dup_domain: str,
                          dup_url: str, dup_title: str,
                          event_title: str = "") -> tuple | None:
    """Everything _record_corroboration decides WITHOUT touching the database.

    Split out so the batched path and the single-row path apply exactly the same
    refusals — an outlet republishing itself, a carrier that is not a witness, one
    newsroom's filing under a second masthead — and so a future rule can only be
    added in one place. Returns the UPDATE's parameter
    tuple, or None when this duplicate must not be recorded at all.
    """
    from src.core.sitrep_verify import registrable_domain
    if event_id is None or not dup_domain:
        return None
    if registrable_domain(dup_domain) == registrable_domain(event_domain or ""):
        return None
    if not is_independent_publisher(dup_domain):
        logger.debug("Corroboration from carrier %s not recorded", dup_domain)
        return None
    if _is_syndicated_filing(event_title, dup_title):
        logger.debug("Syndicated filing from %s not recorded as corroboration: %s",
                     dup_domain, (dup_title or "")[:70])
        return None
    entry = json.dumps([{"domain": dup_domain, "url": _citable_url(dup_url),
                         "title": (dup_title or "")[:200],
                         "seen_at": datetime.now(timezone.utc).isoformat()}])
    probe = json.dumps([{"domain": dup_domain}])
    return (entry, event_id, _MAX_CORROBORATING_SOURCES, probe)


# ---------------------------------------------------------------------------
# Article fetch: parallel, without moving a single decision
# ---------------------------------------------------------------------------
#
# The fetch was the largest phase in Pass A (145-216s over recent runs) and it is
# pure network wait, so it parallelises — but naively deferring an insert while its
# fetch is in flight does NOT come free. This run's own inserts are PREPENDED to
# recent_events, so they are compared first and win the match; an insert still in
# flight is an insert the next candidates cannot see.
#
# That exposure was measured before any of this was written (run 33721288075):
# of 896 content duplicates, 11 matched an event inserted earlier in the same run
# and 6 of those matched within the last 8 inserts. Small, but not zero — a window
# of 8 would have silently changed 6 dedup decisions in one run, roughly 66 a day,
# each one a duplicate event that should have been merged.
#
# So the window is drained rather than gambled with, on every condition that could
# make its contents matter:
#   * a candidate that matches something in flight (settle, then re-run dedup)
#   * an insert that skips the fetch entirely (keeps corpus ORDER exactly sequential,
#     which decides WHICH event a later duplicate corroborates)
#   * an item whose canonical_text will grow by its article body (priority >=
#     _FULL_TEXT_MIN_PRIORITY), because its corpus entry would otherwise be missing
#     the very text a later candidate might match on
#   * the window filling, and the end of the loop
#
# What is left running in parallel is the common case, and the common case is ~99%
# of it.


def _pending_matches(pending: list, title: str, canonical: str) -> bool:
    """True when this candidate looks like something already in flight.

    Compared with the same function the settled corpus uses, so the two answers
    cannot drift apart: the in-flight entries are simply a corpus that has not
    landed yet.
    """
    if not pending:
        return False
    corpus = [(p["title"], p["canonical"], "") for p in pending]
    return find_content_duplicate(corpus, title, canonical) is not None


def _flush_corroborations(db_conn, pending: list[tuple]) -> int:
    """Write a whole run's corroborations in one pipelined round trip.

    Measured 2026-09-03 over 30 runs: this was 90 s/run — 20% of Pass A and 11% of
    the entire pipeline — spent as ~730 separate UPDATEs against a remote pooler.
    The round trips were the cost, not the work, which is the same finding and the
    same fix as load_domain_penalties (210 s/run as a per-item query).

    psycopg3 runs executemany in pipeline mode, so the statements still execute
    server-side in order, one per duplicate. That ordering is load-bearing: the
    `NOT corroborating_sources @> probe` guard is what makes a second duplicate from
    an already-credited domain a no-op, and it only works if the first one's append
    is already visible. A single UPDATE ... FROM (VALUES ...) would be one round trip
    too, and WRONG — Postgres applies one row per target, so the second and later
    duplicates of the same event would be silently dropped, and ~730 duplicates
    across ~100 events means most of them share a target.

    The trade is durability: corroborations now land at the end of Pass A instead of
    as they are found, so a mid-loop crash loses them. Acceptable for a signal the
    module already calls "a bonus signal, never worth failing an ingest run over" —
    the events themselves are committed as they go, unchanged.
    """
    if not pending:
        return 0
    try:
        with db_conn.transaction(), db_conn.cursor() as cur:
            cur.executemany(_CORROBORATION_SQL, pending, returning=True)
            recorded = 0
            while True:
                if cur.rowcount and cur.rowcount > 0:
                    recorded += cur.rowcount
                if not cur.nextset():
                    break
            return recorded
    except Exception:
        logger.debug("Corroboration flush failed for %d entries", len(pending))
        return 0


def load_domain_penalties(db_conn) -> dict[str, float] | None:
    """Snapshot the penalty table once, so the ingest loop needs no DB round trips.

    domain_penalties is only written by update_domain_penalty() in Pass C, which runs
    after Pass A has finished, so the table is static for the length of a run and one
    snapshot answers every lookup the loop makes. Measured at 210 s/run (25% of Pass A)
    as a per-item query, against ~700 eligible rows — the round trips were the cost,
    not the data. Returns None on failure so callers fall back to querying per item;
    an empty dict would silently mean "nothing is penalized".
    """
    try:
        with db_conn.transaction():
            rows = db_conn.execute(
                # penalty_score is nullable with a 0.0 default; coalesced here so one
                # NULL row cannot put a None into the map and blow up the `> 0.8`
                # comparison at the call site.
                "SELECT domain, COALESCE(penalty_score, 0.0) FROM domain_penalties"
                " WHERE total_events >= 5"
            ).fetchall()
        return {row[0]: float(row[1]) for row in rows}
    except Exception:
        logger.warning("Domain penalty preload failed; falling back to per-item lookups")
        return None


def check_domain_penalty(db_conn, domain: str,
                         penalties: dict[str, float] | None = None) -> float:
    """Get penalty score for a domain. Returns 0.0 if not found, if total_events < 5, or if whitelisted.

    `penalties` is a load_domain_penalties() snapshot; it already excludes rows under
    the 5-event floor, so a miss is 0.0 for the same reason the query path returns 0.0.
    """
    TRUSTED_DOMAINS = {
        "reuters.com", "bbc.co.uk", "travel.state.gov", "defense.gov",
        "timesofisrael.com", "aljazeera.com", "jpost.com", "haaretz.com",
        "ynetnews.com", "breakingdefense.com", "militarytimes.com",
        "warontherocks.com", "longwarjournal.org", "centcom.mil",
        "cnn.com", "foxnews.com", "wsj.com", "nytimes.com", "dropsitenews.com",
        "presstv.ir", "france24.com", "theguardian.com", "ukrinform.net",
        "kyivindependent.com", "crisisgroup.org", "bellingcat.com",
        "thecipherbrief.com", "foreignpolicy.com", "defenseone.com",
        "twz.com", "defensenews.com", "al-monitor.com", "themoscowtimes.com",
        "meduza.io", "warsawinstitute.org", "un.org",
        "jamestown.org", "thesoufancenter.org", "ctc.westpoint.edu",
        "counterextremism.com",
    }
    if domain in TRUSTED_DOMAINS:
        return 0.0

    if penalties is not None:
        return penalties.get(domain, 0.0)

    try:
        with db_conn.transaction():
            row = db_conn.execute(
                "SELECT penalty_score, total_events FROM domain_penalties WHERE domain = %s",
                (domain,),
            ).fetchone()
            if row:
                penalty, total = row[0], row[1]
                if total >= 5:
                    return penalty
            return 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main Pass A runner
# ---------------------------------------------------------------------------

# A priority_score at or above this is treated as important enough to outrank source
# diversity. 4 is one distinct critical-term hit — the smallest score that cannot be
# produced by generic security vocabulary alone (see priority_score).
PRIORITY_BAND_MIN = 4


def _round_robin(buckets: list[list[dict]], epoch_min: datetime) -> list[dict]:
    """One item from each domain per round, most important lead item first."""
    ordered = sorted(
        (b for b in buckets if b),
        key=lambda b: (b[0]["_priority"], b[0].get("pub_dt") or epoch_min),
        reverse=True,
    )
    out: list[dict] = []
    depth = 0
    while True:
        row = [b[depth] for b in ordered if depth < len(b)]
        if not row:
            return out
        out.extend(row)
        depth += 1


def _interleave_by_domain(items: list[dict]) -> list[dict]:
    """
    Order candidates for the per-run insert budget: importance across bands, source
    diversity within each band.

    Two failure modes the round-robin prevents:
      - A plain newest-first fill let whichever story dominated the global news
        cycle (and got reprinted by every outlet) eat the entire per-run insert
        budget, crowding out quieter regions. Interleaving guarantees every
        domain that delivered items gets a first slot before any domain gets a
        second.
      - Within a domain, feed order used to decide which items survived the
        per-domain cap — so a routine post could claim a capped domain's slot
        while a mass-casualty report behind it was dropped.

    …and the failure mode the BANDS prevent. A single round-robin is depth-first:
    round 0 takes one item from every domain before any domain gets a second. SIM
    draws on more contributing domains than max_events_per_run (100), so round 0 alone
    exhausted the budget and nothing at depth 1 was ever reachable. Priority then had
    no say at all in what survived: measured over the runs to 2026-08-10, the three
    that hit the cap inserted items with a MEDIAN priority of 1 while dropping items
    scoring 5, 7 and 9 — the exact inversion priority_score exists to prevent.

    Banding fixes it without giving up diversity: high-priority items round-robin
    across domains first, everything else round-robins after. Diversity still decides
    the order inside a band; importance decides which band gets served first.
    """
    _EPOCH_MIN = datetime.min.replace(tzinfo=timezone.utc)
    buckets: dict[str, list[dict]] = {}
    for item in items:
        domain = item.get("domain") or extract_domain(item.get("link", ""))
        item["_priority"] = priority_score(item.get("title", ""), item.get("description", ""))
        buckets.setdefault(domain, []).append(item)

    for bucket in buckets.values():
        bucket.sort(
            key=lambda x: (x["_priority"], x.get("pub_dt") or _EPOCH_MIN),
            reverse=True,
        )

    high = [[i for i in b if i["_priority"] >= PRIORITY_BAND_MIN] for b in buckets.values()]
    rest = [[i for i in b if i["_priority"] < PRIORITY_BAND_MIN] for b in buckets.values()]
    return _round_robin(high, _EPOCH_MIN) + _round_robin(rest, _EPOCH_MIN)


def run_pass_a(db_conn, max_events: int | None = None) -> dict:
    """
    Execute Pass A: Ingest & Canonicalization.

    1. Fetch RSS feeds for all keyword queries across geo regions
    2. Filter by age (max_article_age_days)
    3. Canonicalize text, filter noise
    4. Content dedup (title similarity against recent DB events)
    5. Optionally fetch full article text
    6. Insert new events with NOT EXISTS guard (idempotent)

    Returns: stats dict with counts
    """
    max_events = max_events or SETTINGS["pipeline"]["max_events_per_run"]

    stats = {
        "queries_executed": 0,
        "items_fetched": 0,
        "age_filtered": 0,
        "noise_filtered": 0,
        "content_farm_filtered": 0,
        "duplicates_skipped": 0,
        "content_duplicates_skipped": 0,
        "domain_penalized": 0,
        "domain_capped": 0,
        "social_publisher_resolved": 0,
        "social_publisher_unresolved": 0,
        "corroborations_recorded": 0,
        "events_inserted": 0,
        "full_text_attempted": 0,
        "full_text_fetched": 0,
        "urls_resolved": 0,
        "publish_dates_verified": 0,
        "republished_filtered": 0,
        "unverified_aggregator_inserts": 0,
        # Subset of the line above: aggregator items whose page we never managed to
        # read. They keep a verified-by-default date because a failed request says
        # nothing about the publisher — this counter is how the failure rate stays
        # visible instead of hiding inside the exposure number.
        "article_fetch_failed": 0,
        # Sizing telemetry for parallelising article_fetch. In-run inserts are
        # PREPENDED to recent_events (see the insert branch), so they are compared
        # first and win the match. Any design that defers an insert while its fetch
        # is in flight hides those entries from the next K candidates, which would
        # silently change dedup and corroboration outcomes. These counters say how
        # wide that exposure actually is: how many duplicate hits matched an event
        # this run inserted, and how recent that event was.
        # Duplicates whose source was NOT credited to the survivor: a self-republish,
        # a carrier, or a syndicated filing under a second masthead. Counted because
        # every one of these used to inflate the "Çoklu kaynak" label and the
        # corroboration ALERT floor, and a silent refusal is indistinguishable from
        # a run where nothing was refused.
        "corroborations_refused": 0,
        # How often a candidate duplicated an item still in flight and the window
        # had to land before the question could be answered. This is the price of
        # the parallel fetch being exact rather than approximate; a number that
        # climbs toward the fetch count means the window is buying nothing.
        "dedup_window_stalls": 0,
        # Why the fetch window was drained. The first parallel run cut article_fetch
        # from 145s to 88s but recorded ZERO stalls against 21 in-run duplicate
        # matches, which means the window was usually empty when those arrived —
        # it is being drained more often than correctness requires, and the
        # remaining time is sitting behind whichever of these dominates.
        "window_drain_full": 0,
        "window_drain_same_domain": 0,
        "window_drain_body_grows": 0,
        "window_drain_no_fetch": 0,
        "window_drain_stall": 0,
        "content_dup_matched_in_run": 0,
        "content_dup_in_run_within_4": 0,
        "content_dup_in_run_within_8": 0,
        "content_dup_in_run_within_16": 0,
    }
    now_utc = datetime.now(timezone.utc)
    timings: dict[str, float] = {}
    pending_corroborations: list[tuple] = []
    pass_started = time.perf_counter()
    reset_translation_counter()

    with _timed(timings, "build_queries"):
        queries = build_search_queries(db_conn)
    all_items = []

    # Execute up to 50 queries per run. Active storyline queries always run first;
    # the remaining slots rotate through the static tiers by hour-of-day so that
    # every tier gets coverage across runs (a fixed [:50] slice permanently
    # starved tiers beyond the first ~50 queries).
    MAX_QUERIES_PER_RUN = 50
    dynamic_count = sum(1 for q in queries if q.get("dynamic"))
    static_queries = queries[dynamic_count:]
    selected_queries = queries[:dynamic_count]
    remaining_slots = max(0, MAX_QUERIES_PER_RUN - len(selected_queries))
    if static_queries and remaining_slots:
        offset = (datetime.now(timezone.utc).hour * remaining_slots) % len(static_queries)
        rotated = static_queries[offset:] + static_queries[:offset]
        selected_queries.extend(rotated[:remaining_slots])

    for query_info in selected_queries:
        with _timed(timings, "fetch_query_feeds"):
            items = fetch_rss_feed(query_info, is_direct_url=False, stats=stats)
        all_items.extend(items)
        stats["queries_executed"] += 1

    # Configured feeds, keyword-gated. Two lists, same handling: publisher_feeds
    # are publishers' own RSS, news_queries are standing Google News searches.
    # They are kept apart because their yield profiles differ by roughly 2x per
    # source, so they are worth measuring — and tuning — separately.
    configured = (SETTINGS.get("sources", {}).get("publisher_feeds", [])
                  + SETTINGS.get("sources", {}).get("news_queries", []))
    for feed_url in configured:
        with _timed(timings, "fetch_configured_feeds"):
            items = fetch_rss_feed(feed_url, is_direct_url=True, stats=stats)
        # Apply keyword filter: only keep items matching security keywords
        filtered_items = []
        for it in items:
            if _matches_security_keywords(it.get("title", ""), it.get("description", "")):
                filtered_items.append(it)
            else:
                if stats is not None:
                    stats["noise_filtered"] = stats.get("noise_filtered", 0) + 1
        all_items.extend(filtered_items)
        stats["queries_executed"] += 1

    # Fetch official travel advisories (US State Dept + UK FCDO) — Level 3-4 / "do not travel"
    try:
        with _timed(timings, "fetch_advisories"):
            advisory_items = fetch_travel_advisories(stats=stats)
        all_items.extend(advisory_items)
        if advisory_items:
            stats["queries_executed"] += 1
    except Exception:
        logger.warning("Travel advisory fetch skipped due to errors")

    stats["items_fetched"] = len(all_items)
    logger.info("Pass A: Fetched %d items from %d sources/queries", len(all_items), stats["queries_executed"])

    # Run-level URL dedup — same URL may appear from multiple queries
    seen_urls = set()
    deduped_items = []
    for item in all_items:
        url = item.get("link", "")
        if not url:
            continue
        norm_url = url.strip().lower().split("?")[0].split("#")[0]
        if norm_url in seen_urls:
            continue
        seen_urls.add(norm_url)
        deduped_items.append(item)

    # Diversity-aware ordering: round-robin across source domains instead of a
    # global newest-first sort, so one loud story can't monopolize max_events.
    deduped_items = _interleave_by_domain(deduped_items)

    # Fetch recent events for comparison once (texts and id/domain meta are
    # index-aligned; in-run inserts are prepended to both)
    with _timed(timings, "load_dedup_corpus"):
        recent_events, recent_meta = _fetch_recent_events_for_dedup(db_conn)

    # One read for the whole run instead of one per candidate (see
    # load_domain_penalties). Timed under the same key as the loop lookups it
    # replaces, so the phase stays comparable across runs.
    with _timed(timings, "domain_penalty_db"):
        domain_penalties = load_domain_penalties(db_conn)

    inserted = 0
    domain_inserts: dict[str, int] = {}
    # Fetched-but-not-yet-inserted items, in loop order. Drained by settle_pending().
    pending_inserts: list[dict] = []
    executor = ThreadPoolExecutor(max_workers=_ARTICLE_FETCH_WORKERS,
                                  thread_name_prefix="article-fetch")
    fetch_deadline = time.monotonic() + _ARTICLE_FETCH_DEADLINE_S
    # Triage-quality telemetry: what priorities made it in vs. got cut. A high
    # priority_dropped_max means the budget/caps are cutting into items the
    # scorer considers important — the signal to revisit cap sizes.
    inserted_priorities: list[int] = []
    dropped_priority_max = 0

    def _finalize_item(item, article, url, url_hash, domain, canonical, raw_text,
                       pub_dt, from_aggregator, date_verified) -> bool:
        """Everything the fetch's answer decides, plus the insert. Returns inserted.

        Lifted out of the loop verbatim so the sequential path and the windowed
        path cannot drift: there is exactly one copy of the reprint rule, the
        date-provenance rule and the insert. ``article`` is None when this item
        never earned a fetch, which is the pre-existing behaviour for a run that
        has spent its fetch budget.
        """
        nonlocal inserted
        if article is not None:
            # Prefer the publisher's URL: it is what a reader should be handed in
            # a report, and it makes source_url_hash stable — the same story
            # reached through two Google News queries carries two different
            # opaque handles and used to survive as two events.
            if article["url"] and article["url"] != url:
                stats["urls_resolved"] += 1
                url = article["url"]
                url_hash = compute_url_hash(url)

            page_dt = article["published_at"] if _VERIFY_PUBLISH_DATE else None
            # A page dated in the future is a broken CMS, not evidence — ignore
            # it rather than letting it override a sane feed date.
            if page_dt and page_dt <= now_utc + timedelta(days=_MAX_EVENT_FUTURE_DAYS):
                age_days = (now_utc - page_dt).total_seconds() / 86400
                if age_days > _MAX_ARTICLE_AGE_DAYS:
                    stats["republished_filtered"] += 1
                    logger.info(
                        "Reprint dropped: page says %s, feed claimed %s — %s | %s",
                        page_dt.date(), pub_dt.date() if pub_dt else "?",
                        domain, item.get("title", "")[:70],
                    )
                    return False
                stats["publish_dates_verified"] += 1
                pub_dt = page_dt
            elif from_aggregator:
                # Publisher blocked the fetch and left no date in the path, so
                # this row keeps Google's crawl stamp. Counted, not dropped:
                # dropping every unverifiable aggregator item would cost far more
                # real coverage than the reprints it would catch. This number is
                # the size of the remaining exposure — watch it.
                stats["unverified_aggregator_inserts"] += 1
                if article["fetch_ok"]:
                    # We DID read the page and it named no date — the one case where
                    # Google's stamp stands alone and the freshness gates must not
                    # treat it as evidence.
                    date_verified = False
                else:
                    # Never read it. Split out from the exposure count above because
                    # the two need different fixes: a dateless publisher is permanent,
                    # a failed fetch is a retry away, and only the rate of the second
                    # tells us whether the fetch layer is degrading.
                    stats["article_fetch_failed"] += 1

            full_text = article["text"]
            if full_text:
                stats["full_text_fetched"] += 1
                # canonical_text is what Pass C shows the classifier (truncated to
                # BATCH_TEXT_CHARS), so the body lands in front of the LLM — an RSS
                # description alone stops at the headline and never says which route
                # or until when. Still priority-gated: every insert now fetches an
                # article, and attaching every body would inflate Pass C's token
                # bill well past what the LLM quotas allow.
                if (_FETCH_FULL_TEXT
                        and item.get("_priority", 0) >= _FULL_TEXT_MIN_PRIORITY):
                    canonical = canonicalize_text(f"{canonical} {full_text}")

        # Idempotent insert — NOT EXISTS guard, wrapped in savepoint
        try:
            with _timed(timings, "insert_db"), db_conn.transaction():
                result = db_conn.execute(
                    """INSERT INTO events (source_url, source_url_hash, source_domain,
                                           source_title, raw_text, canonical_text, status,
                                           published_at, date_verified)
                       SELECT %s, %s, %s, %s, %s, %s, 'raw', %s, %s
                       WHERE NOT EXISTS (
                           SELECT 1 FROM events WHERE source_url_hash = %s
                       )
                       RETURNING id""",
                    (url, url_hash, domain, item.get("title", ""),
                     raw_text, canonical, pub_dt, date_verified, url_hash),
                )
                new_row = result.fetchone()
                if new_row:
                    inserted += 1
                    stats["events_inserted"] += 1
                    inserted_priorities.append(item.get("_priority", 0))
                    domain_inserts[domain] = domain_inserts.get(domain, 0) + 1
                    # Inline dedup: add to recent_events (and aligned meta) so later
                    # items in this run are compared — and corroborated — against it
                    # No anchor yet — Pass C classifies this event later in the run.
                    recent_events.insert(0, (item.get("title", ""), canonical, ""))
                    recent_meta.insert(0, (new_row[0], domain))
                    if len(recent_events) > 2500:
                        recent_events.pop()
                        recent_meta.pop()
                    return True
                stats["duplicates_skipped"] += 1
        except Exception:
            logger.exception("Insert error for URL: %s", url[:80])
        return False

    def settle_pending() -> None:
        """Complete every in-flight insert, in the order it was submitted.

        Order matters and is not cosmetic: recent_events is prepended to and
        find_content_duplicate returns the FIRST match, so the sequence in which
        entries land decides which event a later duplicate corroborates. Draining
        in submission order reproduces exactly what the sequential loop produced.
        """
        if not pending_inserts:
            return
        batch, pending_inserts[:] = list(pending_inserts), []
        with _timed(timings, "article_fetch"):
            for entry in batch:
                try:
                    entry["article"] = entry["future"].result(
                        timeout=_ARTICLE_FETCH_DEADLINE_S)
                except Exception:
                    # A fetch that never answered says nothing about the publisher,
                    # so the item is inserted on what the feed gave us — the same
                    # outcome the sequential path produced for a failed fetch.
                    logger.debug("Article fetch failed in pool for %s",
                                 entry["url"][:80])
                    entry["article"] = {"url": "", "text": "",
                                        "published_at": None, "fetch_ok": False}
        for entry in batch:
            _finalize_item(entry["item"], entry["article"], entry["url"],
                           entry["url_hash"], entry["domain"], entry["canonical"],
                           entry["raw_text"], entry["pub_dt"],
                           entry["from_aggregator"], entry["date_verified"])

    for item_idx, item in enumerate(deduped_items):
        # In-flight items are inserts that have not landed yet, so the budget has
        # to count them or the window would overshoot max_events.
        if inserted + len(pending_inserts) >= max_events:
            leftover = deduped_items[item_idx:]
            if leftover:
                dropped_priority_max = max(
                    dropped_priority_max,
                    max(it.get("_priority", 0) for it in leftover),
                )
            break

        url = item.get("link", "")
        if not url:
            continue

        # Auto-translate title and description if needed
        title = item.get("title", "")
        description = item.get("description", "")
        with _timed(timings, "translate"):
            if title:
                item["title"] = translate_to_english_if_needed(title)
            if description:
                item["description"] = translate_to_english_if_needed(description)

        # Canonicalize
        raw_text = f"{item.get('title', '')} {item.get('description', '')}"
        canonical = canonicalize_text(raw_text)

        # Noise filter — skip for official travel advisory items
        with _timed(timings, "noise_filter"):
            noisy = not item.get("_skip_noise_filter") and is_noise(canonical)
        if noisy:
            stats["noise_filtered"] += 1
            continue

        # URL hash for dedup
        url_hash = compute_url_hash(url)

        # Domain extraction and penalty check
        # For travel advisory items, preserve the real domain
        if item.get("source") == "travel_advisory":
            domain = "travel.state.gov"
        elif item.get("domain"):
            domain = item["domain"]
        else:
            domain = extract_domain(url)
        # A social platform is a carrier, not a publisher: Google News files a
        # publisher's own post under the platform's domain, collapsing every outlet
        # that posts there into one identity for corroboration, verification labels
        # and penalties. Recover the publisher from the page slug where we know it.
        # Google News links hide the slug behind a redirect, so those are resolved
        # first — only for social items, which run about five a day.
        if is_social_platform(domain):
            social_url = url
            if "news.google.com" in social_url:
                from src.services.google_news_resolver import resolve_url
                social_url = resolve_url(social_url) or social_url
            publisher = social_publisher_domain(social_url)
            if publisher:
                stats["social_publisher_resolved"] += 1
                logger.debug("Social post attributed to %s (was %s)", publisher, domain)
                domain = publisher
            else:
                # Unrecognised page: keep the platform domain, which is what this
                # item had before. Never invent an identity the corroboration count
                # would then treat as an independent outlet.
                stats["social_publisher_unresolved"] += 1

        # Scraped-content farms, rejected before any scoring can see them. Placed
        # after domain extraction because the check reads the domain, and before the
        # penalty gate because penalty_score is earned over time while this content
        # is never admissible at all.
        if is_content_farm(item.get("title"), url, domain):
            stats["content_farm_filtered"] += 1
            logger.info("Content farm rejected: %s | %.80s", domain, item.get("title") or "")
            continue

        with _timed(timings, "domain_penalty_db"):
            penalty = check_domain_penalty(db_conn, domain, domain_penalties)
        if penalty > 0.8:
            stats["domain_penalized"] += 1
            continue

        # Per-domain insert cap — hard ceiling on how much of the run budget a
        # single outlet can claim, on top of the round-robin ordering.
        domain_cap = _PER_DOMAIN_CAPS.get(domain, _MAX_EVENTS_PER_DOMAIN)
        # The cap counts inserts, and an in-flight item of this domain is an insert
        # the counter cannot see yet. Settling is exact where adding a provisional
        # +1 would not be: a fetch can still end in a reprint drop, which consumes
        # no slot. Cheap because _interleave_by_domain makes a same-domain
        # collision inside one window rare (domain_capped was 0 in the last run).
        if any(pend["domain"] == domain for pend in pending_inserts):
            stats["window_drain_same_domain"] += 1
            settle_pending()
        if domain_inserts.get(domain, 0) >= domain_cap:
            stats["domain_capped"] += 1
            dropped_priority_max = max(dropped_priority_max, item.get("_priority", 0))
            continue

        # Content dedup: a similar article already exists → don't re-insert, but
        # credit its source to the surviving event as corroboration (the dropped
        # duplicate IS the multi-source verification evidence).
        with _timed(timings, "content_dedup_cpu"):
            dup_idx = find_content_duplicate(recent_events, item.get("title", ""), canonical)
            if dup_idx is None and _pending_matches(pending_inserts,
                                                    item.get("title", ""), canonical):
                # This candidate duplicates something still in flight. Its event id
                # does not exist yet, and the fetch could still drop that item as a
                # reprint, so guessing either way would be wrong — land the window
                # and ask the real corpus.
                stats["dedup_window_stalls"] += 1
                stats["window_drain_stall"] += 1
                settle_pending()
                dup_idx = find_content_duplicate(recent_events,
                                                 item.get("title", ""), canonical)
        if dup_idx is not None:
            stats["content_duplicates_skipped"] += 1
            # dup_idx counts back from the head, and this run prepended exactly
            # `inserted` entries, so dup_idx < inserted means the match was made
            # against an event inserted earlier in THIS run — and dup_idx is then
            # how many inserts have happened since that one.
            if dup_idx < inserted:
                stats["content_dup_matched_in_run"] += 1
                for window in (4, 8, 16):
                    if dup_idx < window:
                        stats[f"content_dup_in_run_within_{window}"] += 1
            dup_event_id, dup_event_domain = recent_meta[dup_idx]
            # CPU-only here; the write is deferred to one pipelined flush after the
            # loop (see _flush_corroborations). The refusals still run per item, so
            # a carrier, a self-republish or a syndicated filing never reaches the
            # batch at all. recent_events[dup_idx][0] is the surviving event's own
            # headline — the side _is_syndicated_filing compares against.
            with _timed(timings, "corroboration_cpu"):
                params = _corroboration_params(dup_event_id, dup_event_domain,
                                               domain, url, item.get("title", ""),
                                               recent_events[dup_idx][0])
            if params is not None:
                pending_corroborations.append(params)
            else:
                stats["corroborations_refused"] += 1
            continue

        # Get published_at date
        pub_dt = item.get("pub_dt")
        # Only aggregator links carry the restamping risk: a publisher's own feed
        # reports its own CMS date, while Google News reports when IT last crawled the
        # page. Read BEFORE the fetch below, which rewrites `url` to the resolved
        # publisher address and would erase the evidence of where the item came from.
        from_aggregator = "news.google.com" in url
        # Provenance of pub_dt (migration 021). FALSE is a POSITIVE finding — "we read
        # the publisher's page and it declares no date of its own" — not the absence of
        # one. So it starts TRUE and is only cleared below, after a successful fetch
        # comes back without a date: an item we never fetched, or whose fetch failed,
        # has told us nothing about its publisher and must not be penalised for it
        # (measured 2026-08-12, a live Libya Observer report lost its page to a
        # transient fetch failure while its page declared the date all along).
        date_verified = True

        # Fetch the article itself: resolves the Google News handle to the
        # publisher's URL, and returns the page's own date and body in ONE round
        # trip. Deliberately placed here, after every cheap filter has run, so
        # the per-run network cost is bounded by how many events we can actually
        # insert rather than by how many items the feeds returned.
        #
        # The fetch is submitted to a pool and the loop moves on; the insert is
        # completed later by _settle_pending. Everything that could make an
        # in-flight insert visible to a later decision drains the window first —
        # see the note above _pending_matches.
        # The deadline retires the fetch rather than serialising it. Past the bound
        # the item inserts on what the feed gave us — the same, already-exercised
        # path a run takes once _ARTICLE_FETCH_MAX_PER_RUN is spent. Falling back to
        # sequential fetching instead would keep paying the cost the bound exists
        # to stop.
        wants_fetch = ((_VERIFY_PUBLISH_DATE or _FETCH_FULL_TEXT)
                       and stats["full_text_attempted"] < _ARTICLE_FETCH_MAX_PER_RUN
                       and time.monotonic() < fetch_deadline)
        # An item whose canonical_text will grow by its article body cannot go in the
        # window: its corpus entry would be missing the very text a later candidate
        # might match on, and unlike the ordering questions that is not something
        # draining later can repair.
        body_will_grow = (_FETCH_FULL_TEXT
                          and item.get("_priority", 0) >= _FULL_TEXT_MIN_PRIORITY)

        if not wants_fetch or body_will_grow:
            # Sequential path. Draining first keeps corpus order identical to the
            # loop this replaced.
            if pending_inserts:
                stats["window_drain_body_grows" if body_will_grow
                      else "window_drain_no_fetch"] += 1
            settle_pending()
            article = None
            if wants_fetch:
                stats["full_text_attempted"] += 1
                with _timed(timings, "article_fetch"):
                    article = fetch_article(url)
            _finalize_item(item, article, url, url_hash, domain, canonical,
                           raw_text, pub_dt, from_aggregator, date_verified)
            continue

        stats["full_text_attempted"] += 1
        pending_inserts.append({
            "future": executor.submit(fetch_article, url),
            "item": item, "url": url, "url_hash": url_hash, "domain": domain,
            "canonical": canonical, "raw_text": raw_text, "pub_dt": pub_dt,
            "from_aggregator": from_aggregator, "date_verified": date_verified,
            "title": item.get("title", ""),
        })
        if len(pending_inserts) >= _ARTICLE_FETCH_WINDOW:
            stats["window_drain_full"] += 1
            settle_pending()
        continue


    # Nothing may outlive the loop: the last window still holds real inserts.
    settle_pending()
    executor.shutdown(wait=True)

    with _timed(timings, "corroboration_db"):
        stats["corroborations_recorded"] = _flush_corroborations(
            db_conn, pending_corroborations)

    if inserted_priorities:
        import statistics
        stats["priority_inserted_max"] = max(inserted_priorities)
        stats["priority_inserted_median"] = int(statistics.median(inserted_priorities))
    stats["priority_dropped_max"] = dropped_priority_max

    # Phase timings, seconds. Reported as a nested block so the existing counters
    # keep their shape, and rounded because nothing here is worth sub-ms precision.
    # 'unaccounted' is deliberate: it is the part of Pass A no timer covers yet, and
    # a large value means the next place to look is not on this list.
    elapsed = time.perf_counter() - pass_started
    measured = sum(timings.values())
    block = {k: round(v, 1) for k, v in sorted(timings.items())}
    block["translations_performed"] = translation_call_count()
    block["translations_failed"] = translation_failure_count()
    block["measured_total"] = round(measured, 1)
    block["pass_a_total"] = round(elapsed, 1)
    block["unaccounted"] = round(elapsed - measured, 1)
    stats["timings_sec"] = block

    # Log telemetry
    try:
        db_conn.execute(
            "INSERT INTO system_telemetry(event_type, value_json) VALUES ('pass_a', %s)",
            (json.dumps(stats),),
        )
        db_conn.commit()
    except Exception:
        logger.exception("Failed to log Pass A telemetry")

    logger.info("Pass A complete: %s", stats)
    return stats

