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
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Re-exported: historical import surface of this module (consumers: orchestrator,
# pass_c_classify, tests). Keep these names importable from pass_a_ingest.
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
    translate_to_english_if_needed,
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

def _fetch_recent_events_for_dedup(db_conn) -> tuple[list[tuple[str, str]], list[tuple]]:
    """Fetch recent events once to avoid O(N) database queries during ingestion.

    Returns (texts, meta) as two INDEX-ALIGNED lists: texts feeds the similarity
    matcher (title, canonical_text); meta carries (event_id, source_domain) so a
    detected duplicate can be credited back to the surviving event as
    corroboration.
    """
    try:
        rows = db_conn.execute(
            """SELECT id, source_domain, source_title, canonical_text
               FROM events
               WHERE ingested_at > NOW() - (%s * INTERVAL '1 day')
               ORDER BY ingested_at DESC
               LIMIT 2000""",
            (_MAX_ARTICLE_AGE_DAYS,),
        ).fetchall()
        texts = [(row[2] or "", row[3] or "") for row in rows]
        meta = [(row[0], row[1] or "") for row in rows]
        return texts, meta
    except Exception:
        logger.exception("Failed to fetch recent events for dedup")
        return [], []


# Max corroborating sources kept per event — enough for a Çoklu Kaynak/Resmî
# upgrade; beyond that more entries add bytes, not information.
_MAX_CORROBORATING_SOURCES = 5


def _record_corroboration(db_conn, event_id, event_domain: str,
                          dup_domain: str, dup_url: str, dup_title: str) -> bool:
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
    entry = json.dumps([{"domain": dup_domain, "url": dup_url[:500],
                         "title": (dup_title or "")[:200],
                         "seen_at": datetime.now(timezone.utc).isoformat()}])
    probe = json.dumps([{"domain": dup_domain}])
    try:
        with db_conn.transaction():
            result = db_conn.execute(
                """UPDATE events
                   SET corroborating_sources = corroborating_sources || %s::jsonb
                   WHERE id = %s
                     AND jsonb_array_length(corroborating_sources) < %s
                     AND NOT corroborating_sources @> %s::jsonb""",
                (entry, event_id, _MAX_CORROBORATING_SOURCES, probe),
            )
            return result.rowcount > 0
    except Exception:
        # Pre-migration DBs lack the column — corroboration is a bonus signal,
        # never worth failing an ingest run over.
        logger.debug("Corroboration record failed for event %s", event_id)
        return False


def check_domain_penalty(db_conn, domain: str) -> float:
    """Get penalty score for a domain. Returns 0.0 if not found, if total_events < 5, or if whitelisted."""
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
    }
    now_utc = datetime.now(timezone.utc)

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
    recent_events, recent_meta = _fetch_recent_events_for_dedup(db_conn)

    inserted = 0
    domain_inserts: dict[str, int] = {}
    # Triage-quality telemetry: what priorities made it in vs. got cut. A high
    # priority_dropped_max means the budget/caps are cutting into items the
    # scorer considers important — the signal to revisit cap sizes.
    inserted_priorities: list[int] = []
    dropped_priority_max = 0
    for item_idx, item in enumerate(deduped_items):
        if inserted >= max_events:
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
        if title:
            item["title"] = translate_to_english_if_needed(title)
        if description:
            item["description"] = translate_to_english_if_needed(description)

        # Canonicalize
        raw_text = f"{item.get('title', '')} {item.get('description', '')}"
        canonical = canonicalize_text(raw_text)

        # Noise filter — skip for official travel advisory items
        if not item.get("_skip_noise_filter") and is_noise(canonical):
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
        # Scraped-content farms, rejected before any scoring can see them. Placed
        # after domain extraction because the check reads the domain, and before the
        # penalty gate because penalty_score is earned over time while this content
        # is never admissible at all.
        if is_content_farm(item.get("title"), url, domain):
            stats["content_farm_filtered"] += 1
            logger.info("Content farm rejected: %s | %.80s", domain, item.get("title") or "")
            continue

        penalty = check_domain_penalty(db_conn, domain)
        if penalty > 0.8:
            stats["domain_penalized"] += 1
            continue

        # Per-domain insert cap — hard ceiling on how much of the run budget a
        # single outlet can claim, on top of the round-robin ordering.
        domain_cap = _PER_DOMAIN_CAPS.get(domain, _MAX_EVENTS_PER_DOMAIN)
        if domain_inserts.get(domain, 0) >= domain_cap:
            stats["domain_capped"] += 1
            dropped_priority_max = max(dropped_priority_max, item.get("_priority", 0))
            continue

        # Content dedup: a similar article already exists → don't re-insert, but
        # credit its source to the surviving event as corroboration (the dropped
        # duplicate IS the multi-source verification evidence).
        dup_idx = find_content_duplicate(recent_events, item.get("title", ""), canonical)
        if dup_idx is not None:
            stats["content_duplicates_skipped"] += 1
            dup_event_id, dup_event_domain = recent_meta[dup_idx]
            if _record_corroboration(db_conn, dup_event_id, dup_event_domain,
                                     domain, url, item.get("title", "")):
                stats["corroborations_recorded"] += 1
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
        if (_VERIFY_PUBLISH_DATE or _FETCH_FULL_TEXT) and \
                stats["full_text_attempted"] < _ARTICLE_FETCH_MAX_PER_RUN:
            stats["full_text_attempted"] += 1
            article = fetch_article(url)

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
                    continue
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
            with db_conn.transaction():
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
                    recent_events.insert(0, (item.get("title", ""), canonical))
                    recent_meta.insert(0, (new_row[0], domain))
                    if len(recent_events) > 2500:
                        recent_events.pop()
                        recent_meta.pop()
                else:
                    stats["duplicates_skipped"] += 1
        except Exception:
            logger.exception("Insert error for URL: %s", url[:80])
            continue


    if inserted_priorities:
        import statistics
        stats["priority_inserted_max"] = max(inserted_priorities)
        stats["priority_inserted_median"] = int(statistics.median(inserted_priorities))
    stats["priority_dropped_max"] = dropped_priority_max

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

