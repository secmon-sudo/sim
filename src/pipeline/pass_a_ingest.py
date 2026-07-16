"""
SIM — Pass A: Ingest & Canonicalization
Blueprint V20.1 §4 PASS A

Orchestrates ingest: builds the query set, fans out to the source fetchers,
applies noise/age/dedup filtering and inserts raw events. The heavy lifting
lives in focused modules (split from this 1.9K-line monolith on 2026-07-16):

  - ingest_queries   — search-query construction (static tiers + storyline queries)
  - ingest_sources   — all network I/O (RSS, Nitter, advisories, GDELT, translate)
  - ingest_filters   — pure text filters, canonicalization, similarity dedup

This module keeps only the DB-touching pieces and run_pass_a itself. The
re-exports below preserve the historical import surface of pass_a_ingest.
"""

import json
import logging
from datetime import datetime, timezone
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
    is_noise,
    normalize_title,
    title_similarity,
    title_token_similarity,
)
from src.pipeline.ingest_queries import (  # noqa: F401
    MAX_DYNAMIC_QUERIES,
    build_gdelt_queries,
    build_search_queries,
)
from src.pipeline.ingest_sources import (  # noqa: F401
    GOOGLE_NEWS_RSS,
    fetch_full_text,
    fetch_gdelt_articles,
    fetch_nitter_feeds,
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
_MAX_EVENTS_PER_DOMAIN = _INGESTION.get("max_events_per_domain", 8)
_GDELT_ENABLED = SETTINGS.get("sources", {}).get("gdelt_enabled", False)

# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def _fetch_recent_events_for_dedup(db_conn) -> list[tuple[str, str]]:
    """Fetch recent events once to avoid O(N) database queries during ingestion."""
    try:
        rows = db_conn.execute(
            """SELECT source_title, canonical_text
               FROM events
               WHERE ingested_at > NOW() - (%s * INTERVAL '1 day')
               ORDER BY ingested_at DESC
               LIMIT 2000""",
            (_MAX_ARTICLE_AGE_DAYS,),
        ).fetchall()
        return [(row[0] or "", row[1] or "") for row in rows]
    except Exception:
        logger.exception("Failed to fetch recent events for dedup")
        return []


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

def _interleave_by_domain(items: list[dict]) -> list[dict]:
    """
    Round-robin items across source domains, newest first within each domain.

    A plain newest-first fill let whichever story dominated the global news
    cycle (and got reprinted by every outlet) eat the entire per-run insert
    budget, crowding out quieter regions. Interleaving guarantees every domain
    that delivered items gets a first slot before any domain gets a second.
    """
    _EPOCH_MIN = datetime.min.replace(tzinfo=timezone.utc)
    buckets: dict[str, list[dict]] = {}
    for item in items:
        domain = item.get("domain") or extract_domain(item.get("link", ""))
        buckets.setdefault(domain, []).append(item)

    for bucket in buckets.values():
        bucket.sort(key=lambda x: x.get("pub_dt") or _EPOCH_MIN, reverse=True)

    # Domains with the freshest lead item go first within each round
    ordered = sorted(
        buckets.values(),
        key=lambda b: b[0].get("pub_dt") or _EPOCH_MIN,
        reverse=True,
    )
    interleaved = []
    depth = 0
    while True:
        row = [b[depth] for b in ordered if depth < len(b)]
        if not row:
            return interleaved
        interleaved.extend(row)
        depth += 1


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
        "duplicates_skipped": 0,
        "content_duplicates_skipped": 0,
        "domain_penalized": 0,
        "domain_capped": 0,
        "events_inserted": 0,
        "full_text_fetched": 0,
    }

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

    # Fetch from static hardcoded feeds with keyword post-filter
    static_feeds = SETTINGS.get("sources", {}).get("static_feeds", [])
    for feed_url in static_feeds:
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

    # Fetch from Nitter (Twitter/X) feeds — with mirror fallback & 3 retries
    try:
        nitter_items = fetch_nitter_feeds(stats=stats)
        all_items.extend(nitter_items)
        nitter_count = len(SETTINGS.get("sources", {}).get("nitter_feeds", []))
        stats["queries_executed"] += nitter_count
        if nitter_items:
            logger.info("Nitter: Total %d items from %d accounts", len(nitter_items), nitter_count)
    except Exception:
        logger.warning("Nitter fetch skipped due to errors")

    # Fetch from GDELT — disabled by default (sources.gdelt_enabled): constant
    # 429s/errors on cloud IPs made it noise, never signal.
    if _GDELT_ENABLED:
        try:
            gdelt_queries = build_gdelt_queries()
            # Pick 1 query per run to minimize rate-limit exposure. Rotate by hour so
            # all regions get coverage over a day (seeding the GLOBAL random module by
            # minute both polluted other random users and biased the selection).
            selected = gdelt_queries[datetime.now(timezone.utc).hour % len(gdelt_queries)]

            items = fetch_gdelt_articles(
                query=selected["query"],
                max_age_days=_MAX_ARTICLE_AGE_DAYS,
                tone=selected.get("tone"),
                source_countries=selected.get("countries"),
            )
            if items:
                all_items.extend(items)
                stats["queries_executed"] += 1
                logger.info("GDELT: Got %d articles (bonus)", len(items))
        except Exception:
            # GDELT failure is expected on cloud IPs — never block pipeline
            pass

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

    # Fetch recent events for comparison once
    recent_events = _fetch_recent_events_for_dedup(db_conn)

    inserted = 0
    domain_inserts: dict[str, int] = {}
    for item in deduped_items:
        if inserted >= max_events:
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
        penalty = check_domain_penalty(db_conn, domain)
        if penalty > 0.8:
            stats["domain_penalized"] += 1
            continue

        # Per-domain insert cap — hard ceiling on how much of the run budget a
        # single outlet can claim, on top of the round-robin ordering.
        if domain_inserts.get(domain, 0) >= _MAX_EVENTS_PER_DOMAIN:
            stats["domain_capped"] += 1
            continue

        # Optional: fetch full text
        full_text = ""
        if _FETCH_FULL_TEXT:
            full_text = fetch_full_text(url)
            if full_text:
                stats["full_text_fetched"] += 1
                canonical = canonicalize_text(f"{canonical} {full_text}")

        # Content dedup: check if similar title/text already exists
        if check_content_duplicate(recent_events, item.get("title", ""), canonical):
            stats["content_duplicates_skipped"] += 1
            continue

        # Get published_at date
        pub_dt = item.get("pub_dt")

        # Idempotent insert — NOT EXISTS guard, wrapped in savepoint
        try:
            with db_conn.transaction():
                result = db_conn.execute(
                    """INSERT INTO events (source_url, source_url_hash, source_domain,
                                           source_title, raw_text, canonical_text, status, published_at)
                       SELECT %s, %s, %s, %s, %s, %s, 'raw', %s
                       WHERE NOT EXISTS (
                           SELECT 1 FROM events WHERE source_url_hash = %s
                       )""",
                    (url, url_hash, domain, item.get("title", ""),
                     raw_text, canonical, pub_dt, url_hash),
                )
                if result.rowcount > 0:
                    inserted += 1
                    stats["events_inserted"] += 1
                    domain_inserts[domain] = domain_inserts.get(domain, 0) + 1
                    # Inline dedup: add to recent_events so later items in this run are compared against it
                    recent_events.insert(0, (item.get("title", ""), canonical))
                    if len(recent_events) > 2500:
                        recent_events.pop()
                else:
                    stats["duplicates_skipped"] += 1
        except Exception:
            logger.exception("Insert error for URL: %s", url[:80])
            continue


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

