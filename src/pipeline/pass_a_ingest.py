"""
SIM — Pass A: Ingest & Canonicalization
Blueprint V20.1 §4 PASS A

Collects aviation incident articles from multi-region Google News RSS,
normalizes text, filters noise, applies age filter, content dedup,
optionally fetches full text, and inserts raw events.
"""

import difflib
import email.utils
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import tldextract

logger = logging.getLogger(__name__)

# Load configuration
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
with open(_CONFIG_DIR / "keywords.json", encoding="utf-8") as f:
    KEYWORDS_CONFIG = json.load(f)
with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as f:
    SETTINGS = json.load(f)

# Settings lookups
_INGESTION = SETTINGS.get("ingestion", {})
_DEDUP = SETTINGS.get("dedup", {})
_MAX_ARTICLE_AGE_DAYS = _INGESTION.get("max_article_age_days", 4)
_FETCH_FULL_TEXT = _INGESTION.get("fetch_full_text", True)
_GEO_REGIONS = _INGESTION.get("geo_regions", [
    {"gl": "US", "hl": "en", "ceid": "US:en"}
])
_CONTENT_SIM_THRESHOLD = _DEDUP.get("content_similarity_threshold", 0.82)
_TITLE_SIM_THRESHOLD = _DEDUP.get("title_similarity_threshold", 0.85)

# Prompt injection patterns to strip before LLM classification
PROMPT_INJECTION_PATTERNS = re.compile(
    r"\[INST\]|<\|system\|>|<\|user\|>|<\|assistant\|>|IGNORE PREVIOUS INSTRUCTIONS|"
    r"FORGET ALL PRIOR|YOU ARE NOW|SYSTEM OVERRIDE",
    re.IGNORECASE,
)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en"

# GDELT 2.0 Article List API
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_RECORDS = 25  # Per query to stay within API limits


def fetch_gdelt_articles(query: str, max_age_days: int = 3) -> list[dict]:
    """
    Fetch article URLs from GDELT 2.0 API for a given keyword query.
    GDELT returns global news with tone analysis and precise timestamps.

    Query syntax examples:
    - 'airport AND attack' (AND/OR/NOT supported)
    - 'theme:TAX_TERROR' (GDELT CAMEO themes)
    - 'sourcecountry:NG' (Nigerian sources only)
    - 'tone<-5' (highly negative tone)
    """
    now = datetime.now(timezone.utc)
    end_dt = now.strftime("%Y%m%d%H%M%S")
    start_dt = (now - timedelta(days=max_age_days)).strftime("%Y%m%d%H%M%S")

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": GDELT_MAX_RECORDS,
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "sort": "seendate",  # Most recent first
    }

    try:
        resp = httpx.get(GDELT_DOC_API, params=params, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("GDELT fetch failed for query: %s", query[:60])
        return []

    items = []
    for article in data.get("articles", []):
        seendate_str = article.get("seendate", "")
        title = article.get("title", "")
        url = article.get("url", "")
        domain = article.get("domain", "")

        if not url or not title:
            continue

        # Parse GDELT seendate (YYYYMMDDHHMMSS) → datetime
        pub_dt = None
        if seendate_str and len(seendate_str) >= 8:
            try:
                pub_dt = datetime.strptime(seendate_str[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except Exception:
                pass

        # Strict age filter — reject if no date or too old
        if pub_dt is None:
            continue
        age_days = (now - pub_dt).total_seconds() / 86400
        if age_days > max_age_days:
            continue

        items.append({
            "title": title,
            "link": url,
            "pub_date": seendate_str,
            "pub_dt": pub_dt,
            "description": "",  # GDELT ArtList does not provide descriptions
            "source": "gdelt",
            "domain": domain,
        })

    if items:
        logger.info("GDELT: %d articles for '%s...' (last %dh)", len(items), query[:40], max_age_days * 24)

    return items


def _compile_noise_patterns() -> list[re.Pattern]:
    """Compile noise filters with word boundaries to reduce false positives."""
    patterns = []
    for pattern in KEYWORDS_CONFIG.get("noise_filters", []):
        # Escape regex special chars and wrap with word boundaries
        escaped = re.escape(pattern)
        try:
            patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
        except re.error:
            # Fallback to plain substring if word-boundary fails
            patterns.append(re.compile(re.escape(pattern), re.IGNORECASE))
    return patterns


NOISE_PATTERNS = _compile_noise_patterns()


def build_search_queries() -> list[dict]:
    """Build search queries from keywords config — GLOBAL (no region restriction).

    Strategy:
    - Broad security keywords: search globally without aviation qualifier
    - Specific aviation keywords: search with 'airport OR aviation' qualifier
    - Geopolitical keywords: always broad
    """
    queries = []

    # Keywords that should be searched broadly (security incidents anywhere)
    broad_keywords = {
        "bomb threat", "active shooter", "hijack", "explosion",
        "security breach", "evacuation", "terrorism", "suspicious package",
        "hostage", "weapon", "airport attack", "airport shooting",
        "airport bombing", "hotel attack", "hotel bombing", "hotel shooting",
        "hotel siege", "resort attack", "airline crew attack",
    }

    # Build base queries — deduplicate to avoid redundant calls
    seen_queries = set()

    for lang, keywords in KEYWORDS_CONFIG.get("emergency_keywords", {}).items():
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in seen_queries:
                continue
            seen_queries.add(kw_lower)
            if any(bk in kw_lower for bk in broad_keywords):
                # Broad search — no aviation qualifier
                queries.append({"query": f'"{kw}"', "broad": True})
            else:
                # Narrow search — aviation context
                queries.append({"query": f'"{kw}" airport OR aviation', "broad": False})

    # Geopolitical keywords are always broad
    for lang, keywords in KEYWORDS_CONFIG.get("geopolitical_keywords", {}).items():
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in seen_queries:
                continue
            seen_queries.add(kw_lower)
            queries.append({"query": f'"{kw}"', "broad": True})

    return queries


def build_gdelt_queries() -> list[str]:
    """
    Build optimized GDELT query strings.
    GDELT supports AND/OR/NOT and phrase search.
    Each query targets a distinct threat category.
    """
    return [
        # Drone attacks on critical infrastructure
        '"drone attack" OR "UAV attack" OR "drone bombing" OR "drone strike"',
        # Mass casualty events
        '"mass shooting" OR "mass stabbing" OR "mass casualty" OR "massacre" OR "suicide bombing"',
        # Airport attacks
        '"airport attack" OR "airport bombing" OR "airport shooting" OR "airport terror"',
        # Hotel / resort / tourism attacks
        '"hotel attack" OR "hotel bombing" OR "resort attack" OR "beach attack" OR "cruise ship attack"',
        # Aviation personnel attacks
        '"pilot attacked" OR "cabin crew attacked" OR "ground staff attacked" OR "airline personnel attack"',
        # African terrorism
        '"Boko Haram" OR "Al-Shabaab" OR "jihadist attack" OR "ISIS Africa" OR "Sahel crisis"',
        # War escalation & civilian casualties
        '"missile strike" OR "airstrike" OR "war escalation" OR "ceasefire broken" OR "civilian casualties"',
        # General terrorism & security
        '"bomb threat" OR "explosion" OR "terrorism" OR "hostage" OR "active shooter"',
        # Aviation incidents
        '"hijack" OR "runway incursion" OR "emergency landing" OR "security breach"',
        # Geopolitical conflict (broader)
        '"military action" OR "invasion" OR "border clash" OR "troop buildup" OR "artillery shelling"',
    ]


def fetch_rss_feed(query_info: dict, is_direct_url: bool = False) -> list[dict]:
    """Fetch and parse an RSS feed. Returns items with parsed pub_date."""
    import xml.etree.ElementTree as ET

    if is_direct_url:
        url = query_info if isinstance(query_info, str) else query_info.get("url", "")
    else:
        # Global Google News search — no region lock
        url = GOOGLE_NEWS_RSS.format(query=query_info["query"])

    try:
        # Reddit requires a descriptive User-Agent
        headers = {
            "User-Agent": "SIM-OSINT-Bot/1.0 (Security Incident Monitor; contact@sim-osint.app)"
        }
        resp = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
    except Exception:
        logger.warning("RSS fetch failed for: %s", url[:80])
        return []

    items = []
    now_utc = datetime.now(timezone.utc)
    max_age = _MAX_ARTICLE_AGE_DAYS

    try:
        root = ET.fromstring(resp.text)
        for item in root.findall(".//item"):
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date_str = item.findtext("pubDate", "")
            description = item.findtext("description", "")

            # Parse pubDate and apply age filter
            pub_dt = None
            if pub_date_str:
                try:
                    pub_dt = email.utils.parsedate_to_datetime(pub_date_str)
                except Exception:
                    pass

            # STRICT: Reject items with missing or unparseable pubDate
            # This prevents old news without dates from entering the system
            if pub_dt is None:
                stats["age_filtered"] += 1
                continue

            age_days = (now_utc - pub_dt).total_seconds() / 86400
            if age_days > max_age:
                stats["age_filtered"] += 1
                continue  # Skip old articles

            items.append({
                "title": title,
                "link": link,
                "pub_date": pub_date_str,
                "pub_dt": pub_dt,
                "description": description,
            })
    except Exception:
        logger.exception("RSS parse error for: %s", url[:80])

    return items


def fetch_full_text(url: str) -> str:
    """Attempt to fetch full article text from URL using trafilatura."""
    try:
        import trafilatura
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        downloaded = trafilatura.fetch_url(url, headers=headers)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            return text or ""
    except ImportError:
        logger.debug("trafilatura not installed, skipping full-text fetch")
    except Exception:
        logger.warning("Full-text fetch failed for %s", url[:80])
    return ""


def extract_domain(url: str) -> str:
    """Extract eTLD+1 domain from URL."""
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def compute_url_hash(url: str) -> str:
    """SHA-256 hash of normalized URL for deduplication."""
    normalized = url.strip().lower().split("?")[0].split("#")[0]
    return hashlib.sha256(normalized.encode()).hexdigest()


def is_noise(text: str) -> bool:
    """Check if text matches known noise patterns using word boundaries."""
    text_lower = text.lower()
    for pattern in NOISE_PATTERNS:
        if pattern.search(text_lower):
            return True
    return False


def canonicalize_text(raw_text: str) -> str:
    """
    Clean and normalize raw article text.
    - Strips prompt injection patterns
    - Normalizes whitespace
    - Removes HTML tags
    """
    # Strip HTML tags (simple regex, acceptable for RSS snippets)
    text = re.sub(r"<[^>]+>", " ", raw_text)
    # Strip prompt injection patterns
    text = PROMPT_INJECTION_PATTERNS.sub("", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title(title: str) -> str:
    """Normalize title for deduplication comparison."""
    text = title.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    # Remove common suffixes/prefixes from news outlets
    text = re.sub(r"\s*-\s*(bbc news|cnn|reuters|ap news|the guardian|al jazeera).*", "", text)
    return text


def title_similarity(title_a: str, title_b: str) -> float:
    """Compute similarity between two normalized titles."""
    norm_a = normalize_title(title_a)
    norm_b = normalize_title(title_b)
    if not norm_a or not norm_b:
        return 0.0
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()


def check_content_duplicate(db_conn, title: str, canonical_text: str) -> bool:
    """
    Check if a similar article already exists in the DB.
    Compares against recent events using title similarity.
    """
    try:
        rows = db_conn.execute(
            """SELECT source_title, canonical_text
               FROM events
               WHERE ingested_at > NOW() - INTERVAL '%s days'
               ORDER BY ingested_at DESC
               LIMIT 200""",
            (_MAX_ARTICLE_AGE_DAYS,),
        ).fetchall()

        for row in rows:
            existing_title = row[0] or ""
            existing_text = row[1] or ""
            sim = title_similarity(title, existing_title)
            if sim >= _TITLE_SIM_THRESHOLD:
                return True
            # Also check canonical text similarity for short texts
            if len(canonical_text) < 500 and len(existing_text) < 500:
                text_sim = difflib.SequenceMatcher(None, canonical_text, existing_text).ratio()
                if text_sim >= _CONTENT_SIM_THRESHOLD:
                    return True
        return False
    except Exception:
        logger.exception("Content duplicate check failed")
        return False


def check_domain_penalty(db_conn, domain: str) -> float:
    """Get penalty score for a domain. Returns 0.0 if not found."""
    try:
        row = db_conn.execute(
            "SELECT penalty_score FROM domain_penalties WHERE domain = %s",
            (domain,),
        ).fetchone()
        return row[0] if row else 0.0
    except Exception:
        return 0.0


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
        "events_inserted": 0,
        "full_text_fetched": 0,
    }

    queries = build_search_queries()
    all_items = []

    # Execute global queries — no region restriction
    # Limit to top 50 most important queries per run to stay within time budget
    for query_info in queries[:50]:
        items = fetch_rss_feed(query_info, is_direct_url=False)
        all_items.extend(items)
        stats["queries_executed"] += 1

    # Fetch from static hardcoded feeds (Reddit)
    static_feeds = SETTINGS.get("sources", {}).get("static_feeds", [])
    for feed_url in static_feeds:
        items = fetch_rss_feed(feed_url, is_direct_url=True)
        all_items.extend(items)
        stats["queries_executed"] += 1

    # Fetch from GDELT — global news database with tone analysis
    gdelt_queries = build_gdelt_queries()
    for gdelt_query in gdelt_queries[:20]:  # Limit GDELT queries per run
        items = fetch_gdelt_articles(gdelt_query, max_age_days=_MAX_ARTICLE_AGE_DAYS)
        all_items.extend(items)
        stats["queries_executed"] += 1

    stats["items_fetched"] = len(all_items)
    logger.info("Pass A: Fetched %d items from %d sources/queries", len(all_items), stats["queries_executed"])

    inserted = 0
    for item in all_items:
        if inserted >= max_events:
            break

        url = item.get("link", "")
        if not url:
            continue

        # Canonicalize
        raw_text = f"{item.get('title', '')} {item.get('description', '')}"
        canonical = canonicalize_text(raw_text)

        # Noise filter
        if is_noise(canonical):
            stats["noise_filtered"] += 1
            continue

        # URL hash for dedup
        url_hash = compute_url_hash(url)

        # Domain extraction and penalty check
        domain = extract_domain(url)
        penalty = check_domain_penalty(db_conn, domain)
        if penalty > 0.8:
            stats["domain_penalized"] += 1
            continue

        # Optional: fetch full text
        full_text = ""
        if _FETCH_FULL_TEXT:
            full_text = fetch_full_text(url)
            if full_text:
                stats["full_text_fetched"] += 1
                # Merge full text into canonical
                canonical = canonicalize_text(f"{canonical} {full_text}")

        # Content dedup: check if similar title/text already exists
        if check_content_duplicate(db_conn, item.get("title", ""), canonical):
            stats["content_duplicates_skipped"] += 1
            continue

        # Idempotent insert — NOT EXISTS guard
        try:
            result = db_conn.execute(
                """INSERT INTO events (source_url, source_url_hash, source_domain,
                                       source_title, raw_text, canonical_text, status)
                   SELECT %s, %s, %s, %s, %s, %s, 'raw'
                   WHERE NOT EXISTS (
                       SELECT 1 FROM events WHERE source_url_hash = %s
                   )""",
                (url, url_hash, domain, item.get("title", ""),
                 raw_text, canonical, url_hash),
            )
            if result.rowcount > 0:
                inserted += 1
                stats["events_inserted"] += 1
            else:
                stats["duplicates_skipped"] += 1
        except Exception:
            logger.exception("Insert error for URL: %s", url[:80])
            db_conn.rollback()
            continue

    db_conn.commit()

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
