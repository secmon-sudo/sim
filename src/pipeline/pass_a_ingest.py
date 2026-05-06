"""
SIM — Pass A: Ingest & Canonicalization
Blueprint V20.1 §4 PASS A

Collects aviation incident articles from Google News RSS,
normalizes text, filters noise, and inserts raw events.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
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

# Prompt injection patterns to strip before LLM classification
PROMPT_INJECTION_PATTERNS = re.compile(
    r"\[INST\]|<\|system\|>|<\|user\|>|<\|assistant\|>|IGNORE PREVIOUS INSTRUCTIONS|"
    r"FORGET ALL PRIOR|YOU ARE NOW|SYSTEM OVERRIDE",
    re.IGNORECASE,
)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"


def build_search_queries() -> list[str]:
    """Build search queries from keywords config."""
    queries = []
    for lang, keywords in KEYWORDS_CONFIG.get("emergency_keywords", {}).items():
        for kw in keywords:
            queries.append(f'"{kw}" airport OR aviation')
    return queries


def fetch_rss_feed(query: str) -> list[dict]:
    """Fetch and parse a Google News RSS feed for a query."""
    import xml.etree.ElementTree as ET

    url = GOOGLE_NEWS_RSS.format(query=query)
    try:
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
    except Exception:
        logger.warning("RSS fetch failed for query: %s", query[:50])
        return []

    items = []
    try:
        root = ET.fromstring(resp.text)
        for item in root.findall(".//item"):
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            description = item.findtext("description", "")
            items.append({
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "description": description,
            })
    except Exception:
        logger.exception("RSS parse error for query: %s", query[:50])

    return items


def extract_domain(url: str) -> str:
    """Extract eTLD+1 domain from URL."""
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def compute_url_hash(url: str) -> str:
    """SHA-256 hash of normalized URL for deduplication."""
    normalized = url.strip().lower().split("?")[0].split("#")[0]
    return hashlib.sha256(normalized.encode()).hexdigest()


def is_noise(text: str) -> bool:
    """Check if text matches known noise patterns."""
    text_lower = text.lower()
    for pattern in KEYWORDS_CONFIG.get("noise_filters", []):
        if pattern.lower() in text_lower:
            return True
    return False


def canonicalize_text(raw_text: str) -> str:
    """
    Clean and normalize raw article text.
    - Strips prompt injection patterns
    - Normalizes whitespace
    - Removes HTML tags
    """
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", raw_text)
    # Strip prompt injection patterns
    text = PROMPT_INJECTION_PATTERNS.sub("", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


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

    1. Fetch RSS feeds for all keyword queries
    2. Canonicalize text, filter noise
    3. Insert new events with NOT EXISTS guard (idempotent)

    Returns: stats dict with counts
    """
    max_events = max_events or SETTINGS["pipeline"]["max_events_per_run"]

    stats = {
        "queries_executed": 0,
        "items_fetched": 0,
        "noise_filtered": 0,
        "duplicates_skipped": 0,
        "domain_penalized": 0,
        "events_inserted": 0,
    }

    queries = build_search_queries()
    all_items = []

    # Fetch from RSS feeds
    for query in queries[:20]:  # Limit queries per run to avoid rate limiting
        items = fetch_rss_feed(query)
        all_items.extend(items)
        stats["queries_executed"] += 1

    stats["items_fetched"] = len(all_items)
    logger.info("Pass A: Fetched %d items from %d queries", len(all_items), stats["queries_executed"])

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
