"""
SIM — Pass A ingest: source fetchers (all network I/O)

RSS/Google News, official travel advisories, full-text
fetch and best-effort translation. Everything that talks to the outside world
during ingest lives here. Split out of pass_a_ingest.py on 2026-07-16.
"""

import calendar
import email.utils
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.pipeline.ingest_filters import extract_domain

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as f:
    SETTINGS = json.load(f)

_MAX_ARTICLE_AGE_DAYS = SETTINGS.get("ingestion", {}).get("max_article_age_days", 4)

# Global Google News RSS — no geo lock.
# Google auto-redirects to US if hl=en alone; append gl=US is removed
# to let Google serve regionally mixed results.  We still force hl=en.
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"

# Google News search feeds are RELEVANCE-ranked, not date-ranked, so a bare
# query returns its 100 all-time best matches and buries today's coverage.
# Measured 2026-07-23 across 12 live queries: 885 items fetched, 6 of them from
# the last 48h. The same queries with `when:2d` returned 65 fresh items and
# nothing stale. The window follows max_article_age_days so the search operator
# and the age filter can never drift apart — asking Google for more than the
# filter accepts just wastes the feed's 100-item budget on rows we then drop.
_RECENCY_OPERATOR = f"when:{max(1, _MAX_ARTICLE_AGE_DAYS)}d"


def with_recency(query: str) -> str:
    """Append the Google News recency operator unless the query sets its own."""
    return query if "when:" in query else f"{query} {_RECENCY_OPERATOR}"



# ---------------------------------------------------------------------------
# HTTP helpers with retry / backoff
# ---------------------------------------------------------------------------

# Bot-protection interstitials answer with a RETRYABLE-LOOKING status (429/403) but serve
# an HTML challenge page instead of the feed. Backoff can never clear one — the challenge
# wants a JS-capable browser — so every retry is guaranteed-wasted wall clock, every run,
# forever. humanglemedia.com went behind Vercel's checkpoint and answered 429 on EVERY path
# including "/" (found 2026-08-06; the feed had been silently dead for weeks while the logs
# read like an ordinary rate limit). Detecting the challenge turns a permanent failure into
# an honest one-line diagnosis instead of a misleading "Rate limit (429), retry in 2.0s".
_CHALLENGE_MARKERS = (
    "vercel security checkpoint",
    "just a moment",
    "checking your browser",
    "cf-browser-verification",
    "attention required",
    "captcha",
)


def _is_bot_challenge(resp: httpx.Response) -> bool:
    """True when the body is a bot-protection interstitial rather than real content.

    Requires an HTML content-type: a feed endpoint serving text/html is already wrong,
    and that guard keeps the marker scan from ever firing on a legitimate XML feed that
    happens to quote one of these phrases in an article title.
    """
    if "html" not in resp.headers.get("content-type", "").lower():
        return False
    try:
        head = resp.text[:4000].lower()
    except Exception:  # undecodable body — can't be a challenge page we could read
        return False
    return any(marker in head for marker in _CHALLENGE_MARKERS)


def _http_get_with_retry(url: str, headers: dict | None = None, timeout: float = 15.0,
                         max_retries: int = 4, backoff_base: float = 8.0,
                         params: dict | None = None) -> httpx.Response | None:
    """Perform GET with exponential backoff on 429 / 5xx / network errors.

    Returns None when the URL is unusable — retries exhausted, or a bot-protection
    challenge that retrying could never clear.
    """
    headers = headers or {}
    for attempt in range(max_retries):
        try:
            resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True, params=params)
            if resp.status_code in (429, 403) and _is_bot_challenge(resp):
                logger.warning(
                    "Bot-protection challenge on %s (HTTP %d) — not retryable, giving up; "
                    "drop the source if this persists",
                    url[:80], resp.status_code,
                )
                return None
            if resp.status_code == 429:
                wait = backoff_base ** attempt
                logger.warning("Rate limit (429) on %s, retry in %.1fs (attempt %d/%d)",
                               url[:80], wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500 and attempt < max_retries - 1:
                wait = backoff_base ** attempt
                logger.warning("Server error %d on %s, retry in %.1fs",
                               exc.response.status_code, url[:80], wait)
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < max_retries - 1:
                wait = backoff_base ** attempt
                time.sleep(wait)
                continue
            raise
    return None


# ---------------------------------------------------------------------------
# Travel Advisory fetch (US State Dept)
# ---------------------------------------------------------------------------

_LEVEL_RE = re.compile(r"Level\s+(\d)", re.IGNORECASE)
_ADVISORY_NO_CHANGE_RE = re.compile(
    r"no changes?\s+to\s+the\s+advisory\s+level",
    re.IGNORECASE,
)
_ADVISORY_DOWNGRADE_RE = re.compile(
    r"\b(downgraded?|lowered?|decreased?|reduced?)\b",
    re.IGNORECASE,
)
_ADVISORY_UPGRADE_RE = re.compile(
    r"\b(upgraded?|raised?|elevated?|increased?\s+to\s+level|changed?\s+to\s+level\s+[3-4])\b",
    re.IGNORECASE,
)


# Phrase-based highest-tier wording, so non-US agencies (UK FCDO, Canada, Australia,
# New Zealand) that don't use "Level N" are still understood. Mapped to the US 1-4 scale.
_ADVISORY_L4_RE = re.compile(
    r"\b(do not travel"                          # US L4 / Australia / NZ
    r"|advise against all travel"                # UK FCDO highest
    r"|avoid all travel)\b",                     # Canada highest
    re.IGNORECASE,
)
_ADVISORY_L3_RE = re.compile(
    r"\b(reconsider (your )?(need to )?travel"    # US L3 / Australia
    r"|advise against all but essential travel"   # UK FCDO second
    r"|avoid (all )?non[- ]essential travel)\b",  # Canada second
    re.IGNORECASE,
)


def _parse_advisory_level(title: str, description: str = "") -> int:
    """Highest advisory level implied by the text, on the US 1-4 scale.

    Combines the numeric "Level N" form (US) with phrase-based wording used by other
    agencies (UK/CA/AU/NZ). Returns the max of the two, or 0 if none is found.
    """
    text = f"{title} {description}"
    m = _LEVEL_RE.search(text)
    numeric = int(m.group(1)) if m else 0
    phrase = 4 if _ADVISORY_L4_RE.search(text) else (3 if _ADVISORY_L3_RE.search(text) else 0)
    return max(numeric, phrase)


def _is_advisory_worth_ingesting(title: str, description: str) -> bool:
    """
    Return True only if this advisory is a level INCREASE or a high-risk Level 3/4 entry.

    Rules:
      1. 'No changes to the advisory level' → skip.
      2. Downgrade keywords in description → skip.
      3. Level >= 3 (Do Not Travel / Reconsider) without downgrade → ingest.
      4. Level < 3 with explicit upgrade keywords → ingest.
      5. Otherwise → skip (conservative).
    """
    desc_plain = re.sub(r"<[^>]+>", " ", description)  # strip HTML
    # Rule 1: no level change
    if _ADVISORY_NO_CHANGE_RE.search(desc_plain):
        return False
    # Rule 2: downgrade
    if _ADVISORY_DOWNGRADE_RE.search(desc_plain):
        return False
    level = _parse_advisory_level(title, desc_plain)
    # Rule 3: high-risk level
    if level >= 3:
        return True
    # Rule 4: explicit upgrade for lower levels
    if _ADVISORY_UPGRADE_RE.search(desc_plain):
        return True
    return False


def fetch_travel_advisories(stats: dict | None = None) -> list[dict]:
    """
    Fetch official travel advisory RSS feeds.
    Only ingests items where the advisory level has INCREASED or is Level 3/4.
    Items are tagged with source='travel_advisory' to bypass noise filters.
    """
    import xml.etree.ElementTree as ET
    from email.utils import parsedate

    advisory_feeds = SETTINGS.get("sources", {}).get("travel_advisory_feeds", [])
    if not advisory_feeds:
        return []

    all_items: list[dict] = []
    now_utc = datetime.now(timezone.utc)

    for feed_url in advisory_feeds:
        try:
            resp = _http_get_with_retry(
                feed_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "Accept": "application/rss+xml, */*"},
                timeout=20.0,
                max_retries=2,
                backoff_base=3.0,
            )
            if resp is None:
                logger.warning("Travel advisory feed unreachable: %s", feed_url)
                continue

            root = ET.fromstring(resp.text)
            # UK FCDO feeds are curated to high-risk countries (the country selection IS
            # the L3-4 filter) AND are Atom, not RSS. US State Dept is RSS + "Level N".
            is_uk = "gov.uk/foreign-travel-advice" in feed_url
            _ATOM = "{http://www.w3.org/2005/Atom}"
            entries = root.findall(".//item") or root.findall(f".//{_ATOM}entry")
            for item in entries:
                if item.tag.endswith("entry"):  # Atom (UK FCDO)
                    title = (item.findtext(f"{_ATOM}title") or "").strip()
                    link_el = item.find(f"{_ATOM}link")
                    link = ((link_el.get("href") if link_el is not None else "") or "").strip()
                    pub_date_str = (item.findtext(f"{_ATOM}updated")
                                    or item.findtext(f"{_ATOM}published") or "").strip()
                    description = (item.findtext(f"{_ATOM}summary")
                                   or item.findtext(f"{_ATOM}content") or "").strip()
                else:  # RSS (US State Dept)
                    title = (item.findtext("title") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    pub_date_str = (item.findtext("pubDate") or "").strip()
                    description = (item.findtext("description") or "").strip()

                if not title or not link:
                    continue

                # UK entries are change-notes without level wording, so ingest every
                # recent entry (the curated high-risk country is the filter). US/level
                # feeds keep the level-increase / Level 3-4 gate.
                if not is_uk and not _is_advisory_worth_ingesting(title, description):
                    continue

                # Parse date — RSS uses "Mon, 19 May 2026"; Atom uses ISO 8601.
                pub_dt: datetime | None = None
                if pub_date_str:
                    try:
                        pub_dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                    except Exception:
                        pass
                    if pub_dt is None:
                        try:
                            pub_dt = email.utils.parsedate_to_datetime(pub_date_str)
                        except Exception:
                            pass
                    if pub_dt is None:
                        try:
                            t = parsedate(pub_date_str)
                            if t:
                                pub_dt = datetime(*t[:6], tzinfo=timezone.utc)
                        except Exception:
                            pass
                    # Fallback: try strptime for date-only format "Day, DD Mon YYYY"
                    if pub_dt is None:
                        for fmt in ("%a, %d %b %Y", "%d %b %Y", "%Y-%m-%d"):
                            try:
                                pub_dt = datetime.strptime(pub_date_str.strip(), fmt).replace(tzinfo=timezone.utc)
                                break
                            except ValueError:
                                continue
                if pub_dt is None:
                    # Cannot determine date — skip rather than assume today
                    continue

                age_days = (now_utc - pub_dt).total_seconds() / 86400
                # Advisories update less often than news — use 7-day window
                if age_days > 7:
                    if stats is not None:
                        stats["age_filtered"] += 1
                    continue

                # Strip HTML from description for canonical text
                desc_plain = re.sub(r"<[^>]+>", " ", description)
                desc_plain = re.sub(r"\s+", " ", desc_plain).strip()

                # Advisory page URLs are stable across updates, but the pipeline dedups
                # permanently by URL hash (INSERT ... WHERE NOT EXISTS source_url_hash).
                # Stamp the update date onto the link so each genuine update becomes a
                # distinct event/alert, while the same update seen on repeated runs stays
                # deduped (no re-alert spam). The fragment is ignored when the link opens.
                dated_link = f"{link}#adv-{pub_dt.strftime('%Y%m%d')}"

                all_items.append({
                    "title": title,
                    "link": dated_link,
                    "pub_date": pub_date_str,
                    "pub_dt": pub_dt,
                    "description": desc_plain,
                    "source": "travel_advisory",
                    "_skip_noise_filter": True,  # official gov source, bypass noise check
                })

        except Exception:
            logger.exception("Error fetching travel advisory feed: %s", feed_url)

    if all_items:
        logger.info("Travel advisories: %d increase/high-risk items fetched", len(all_items))
    return all_items


# ---------------------------------------------------------------------------
# RSS / Atom fetch
# ---------------------------------------------------------------------------

def _parse_feed_lenient(text: str, now_utc: datetime, max_age: float, stats: dict | None) -> list[dict]:
    """Fallback parser for malformed feeds (e.g. raw HTML embedded as markup).

    Uses feedparser, which is far more tolerant than the stdlib XML parser.
    Returns items in the same shape as fetch_rss_feed.
    """
    import feedparser

    items: list[dict] = []
    parsed = feedparser.parse(text)
    for entry in parsed.entries:
        pub_dt = None
        if getattr(entry, "published_parsed", None):
            pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        elif getattr(entry, "updated_parsed", None):
            pub_dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

        if pub_dt is None:
            if stats is not None:
                stats["age_filtered"] += 1
            continue

        age_days = (now_utc - pub_dt).total_seconds() / 86400
        if age_days > max_age:
            if stats is not None:
                stats["age_filtered"] += 1
            continue

        items.append({
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "pub_date": entry.get("published", entry.get("updated", "")),
            "pub_dt": pub_dt,
            "description": entry.get("summary", ""),
        })
    return items


def fetch_rss_feed(query_info: dict, is_direct_url: bool = False, stats: dict | None = None) -> list[dict]:
    """Fetch and parse an RSS or Atom feed. Returns items with parsed pub_date."""
    import xml.etree.ElementTree as ET

    if is_direct_url:
        url = query_info if isinstance(query_info, str) else query_info.get("url", "")
    else:
        from urllib.parse import quote_plus
        url = GOOGLE_NEWS_RSS.format(query=quote_plus(with_recency(query_info["query"])))

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Several feeds (e.g. breakingdefense, crisisgroup, al-monitor) reject
            # requests without an explicit feed Accept header (403 / 415).
            "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.9, */*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = _http_get_with_retry(url, headers=headers, timeout=15.0, max_retries=2, backoff_base=2.0)
        if resp is None:
            logger.warning("RSS fetch failed for: %s", url[:80])
            return []
    except Exception:
        logger.warning("RSS fetch failed for: %s", url[:80])
        return []

    items = []
    now_utc = datetime.now(timezone.utc)
    max_age = _MAX_ARTICLE_AGE_DAYS

    try:
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            # Some feeds embed raw HTML as markup or contain unescaped tokens that
            # the strict stdlib parser rejects (e.g. warsawinstitute). Fall back to
            # feedparser, which is far more tolerant.
            return _parse_feed_lenient(resp.text, now_utc, max_age, stats)
        tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

        # Detect Atom vs RSS by root tag
        is_atom = tag == "feed"

        if is_atom:
            entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            if not entries:
                entries = root.findall(".//entry")
        else:
            entries = root.findall(".//item")

        for entry in entries:
            title = ""
            link = ""
            pub_date_str = ""
            description = ""

            if is_atom:
                # Atom: <title>, <link href="..."/>, <updated> or <published>, <content> or <summary>
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                if title_elem is None:
                    title_elem = entry.find("title")
                title = (title_elem.text or "") if title_elem is not None else ""

                link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
                if link_elem is None:
                    link_elem = entry.find("link")
                if link_elem is not None:
                    link = link_elem.get("href", "")

                pub_elem = entry.find("{http://www.w3.org/2005/Atom}published")
                if pub_elem is None:
                    pub_elem = entry.find("{http://www.w3.org/2005/Atom}updated")
                if pub_elem is None:
                    pub_elem = entry.find("published")
                if pub_elem is None:
                    pub_elem = entry.find("updated")
                pub_date_str = (pub_elem.text or "") if pub_elem is not None else ""

                content_elem = entry.find("{http://www.w3.org/2005/Atom}content")
                if content_elem is None:
                    content_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                if content_elem is None:
                    content_elem = entry.find("content")
                if content_elem is None:
                    content_elem = entry.find("summary")
                description = (content_elem.text or "") if content_elem is not None else ""
            else:
                # RSS 2.0
                title = entry.findtext("title", "")
                link = entry.findtext("link", "")
                pub_date_str = entry.findtext("pubDate", "")
                description = entry.findtext("description", "")
                # Google News links are news.google.com redirects; the real
                # publisher lives in <source url="...">. Without it every Google
                # query item collapses into one "news.google.com" domain for
                # dedup, penalties and the per-domain diversity cap.
                source_elem = entry.find("source")
                source_feed_url = source_elem.get("url", "") if source_elem is not None else ""

            # Parse date
            pub_dt = None
            if pub_date_str:
                try:
                    pub_dt = email.utils.parsedate_to_datetime(pub_date_str)
                except Exception:
                    pass
                if pub_dt is None:
                    # Try ISO 8601 (Atom)
                    try:
                        pub_dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                    except Exception:
                        pass

            if pub_dt is None:
                if stats is not None:
                    stats["age_filtered"] += 1
                continue

            age_days = (now_utc - pub_dt).total_seconds() / 86400
            if age_days > max_age:
                if stats is not None:
                    stats["age_filtered"] += 1
                continue

            item = {
                "title": title,
                "link": link,
                "pub_date": pub_date_str,
                "pub_dt": pub_dt,
                "description": description,
            }
            if not is_atom and source_feed_url:
                item["domain"] = extract_domain(source_feed_url)
            items.append(item)
    except Exception:
        logger.exception("RSS parse error for: %s", url[:80])

    return items


# ---------------------------------------------------------------------------
# Publication date extracted from the article itself
# ---------------------------------------------------------------------------

# Browser-shaped headers. trafilatura's own downloader announces itself and is
# served a 403 by a good share of publishers, so the article body — and with it
# the page's own publication date — never arrived.
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Publisher-declared dates, most trustworthy first. schema.org datePublished is
# what CMSes emit and is the field that caught the 2016 reprints; the OpenGraph
# meta tag is the common second source; <time datetime> is last because it is
# also used for "related article" teasers.
_JSONLD_PUBLISHED_RE = re.compile(r'"datePublished"\s*:\s*"([^"]{8,40})"')
_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_META_KEY_RE = re.compile(r'(?:property|name|itemprop)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_META_CONTENT_RE = re.compile(r'content\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_TIME_TAG_RE = re.compile(r'<time\b[^>]*\bdatetime\s*=\s*["\']([^"\']{8,40})["\']', re.IGNORECASE)

_PUBLISHED_META_KEYS = {
    "article:published_time", "article:published", "og:published_time",
    "datepublished", "publishdate", "pubdate", "date", "dc.date.issued",
    "parsely-pub-date", "sailthru.date",
}

# The same declaration under a house prefix: Hankyoreh serves `h:published_time` and
# nothing else machine-readable, which cost a severity-100 Zaporizhzhia strike its page
# on 2026-08-12 — the date was right there, under a key the set above did not list.
# Matching the suffix is still a metadata match: the publisher is naming the field, not
# us inferring one from page text, so this does not reopen the htmldate hole.
_NAMESPACED_PUBLISHED_KEY_SUFFIX = ":published_time"


def _parse_iso_like(value: str) -> datetime | None:
    """Parse the date formats publishers actually emit, as aware UTC."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(value)
        except Exception:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%Y%m%d"):
                try:
                    dt = datetime.strptime(value[:len(fmt) + 2].strip(), fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Date-only values carry no time of day. Reading them as midnight ages the
    # article by up to 24h and pushes genuinely fresh stories over the freshness
    # window — measured 2026-08-05, two same-day France24 reports were dropped as
    # reprints on exactly this. End-of-day keeps the estimate conservative.
    if ":" not in value:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.astimezone(timezone.utc)


def extract_published_date(html: str) -> datetime | None:
    """Best-effort publication date declared by the article page itself.

    This is the only date in the pipeline the publisher can't get wrong by
    accident: the feed's pubDate is whatever the aggregator believes, and Google
    News stamps re-crawled archive pages with the crawl date (measured
    2026-08-05: a 2016-10-25 Yeni Safak story served as 2026-08-04).
    """
    if not html:
        return None

    match = _JSONLD_PUBLISHED_RE.search(html)
    if match:
        dt = _parse_iso_like(match.group(1))
        if dt:
            return dt

    for tag in _META_TAG_RE.findall(html):
        key = _META_KEY_RE.search(tag)
        if not key:
            continue
        name = key.group(1).strip().lower()
        if name not in _PUBLISHED_META_KEYS \
                and not name.endswith(_NAMESPACED_PUBLISHED_KEY_SUFFIX):
            continue
        content = _META_CONTENT_RE.search(tag)
        if content:
            dt = _parse_iso_like(content.group(1))
            if dt:
                return dt

    match = _TIME_TAG_RE.search(html)
    if match:
        dt = _parse_iso_like(match.group(1))
        if dt:
            return dt

    # Deliberately NO heuristic guess here. htmldate used to backstop this chain
    # and it was the sole cause of every false drop in the first production run
    # (#1349, 2026-08-05): on pages carrying no date metadata it inferred one
    # from the copyright line, so three same-day reports of the Kyiv strike were
    # discarded as reprints — two dated "2026-01-01" (year-only guess) and one
    # "2026-01-23". Every genuine reprint that run caught had a publisher-declared
    # date, so the guess bought nothing and cost today's top story. An article
    # that declares no date is UNKNOWN, and unknown must never justify a drop.
    return None


# Dates publishers put in the path: /2016/10/25/, /20260805-slug, /2026-08-05/.
# Free, and it is the only date signal left when the page itself 403s — France24
# blocks our fetch outright but stamps every article path with YYYYMMDD.
_URL_DATE_RES = (
    re.compile(r"/(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?:[/-]|$)"),
    re.compile(r"/(20\d{2})(\d{2})(\d{2})[-_/]"),
)

# Month precision: /2022/09/slug — a very common WordPress permalink shape, and one the
# day-precision patterns above silently ignored. Measured cost of ignoring it (30 days
# to 2026-08-12): 15 events whose path month predated their Google-stamped date by more
# than a month, three of which fired ALERT — including a 2022 Waziristan bombing sent as
# an 2026-08-05 incident at severity 95. Tried only after the day patterns miss.
_URL_MONTH_RE = re.compile(r"/(20\d{2})/(0?[1-9]|1[0-2])/")


def extract_date_from_url(url: str) -> datetime | None:
    """Publication date encoded in the article path, or None.

    Coarse precision, so it resolves to the END of the period it names — end of day for
    /2026/08/05/, end of month for /2026/08/. The age check must never make a story look
    older than it is, and this date only ever decides whether something is inside the
    freshness window, not when an incident happened.

    End-of-month also makes the month pattern self-limiting: for the CURRENT month the
    result lands past the caller's future-date guard and is discarded, so a path month
    can only ever age an article that is already at least a month old.
    """
    for pattern in _URL_DATE_RES:
        match = pattern.search(url or "")
        if not match:
            continue
        year, month, day = (int(g) for g in match.groups())
        try:
            return datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
        except ValueError:
            continue

    match = _URL_MONTH_RE.search(url or "")
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        last_day = calendar.monthrange(year, month)[1]
        return datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return None


# ---------------------------------------------------------------------------
# Full-text fetch
# ---------------------------------------------------------------------------

def fetch_article(url: str, timeout: float = 20.0) -> dict:
    """Fetch one article: resolve aggregator redirects, extract text and date.

    Returns {"url", "text", "published_at", "fetch_ok"} — `url` is the publisher's
    URL when the Google News handle could be resolved (otherwise the input
    unchanged), `published_at` is the page's own declared date or None.

    `fetch_ok` separates "the publisher's page declares no date" from "we never got
    to read the publisher's page". Both produce published_at=None, and treating them
    alike cost a live Libya Observer report its page on 2026-08-12: the page declares
    2026-08-12T10:16 and a retry minutes later read it fine, but the fetch inside that
    run failed and the event was recorded as having an unconfirmable date. A transient
    403 is not evidence about a publisher's metadata.

    One HTTP round trip serves both consumers: Pass A needs the body for the
    classifier and the date to reject reprints, and fetching twice would double
    the per-run network budget for no gain.
    """
    result = {"url": url, "text": "", "published_at": None, "fetch_ok": False}
    if not url:
        return result

    # Same resolver the SITREP citations use — one implementation, so a break in
    # Google's handle format surfaces in both places at once instead of leaving
    # ingest quietly running on a stale copy.
    from src.services.google_news_resolver import resolve_url
    result["url"] = resolve_url(url, timeout=min(timeout, 8.0)) or url

    # Path-encoded date first: it costs nothing and it is the ONLY date we get
    # from publishers that 403 our fetch (France24 among them).
    result["published_at"] = extract_date_from_url(result["url"])

    try:
        resp = httpx.get(result["url"], headers=_BROWSER_HEADERS, timeout=timeout,
                         follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.debug("Article fetch failed for %s: %s", result["url"][:80], str(exc))
        return result
    result["fetch_ok"] = True

    # The page's own declaration beats the path: it carries a time of day, and a
    # path date can be the slug's creation day rather than publication.
    result["published_at"] = extract_published_date(html) or result["published_at"]
    try:
        import trafilatura
        result["text"] = trafilatura.extract(
            html, include_comments=False, include_tables=False, no_fallback=False
        ) or ""
    except Exception:
        logger.debug("Text extraction failed for %s", result["url"][:80])
    return result


def fetch_full_text(url: str) -> str:
    """Article body text only. Thin wrapper over fetch_article."""
    return fetch_article(url)["text"]


def google_translate(text: str, target: str = "en") -> str:
    """Translate text using public Google Translate endpoint (no credentials needed)."""
    if not text or not text.strip():
        return text
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": target,
            "dt": "t",
            "q": text
        }
        resp = httpx.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        # The response is a nested JSON list: [[[translated_text, original_text, ...]]]
        data = resp.json()
        if data and len(data) > 0 and data[0]:
            translated_segments = [seg[0] for seg in data[0] if seg and seg[0]]
            return "".join(translated_segments)
    except Exception:
        logger.exception("Failed to translate text: %s", text[:80])
    return text


# Scripts that signal the text itself is not English. One class, findall'd so the
# trigger can be ratio-based rather than any-single-character.
_NON_LATIN_CHAR_RE = re.compile(
    r'[\u0370-\u03FF'   # Greek
    r'\u0400-\u04FF'    # Cyrillic (Russian, Ukrainian, ...)
    r'\u0590-\u05FF'    # Hebrew
    r'\u0600-\u06FF'    # Arabic / Farsi
    r'\u0900-\u097F'    # Devanagari (Hindi)
    r'\u0E00-\u0E7F'    # Thai
    r'\u3040-\u30FF'    # Japanese kana
    r'\u4E00-\u9FFF'    # CJK unified ideographs (Chinese, Japanese kanji)
    r'\uAC00-\uD7AF]'   # Hangul (Korean)
)

# Translate only when at least this share of the LETTERS is non-Latin script.
# Any-single-character triggering sent already-English headlines to Google
# Translate whenever Google News appended a foreign outlet name (e.g.
# "Turkiye marks anniversary \u2026 - \u0634\u0641\u0642 \u0646\u064A\u0648\u0632", seen every run in the logs).
# 0.3: a long Cyrillic outlet name on a short English headline measures ~0.25
# ("\u2026 - \u041A\u043E\u0440\u0430\u0431\u0435\u043B\u043E\u0432.\u0406\u041D\u0424\u041E"), genuinely mixed-language text ~0.5, foreign ~1.0.
_TRANSLATE_MIN_NON_LATIN_RATIO = 0.3


# Translation is the one network call Pass A makes per ITEM rather than per
# inserted event, so its volume is invisible in every existing counter. Counted
# here rather than at the call site because only this function knows whether the
# ratio gate actually fired.
_TRANSLATE_CALLS = 0


def translation_call_count() -> int:
    """Network translations performed since the last reset."""
    return _TRANSLATE_CALLS


def reset_translation_counter() -> None:
    global _TRANSLATE_CALLS
    _TRANSLATE_CALLS = 0


def translate_to_english_if_needed(text: str) -> str:
    """Translate to English when the text is substantially non-Latin script.

    Ratio-gated (share of letters, not raw chars, so punctuation/digits don't
    dilute it): a mostly-English headline with a foreign outlet suffix passes
    through untouched, while a genuinely foreign-language headline \u2014 including
    CJK/Greek/Thai/Devanagari, which the old 3-range check missed \u2014 is translated.
    """
    if not text:
        return text
    non_latin = len(_NON_LATIN_CHAR_RE.findall(text))
    if not non_latin:
        return text
    letters = sum(1 for ch in text if ch.isalpha())
    if letters and non_latin / letters >= _TRANSLATE_MIN_NON_LATIN_RATIO:
        global _TRANSLATE_CALLS
        _TRANSLATE_CALLS += 1
        return google_translate(text, target="en")
    return text

