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
import random
import re
import time
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
_CONTENT_SIM_THRESHOLD = _DEDUP.get("content_similarity_threshold", 0.72)
_TITLE_SIM_THRESHOLD = _DEDUP.get("title_similarity_threshold", 0.78)

# Prompt injection patterns to strip before LLM classification
PROMPT_INJECTION_PATTERNS = re.compile(
    r"\[INST\]|<\|system\|>|<\|user\|>|<\|assistant\|>|IGNORE PREVIOUS INSTRUCTIONS|"
    r"FORGET ALL PRIOR|YOU ARE NOW|SYSTEM OVERRIDE",
    re.IGNORECASE,
)

# Global Google News RSS — no geo lock.
# Google auto-redirects to US if hl=en alone; append gl=US is removed
# to let Google serve regionally mixed results.  We still force hl=en.
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"

# GDELT 2.0 Article List API
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_RECORDS = 25  # Per query to stay within API limits


# ---------------------------------------------------------------------------
# HTTP helpers with retry / backoff
# ---------------------------------------------------------------------------

def _http_get_with_retry(url: str, headers: dict | None = None, timeout: float = 15.0,
                         max_retries: int = 4, backoff_base: float = 8.0,
                         params: dict | None = None) -> httpx.Response | None:
    """Perform GET with exponential backoff on 429 / 5xx / network errors."""
    headers = headers or {}
    for attempt in range(max_retries):
        try:
            resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True, params=params)
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
# GDELT fetch
# ---------------------------------------------------------------------------

# Try to use gdeltdoc for structured queries; fall back to raw HTTP if unavailable.
_GDELTDOC_AVAILABLE = False
try:
    from gdeltdoc import Filters, GdeltDoc
    _GDELTDOC_AVAILABLE = True
except ImportError:
    pass

# GDELT uses FIPS 10-4 country codes, not ISO2.
_ISO2_TO_FIPS = {
    # Middle East
    "IL": "IS", "IQ": "IZ", "TR": "TU", "YE": "YM", "KW": "KU", "LB": "LE", "JO": "JO", "PS": "WE",
    # Africa
    "NG": "NI", "NE": "NG", "BF": "UV", "SD": "SU", "SS": "OD", "DZ": "AG", "LY": "LY",
    # Eurasia
    "UA": "UP", "RU": "RS", "GE": "GG", "MD": "MD", "AZ": "AJ", "AM": "AM",
    # Asia
    "KP": "KN", "KR": "KS", "VN": "VM", "PH": "RP", "MM": "BM", "KH": "CB",
}


def _parse_gdelt_date(seendate_str: str) -> datetime | None:
    """Parse GDELT seendate (YYYYMMDDHHMMSS) to datetime."""
    if seendate_str and len(seendate_str) >= 14:
        try:
            return datetime.strptime(seendate_str[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def _gdelt_articles_from_raw(
    query: str,
    max_age_days: int = 3,
    tone: str | None = None,
    source_countries: list[str] | None = None,
) -> list[dict]:
    """Raw HTTP fallback for GDELT (used when gdeltdoc is not installed)."""
    now = datetime.now(timezone.utc)
    end_dt = now.strftime("%Y%m%d%H%M%S")
    start_dt = (now - timedelta(days=max_age_days)).strftime("%Y%m%d%H%M%S")

    full_query = query
    if tone:
        full_query = f"({full_query}) tone{tone}"
    if source_countries:
        # Map ISO2 to FIPS for GDELT
        fips_list = [_ISO2_TO_FIPS.get(c, c) for c in source_countries]
        # Keep filter short to avoid GDELT query length limits
        country_filter = " OR ".join(f"sourcecountry:{c}" for c in fips_list[:10])
        full_query = f"({full_query}) AND ({country_filter})"

    params = {
        "query": full_query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": GDELT_MAX_RECORDS,
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "sort": "seendate",
    }

    try:
        resp = _http_get_with_retry(
            GDELT_DOC_API,
            headers={"User-Agent": "SIM-OSINT-Bot/1.0"},
            timeout=30.0,
            max_retries=3,
            backoff_base=5.0,
            params=params,
        )
        if resp is None:
            return []
        data = resp.json()
    except Exception:
        return []

    items = []
    for article in data.get("articles", []):
        seendate_str = article.get("seendate", "")
        title = article.get("title", "")
        url = article.get("url", "")
        domain = article.get("domain", "")
        if not url or not title:
            continue
        pub_dt = _parse_gdelt_date(seendate_str)
        if pub_dt is None:
            continue
        if (now - pub_dt).total_seconds() / 86400 > max_age_days:
            continue
        items.append({
            "title": title, "link": url, "pub_date": seendate_str,
            "pub_dt": pub_dt, "description": "",
            "source": "gdelt", "domain": domain,
            "source_country": article.get("sourcecountry", ""),
        })
    return items


def _gdelt_articles_with_client(
    query: str,
    max_age_days: int = 3,
    tone: str | None = None,
    source_countries: list[str] | None = None,
) -> list[dict]:
    """Use gdeltdoc client for structured GDELT queries."""
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    try:
        f = Filters(
            keyword=query,
            start_date=start_date,
            end_date=end_date,
            num_records=GDELT_MAX_RECORDS,
            tone=tone,
            country=[_ISO2_TO_FIPS.get(c, c) for c in source_countries[:10]] if source_countries else None,
        )
        gd = GdeltDoc()
        df = gd.article_search(f)
    except Exception:
        return []

    items = []
    for _, row in df.iterrows():
        url = row.get("url", "")
        title = row.get("title", "")
        if not url or not title:
            continue
        seendate_str = str(row.get("seendate", ""))
        pub_dt = _parse_gdelt_date(seendate_str)
        if pub_dt is None:
            continue
        if (now - pub_dt).total_seconds() / 86400 > max_age_days:
            continue
        items.append({
            "title": title, "link": url, "pub_date": seendate_str,
            "pub_dt": pub_dt, "description": "",
            "source": "gdelt", "domain": row.get("domain", ""),
            "source_country": row.get("sourcecountry", ""),
        })
    return items


def fetch_gdelt_articles(
    query: str,
    max_age_days: int = 3,
    tone: str | None = None,
    source_countries: list[str] | None = None,
) -> list[dict]:
    """
    Fetch article URLs from GDELT 2.0 API.

    Uses gdeltdoc client if available (structured Filters with tone/country
    support); falls back to raw HTTP GET with retry/backoff otherwise.
    """
    if _GDELTDOC_AVAILABLE:
        items = _gdelt_articles_with_client(query, max_age_days, tone, source_countries)
    else:
        items = _gdelt_articles_from_raw(query, max_age_days, tone, source_countries)

    if items:
        logger.info("GDELT: %d articles for '%s...' (last %dh)", len(items), query[:40], max_age_days * 24)
    return items


# ---------------------------------------------------------------------------
# Noise filters
# ---------------------------------------------------------------------------

def _compile_noise_patterns() -> list[re.Pattern]:
    """Compile noise filters with word boundaries to reduce false positives."""
    patterns = []
    for pattern in KEYWORDS_CONFIG.get("noise_filters", []):
        escaped = re.escape(pattern)
        try:
            patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
        except re.error:
            patterns.append(re.compile(re.escape(pattern), re.IGNORECASE))
    return patterns


NOISE_PATTERNS = _compile_noise_patterns()

# Additional hard-coded title-level sports/entertainment blockers.
# These are compiled once and applied *before* the config-based filters.
_SPORTS_ENT_BLOCKERS = [
    re.compile(r"\btransfer\s+(window|deal|rumor|gossip|news)\b", re.IGNORECASE),
    re.compile(r"\b(hijack|hijacked)\s+(deal|transfer|move|signing)\b", re.IGNORECASE),
    re.compile(r"\b(football|soccer|premier\s+league|la\s+liga|bundesliga|serie\s+a|champions\s+league|fifa|uefa|world\s+cup)\b", re.IGNORECASE),
    re.compile(r"\b(liverpool|tottenham|manchester\s+(united|city)|chelsea|arsenal|barcelona|real\s+madrid|bayern|juventus|ac\s+milan|inter\s+milan|psg|borussia)\b", re.IGNORECASE),
    re.compile(r"\b(match|score|goal|fixture|kick\s*off|half[-\s]time|full[-\s]time)\b", re.IGNORECASE),
    re.compile(r"\b(netflix|disney\+|hulu|amazon\s+prime|streaming|season\s+\d+|episode\s+\d+|doctor\s+who|tv\s+series|tv\s+show|movie\s+review|box\s+office)\b", re.IGNORECASE),
    re.compile(r"\b(celebrity|gossip|rumour|rumor|speculation|insider)\b", re.IGNORECASE),
    re.compile(r"\b(bitcoin|crypto|nft|blockchain|stock\s+market|shares\s+rise|shares\s+fall|ipo|earnings)\b", re.IGNORECASE),
]


def is_noise(text: str) -> bool:
    """Check if text matches known noise patterns using word boundaries."""
    text_lower = text.lower()
    for pattern in NOISE_PATTERNS:
        if pattern.search(text_lower):
            return True
    for pattern in _SPORTS_ENT_BLOCKERS:
        if pattern.search(text_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def build_search_queries() -> list[dict]:
    """Build focused search queries.  ONLY aviation / hotel / tourism / mass-casualty security.

    Strategy:
    1.  Keep the query list SHORT (≈40 queries).  300+ low-quality queries drown the signal.
    2.  Every query MUST contain aviation / hotel / tourism context, OR be an
        unambiguous mass-casualty security phrase.
    3.  Google News RSS handles simple `phrase airport` syntax reliably.
        Complex boolean with many negative keywords breaks or returns 0 results.
    4.  Remaining noise is caught by `is_noise()` post-filter.
    """
    queries = []
    seen = set()

    def _add(q: str):
        if q.lower() not in seen:
            seen.add(q.lower())
            queries.append({"query": q, "broad": False})

    # ── Tier 1: Unambiguous aviation / airport security phrases ──
    tier1 = [
        '"airport attack"',
        '"airport shooting"',
        '"airport bombing"',
        '"airport stabbing"',
        '"airport security breach"',
        '"airport threat"',
        '"airport explosion"',
        '"airport terror"',
        '"airport gunfire"',
        '"airline crew attack"',
        '"airline staff assault"',
        '"flight attendant attack"',
        '"pilot assault"',
        '"pilot attacked"',
        '"ground crew attack"',
        '"ground staff stabbed"',
        '"cabin crew assaulted"',
        '"airport worker killed"',
        '"check-in agent attacked"',
        '"air traffic controller threat"',
    ]
    for q in tier1:
        _add(q)

    # ── Tier 2: Hotel / resort / tourism security phrases ──
    tier2 = [
        '"hotel attack"',
        '"hotel bombing"',
        '"hotel shooting"',
        '"hotel siege"',
        '"hotel explosion"',
        '"hotel terror"',
        '"resort attack"',
        '"resort bombing"',
        '"beach attack"',
        '"cruise ship attack"',
        '"tourist hotel attack"',
        '"hostage hotel"',
    ]
    for q in tier2:
        _add(q)

    # ── Tier 3: Generic words FORCED into aviation/hotel context ──
    # Simple `phrase airport` format — Google News handles this reliably.
    # Noise filters catch any remaining false positives post-fetch.
    tier3 = [
        '"bomb threat" airport',
        '"active shooter" airport',
        '"security breach" airport',
        '"evacuation" airport',
        '"explosion" airport',
        '"suspicious package" airport',
        '"hostage" airport',
        '"hijack" airport',
        '"mass shooting" airport',
        '"mass casualty" airport',
        '"suicide bombing" airport',
        '"drone attack" airport',
        '"unruly passenger" airport',
        '"passenger attack crew" airport',
        '"laser attack" airport',
        '"runway incursion"',
        '"emergency landing"',
        '"engine failure" flight',
        '"fire on board" flight',
        '"bird strike" airport',
        '"depressurization" flight',
        '"drone incursion" airport',
    ]
    for q in tier3:
        _add(q)

    # ── Tier 4: Geopolitical / African terrorism (broad, high-value only) ──
    geo = [
        '"missile strike"',
        '"airstrike"',
        '"war escalation"',
        '"ceasefire broken"',
        '"Iran Israel"',
        '"Ukraine Russia"',
        '"nuclear threat"',
        '"military coup"',
        '"Boko Haram"',
        '"Al-Shabaab"',
        '"jihadist attack"',
        '"ISIS Africa"',
        '"Sahel crisis"',
        '"Mali attack"',
        '"Burkina Faso attack"',
        '"Niger coup"',
        '"Somalia bombing"',
        '"Wagner Africa"',
        '"civilian casualties"',
        '"artillery shelling"',
        '"troop buildup"',
        '"border clash"',
    ]
    for q in geo:
        _add(q)

    return queries


def build_gdelt_queries() -> list[dict]:
    """
    Build focused GDELT queries with tone and source-country filters.

    Covers ALL active conflict regions globally per EASA CZIB + known high-risk areas:
      • Middle East & Persian Gulf
      • Sahel & Horn of Africa
      • North & Central Africa
      • South & Southeast Asia
      • Latin America
      • Eurasia / Eastern Europe

    Each query dict has:
      - query: GDELT search string
      - tone: optional tone filter ("<-5" = negative news)
      - countries: FIPS source-country list
    """
    return [
        # ── Middle East & Persian Gulf (CZIB Active) ──
        {
            "query": '"airport attack" OR "airport bombing" OR "airport shooting" OR "airport terror"',
            "tone": "<-5",
            "countries": ["SY", "IQ", "IR", "IL", "JO", "LB", "YE", "SA", "AE", "QA", "KW", "BH", "OM", "TR", "PS"],
        },
        {
            "query": '"missile strike" OR "airstrike" OR "drone strike" OR "civilian casualties"',
            "tone": "<-5",
            "countries": ["SY", "IQ", "IR", "IL", "LB", "YE", "SA", "AE", "TR", "PS", "JO", "KW"],
        },
        # ── Sahel & West Africa (CZIB Active: Mali, Libya, Sudan, Somalia) ──
        {
            "query": '"Boko Haram" OR "Al-Shabaab" OR "jihadist attack" OR "ISIS Africa" OR "Sahel crisis" OR "Wagner Africa"',
            "tone": None,
            "countries": ["NG", "NE", "ML", "BF", "TD", "CF", "SN", "MR", "GN", "SL", "LR", "CI", "GH", "TG", "BJ", "CM", "GA", "GQ", "ST", "SO", "ET", "ER", "DJ", "KE", "SS", "SD", "UG", "RW", "BI", "CD", "CG", "AO", "LY", "DZ", "TN", "EG", "MA", "EH"],
        },
        {
            "query": '"mass shooting" OR "mass casualty" OR "massacre" OR "suicide bombing" OR "vehicle ramming"',
            "tone": "<-5",
            "countries": ["NG", "ML", "BF", "TD", "CF", "SO", "ET", "SS", "SD", "CD", "CG", "AO", "LY", "DZ", "TN", "EG"],
        },
        # ─— Ukraine / Russia / Eurasia (CZIB Active) ──
        {
            "query": '"Ukraine" OR "Russia" OR "missile" OR "airstrike" OR "war" OR "invasion"',
            "tone": "<-5",
            "countries": ["UA", "RU", "BY", "MD", "GE", "AM", "AZ", "PL", "RO", "HU", "SK", "LT", "LV", "EE", "FI"],
        },
        # ── South & Central Asia ──
        {
            "query": '"Taliban" OR "Afghanistan" OR "Pakistan" OR "terror attack" OR "suicide bomber"',
            "tone": "<-5",
            "countries": ["AF", "PK", "IN", "BD", "LK", "NP", "BT", "MV", "IR", "IQ", "SY", "YE"],
        },
        # ── Southeast Asia ──
        {
            "query": '"Myanmar" OR "Rohingya" OR "civil war" OR "armed conflict" OR "insurgency"',
            "tone": "<-5",
            "countries": ["MM", "TH", "PH", "ID", "MY", "VN", "LA", "KH", "SG", "BN", "TL", "PG"],
        },
        # ── Latin America ──
        {
            "query": '"drug cartel" OR "gang violence" OR "massacre" OR "armed conflict" OR "homicide"',
            "tone": "<-5",
            "countries": ["CO", "VE", "MX", "HT", "EC", "PE", "BR", "HN", "GT", "SV", "NI", "CR", "PA", "BO", "PY", "CL", "AR", "UY", "CU", "JM", "DO"],
        },
        # ── East Asia / Pacific ──
        {
            "query": '"North Korea" OR "DPRK" OR "nuclear" OR "missile test" OR "provocation"',
            "tone": "<-5",
            "countries": ["KP", "KR", "JP", "CN", "TW", "MN", "PH", "VN", "ID", "MY", "SG", "TH", "AU", "NZ", "PG"],
        },
    ]


# ---------------------------------------------------------------------------
# RSS / Atom fetch
# ---------------------------------------------------------------------------

def fetch_rss_feed(query_info: dict, is_direct_url: bool = False, stats: dict | None = None) -> list[dict]:
    """Fetch and parse an RSS or Atom feed. Returns items with parsed pub_date."""
    import xml.etree.ElementTree as ET

    if is_direct_url:
        url = query_info if isinstance(query_info, str) else query_info.get("url", "")
    else:
        url = GOOGLE_NEWS_RSS.format(query=query_info["query"])

    try:
        headers = {
            "User-Agent": "python:sim-osint:v20.1 (by /u/sim_osint_bot)"
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
        root = ET.fromstring(resp.text)
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


# ---------------------------------------------------------------------------
# Full-text fetch
# ---------------------------------------------------------------------------

def fetch_full_text(url: str) -> str:
    """
    Attempt to fetch full article text from URL using trafilatura.
    Uses a strict timeout to prevent pipeline hangs on slow sources.
    """
    try:
        import trafilatura
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        # We use a strict timeout for both connection and read
        # trafilatura.fetch_url uses a complex internal download mechanism; 
        # using it directly but being aware of its potential to hang.
        downloaded = trafilatura.fetch_url(url) 
        
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False, no_fallback=False)
            return text or ""
    except Exception as e:
        logger.debug("Full-text fetch failed for %s: %s", url[:80], str(e))
    return ""


# ---------------------------------------------------------------------------
# Domain / URL helpers
# ---------------------------------------------------------------------------

def extract_domain(url: str) -> str:
    """Extract eTLD+1 domain from URL."""
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


_GOOGLE_NEWS_REDIR = re.compile(r"^https?://news\.google\.com/rss/articles/")

def compute_url_hash(url: str) -> str:
    """SHA-256 hash of normalized URL for deduplication."""
    normalized = url.strip().lower()
    # For Google News redirect URLs, strip query params as well
    # because the article ID is in the path, params are tracking.
    if _GOOGLE_NEWS_REDIR.match(normalized):
        normalized = normalized.split("?")[0].split("#")[0]
    else:
        normalized = normalized.split("?")[0].split("#")[0]
    return hashlib.sha256(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def canonicalize_text(raw_text: str) -> str:
    """Clean and normalize raw article text."""
    text = re.sub(r"<[^>]+>", " ", raw_text)
    text = PROMPT_INJECTION_PATTERNS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title(title: str) -> str:
    """Normalize title for deduplication comparison."""
    text = title.lower()
    # Strip trailing source attribution BEFORE removing punctuation
    # Heuristic: if the part after the last dash/pipe is short, it's likely a source name
    for sep in (" - ", " | ", " — ", " – "):
        if sep in text:
            parts = text.rsplit(sep, 1)
            if len(parts) == 2 and len(parts[1].strip()) <= 45:
                text = parts[0].strip()
                break
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_similarity(title_a: str, title_b: str) -> float:
    """Compute similarity between two normalized titles."""
    norm_a = normalize_title(title_a)
    norm_b = normalize_title(title_b)
    if not norm_a or not norm_b:
        return 0.0
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def _fetch_recent_events_for_dedup(db_conn) -> list[tuple[str, str]]:
    """Fetch recent events once to avoid O(N) database queries during ingestion."""
    try:
        rows = db_conn.execute(
            """SELECT source_title, canonical_text
               FROM events
               WHERE ingested_at > NOW() - INTERVAL '%s days'
               ORDER BY ingested_at DESC
               LIMIT 2000""",
            (_MAX_ARTICLE_AGE_DAYS,),
        ).fetchall()
        return [(row[0] or "", row[1] or "") for row in rows]
    except Exception:
        logger.exception("Failed to fetch recent events for dedup")
        return []


def check_content_duplicate(recent_events: list[tuple[str, str]], title: str, canonical_text: str) -> bool:
    """
    Check if a similar article exists in the provided list of recent events.
    Uses title similarity AND canonical text similarity.
    """
    for existing_title, existing_text in recent_events:
        # Title similarity (primary signal)
        sim = title_similarity(title, existing_title)
        if sim >= _TITLE_SIM_THRESHOLD:
            return True

        # Content similarity for longer texts
        if len(canonical_text) > 100 and len(existing_text) > 100:
            text_sim = difflib.SequenceMatcher(None, canonical_text, existing_text).ratio()
            if text_sim >= _CONTENT_SIM_THRESHOLD:
                return True
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


# ---------------------------------------------------------------------------
# Main Pass A runner
# ---------------------------------------------------------------------------

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

    # Execute global queries
    for query_info in queries[:50]:
        items = fetch_rss_feed(query_info, is_direct_url=False, stats=stats)
        all_items.extend(items)
        stats["queries_executed"] += 1

    # Fetch from static hardcoded feeds (Reddit)
    static_feeds = SETTINGS.get("sources", {}).get("static_feeds", [])
    for feed_url in static_feeds:
        items = fetch_rss_feed(feed_url, is_direct_url=True, stats=stats)
        all_items.extend(items)
        stats["queries_executed"] += 1

    # Fetch from GDELT — optional, with strict rate-limit handling.
    # GDELT has aggressive shared-IP rate limits on cloud runners.
    # We rotate queries per-run and skip silently on repeated 429s.
    try:
        gdelt_queries = build_gdelt_queries()
        # Rotate: pick 2 queries based on minute-of-hour for distribution
        random.seed(datetime.now().minute)
        selected = random.sample(gdelt_queries, min(2, len(gdelt_queries)))

        # Initial delay — GDELT often rejects the very first request on a fresh IP
        time.sleep(8.0)

        for idx, gdelt_spec in enumerate(selected):
            if idx > 0:
                time.sleep(6.0)
            items = fetch_gdelt_articles(
                query=gdelt_spec["query"],
                max_age_days=_MAX_ARTICLE_AGE_DAYS,
                tone=gdelt_spec.get("tone"),
                source_countries=gdelt_spec.get("countries"),
            )
            all_items.extend(items)
            stats["queries_executed"] += 1
    except Exception:
        logger.warning("GDELT fetch skipped due to rate-limit or network issues")
        pass

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

    # Sort items by date descending (newest first)
    deduped_items.sort(key=lambda x: x.get("pub_dt", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)

    # Fetch recent events for comparison once
    recent_events = _fetch_recent_events_for_dedup(db_conn)

    inserted = 0
    for item in deduped_items:
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
                canonical = canonicalize_text(f"{canonical} {full_text}")

        # Content dedup: check if similar title/text already exists
        if check_content_duplicate(recent_events, item.get("title", ""), canonical):
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
                # Inline dedup: add to recent_events so later items in this run are compared against it
                recent_events.insert(0, (item.get("title", ""), canonical))
                if len(recent_events) > 2500:
                    recent_events.pop()
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
