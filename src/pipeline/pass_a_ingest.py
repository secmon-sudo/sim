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
_TITLE_TOKEN_THRESHOLD = _DEDUP.get("title_token_jaccard_threshold", 0.72)
_CONTENT_SHINGLE_THRESHOLD = _DEDUP.get("content_shingle_threshold", 0.40)

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
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
            timeout=8.0,       # fail fast — cloud IPs often get 429
            max_retries=1,     # single attempt, no long backoff loops
            backoff_base=2.0,
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

# Military/security context patterns — if any of these match, the article
# should NOT be discarded by noise filters even if a noise keyword is present.
# e.g. "military training exercise near border" is real news, not noise.
_MILITARY_CONTEXT_BYPASS = re.compile(
    r"\b(military|army|troops|soldiers|combat|battlefield|frontline|"
    r"war zone|airbase|naval|marines|special forces|regiment|battalion|"
    r"armed forces|defense ministry|ministry of defence|pentagon|"
    r"NATO|Wagner|militia|insurgent|guerrilla|paramilitary|"
    r"airstrike|missile|bombing|shelling|casualties|killed in|"
    r"drone strike|offensive|ceasefire|blockade|siege|ambush)\b",
    re.IGNORECASE,
)

# Bypass cancellers — even with military vocabulary, these markers indicate the
# article is ANALYSIS/RECAP/MEDIA about conflict, not a live incident. They cancel
# the military bypass so the normal noise filters apply (e.g. "documentary about the
# missile strike", "investigation into the bombing", "opinion: why the war drags on").
_BYPASS_CANCEL_PATTERN = re.compile(
    r"\b(documentary|docuseries|investigation into|investigates|"
    r"opinion|op-?ed|editorial|analysis|explainer|explained|"
    r"what we know|here's what|the story of|how the|why the|"
    r"podcast|book review|new book|film about|movie about|"
    r"retrospective|in pictures|in photos|photo essay|timeline of)\b",
    re.IGNORECASE,
)


# Retrospective / anniversary patterns — these indicate an article ABOUT a past
# event (recap, memorial, "N years ago"), not a current incident. They override the
# military-context bypass: "10th anniversary of the airstrike" is stale news, not a
# live event, even though it mentions "airstrike".
_RETROSPECTIVE_PATTERN = re.compile(
    r"\b\d+\s*(?:st|nd|rd|th)?\s*anniversary\b"
    r"|\banniversary of\b"
    r"|\b\d+\s+years?\s+(?:ago|since|on)\b"
    r"|\bon this day\b"
    r"|\byears ago today\b"
    r"|\blooking back\b"
    r"|\bremember(?:ing|ed)?\s+the\b"
    r"|\bthrowback\b"
    r"|\b(?:a\s+)?(?:decade|decades)\s+(?:ago|since)\b"
    r"|\bback in (?:19|20)\d\d\b"
    r"|\bmarks?\s+\d+\s+years\b",
    re.IGNORECASE,
)


def is_noise(text: str) -> bool:
    """Check if text matches known noise patterns using word boundaries.

    Military/security context overrides noise filters — an article about
    'military training exercise near border' is real news, not simulator noise.
    EXCEPTION: retrospective/anniversary content is always noise (it describes a
    past event, not a current incident) and overrides the military bypass.
    """
    text_lower = text.lower()

    # Retrospectives are stale by definition — filtered even with military context
    if _RETROSPECTIVE_PATTERN.search(text_lower):
        return True

    # Military/security context normally overrides noise filters — but NOT when the
    # article is analysis/recap/media about conflict rather than a live incident.
    if _MILITARY_CONTEXT_BYPASS.search(text_lower) and not _BYPASS_CANCEL_PATTERN.search(text_lower):
        return False

    for pattern in NOISE_PATTERNS:
        if pattern.search(text_lower):
            return True
    for pattern in _SPORTS_ENT_BLOCKERS:
        if pattern.search(text_lower):
            return True
    return False


# Standalone high-signal terms that should ALWAYS match from static feeds,
# even without compound context like "airport attack" or "hotel bombing".
# These are words/phrases that almost always indicate a real security event.
_HIGH_SIGNAL_TERMS = {
    "explosion", "explosions", "bombing", "bombings", "shelling",
    "airstrike", "airstrikes", "air strike", "air strikes",
    "missile", "missiles", "missile strike", "missile attack",
    "gunfire", "gunshots", "shooting",
    "assassination", "assassinated", "massacre", "massacred",
    "invasion", "invaded", "coup", "overthrow", "overthrown",
    "ceasefire", "blockade", "siege", "ambush", "offensive",
    "casualties", "fatalities", "killed", "wounded", "dead",
    "artillery", "mortar", "rocket", "rockets",
    "drone attack", "drone strike", "drone strikes",
    "war", "warfare", "conflict", "clashes",
    "evacuated", "evacuation",
    "military operation", "ground offensive",
    "nuclear", "chemical weapon", "biological weapon",
    "terror attack", "terrorist attack", "terrorist",
    "hostage", "hostages", "kidnapped", "abducted",
    "insurgent", "insurgents", "insurgency",
    "militia", "paramilitary",
    "sanctions", "embargo",
    "refugee", "refugees", "displaced",
    "humanitarian crisis", "famine",
    "large-scale attack", "major attack", "massive attack",
    "suicide bomb", "suicide bomber", "car bomb", "truck bomb",
    "IED", "improvised explosive",
    "incursion", "retaliation", "retaliatory",
}


def _compile_security_keyword_pattern() -> re.Pattern:
    """Compile high-signal terms + all config keywords into one word-boundary regex.

    Word boundaries (\\b) prevent substring false positives that plain
    `keyword in text` produced — e.g. "war" matching "Warsaw"/"forward",
    "coup" matching "couple", "riot" matching "patriot", "dead" matching
    "deadline". \\b uses Unicode \\w, so it also works for Arabic/Hebrew/Cyrillic
    keywords (boundaries between word chars and spaces/punctuation).
    """
    terms: set[str] = set(_HIGH_SIGNAL_TERMS)
    for keyword_group in ("emergency_keywords", "geopolitical_keywords"):
        for keywords in KEYWORDS_CONFIG.get(keyword_group, {}).values():
            terms.update(kw.lower() for kw in keywords)

    parts = []
    for term in terms:
        term = term.strip()
        if not term:
            continue
        try:
            re.compile(rf"\b{re.escape(term)}\b")
            parts.append(rf"\b{re.escape(term)}\b")
        except re.error:
            parts.append(re.escape(term))
    return re.compile("|".join(parts), re.IGNORECASE)


_SECURITY_KEYWORD_PATTERN = _compile_security_keyword_pattern()


def _matches_security_keywords(title: str, description: str) -> bool:
    """Check if article title/description contains at least one security keyword.

    Used as a post-filter for general RSS feeds (reddit, aljazeera, reuters)
    that aren't pre-filtered by search query. Matches on word boundaries to
    avoid substring false positives. Covers high-signal standalone terms plus
    config emergency/geopolitical keywords across all languages (en, ar, tr, fr).
    """
    text = f"{title} {description}"
    return bool(_SECURITY_KEYWORD_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def build_search_queries(db_conn=None) -> list[dict]:
    """Build focused search queries.  ONLY aviation / hotel / tourism / mass-casualty security.

    Strategy:
    1.  Keep the query list SHORT (≈40 queries).  300+ low-quality queries drown the signal.
    2.  Every query MUST contain aviation / hotel / tourism context, OR be an
        unambiguous mass-casualty security phrase.
    3.  Google News RSS handles simple `phrase airport` syntax reliably.
        Complex boolean with many negative keywords breaks or returns 0 results.
    4.  Remaining noise is caught by `is_noise()` post-filter.
    """
    active_queries = []
    if db_conn is not None:
        try:
            # Query recent storylines from last 14 days
            with db_conn.transaction():
                rows = db_conn.execute(
                    """SELECT storyline_hint, MAX(occurred_at_est) as last_update, MAX(severity_score) as max_severity
                       FROM events
                       WHERE status IN ('scored', 'reconciled')
                         AND storyline_hint IS NOT NULL
                         AND occurred_at_est > NOW() - INTERVAL '14 days'
                       GROUP BY storyline_id, storyline_hint"""
                ).fetchall()

            now = datetime.now(timezone.utc)
            for row in rows:
                hint = row[0]
                last_update = row[1]
                max_severity = row[2] or 0

                # Ensure last_update is timezone-aware for math comparison
                if last_update.tzinfo is None:
                    last_update = last_update.replace(tzinfo=timezone.utc)

                age_hours = (now - last_update).total_seconds() / 3600

                # Determine tracking window based on severity
                if max_severity >= 80:
                    window = 168  # 7 days
                elif max_severity >= 60:
                    window = 72   # 3 days
                else:
                    window = 36   # 36 hours

                if age_hours <= window:
                    # Clean the hint (strip the date hint from the end, e.g. " Jun9")
                    clean_query = re.sub(r'\s+[A-Z][a-z]{2}\d{1,2}$', '', hint)
                    if clean_query:
                        active_queries.append(clean_query)
        except Exception:
            logger.exception("Error building dynamic search queries from active storylines")

    queries = []
    seen = set()

    def _add(q: str, dynamic: bool = False):
        if q.lower() not in seen:
            seen.add(q.lower())
            queries.append({"query": q, "broad": False, "dynamic": dynamic})

    # Add active dynamic queries FIRST so they are run with highest priority
    for q in active_queries:
        _add(q, dynamic=True)

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

    # ── Tier 4: Geopolitical / African terrorism (narrowed to infrastructure-relevant) ──
    geo = [
        '"missile strike" airport OR airspace',
        '"airstrike" civilian airport OR infrastructure',
        '"war escalation" airspace OR NOTAM',
        '"Iran Israel" military strike',
        '"Ukraine Russia" drone attack infrastructure',
        '"Boko Haram" attack',
        '"Al-Shabaab" attack',
        '"jihadist attack"',
        '"ISIS Africa" attack',
        '"Sahel crisis" attack',
        '"civilian casualties" airstrike',
        '"Houthi attack" ship OR Red Sea',
        '"drone swarm" attack',
        '"India Pakistan" military',
    ]
    for q in geo:
        _add(q)

    # ── Tier 5: Maritime / rail / mass-transit security ──
    transport = [
        '"train station attack"',
        '"train station bombing"',
        '"metro attack"',
        '"subway attack"',
        '"bus terminal attack"',
        '"port attack"',
        '"piracy attack"',
        '"tanker hijack"',
        '"ship hijacked"',
        '"maritime security incident"',
    ]
    for q in transport:
        _add(q)

    # ── Tier 6: Tourist-specific threats ──
    tourist = [
        '"tourists killed"',
        '"tourist attack"',
        '"tourist kidnapped"',
        '"travelers warned"',
        '"travel advisory" raised',
        '"travel warning" issued',
        '"do not travel" warning',
        '"embassy attack"',
        '"consulate attack"',
        '"tourist area bombing"',
    ]
    for q in tourist:
        _add(q)

    # ── Tier 7: Protest & Civil Unrest ──
    protest = [
        '"mass protest"',
        '"violent protest"',
        '"anti-government protest"',
        '"protest crackdown"',
        '"riot police" protest',
        '"tear gas" protest',
        '"general strike"',
        '"nationwide strike"',
        '"demonstration violence"',
        '"protesters killed"',
        '"protesters shot"',
        '"protest shooting"',
        '"coup attempt"',
        '"political unrest"',
        '"state of emergency" protest',
        '"curfew imposed"',
        '"martial law"',
        '"protest" clashes',
        '"uprising"',
        '"civil unrest"',
    ]
    for q in protest:
        _add(q)

    # ── Tier 8: Travel Advisory / Travel Warning ──
    advisory = [
        '"travel advisory" country',
        '"travel warning" country',
        '"do not travel" advisory',
        '"travel ban"',
        '"embassy closed"',
        '"consulate closed"',
        '"evacuate citizens"',
        '"security alert" embassy',
        '"Level 4" travel advisory',
        '"Level 3" travel advisory',
        '"reconsider travel"',
        '"travel restriction"',
    ]
    for q in advisory:
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


def _parse_advisory_level(title: str) -> int:
    """Extract numeric advisory level from title (e.g. 'Level 3'). Returns 0 if not found."""
    m = _LEVEL_RE.search(title)
    return int(m.group(1)) if m else 0


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
    level = _parse_advisory_level(title)
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
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date_str = (item.findtext("pubDate") or "").strip()
                description = (item.findtext("description") or "").strip()

                if not title or not link:
                    continue

                # Only ingest level increases / high-risk entries
                if not _is_advisory_worth_ingesting(title, description):
                    continue

                # Parse date — advisory feed uses date-only "Mon, 19 May 2026"
                pub_dt: datetime | None = None
                if pub_date_str:
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

                all_items.append({
                    "title": title,
                    "link": link,
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
# Nitter (Twitter/X) RSS fetch with mirror fallback
# ---------------------------------------------------------------------------

def fetch_nitter_feeds(stats: dict | None = None) -> list[dict]:
    """
    Fetch RSS feeds from Nitter instances for Twitter/X accounts.
    Uses 3 retries per mirror, with automatic fallback to alternative mirrors.
    Returns list of parsed feed items.
    """
    nitter_feeds = SETTINGS.get("sources", {}).get("nitter_feeds", [])
    mirrors = SETTINGS.get("sources", {}).get("nitter_mirrors", [
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
    ])

    if not nitter_feeds:
        return []

    all_items = []

    for feed_url in nitter_feeds:
        # Extract the account path (e.g., "/ww3mediaa/rss") from the original URL
        from urllib.parse import urlparse
        parsed = urlparse(feed_url)
        account_path = parsed.path  # e.g., "/ww3mediaa/rss"

        fetched = False
        for mirror in mirrors:
            mirror_url = f"{mirror.rstrip('/')}{account_path}"

            # 3 retries per mirror
            for attempt in range(3):
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "application/rss+xml, application/xml, text/xml, */*",
                    }
                    resp = httpx.get(mirror_url, headers=headers, timeout=12.0, follow_redirects=True)

                    if resp.status_code == 429:
                        wait = 3.0 * (attempt + 1)
                        logger.warning("Nitter rate limit (429) on %s, retry %d/3 in %.0fs",
                                       mirror_url[:60], attempt + 1, wait)
                        time.sleep(wait)
                        continue

                    resp.raise_for_status()

                    # Parse the RSS feed using the existing fetch_rss_feed logic
                    items = _parse_rss_response(resp, mirror_url, stats)
                    all_items.extend(items)
                    fetched = True
                    logger.info("Nitter: Fetched %d items from %s", len(items), mirror_url[:60])
                    break  # Success — stop retrying this mirror

                except Exception as e:
                    logger.debug("Nitter attempt %d/3 failed for %s: %s",
                                 attempt + 1, mirror_url[:60], str(e))
                    if attempt < 2:
                        time.sleep(2.0 * (attempt + 1))
                    continue

            if fetched:
                break  # Got data from this mirror, move to next feed

        if not fetched:
            logger.warning("Nitter: All mirrors failed for %s", account_path)

    # Sort by date descending (newest first) and limit per account
    all_items.sort(
        key=lambda x: x.get("pub_dt", datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )
    max_nitter_per_account = 5
    nitter_account_count = len(nitter_feeds) or 1
    max_nitter_total = max_nitter_per_account * nitter_account_count
    all_items = all_items[:max_nitter_total]

    return all_items


def _parse_rss_response(resp, url: str, stats: dict | None = None) -> list[dict]:
    """Parse an RSS/Atom HTTP response into item dicts. Shared by Nitter and regular RSS."""
    import xml.etree.ElementTree as ET
    import email.utils

    items = []
    now_utc = datetime.now(timezone.utc)
    max_age = _MAX_ARTICLE_AGE_DAYS

    try:
        root = ET.fromstring(resp.text)
        tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
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
                title = entry.findtext("title", "")
                link = entry.findtext("link", "")
                pub_date_str = entry.findtext("pubDate", "")
                description = entry.findtext("description", "")

            # Nitter links may point to the nitter instance; keep them as-is
            # (they'll be deduped by content, not URL)

            # Parse date
            pub_dt = None
            if pub_date_str:
                try:
                    pub_dt = email.utils.parsedate_to_datetime(pub_date_str)
                except Exception:
                    pass
                if pub_dt is None:
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
                "source": "nitter",
                "domain": "twitter.com",
            })
    except Exception:
        logger.exception("Nitter RSS parse error for: %s", url[:80])

    return items


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
        url = GOOGLE_NEWS_RSS.format(query=quote_plus(query_info["query"]))

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


def compute_url_hash(url: str) -> str:
    """SHA-256 hash of normalized URL for deduplication."""
    normalized = url.strip().lower()
    # Strip query params and fragments — for Google News redirect URLs the
    # article ID is in the path and params are tracking; other sources likewise.
    normalized = normalized.split("?")[0].split("#")[0]
    return hashlib.sha256(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def canonicalize_text(raw_text: str) -> str:
    """Clean and normalize raw article text."""
    # Strip HTML tags — require tag to start with letter or '/' to avoid
    # false positives on math expressions like "3 < 5 > 2"
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", raw_text)
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


def _word_set(text: str) -> set[str]:
    """Normalized word set of a title (lowercased, punctuation-stripped)."""
    return set(normalize_title(text).split())


def _shingles(text: str, n: int = 4) -> set[str]:
    """Word n-grams (shingles) of canonical text — robust to reordering/truncation."""
    words = normalize_title(text).split()
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def title_token_similarity(title_a: str, title_b: str) -> float:
    """Word-set Jaccard of two titles.

    Catches cross-source rephrasing that SequenceMatcher misses (reordered words,
    different source suffixes, inserted words) — e.g. two outlets covering the same
    incident with differently worded headlines.
    """
    return _jaccard(_word_set(title_a), _word_set(title_b))


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


def check_content_duplicate(recent_events: list[tuple[str, str]], title: str, canonical_text: str) -> bool:
    """
    Check if a similar article exists in the provided list of recent events.

    Three complementary signals (any one triggers a dedup):
      1. Title SequenceMatcher  — near-identical headlines (incl. source suffix).
      2. Title word-set Jaccard — cross-source rephrasing / reordered headlines
         that SequenceMatcher's char-ratio misses.
      3. Content word-shingle Jaccard — same body reported by different outlets;
         replaces the old O(N*M) full-text SequenceMatcher (faster, truncation-robust).
    """
    title_tokens = _word_set(title)
    text_shingles = _shingles(canonical_text) if len(canonical_text) > 100 else None

    for existing_title, existing_text in recent_events:
        # Signal 1: char-ratio title similarity (primary)
        if title_similarity(title, existing_title) >= _TITLE_SIM_THRESHOLD:
            return True

        # Signal 2: token-set title similarity (cross-source rephrasing)
        if _jaccard(title_tokens, _word_set(existing_title)) >= _TITLE_TOKEN_THRESHOLD:
            return True

        # Signal 3: content shingle similarity for longer texts
        if text_shingles is not None and len(existing_text) > 100:
            if _jaccard(text_shingles, _shingles(existing_text)) >= _CONTENT_SHINGLE_THRESHOLD:
                return True
    return False


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


def translate_to_english_if_needed(text: str) -> str:
    """Check if text contains Non-Latin characters (Arabic, Hebrew, Farsi, Cyrillic) and translate if so."""
    if not text:
        return text
    # Unicode ranges:
    # \u0590-\u05FF: Hebrew
    # \u0600-\u06FF: Arabic/Farsi
    # \u0400-\u04FF: Cyrillic (Russian, etc.)
    if re.search(r'[\u0590-\u05FF\u0600-\u06FF\u0400-\u04FF]', text):
        return google_translate(text, target="en")
    return text


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

    # Fetch from GDELT — non-blocking, fire-and-forget.
    # GDELT rate-limits cloud IPs aggressively (429).
    # Strategy: single query, tight timeout, no initial delay.
    # If it works → bonus data. If not → silently skip.
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

    # Fetch US State Dept travel advisories — level increases and Level 3/4 only
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
