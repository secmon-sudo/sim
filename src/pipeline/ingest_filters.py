"""
SIM — Pass A ingest: text filters & dedup primitives (pure, no network/DB)

Noise/keyword filtering, canonicalization, URL/domain helpers and the
title/content similarity machinery used for ingest-time dedup.
Split out of pass_a_ingest.py on 2026-07-16 (was a 1.9K-line monolith).
"""

import difflib
import hashlib
import json
import logging
import re
from pathlib import Path

import tldextract

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
with open(_CONFIG_DIR / "keywords.json", encoding="utf-8") as f:
    KEYWORDS_CONFIG = json.load(f)
with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as f:
    SETTINGS = json.load(f)

_DEDUP = SETTINGS.get("dedup", {})
_TITLE_SIM_THRESHOLD = _DEDUP.get("title_similarity_threshold", 0.78)
_TITLE_TOKEN_THRESHOLD = _DEDUP.get("title_token_jaccard_threshold", 0.72)
_CONTENT_SHINGLE_THRESHOLD = _DEDUP.get("content_shingle_threshold", 0.40)

# Prompt injection patterns to strip before LLM classification
PROMPT_INJECTION_PATTERNS = re.compile(
    r"\[INST\]|<\|system\|>|<\|user\|>|<\|assistant\|>|IGNORE PREVIOUS INSTRUCTIONS|"
    r"FORGET ALL PRIOR|YOU ARE NOW|SYSTEM OVERRIDE",
    re.IGNORECASE,
)


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

