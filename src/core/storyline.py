"""
SIM — Storyline Matching
Blueprint V20.1 §PASS D

Bigram-enhanced Jaccard similarity for linking related aviation events.
"""

import re
from typing import Set

from src.core.geo import geo_key

# Context-independent words and generic incident types that dilute Jaccard signal
AVIATION_STOPWORDS = {
    # Common English stopwords
    "the", "a", "an", "at", "in", "on", "of", "to", "and", "or",
    # Aviation generic terms
    "airport", "terminal", "flight", "gate", "apron",
    "emergency", "landing", "bomb", "threat", "crash", "incident",
    "attack", "plane", "aircraft", "passenger", "crew", "pilot",
    "drone", "laser", "evacuation", "security", "issue", "small",
    # News/media generic terms that dilute similarity signal
    "report", "reports", "breaking", "news", "update", "source",
    "military", "strike", "killed", "dead", "injured",
    "new", "latest", "just", "now", "says", "official",
    "according", "confirmed", "reported", "sources",
}

# Incident-type vocabulary: words that say WHAT KIND of event this is, never
# WHICH event it is. They are NOT removed from the similarity signal — doing that
# shrinks hints so far that "Philippines high school shooting" and "Philippines
# school shooting" stop matching. They are used as a gate instead: a lexical link
# whose entire overlap is drawn from this list is not evidence of the same
# incident. On 1 Aug 2026 "seattle mass shooting" and "breckenridge mass
# shooting" scored 0.43 on nothing but {mass, shooting} and were merged, which
# then credited an unrelated Arkansas report as Seattle's corroborating source.
GENERIC_INCIDENT_TOKENS = {
    "shooting", "shootings", "shot", "gunman", "gunfire", "stabbing",
    "mass", "casualties", "wounded", "victim", "victims", "fatal", "dead",
    "missile", "missiles", "rocket", "rockets", "shelling", "airstrike",
    "airstrikes", "blast", "explosion", "bombing", "raid", "raids",
    "terror", "terrorist", "terrorists", "terrorism", "militant", "militants",
    "protest", "protests", "clash", "clashes", "unrest", "riot",
    "police", "arrested", "suspect", "suspects", "man", "woman", "people",
    "video", "footage", "war", "conflict", "operation", "school", "festival",
}


# Date-hint tokens (e.g. "jun8", "may15", bare years/days) are REQUIRED in the LLM
# storyline_hint format but distort Jaccard: the same event reported across two days
# gets different date tokens (lower sim), while two different events on the same day
# share one (inflated sim). They are stripped from the similarity signal — time is
# handled separately by the occurred_at window. Flight numbers ("dl54") are kept.
_DATE_TOKEN = re.compile(
    r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:\d{0,2}|unknown|tbd)$|^\d{1,4}$"
)

# In-text variant of _DATE_TOKEN (word-boundary, month+suffix only — bare numbers
# like flight "54" are left alone). Used to scrub the fabricated date hints older
# classifications baked into storyline_hint before showing them to users.
_DATE_HINT_IN_TEXT = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:\d{1,2}|unknown|tbd)\b",
    re.IGNORECASE,
)


def strip_date_hint(text: str) -> str:
    """Remove LLM date-hint tokens ("Jun8", "nov20", "JunUnknown") from a
    storyline hint. The pre-2026-07-09 classifier prompt REQUIRED a MonDD token
    and the model fabricated one when the article stated no date; the token was
    never part of the matching signal (see _DATE_TOKEN above), so dropping it is
    purely cosmetic-safe."""
    return re.sub(r"\s+", " ", _DATE_HINT_IN_TEXT.sub("", text)).strip()


def tokenize_storyline_hint(text: str) -> Set[str]:
    """
    Bigram-enhanced tokenization.
    Example: "runway incursion CAI" → {"runway", "incursion", "cai",
                                        "runway incursion", "incursion cai"}
    """
    clean = re.sub(r"[^\w\s]", "", text.lower())
    tokens = [
        t for t in clean.split()
        if t not in AVIATION_STOPWORDS and not _DATE_TOKEN.match(t)
    ]
    unigrams = set(tokens)
    bigrams = {f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)}
    return unigrams | bigrams


def jaccard_similarity(hint_a: str, hint_b: str) -> float:
    """Compute Jaccard similarity between two storyline hints."""
    set_a = tokenize_storyline_hint(hint_a)
    set_b = tokenize_storyline_hint(hint_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def lexical_kinship(hint_a: str, hint_b: str) -> float:
    """Jaccard over the full vocabulary — incident words and all.

    Not a linking signal: `jaccard_similarity` drops the words that dilute the
    decision, which is right for deciding and wrong for RANKING adjudication
    candidates. "kashmir terrorist attack" and "kulgam terror attack" (the same
    incident, one named by region and one by town) share nothing a linking-grade
    score can see, so the true duplicate was dropped from the candidate list
    before the model ever saw it. Ranking is not deciding — the LLM still has to
    call it the same incident.
    """
    def _tokens(text: str) -> Set[str]:
        return {t for t in re.sub(r"[^\w\s]", "", text.lower()).split()
                if t not in _FUNCTION_WORDS and not _DATE_TOKEN.match(t)}

    set_a, set_b = _tokens(hint_a), _tokens(hint_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


_FUNCTION_WORDS = {"the", "a", "an", "at", "in", "on", "of", "to", "and", "or"}


def has_discriminating_overlap(hint_a: str, hint_b: str) -> bool:
    """Whether two hints share anything that identifies WHICH incident this is.

    A token (or bigram) counts only if it is not drawn entirely from
    GENERIC_INCIDENT_TOKENS: places, actors, named entities and numbers identify
    an incident; "mass shooting" describes a category that thousands of distinct
    incidents belong to.
    """
    shared = tokenize_storyline_hint(hint_a) & tokenize_storyline_hint(hint_b)
    return any(
        any(word not in GENERIC_INCIDENT_TOKENS for word in token.split())
        for token in shared
    )


def should_link_storyline(
    event_a: dict,
    event_b: dict,
    threshold: float = 0.4,
    max_days: int = 14,
    country_match_required: bool = True,
    anchor_assist_threshold: float = 0.2,
    anchor_assist_max_hours: float = 72.0,
) -> bool:
    """Decide whether two events belong to the same storyline.

    Requires (time window) AND (country, when required) AND
    (lexical similarity OR shared-anchor identity). Defaults mirror
    config/settings.json -> storyline.* so callers that pass no overrides behave
    the same as a config-driven call.

    country_match_required:
      - True  (default): if BOTH events have a country_iso they must be equal;
        a missing iso on either side stays lenient (still allowed).
      - False: country is ignored entirely.
    anchor_assist_threshold:
      Minimum lexical similarity for the shared-anchor rescue path when the two
      events are far apart in time (links paraphrased same-location reports that
      fall below the main threshold).
    anchor_assist_max_hours:
      Within this tight window, a shared anchor alone links the events regardless
      of wording (same place + same time ≈ same developing story). Beyond it, the
      anchor path additionally requires anchor_assist_threshold lexical overlap so
      two DISTINCT incidents at the same location aren't merged.
    """
    # ── Time gate (hard) — guard against None datetimes ──
    dt_a = event_a.get("occurred_at_est")
    dt_b = event_b.get("occurred_at_est")
    if dt_a is None or dt_b is None:
        return False
    try:
        # Use total_seconds, not .days: timedelta.days truncates toward -inf, so
        # the window was asymmetric depending on which event came first
        # (e.g. -14.5 days → -15 → excluded, +14.5 days → 14 → included).
        within_window = abs((dt_a - dt_b).total_seconds()) <= max_days * 86400
    except Exception:
        return False
    if not within_window:
        return False

    # ── Country gate (hard when required) ──
    iso_a = event_a.get("country_iso")
    iso_b = event_b.get("country_iso")
    if country_match_required and iso_a and iso_b and iso_a != iso_b:
        return False

    # ── Lexical similarity ──
    similarity = jaccard_similarity(
        event_a.get("storyline_hint") or "",
        event_b.get("storyline_hint") or "",
    )
    # A high score built purely out of incident-type words ("mass shooting",
    # "missile strike") says the two reports are the same KIND of event, not the
    # same event. Those pairs are handed to the LLM adjudicator instead of being
    # merged for free.
    if similarity > threshold and has_discriminating_overlap(
        event_a.get("storyline_hint") or "", event_b.get("storyline_hint") or ""
    ):
        return True

    # ── Hybrid anchor-assist (zero-LLM) ──
    # Two paraphrased reports of the SAME incident often share the SAME physical
    # location (airport/base IATA) even when their hints word differently and fall
    # below the lexical threshold. A matching normalized anchor + a minimum lexical
    # overlap rescues these links — solving cross-source paraphrase duplication that
    # pure Jaccard misses. The overlap floor prevents merging two DISTINCT incidents
    # that merely happened at the same place.
    anchor_a = (event_a.get("anchor_name_norm") or "").strip().upper()
    anchor_b = (event_b.get("anchor_name_norm") or "").strip().upper()
    if anchor_a and anchor_a == anchor_b:
        try:
            hours_apart = abs((dt_a - dt_b).total_seconds()) / 3600.0
        except Exception:
            hours_apart = float("inf")
        # Same place + same time → same story regardless of wording.
        if hours_apart <= anchor_assist_max_hours:
            return True
        # Same place, far apart in time → require some lexical overlap.
        if similarity >= anchor_assist_threshold:
            return True

    # ── Coarse geo-assist (city-level, for events without a shared IATA anchor) ──
    # Most Russia–Ukraine / Middle-East volume is city-level and never resolves to an
    # airport IATA, so the anchor path above never fires for it. Fall back to a coarse,
    # paraphrase-stable geo_key (e.g. "Kyiv"/"Kiev"/"Ukraine capital" → KYIV) and link
    # when the SAME place shows a minimum lexical overlap. Unlike the IATA path there is
    # deliberately NO pure-time auto-link: a city is coarser than a specific airport, so
    # lexical support is always required to avoid merging two DISTINCT same-city events.
    # Zero-overlap same-city candidates are left for the LLM adjudicator (Layer 2).
    if not (anchor_a and anchor_a == anchor_b):
        geo_a = geo_key(event_a.get("anchor_name_raw"), iso_a)
        geo_b = geo_key(event_b.get("anchor_name_raw"), iso_b)
        if geo_a and geo_a == geo_b and similarity >= anchor_assist_threshold:
            return True

    return False
