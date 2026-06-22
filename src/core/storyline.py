"""
SIM — Storyline Matching
Blueprint V20.1 §PASS D

Bigram-enhanced Jaccard similarity for linking related aviation events.
"""

import re
from typing import Set

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


# Date-hint tokens (e.g. "jun8", "may15", bare years/days) are REQUIRED in the LLM
# storyline_hint format but distort Jaccard: the same event reported across two days
# gets different date tokens (lower sim), while two different events on the same day
# share one (inflated sim). They are stripped from the similarity signal — time is
# handled separately by the occurred_at window. Flight numbers ("dl54") are kept.
_DATE_TOKEN = re.compile(
    r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:\d{0,2}|unknown|tbd)$|^\d{1,4}$"
)


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
        within_window = abs((dt_a - dt_b).days) <= max_days
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
    if similarity > threshold:
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

    return False
