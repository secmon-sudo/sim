"""
SIM — Storyline Matching
Blueprint V20.1 §PASS D

Bigram-enhanced Jaccard similarity for linking related aviation events.
"""

import re
from functools import lru_cache
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
    "shooter", "shooters",
    # Classifier placeholders that leak into hints when it cannot name the place
    # ("location mass shooting", run #24) — the opposite of discriminating.
    "location", "unknown",
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


# Cached because linking asks the same question thousands of times per run: every
# scored event is compared against the whole candidate pool, and the pool now carries
# every member of every storyline rather than one representative each (see
# _fetch_recent_events_for_linking). The hints repeat exactly — that is the entire
# point of a storyline — so this turns all but the first tokenization of a given
# string into a dict lookup. Same reasoning as _place_keys_cached in core.geo.
@lru_cache(maxsize=16384)
def _tokenize_storyline_hint_cached(text: str) -> frozenset:
    clean = re.sub(r"[^\w\s]", "", text.lower())
    tokens = [
        t for t in clean.split()
        if t not in AVIATION_STOPWORDS and not _DATE_TOKEN.match(t)
    ]
    unigrams = set(tokens)
    bigrams = {f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)}
    return frozenset(unigrams | bigrams)


def tokenize_storyline_hint(text: str) -> Set[str]:
    """
    Bigram-enhanced tokenization.
    Example: "runway incursion CAI" → {"runway", "incursion", "cai",
                                        "runway incursion", "incursion cai"}
    """
    return set(_tokenize_storyline_hint_cached(text))


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


def _unigrams(text: str) -> Set[str]:
    """Single tokens only — the bigrams `tokenize_storyline_hint` adds are an
    amplifier for Jaccard and noise for containment: inserting one word ("idaho
    in-n-out BURGER shooting") rewrites every bigram that touches it."""
    return {t for t in tokenize_storyline_hint(text) if " " not in t}


def containment_similarity(hint_a: str, hint_b: str) -> float:
    """Shared words as a fraction of the SHORTER hint.

    Jaccard penalizes a hint for being more specific, which is the wrong reflex
    when two sources name the same incident at different zoom levels: "twin
    falls in-n-out shooting" and "idaho in-n-out shooting" are one event (town vs
    state) and score 0.33 — below any threshold that also keeps distinct
    incidents apart. Containment asks the question that actually applies: is the
    shorter hint essentially *inside* the longer one (0.67 for that pair).
    """
    set_a, set_b = _unigrams(hint_a), _unigrams(hint_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


# Country and nationality words. Inside this comparison they are worth nothing:
# linking already REQUIRES the two events to share a country, so every candidate
# pair agrees on them by construction. Measured against two days of production
# hints, treating them as evidence merged "kyiv russian airstrike" with "kherson
# region russian airstrike" (different cities) and "us iran talks" with "us
# middle east travel advisory iran". Only the containment path consults this —
# the Jaccard path needs the words to stay in its denominator.
_COUNTRY_LEVEL_TOKENS = {
    "afghan", "afghanistan", "american", "chinese", "china", "colombia",
    "colombian", "egypt", "egyptian", "gaza", "india", "indian", "iran",
    "iranian", "iraq", "iraqi", "israel", "israeli", "lebanese", "lebanon",
    "libya", "libyan", "mexican", "mexico", "myanmar", "nigeria", "nigerian",
    "pakistan", "pakistani", "palestinian", "poland", "polish", "russia",
    "russian", "saudi", "somali", "somalia", "sudan", "sudanese", "syria",
    "syrian", "turkey", "turkish", "ukraine", "ukrainian", "us", "usa",
    "yemen", "yemeni",
}

# Words that name a scale rather than a place or an actor.
_SCALE_TOKENS = {"region", "regions", "city", "country", "national", "forces",
                 "area", "areas", "province", "district", "state", "town"}

# Incident and diplomacy vocabulary the singular security list misses — plurals,
# and the nouns that name a KIND of story ("deal", "talks", "brush fire").
_CATEGORY_TOKENS = {
    "strikes", "attacks", "bombings", "explosions", "drones", "killings",
    "blasts", "fires", "fire", "brush", "wildfire", "crackdown", "deal",
    "talks", "ceasefire", "advisory", "sanctions",
}

_UNINFORMATIVE_FOR_CONTAINMENT = (
    GENERIC_INCIDENT_TOKENS | _COUNTRY_LEVEL_TOKENS | _SCALE_TOKENS
    | _CATEGORY_TOKENS
)


def overexposed_tokens(hints, min_storylines: int = 3) -> Set[str]:
    """Words that recur across many separate storylines in the recent corpus.

    The curated lists above cannot keep up with what a given week is about: on
    2-3 Aug 2026 "idf", "kyiv" and "swat" each named a dozen storylines, so
    sharing one of them says only "same conflict", not "same incident". Counted
    per STORYLINE, not per event, so one heavily-covered incident cannot make its
    own distinctive words look generic.

    `hints` is an iterable of (storyline_id, hint) pairs.
    """
    seen: dict[str, set] = {}
    for sid, hint in hints:
        if not sid or not hint:
            continue
        for token in _unigrams(hint):
            seen.setdefault(token, set()).add(sid)
    return {t for t, sids in seen.items() if len(sids) >= min_storylines}


# One shared word is a coincidence — "gaza airstrike" and "gaza protest" share
# "gaza" and are two different events. Containment only speaks when the hints
# agree on at least this many words.
CONTAINMENT_MIN_SHARED_WORDS = 2


def is_specificity_variant(hint_a: str, hint_b: str, threshold: float = 0.5,
                           common_tokens: Set[str] = frozenset()) -> bool:
    """Whether two hints look like the same incident described at different
    levels of detail, rather than two incidents of the same kind.

    Requires the overlap to carry something that identifies WHICH incident this
    is — a town, a venue, a named actor. Country words do not qualify (the
    country is already a precondition for linking), and neither do words the
    current corpus has made ambient (`common_tokens`, see overexposed_tokens).
    """
    shared = _unigrams(hint_a) & _unigrams(hint_b)
    if len(shared) < CONTAINMENT_MIN_SHARED_WORDS:
        return False
    if not any(word not in _UNINFORMATIVE_FOR_CONTAINMENT and word not in common_tokens
               for word in shared):
        return False
    return containment_similarity(hint_a, hint_b) >= threshold


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
    containment_threshold: float = 0.5,
    containment_max_hours: float = 72.0,
    common_tokens: Set[str] = frozenset(),
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
    containment_threshold / containment_max_hours:
      Rescue path for hints that name the same incident at different zoom levels
      (town vs state, venue vs city). Deliberately time-boxed: two distinct
      incidents at one place can have exactly this shape when they are months
      apart.
    common_tokens:
      Words the recent corpus has made ambient (see overexposed_tokens); the
      containment path refuses to treat them as identifying. Empty by default,
      which only makes the path more permissive, never less.
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

    # ── Specificity-tolerant containment (zero-LLM) ──
    # Run #24 (2 Aug 2026) carried the Twin Falls In-N-Out shooting as TWO
    # storylines — "twin falls in-n-out shooting" and "idaho in-n-out shooting" —
    # each with its own sources, its own verification label and its own airspace
    # card. Jaccard saw 0.33 because one hint names the town and the other the
    # state; containment sees 0.6. The discriminating-overlap gate still applies,
    # so a pair whose whole overlap is "mass shooting" does not qualify here
    # either, and the tight window keeps two distinct incidents at the same place
    # weeks apart from collapsing.
    if (
        abs((dt_a - dt_b).total_seconds()) / 3600.0 <= containment_max_hours
        and is_specificity_variant(
            event_a.get("storyline_hint") or "",
            event_b.get("storyline_hint") or "",
            containment_threshold,
            common_tokens,
        )
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
