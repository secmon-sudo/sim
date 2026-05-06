"""
SIM — Storyline Matching
Blueprint V20.1 §PASS D

Bigram-enhanced Jaccard similarity for linking related aviation events.
"""

import re
from typing import Set

# Context-independent words that dilute Jaccard signal in aviation text
AVIATION_STOPWORDS = {
    "the", "a", "an", "at", "in", "on", "of", "to", "and", "or",
    "airport", "terminal", "flight", "gate", "apron",
}


def tokenize_storyline_hint(text: str) -> Set[str]:
    """
    Bigram-enhanced tokenization.
    Example: "runway incursion CAI" → {"runway", "incursion", "cai",
                                        "runway incursion", "incursion cai"}
    """
    clean = re.sub(r"[^\w\s]", "", text.lower())
    tokens = [t for t in clean.split() if t not in AVIATION_STOPWORDS]
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


def should_link_storyline(event_a: dict, event_b: dict, threshold: float = 0.4, max_days: int = 14) -> bool:
    """True only when ALL three conditions hold."""
    similarity = jaccard_similarity(
        event_a.get("storyline_hint") or "",
        event_b.get("storyline_hint") or "",
    )
    same_country = event_a.get("country_iso") == event_b.get("country_iso")
    within_window = (
        abs((event_a["occurred_at_est"] - event_b["occurred_at_est"]).days)
        <= max_days
    )
    return similarity > threshold and same_country and within_window
