"""
SIM — Coarse Geo Key
Blueprint V20.1 §PASS D (storyline linking support)

The airport `anchor_master` gazetteer is IATA-centric, so city-level events (most of
the Russia–Ukraine and Middle-East volume) never resolve to an anchor and slip past
the anchor-assist path in storyline linking. `geo_key` provides a lightweight,
DB-free coarse location key that is stable across paraphrases:

    "Kyiv"            -> "KYIV"
    "Kiev"           -> "KYIV"   (transliteration alias)
    "Ukraine capital"-> "KYIV"   (country-capital resolution, iso="UA")
    "Gaza City"      -> "GAZA"   (admin-suffix stripped + alias)

It is deliberately curated + extensible rather than a full gazetteer: the goal is to
collapse the handful of high-volume conflict geographies that dominate alert spam, not
to geocode the world. Anything unrecognised falls back to its normalized text so two
identical location strings still share a key.
"""

import re

# Administrative suffixes/prefixes that describe the same place with extra words.
# Stripped before alias lookup so "Kyiv city" / "Kharkiv Oblast" collapse.
_ADMIN_WORDS = {
    "city", "region", "oblast", "province", "governorate", "district",
    "county", "prefecture", "municipality", "metropolitan", "area",
    "greater", "downtown", "central", "old", "new",
}

# Transliteration / naming variants for the high-volume conflict geographies.
# Each key is the canonical form; every value in the list maps to it.
_CITY_ALIASES: dict[str, list[str]] = {
    "KYIV":      ["kyiv", "kiev", "kyev", "kyiiv", "kyivan"],
    "KHARKIV":   ["kharkiv", "kharkov"],
    "ODESA":     ["odesa", "odessa"],
    "ZAPORIZHZHIA": ["zaporizhzhia", "zaporizhia", "zaporozhye", "zaporizhzhya"],
    "DNIPRO":    ["dnipro", "dnepropetrovsk", "dnipropetrovsk"],
    "LVIV":      ["lviv", "lvov"],
    "MYKOLAIV":  ["mykolaiv", "nikolaev"],
    "MOSCOW":    ["moscow", "moskva"],
    "BELGOROD":  ["belgorod"],
    "GAZA":      ["gaza", "gaza strip"],
    "TEL AVIV":  ["tel aviv", "telaviv"],
    "JERUSALEM": ["jerusalem", "al quds", "al-quds"],
    "BEIRUT":    ["beirut"],
    "DAMASCUS":  ["damascus", "dimashq"],
    "BAGHDAD":   ["baghdad"],
    "TEHRAN":    ["tehran", "teheran"],
    "SANAA":     ["sanaa", "sana'a", "sana"],
    "KABUL":     ["kabul"],
    "KHARTOUM":  ["khartoum"],
    "BAMAKO":    ["bamako"],
    "MOGADISHU": ["mogadishu"],
}

# Reverse index: alias -> canonical, built once at import.
_ALIAS_TO_CANON: dict[str, str] = {
    alias: canon for canon, aliases in _CITY_ALIASES.items() for alias in aliases
}

# Country ISO -> canonical capital key, so "<country> capital" / "capital" phrasing
# (common when a source avoids naming the city) collapses onto the real place.
_COUNTRY_CAPITAL: dict[str, str] = {
    "UA": "KYIV",
    "RU": "MOSCOW",
    "IL": "JERUSALEM",
    "LB": "BEIRUT",
    "SY": "DAMASCUS",
    "IQ": "BAGHDAD",
    "IR": "TEHRAN",
    "YE": "SANAA",
    "AF": "KABUL",
    "SD": "KHARTOUM",
    "ML": "BAMAKO",
    "SO": "MOGADISHU",
}


def _clean(text: str) -> str:
    """Lowercase, drop punctuation (keep spaces/hyphens), collapse whitespace."""
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    return " ".join(text.split())


def geo_key(text: str | None, country_iso: str | None = None) -> str | None:
    """Return a coarse, paraphrase-stable location key, or None if unusable.

    Resolution order:
      1. Curated transliteration alias (whole-string, then per-token).
      2. Country-capital resolution for "capital" phrasing.
      3. Fallback: the admin-suffix-stripped normalized text, uppercased.

    country_iso is an optional hint used only for capital resolution.
    """
    if not isinstance(text, str):
        return None
    cleaned = _clean(text)
    if not cleaned:
        return None

    # 1a. Whole-string alias hit (handles multi-word aliases like "gaza strip").
    if cleaned in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[cleaned]

    # 2. Capital phrasing: "capital", "ukrainian capital", "capital of ukraine".
    iso = (country_iso or "").strip().upper()
    if "capital" in cleaned.split() and iso in _COUNTRY_CAPITAL:
        return _COUNTRY_CAPITAL[iso]

    # Strip administrative words, then retry alias lookup on the remainder.
    tokens = [t for t in cleaned.split() if t not in _ADMIN_WORDS]
    stripped = " ".join(tokens)
    if stripped in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[stripped]

    # 1b. Per-token alias hit (e.g. "kyiv" inside "near kyiv suburb").
    for t in tokens:
        if t in _ALIAS_TO_CANON:
            return _ALIAS_TO_CANON[t]

    # 3. Fallback: normalized remainder as a weak key so identical strings still match.
    return stripped.upper() if stripped else None
