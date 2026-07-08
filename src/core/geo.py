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


# Coordinates for the high-volume conflict geographies, keyed by the canonical geo_key
# (so both curated aliases like "Kiev"→KYIV and single-spelling fallbacks like
# "Aleppo"→ALEPPO resolve here). Deliberately curated, same as the alias table: it gives
# city-level events a lat/lon so they participate in spatial features and maps, which the
# IATA-only anchor_master gazetteer never provided. Values are (lat, lon, iso).
_CITY_COORDS: dict[str, tuple[float, float, str]] = {
    # Ukraine
    "KYIV": (50.4501, 30.5234, "UA"),
    "KHARKIV": (49.9935, 36.2304, "UA"),
    "ODESA": (46.4825, 30.7233, "UA"),
    "ZAPORIZHZHIA": (47.8388, 35.1396, "UA"),
    "DNIPRO": (48.4647, 35.0462, "UA"),
    "LVIV": (49.8397, 24.0297, "UA"),
    "MYKOLAIV": (46.9750, 31.9946, "UA"),
    "MARIUPOL": (47.0951, 37.5497, "UA"),
    "BAKHMUT": (48.5946, 38.0027, "UA"),
    "KHERSON": (46.6354, 32.6169, "UA"),
    "DONETSK": (48.0159, 37.8028, "UA"),
    "LUHANSK": (48.5740, 39.3078, "UA"),
    # Russia / Belarus
    "MOSCOW": (55.7558, 37.6173, "RU"),
    "BELGOROD": (50.5997, 36.5983, "RU"),
    "MINSK": (53.9006, 27.5590, "BY"),
    # Israel / Palestine
    "GAZA": (31.5000, 34.4668, "PS"),
    "RAFAH": (31.2968, 34.2432, "PS"),
    "KHAN YUNIS": (31.3469, 34.3061, "PS"),
    "RAMALLAH": (31.9038, 35.2034, "PS"),
    "HEBRON": (31.5326, 35.0998, "PS"),
    "TEL AVIV": (32.0853, 34.7818, "IL"),
    "JERUSALEM": (31.7683, 35.2137, "IL"),
    # Levant / Iraq / Iran
    "BEIRUT": (33.8938, 35.5018, "LB"),
    "DAMASCUS": (33.5138, 36.2765, "SY"),
    "ALEPPO": (36.2021, 37.1343, "SY"),
    "HOMS": (34.7324, 36.7137, "SY"),
    "IDLIB": (35.9306, 36.6339, "SY"),
    "RAQQA": (35.9528, 39.0079, "SY"),
    "BAGHDAD": (33.3152, 44.3661, "IQ"),
    "MOSUL": (36.3450, 43.1189, "IQ"),
    "ERBIL": (36.1901, 44.0091, "IQ"),
    "BASRA": (30.5085, 47.7804, "IQ"),
    "TEHRAN": (35.6892, 51.3890, "IR"),
    # Arabian Peninsula
    "SANAA": (15.3694, 44.1910, "YE"),
    "ADEN": (12.7797, 45.0095, "YE"),
    "RIYADH": (24.7136, 46.6753, "SA"),
    "DOHA": (25.2854, 51.5310, "QA"),
    "DUBAI": (25.2048, 55.2708, "AE"),
    # North Africa / Sahel / Horn
    "CAIRO": (30.0444, 31.2357, "EG"),
    "BENGHAZI": (32.1167, 20.0667, "LY"),
    "KHARTOUM": (15.5007, 32.5599, "SD"),
    "MOGADISHU": (2.0469, 45.3182, "SO"),
    "BAMAKO": (12.6392, -8.0029, "ML"),
    "GAO": (16.2666, -0.0400, "ML"),
    "KIDAL": (18.4411, 1.4078, "ML"),
    "NIAMEY": (13.5116, 2.1254, "NE"),
    "OUAGADOUGOU": (12.3714, -1.5197, "BF"),
    "MAIDUGURI": (11.8333, 13.1500, "NG"),
    # South / Central Asia
    "KABUL": (34.5553, 69.2075, "AF"),
    # Turkey
    "ISTANBUL": (41.0082, 28.9784, "TR"),
    "ANKARA": (39.9334, 32.8597, "TR"),
}


def geo_coords(
    text: str | None, country_iso: str | None = None
) -> tuple[float, float, str] | None:
    """Resolve a location string to (lat, lon, iso) via the curated city gazetteer.

    Uses the same canonical key as `geo_key`, so transliterations and admin suffixes are
    handled identically. Returns None for unknown places (the caller keeps lat/lon empty).
    When a country_iso hint is supplied and contradicts the gazetteer entry's country, the
    entry is rejected — a name-collision (e.g. a same-named city in another country) should
    not plant a wrong coordinate; better no coordinate than a misplaced one.
    """
    key = geo_key(text, country_iso)
    if not key:
        return None
    entry = _CITY_COORDS.get(key)
    if not entry:
        return None
    lat, lon, entry_iso = entry
    hint = (country_iso or "").strip().upper()
    if hint and entry_iso and hint != entry_iso:
        return None
    return (lat, lon, entry_iso)


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
