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

import math
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
    # NATO eastern flank / Europe: drone-incursion and airspace-violation
    # reporting names these places constantly, and each has a Polish/Romanian
    # spelling that wire copy strips the diacritics from.
    "WARSAW":    ["warsaw", "warszawa"],
    "KRAKOW":    ["krakow", "kraków", "cracow"],
    "GDANSK":    ["gdansk", "gdańsk"],
    "WROCLAW":   ["wroclaw", "wrocław"],
    "RZESZOW":   ["rzeszow", "rzeszów"],
    "LODZ":      ["lodz", "łódź"],
    "CHISINAU":  ["chisinau", "chișinău", "kishinev"],
    "CONSTANTA": ["constanta", "constanța"],
    "BUCHAREST": ["bucharest", "bucuresti", "bucurești", "bucureşti"],
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
    "BRYANSK": (53.2436, 34.3634, "RU"),
    "KURSK": (51.7373, 36.1874, "RU"),
    "ROSTOV": (47.2357, 39.7015, "RU"),
    "KRASNODAR": (45.0355, 38.9753, "RU"),
    "NOVOROSSIYSK": (44.7239, 37.7686, "RU"),
    "SEVASTOPOL": (44.6166, 33.5254, "RU"),
    "KALININGRAD": (54.7104, 20.4522, "RU"),
    "SAINT PETERSBURG": (59.9311, 30.3609, "RU"),
    "MINSK": (53.9006, 27.5590, "BY"),
    # NATO eastern flank — drone incursions and airspace violations are reported
    # against these places, and none of them resolved to a coordinate before.
    "WARSAW": (52.2297, 21.0122, "PL"),
    "LUBLIN": (51.2465, 22.5684, "PL"),
    "RZESZOW": (50.0413, 21.9990, "PL"),
    "KRAKOW": (50.0647, 19.9450, "PL"),
    "GDANSK": (54.3520, 18.6466, "PL"),
    "WROCLAW": (51.1079, 17.0385, "PL"),
    "POZNAN": (52.4064, 16.9252, "PL"),
    "KATOWICE": (50.2649, 19.0238, "PL"),
    "SZCZECIN": (53.4285, 14.5528, "PL"),
    "BIALYSTOK": (53.1325, 23.1688, "PL"),
    "LODZ": (51.7592, 19.4560, "PL"),
    "VILNIUS": (54.6872, 25.2797, "LT"),
    "KAUNAS": (54.8985, 23.9036, "LT"),
    "RIGA": (56.9496, 24.1052, "LV"),
    "TALLINN": (59.4370, 24.7536, "EE"),
    "HELSINKI": (60.1699, 24.9384, "FI"),
    "STOCKHOLM": (59.3293, 18.0686, "SE"),
    "OSLO": (59.9139, 10.7522, "NO"),
    "COPENHAGEN": (55.6761, 12.5683, "DK"),
    "BERLIN": (52.5200, 13.4050, "DE"),
    "PRAGUE": (50.0755, 14.4378, "CZ"),
    "BRATISLAVA": (48.1486, 17.1077, "SK"),
    "BUDAPEST": (47.4979, 19.0402, "HU"),
    "VIENNA": (48.2082, 16.3738, "AT"),
    "BUCHAREST": (44.4268, 26.1025, "RO"),
    "CONSTANTA": (44.1598, 28.6348, "RO"),
    "CHISINAU": (47.0105, 28.8638, "MD"),
    "SOFIA": (42.6977, 23.3219, "BG"),
    "BELGRADE": (44.7866, 20.4489, "RS"),
    "ZAGREB": (45.8150, 15.9819, "HR"),
    "ATHENS": (37.9838, 23.7275, "GR"),
    "NICOSIA": (35.1856, 33.3823, "CY"),
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


_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km.

    Lives here rather than in a service module because both the flash detector's
    co-location check and the airspace proximity analysis need it, and geo.py is
    the one dependency-free module both can import. Returns infinity on
    unusable input so a bad coordinate can never look like a near miss.
    """
    try:
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return _EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    except (TypeError, ValueError):
        return float("inf")


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
