"""
SIM — Anchor Normalization
Blueprint V20.1 §2.2

Normalizes raw airport/location text to IATA/ICAO codes using
exact match, alias lookup, and trigram fuzzy matching.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


# Words that appear in most airport names and therefore carry no discriminating
# signal. Comparing full names lets this boilerplate dominate the trigram score:
# measured 2026-08-16, "Lal Bahadur Shastri International Airport" scored 0.543
# against BOTH "Sharjah International Airport" and "Bahrain International Airport"
# — an exact tie broken by LIMIT 1 — while every one of the top 8 candidates was
# the wrong country. "International Airport" alone is worth ~0.5, so the old
# 0.5 threshold meant "contains the word Airport".
#
# Worse, the boilerplate actively inverted correct matches: it penalised the
# longer right answer and rewarded a same-shaped wrong one. "Narita Airport"
# resolved to ASF Narimanovo (Russia) even though NRT Narita International
# Airport was in the table all along. Stripping these words first makes NRT score
# 1.000. Same for Erbil→EBL, Dubai→DXB, Denver→DEN.
# "capital" and "city" are here for the same reason as "international": they are
# administrative descriptors, not names. Measured 2026-08-17 on live output, leaving
# them in let strict_word_similarity match "Russian capital" against "Beijing Capital
# International Airport" on the shared word "capital" alone — score 0.50, exactly at
# the acceptance floor — and three Moscow drone events paged as CN/Beijing. Stripping
# them takes that pair to 0.0 while Sana'a (0.571), Catania (1.0) and Kyiv (1.0) are
# untouched.
_BOILERPLATE_SQL = (
    r"\m(international|intl|airport|airfield|air ?base|airbase|"
    r"regional|municipal|the|capital|city)\M"
)
_BOILERPLATE_RE = re.compile(
    r"\b(?:international|intl|airport|airfield|air\s?base|airbase|"
    r"regional|municipal|the|capital|city)\b",
    re.IGNORECASE,
)

# Minimum strict_word_similarity between BOILERPLATE-STRIPPED names.
#
# Plain similarity() is a ratio over the whole trigram set, so a shared SUFFIX scores
# as highly as a shared name: "rochester" vs "manchester" scores 0.400 on the strength
# of "chester" alone — exactly the same as the correct "catania" vs "catania
# fontanarossa". No threshold can separate those two, and Rochester Airport (US) was
# filed under Manchester (GB) five times because of it.
#
# strict_word_similarity asks a different question — "does the query match a WORD of
# the candidate?" — which is the question that actually matters here. Measured over the
# 90-day corpus it separates cleanly, with a wide gap and no overlap:
#
#   correct: sana'a 0.571, catania 1.000, leipzig 1.000, heathrow 1.000, narita 1.000
#   wrong:   rochester/manchester 0.400, bandar abbas 0.300, ankara 0.273,
#            varanasi/varna 0.250, orsk 0.182, pittsburgh/pisa 0.143
#
# It also keeps matches a prefix rule would lose: "heathrow" -> "london heathrow" is
# 1.000 here despite sharing no leading characters.
FUZZY_MIN_SIMILARITY = 0.50

# Reject when the runner-up is nearly as good as the winner. The Shastri tie above
# is the failure this prevents: with two candidates that close, ORDER BY picks one
# arbitrarily and the result is a coin flip presented as a resolved location.
FUZZY_AMBIGUITY_RATIO = 0.90


def _core_name(text: str) -> str:
    """Drop boilerplate airport words, leaving the discriminating part of the name."""
    return " ".join(_BOILERPLATE_RE.sub(" ", text).split()).strip()


def normalize_anchor(raw_text: str, db_conn) -> tuple[str | None, float]:
    """
    Normalize raw location text to IATA/ICAO code.

    Returns:
        (normalized_id, confidence)
        normalized_id: IATA code (preferred), ICAO, or None
        confidence: 1.0 (exact match), 0.8 (alias), 0.50-0.60 (fuzzy), 0.0 (not found)

    A fuzzy hit is never returned below MEDIUM confidence. Callers may therefore
    treat "anchor_name_norm is set" as "the location is trustworthy enough to key
    on" — which build_geo_suppression_key and the storyline adjudicator both do.
    Before this, 42 of 92 LOW-confidence anchors over 30 days put the event in the
    wrong country (46%), 22 of them paged, and none of the 128 MEDIUM/HIGH anchors
    was wrong even once. LOW simply carried no information.
    """
    # Input guard: reject non-string, empty, or excessively long input
    if not isinstance(raw_text, str) or len(raw_text) > 200:
        return None, 0.0
    raw_text = raw_text.strip()
    if not raw_text:
        return None, 0.0

    try:
        # 1. Direct IATA / ICAO exact match (case insensitive)
        if re.match(r"^[A-Za-z]{3,4}$", raw_text):
            upper_text = raw_text.upper()
            row = db_conn.execute(
                "SELECT iata_code FROM anchor_master WHERE iata_code=%s OR icao_code=%s",
                (upper_text, upper_text),
            ).fetchone()
            if row:
                return row[0], 1.0

        # 2. Case-insensitive alias JSONB search
        row = db_conn.execute(
            "SELECT iata_code FROM anchor_master WHERE aliases @> %s::jsonb",
            (json.dumps([raw_text]),),
        ).fetchone()
        if row:
            return row[0], 0.8

        # 3. Trigram fuzzy match (pg_trgm) on boilerplate-stripped names.
        core = _core_name(raw_text)
        if not core:
            # Nothing left but boilerplate ("Airport", "the airfield"). Previously
            # this matched an arbitrary row; there is no location here to resolve.
            return None, 0.0

        # 130 of the 429 rows are hotels and lounges (KABUL STAR HOTEL, Hilton Hotel
        # Dushanbe, TBS CIP LOUNGE) carried for proximity lookups under synthetic
        # 4-character codes: HKB0, HHL5, HCS3. They can never BE an event's location,
        # but they sit on city names and hijack them through the fuzzy path — "Kabul"
        # matched KABUL STAR HOTEL at 0.353 and the event paged with country XX.
        #
        # The code SHAPE is the discriminator, not country_iso. An earlier version of
        # this filter excluded country_iso='XX' and let 45 of them through, because
        # many carry a real-looking country that is simply wrong: "Hilton Hotel
        # Dushanbe" is filed under FI, "Garden Inn Hilton Kuwait" under KR. All 299
        # genuine airports have a 3-letter IATA code and none of them is 'XX', so this
        # test is strictly stronger and loses nothing.
        # Excluded from fuzzy only; an explicit code or alias hit still resolves them.
        #
        # The icao_code disjunct (2026-09-03) admits military air bases, which have no
        # IATA at all and so were taking a generated M-code that this shape test reads
        # as boilerplate — Incirlik, Bagram, Al Udeid and the rest could be resolved by
        # an exact code but never by name. Note this is NOT a widening of the shape test
        # to 4 characters, which is what it looks like from the outside and which would
        # let every HKB0/HHL5 hotel row straight back in: the synthetic codes carry no
        # ICAO, so requiring a real one keeps the discriminator intact while naming the
        # rows that deserve an exception. If a hotel ever arrives with an icao_code
        # populated, that is the row to fix, not this predicate.
        rows = db_conn.execute(
            """SELECT iata_code,
                      strict_word_similarity(
                        %s,
                        btrim(regexp_replace(lower(canonical_name), %s, ' ', 'g'))
                      ) AS sim
               FROM anchor_master
               WHERE btrim(regexp_replace(lower(canonical_name), %s, ' ', 'g')) <> ''
                 AND (iata_code ~ '^[A-Z]{3}$' OR icao_code IS NOT NULL)
               ORDER BY sim DESC LIMIT 2""",
            (core, _BOILERPLATE_SQL, _BOILERPLATE_SQL),
        ).fetchall()

        if rows:
            best_code, best_sim = rows[0][0], float(rows[0][1] or 0.0)
            if best_sim >= FUZZY_MIN_SIMILARITY:
                runner_up = float(rows[1][1] or 0.0) if len(rows) > 1 else 0.0
                if runner_up / best_sim > FUZZY_AMBIGUITY_RATIO:
                    logger.info(
                        "Anchor ambiguous, rejected: %.60s (%s %.3f vs %s %.3f)",
                        raw_text, best_code, best_sim, rows[1][0], runner_up,
                    )
                    return None, 0.0
                return best_code, _fuzzy_confidence(best_sim)

    except Exception:
        logger.exception("Anchor normalization error for: %s", raw_text[:50])

    return None, 0.0


def _fuzzy_confidence(sim: float) -> float:
    """Map an accepted stripped-name similarity onto the MEDIUM confidence band.

    Deliberately spans only 0.50-0.60. The floor keeps every accepted fuzzy hit at
    MEDIUM, so a set anchor is always a trusted anchor. The ceiling preserves the
    existing severity behaviour: compute_severity awards PROXIMITY_BONUS at
    confidence >= 0.6, which under the old sim*0.6 mapping required a perfect
    score, so only a near-exact core match earns it here too.
    """
    span = (sim - FUZZY_MIN_SIMILARITY) / (1.0 - FUZZY_MIN_SIMILARITY)
    return round(0.50 + span * 0.10, 2)


def get_anchor_confidence_level(confidence: float) -> str:
    """Convert numeric confidence to tier string."""
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"
