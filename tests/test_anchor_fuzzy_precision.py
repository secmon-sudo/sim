"""Fuzzy anchor resolution must not invent a location.

Regression for the 2026-08-16 finding: over 30 days, 42 of 92 LOW-confidence anchors
named the wrong COUNTRY (46%) and 22 of them paged. The Varanasi airport shooting
(India) resolved to VAR / Varna, Bulgaria and went out as a Bulgarian alert; a second
report of the same incident resolved to SHJ / Sharjah and paged separately.

Cause: similarity() was computed over full names, where "International Airport" alone
scores ~0.5 — so the 0.5 threshold meant "contains the word Airport", and the
boilerplate penalised the correct longer name while rewarding a same-shaped wrong one.
"""

from src.core.anchor import (
    FUZZY_AMBIGUITY_RATIO,
    FUZZY_MIN_SIMILARITY,
    _core_name,
    _fuzzy_confidence,
    get_anchor_confidence_level,
    normalize_anchor,
)


class FuzzyDB:
    """Stands in for anchor_master: scores candidates on stripped names, as the SQL does."""

    def __init__(self, ranked):
        self.ranked = ranked  # [(iata, similarity), ...] already sorted desc
        self.fuzzy_sql = None

    def execute(self, sql, params=None):
        if "similarity" in sql:
            self.fuzzy_sql = sql
            return FuzzyCursor(self.ranked)
        return FuzzyCursor(None)


class FuzzyCursor:
    def __init__(self, ranked):
        self.ranked = ranked

    def fetchone(self):
        return None  # no exact-code or alias hit

    def fetchall(self):
        return list(self.ranked[:2]) if self.ranked else []


class TestBoilerplateStripping:
    def test_strips_generic_airport_words(self):
        assert _core_name("Lal Bahadur Shastri International Airport") == "Lal Bahadur Shastri"
        assert _core_name("Varanasi airport") == "Varanasi"
        assert _core_name("Erbil International Airport") == "Erbil"

    def test_boilerplate_only_text_has_no_core(self):
        """'Airport' names no place; it used to fuzzy-match an arbitrary row."""
        assert _core_name("Airport") == ""
        assert _core_name("the international airfield") == ""

    def test_boilerplate_only_input_resolves_to_nothing(self):
        db = FuzzyDB([("HCL0", 0.9)])
        assert normalize_anchor("Airport", db) == (None, 0.0)


class TestSimilarityFloor:
    def test_rejects_below_floor(self):
        """Varanasi→Varna scored 0.250 on stripped names. It must not resolve."""
        db = FuzzyDB([("VAR", 0.250), ("CMB", 0.158)])
        assert normalize_anchor("Varanasi airport", db) == (None, 0.0)

    def test_rejects_shared_suffix_match(self):
        """Rochester→Manchester: 0.400, on the strength of a shared "chester" alone.

        Plain similarity() cannot reject this without also rejecting the correct
        Catania→Catania-Fontanarossa, which scores an identical 0.400. That is why
        the query uses strict_word_similarity, where the two separate.
        """
        assert normalize_anchor("Rochester Airport", FuzzyDB([("MAN", 0.400)])) == (None, 0.0)

    def test_accepts_above_floor(self):
        """Sana'a→SAH scored 0.571 and is correct — the lowest measured true match."""
        norm, conf = normalize_anchor("Sanaa airport", FuzzyDB([("SAH", 0.571), ("SCL", 0.2)]))
        assert norm == "SAH"
        assert conf >= 0.5

    def test_floor_sits_between_measured_right_and_wrong(self):
        """Over 90 days: true matches >=0.571, false ones <=0.400."""
        assert 0.400 < FUZZY_MIN_SIMILARITY < 0.571

    def test_query_uses_word_boundary_metric(self):
        db = FuzzyDB([("SAH", 0.9)])
        normalize_anchor("Sanaa airport", db)
        assert "strict_word_similarity" in db.fuzzy_sql


class TestAmbiguityGuard:
    def test_rejects_near_tie(self):
        """Shastri tied 0.5435 against BOTH Sharjah and Bahrain; LIMIT 1 broke it by luck."""
        db = FuzzyDB([("SHJ", 0.5435), ("BAH", 0.5435)])
        assert normalize_anchor("Lal Bahadur Shastri International Airport", db) == (None, 0.0)

    def test_accepts_clear_winner(self):
        norm, _ = normalize_anchor("Narita Airport", FuzzyDB([("NRT", 1.0), ("ASF", 0.286)]))
        assert norm == "NRT"

    def test_ratio_is_strict_enough_to_catch_ties(self):
        assert FUZZY_AMBIGUITY_RATIO < 1.0


class TestHotelRowsExcluded:
    """130 of 429 anchor_master rows are hotels and lounges, not airports.

    They are carried for proximity lookups and can never be an event's location, but
    they sit on city names: "Kabul" matched KABUL STAR HOTEL at 0.353 and the event
    paged with country XX. Excluded from the fuzzy path only.

    Filtering on country_iso='XX' is NOT enough — 45 of them carry a real-looking but
    wrong country ("Hilton Hotel Dushanbe" is filed under FI). The synthetic 4-char
    code (HKB0, HHL5) is the reliable discriminator; all 299 real airports are 3-letter.
    """

    def test_fuzzy_query_matches_only_three_letter_iata_codes(self):
        db = FuzzyDB([("EBL", 1.0), ("ECN", 0.2)])
        normalize_anchor("Erbil airport", db)
        assert "^[A-Z]{3}$" in db.fuzzy_sql


class TestConfidenceBand:
    def test_accepted_fuzzy_is_never_low(self):
        """The invariant every location-key consumer now relies on."""
        for sim in (FUZZY_MIN_SIMILARITY, 0.5, 0.75, 0.9, 1.0):
            assert get_anchor_confidence_level(_fuzzy_confidence(sim)) == "MEDIUM"

    def test_fuzzy_never_reaches_high(self):
        """HIGH stays reserved for exact-code and alias hits."""
        assert get_anchor_confidence_level(_fuzzy_confidence(1.0)) != "HIGH"

    def test_proximity_bonus_needs_a_near_exact_core(self):
        """compute_severity awards PROXIMITY_BONUS at >=0.6; a weak hit must not earn it."""
        assert _fuzzy_confidence(1.0) >= 0.6
        assert _fuzzy_confidence(0.5) < 0.6
