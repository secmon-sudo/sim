"""
Tests for anchor case-insensitive IATA matching.
"""

from src.core.anchor import normalize_anchor


class MockDB:
    """Mock database that returns CAI for any 3-letter code starting with C."""

    def execute(self, query, params):
        return MockCursor(params)


class MockCursor:
    def __init__(self, params):
        self._params = params

    def fetchone(self):
        # Simulate CAI match for 'CAI' or 'cai'
        if self._params and len(self._params) >= 2:
            code = self._params[0] if self._params[0] else self._params[1]
            if code and code.upper() == "CAI":
                return ("CAI",)
        return None


def test_lowercase_iata_matches():
    """Lowercase 'cai' should match uppercase IATA code."""
    norm, conf = normalize_anchor("cai", MockDB())
    assert norm == "CAI"
    assert conf == 1.0


def test_uppercase_iata_matches():
    """Uppercase 'CAI' should still match."""
    norm, conf = normalize_anchor("CAI", MockDB())
    assert norm == "CAI"
    assert conf == 1.0


def test_mixed_case_iata_matches():
    """Mixed case 'Cai' should match."""
    norm, conf = normalize_anchor("Cai", MockDB())
    assert norm == "CAI"
    assert conf == 1.0
