"""
Tests for storyline matching.
Blueprint V20.1 §PASS D
"""

import pytest

from src.core.storyline import jaccard_similarity, tokenize_storyline_hint


class TestTokenize:
    def test_basic_tokenization(self):
        result = tokenize_storyline_hint("runway incursion CAI")
        assert "runway" in result
        assert "incursion" in result
        assert "cai" in result
        assert "runway incursion" in result
        assert "incursion cai" in result

    def test_stopword_removal(self):
        result = tokenize_storyline_hint("the flight at the airport terminal")
        assert "the" not in result
        assert "flight" not in result
        assert "airport" not in result
        assert "terminal" not in result

    def test_empty_input(self):
        assert tokenize_storyline_hint("") == set()

    def test_single_word(self):
        result = tokenize_storyline_hint("hijacking")
        assert "hijacking" in result
        assert len(result) == 1  # No bigrams possible


class TestJaccard:
    def test_identical_hints(self):
        assert jaccard_similarity("runway incursion", "runway incursion") == 1.0

    def test_zero_similarity(self):
        assert jaccard_similarity("bomb threat", "bird strike") == 0.0

    def test_partial_overlap(self):
        sim = jaccard_similarity(
            "runway incursion Cairo",
            "runway closure Cairo weather",
        )
        assert 0.0 < sim < 1.0

    def test_empty_hint(self):
        assert jaccard_similarity("", "something") == 0.0
        assert jaccard_similarity("something", "") == 0.0


class TestAlertTier:
    """Test alert tier evaluation from alerts module."""

    def test_critical_tier(self):
        from src.core.alerts import evaluate_alert_tier
        event = {
            "severity_score": 85,
            "system_confidence": 0.9,
            "anchor_confidence": "HIGH",
            "time_certainty": "same_day",
        }
        assert evaluate_alert_tier(event) == "CRITICAL"

    def test_alert_tier(self):
        from src.core.alerts import evaluate_alert_tier
        event = {
            "severity_score": 70,
            "system_confidence": 0.7,
            "anchor_confidence": "MEDIUM",
            "time_certainty": "previous_day",
        }
        assert evaluate_alert_tier(event) == "ALERT"

    def test_watch_tier(self):
        from src.core.alerts import evaluate_alert_tier
        event = {
            "severity_score": 50,
            "system_confidence": 0.6,
            "anchor_confidence": "LOW",
            "time_certainty": "same_day",
        }
        assert evaluate_alert_tier(event) == "WATCH"

    def test_no_alert(self):
        from src.core.alerts import evaluate_alert_tier
        event = {
            "severity_score": 20,
            "system_confidence": 0.3,
            "anchor_confidence": "LOW",
            "time_certainty": "unknown",
        }
        assert evaluate_alert_tier(event) is None

    def test_high_severity_but_unknown_time(self):
        """CRITICAL requires time_certainty != 'unknown'."""
        from src.core.alerts import evaluate_alert_tier
        event = {
            "severity_score": 95,
            "system_confidence": 0.95,
            "anchor_confidence": "HIGH",
            "time_certainty": "unknown",
        }
        # Should NOT be CRITICAL due to unknown time
        result = evaluate_alert_tier(event)
        assert result != "CRITICAL"
