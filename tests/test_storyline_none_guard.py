"""
Tests for storyline None datetime guard.
"""

from src.core.storyline import should_link_storyline


class TestStorylineNoneGuard:
    def test_none_occurred_at_est_returns_false(self):
        event_a = {
            "storyline_hint": "runway incursion CAI",
            "country_iso": "EG",
            "occurred_at_est": None,
        }
        event_b = {
            "storyline_hint": "runway incursion CAI",
            "country_iso": "EG",
            "occurred_at_est": None,
        }
        assert should_link_storyline(event_a, event_b) is False

    def test_one_none_returns_false(self):
        from datetime import datetime
        event_a = {
            "storyline_hint": "runway incursion CAI",
            "country_iso": "EG",
            "occurred_at_est": datetime(2025, 1, 1),
        }
        event_b = {
            "storyline_hint": "runway incursion CAI",
            "country_iso": "EG",
            "occurred_at_est": None,
        }
        assert should_link_storyline(event_a, event_b) is False

    def test_valid_datetimes_link(self):
        from datetime import datetime
        event_a = {
            "storyline_hint": "runway incursion CAI",
            "country_iso": "EG",
            "occurred_at_est": datetime(2025, 1, 1),
        }
        event_b = {
            "storyline_hint": "runway incursion CAI",
            "country_iso": "EG",
            "occurred_at_est": datetime(2025, 1, 2),
        }
        assert should_link_storyline(event_a, event_b) is True
