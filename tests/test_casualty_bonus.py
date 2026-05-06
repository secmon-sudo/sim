"""
Tests for casualty-based severity bonus.
"""

import pytest

from src.pipeline.pass_d_score import compute_casualty_bonus


class TestCasualtyBonus:
    def test_no_casualties_no_bonus(self):
        assert compute_casualty_bonus({}) == 0
        assert compute_casualty_bonus({"casualties": None}) == 0
        assert compute_casualty_bonus({"casualties": {"deaths": 0, "injuries": 0}}) == 0

    def test_deaths_threshold_met(self):
        # 3+ deaths = +20 bonus
        assert compute_casualty_bonus({"casualties": {"deaths": 3, "injuries": 0}}) == 20
        assert compute_casualty_bonus({"casualties": {"deaths": 5, "injuries": 0}}) == 20
        assert compute_casualty_bonus({"casualties": {"deaths": 10}}) == 20

    def test_injuries_threshold_met(self):
        # 10+ injuries = +20 bonus
        assert compute_casualty_bonus({"casualties": {"deaths": 0, "injuries": 10}}) == 20
        assert compute_casualty_bonus({"casualties": {"deaths": 0, "injuries": 25}}) == 20
        assert compute_casualty_bonus({"casualties": {"injuries": 50}}) == 20

    def test_both_thresholds_met(self):
        # Both met still +20 (not cumulative)
        assert compute_casualty_bonus({"casualties": {"deaths": 5, "injuries": 20}}) == 20

    def test_below_threshold_no_bonus(self):
        assert compute_casualty_bonus({"casualties": {"deaths": 2, "injuries": 5}}) == 0
        assert compute_casualty_bonus({"casualties": {"deaths": 1, "injuries": 9}}) == 0

    def test_string_casualties_parsed(self):
        # Sometimes LLM returns strings
        assert compute_casualty_bonus({"casualties": {"deaths": "5", "injuries": "15"}}) == 20
