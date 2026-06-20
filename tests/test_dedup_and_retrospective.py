"""
Tests for Faz 2.1 (cross-source / reordered / body-repost dedup) and
Faz 4.2 (retrospective-anniversary filtering overrides military bypass).
"""

from src.pipeline.pass_a_ingest import (
    check_content_duplicate,
    is_noise,
    title_token_similarity,
)

_BODY = (
    "A powerful explosion struck the main terminal at Kabul international airport "
    "on Tuesday morning, killing at least ten people and wounding more than thirty "
    "others. Officials said it was a coordinated attack and security forces sealed "
    "off the area."
)
_RECENT = [("Explosion at Kabul airport kills 10", _BODY)]


class TestContentDedup:
    def test_identical_repost_is_dup(self):
        assert check_content_duplicate(_RECENT, "Explosion at Kabul airport kills 10", _BODY) is True

    def test_syndicated_source_suffix_is_dup(self):
        assert check_content_duplicate(
            _RECENT, "Explosion at Kabul airport kills 10 - Reuters", _BODY
        ) is True

    def test_reordered_title_is_dup(self):
        # SequenceMatcher char-ratio penalizes reordering; token Jaccard catches it.
        assert check_content_duplicate(_RECENT, "Kabul airport explosion kills 10", _BODY) is True

    def test_same_body_different_headline_is_dup(self):
        # Content shingle signal catches a reposted body under a new headline.
        assert check_content_duplicate(
            _RECENT, "Totally different headline here", _BODY.replace("on Tuesday morning", "")
        ) is True

    def test_distinct_story_not_dup(self):
        assert check_content_duplicate(
            _RECENT,
            "Flooding displaces thousands in Brazil",
            "Heavy rains caused severe flooding across southern Brazil, forcing "
            "thousands of residents to evacuate their homes this week.",
        ) is False

    def test_token_similarity_reordered_titles(self):
        sim = title_token_similarity(
            "Kabul airport explosion kills 10", "Explosion at Kabul airport kills 10"
        )
        assert sim > 0.7


class TestRetrospectiveNoise:
    def test_anniversary_with_military_term_is_noise(self):
        # Overrides the military-context bypass — stale recap, not a live event.
        assert is_noise(
            "On the 10th anniversary of the airstrike, families gathered to remember the victims"
        ) is True

    def test_years_ago_with_bombing_is_noise(self):
        assert is_noise("5 years ago, a bombing killed dozens in the city") is True

    def test_on_this_day_is_noise(self):
        assert is_noise("On this day in 2014, the conflict began") is True

    def test_live_airstrike_not_noise(self):
        assert is_noise("Airstrike hits military base, casualties reported") is False
