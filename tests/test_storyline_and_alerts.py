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


class TestDateHintPollution:
    """A missing-day date hint ("JunUnknown") must not survive as a Jaccard token."""

    def test_malformed_date_token_stripped(self):
        toks = tokenize_storyline_hint("Philippines school shooting JunUnknown")
        assert "jununknown" not in toks
        assert "shooting jununknown" not in toks
        assert "shooting" in toks

    def test_valid_date_token_still_stripped(self):
        # Well-formed MonDD hints were always dropped from the similarity signal.
        assert "jun8" not in tokenize_storyline_hint("Istanbul bomb threat Jun8")

    def test_normalize_strips_unknown_day(self):
        from src.pipeline.pass_c_classify import _normalize_storyline_hint
        assert _normalize_storyline_hint("Philippines school shooting JunUnknown") == \
            "philippines school shooting"
        # A real day must be preserved, and month-like words must not be over-stripped.
        assert _normalize_storyline_hint("Istanbul Ataturk bomb threat Jun8") == \
            "istanbul ataturk bomb threat jun8"
        assert _normalize_storyline_hint("Junction City may riot") == "junction city may riot"


class TestIntraBatchClustering:
    """Sibling reports of one incident scored in the same Pass D batch must cluster.

    Regression for the bug where recent_events was fetched once per pass and never
    updated, so multi-source reports arriving together each spawned a new storyline.
    """

    def test_siblings_share_one_storyline(self):
        import uuid
        from datetime import datetime
        from src.pipeline.pass_d_score import link_storylines

        t = datetime(2026, 6, 22, 6, 0, 0)
        siblings = [
            {"id": "1", "storyline_hint": "Philippines high school shooting",
             "country_iso": "PH", "occurred_at_est": t, "anchor_name_norm": None},
            {"id": "2", "storyline_hint": "Philippines school shooting",
             "country_iso": "PH", "occurred_at_est": t, "anchor_name_norm": None},
            {"id": "3", "storyline_hint": "Philippines school shooting",
             "country_iso": "PH", "occurred_at_est": t, "anchor_name_norm": None},
        ]

        recent: list[dict] = []
        assigned = []
        for ev in siblings:
            sid = link_storylines(ev, recent) or str(uuid.uuid4())
            ev["storyline_id"] = sid
            assigned.append(sid)
            # Mirror score_single_event advertising the just-scored event.
            recent.append({k: ev.get(k) for k in (
                "id", "storyline_id", "storyline_hint",
                "country_iso", "occurred_at_est", "anchor_name_norm")})

        assert len(set(assigned)) == 1, "all sibling reports should share one storyline"
