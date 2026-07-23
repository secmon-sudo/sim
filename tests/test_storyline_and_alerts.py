"""
Tests for storyline matching.
Blueprint V20.1 §PASS D
"""


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
        # Since 2026-07-09 well-formed MonDD tokens are stripped too: the old prompt
        # forced the LLM to append one, so it FABRICATED dates for undated articles
        # ("nov20" in Telegram cards). Time lives in occurred_at, never in the hint.
        assert _normalize_storyline_hint("Istanbul Ataturk bomb threat Jun8") == \
            "istanbul ataturk bomb threat"
        assert _normalize_storyline_hint("Omsk refinery Ukraine drone strike Nov20") == \
            "omsk refinery ukraine drone strike"
        # Month-like WORDS must not be over-stripped.
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


class TestConfigDrivenTiers:
    """alert.tiers in settings.json must actually drive the gates.

    The thresholds were duplicated as literals in alerts.py, so editing the
    config changed nothing — a silent trap for anyone tuning alert volume.
    """

    @staticmethod
    def _event(sev, conf, anchor="HIGH", time_="same_day"):
        return {"severity_score": sev, "system_confidence": conf,
                "anchor_confidence": anchor, "time_certainty": time_}

    def test_config_and_code_agree(self):
        # The shipped config must reproduce the V19 gates; a mismatch means the
        # file was edited without the intent being reviewed.
        import json
        from pathlib import Path
        from src.core.alerts import TIER_RULES
        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "config" / "settings.json").read_text(encoding="utf-8")
        )["alert"]["tiers"]
        assert cfg["CRITICAL"]["severity_min"] == TIER_RULES["CRITICAL"]["severity_min"] == 80
        assert cfg["ALERT"]["confidence_min"] == TIER_RULES["ALERT"]["confidence_min"] == 0.65
        assert cfg["WATCH"]["severity_min"] == TIER_RULES["WATCH"]["severity_min"] == 45

    def test_raising_a_threshold_takes_effect(self, monkeypatch):
        import src.core.alerts as alerts
        event = self._event(85, 0.9)
        assert alerts.evaluate_alert_tier(event) == "CRITICAL"
        stricter = {k: dict(v) for k, v in alerts.TIER_RULES.items()}
        stricter["CRITICAL"]["severity_min"] = 95
        monkeypatch.setattr(alerts, "TIER_RULES", stricter)
        # Falls through to the next tier it still satisfies, not to None.
        assert alerts.evaluate_alert_tier(event) == "ALERT"

    def test_partial_config_falls_back_to_defaults(self, monkeypatch):
        # A config that sets only severity_min must keep the other gates.
        import src.core.alerts as alerts
        monkeypatch.setattr(
            alerts, "_SETTINGS",
            {"alert": {"tiers": {"CRITICAL": {"severity_min": 70}}}},
        )
        rules = alerts._tier_rules()
        assert rules["CRITICAL"]["severity_min"] == 70
        assert rules["CRITICAL"]["confidence_min"] == 0.8
        assert rules["CRITICAL"]["anchor_confidence"] == ["HIGH"]

    def test_evaluation_order_is_fixed(self):
        # Tiers must be tried most-severe first regardless of config key order,
        # or every CRITICAL event would report as WATCH.
        from src.core.alerts import TIER_ORDER
        assert TIER_ORDER == ("CRITICAL", "ALERT", "WATCH")

    def test_low_anchor_cannot_reach_alert(self):
        from src.core.alerts import evaluate_alert_tier
        assert evaluate_alert_tier(self._event(70, 0.7, anchor="LOW")) == "WATCH"

    def test_unknown_time_blocks_critical(self):
        from src.core.alerts import evaluate_alert_tier
        assert evaluate_alert_tier(self._event(90, 0.9, time_="unknown")) is None
