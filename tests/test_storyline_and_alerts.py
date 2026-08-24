"""
Tests for storyline matching.
Blueprint V20.1 §PASS D
"""


import itertools
from datetime import datetime

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
            "anchor_name_norm": "CAI",  # CRITICAL requires a resolved place
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
    def _event(sev, conf, anchor="HIGH", time_="same_day", located=True):
        ev = {"severity_score": sev, "system_confidence": conf,
              "anchor_confidence": anchor, "time_certainty": time_}
        if located:
            ev["anchor_name_norm"] = "IST"
        return ev

    def test_config_and_code_agree(self):
        # The shipped config must reproduce the calibrated gates; a mismatch means
        # the file was edited without the intent being reviewed.
        import json
        from pathlib import Path
        from src.core.alerts import TIER_RULES
        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "config" / "settings.json").read_text(encoding="utf-8")
        )["alert"]["tiers"]
        assert cfg["CRITICAL"]["severity_min"] == TIER_RULES["CRITICAL"]["severity_min"] == 80
        assert cfg["ALERT"]["confidence_min"] == TIER_RULES["ALERT"]["confidence_min"] == 0.50
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
        assert rules["CRITICAL"]["confidence_min"] == 0.62
        assert rules["CRITICAL"]["require_location"] is True

    def test_evaluation_order_is_fixed(self):
        # Tiers must be tried most-severe first regardless of config key order,
        # or every CRITICAL event would report as WATCH.
        from src.core.alerts import TIER_ORDER
        assert TIER_ORDER == ("CRITICAL", "ALERT", "WATCH")

    def test_low_anchor_no_longer_blocks_alert(self):
        # anchor_confidence only ever means "an IATA airport resolved" and is LOW for
        # ~99% of the corpus, so it must not gate paging. A located, fresh, confident
        # event pages regardless of anchor level.
        from src.core.alerts import evaluate_alert_tier
        assert evaluate_alert_tier(self._event(70, 0.7, anchor="LOW")) == "ALERT"

    def test_alert_needs_location_or_fresh_time(self):
        from src.core.alerts import evaluate_alert_tier
        # Neither located nor fresh → no page, however severe.
        assert evaluate_alert_tier(
            self._event(100, 0.7, time_="unknown", located=False)) is None
        # Fresh but unlocated still pages...
        assert evaluate_alert_tier(
            self._event(70, 0.7, time_="same_day", located=False)) == "ALERT"
        # ...as does located but undated.
        assert evaluate_alert_tier(
            self._event(70, 0.7, time_="unknown", located=True)) == "ALERT"

    def test_critical_requires_both_location_and_fresh_time(self):
        from src.core.alerts import evaluate_alert_tier
        assert evaluate_alert_tier(
            self._event(90, 0.7, time_="same_day", located=False)) == "ALERT"
        assert evaluate_alert_tier(
            self._event(90, 0.7, time_="same_day", located=True)) == "CRITICAL"

    def test_unknown_time_blocks_critical(self):
        # 86% of the corpus carries time_certainty='unknown', so this must cost the
        # event its CRITICAL standing without silencing it altogether.
        from src.core.alerts import evaluate_alert_tier
        assert evaluate_alert_tier(self._event(90, 0.9, time_="unknown")) == "ALERT"


def test_balochistan_fragments_reach_the_adjudicator():
    """Regression: the 7 Aug 2026 SITREP carried ONE Balochistan counter-terrorism
    operation (12 militants, Mastung + Washuk IBOs) as four storylines with three
    different event_types, and paged four separate ALERT cards for it. Every
    deterministic path correctly declined — Jaccard 0.00-0.11, containment 0.33, and
    the coarse geo keys all differ (BALOCHISTAN / WASHUK / MASTUNG / None) — so this
    was the adjudicator's call to make. It never got to make it: 5 of the 6 pairs
    scored under the old 0.15 candidate floor, several at exactly 0.143.
    """
    from src.core.storyline import lexical_kinship
    from src.pipeline.pass_d_score import STORYLINE_ADJUDICATION_LEXICAL_FLOOR

    hints = [
        "pakistan balochistan militant killings",
        "balochistan security forces militants",
        "balochistan ispr terrorist killed",
    ]
    for a, b in itertools.combinations(hints, 2):
        assert lexical_kinship(a, b) >= STORYLINE_ADJUDICATION_LEXICAL_FLOOR, (
            f"{a!r} vs {b!r} would never be offered to the adjudicator"
        )


def test_adjudicator_floor_still_excludes_unrelated_same_country_events():
    """The floor is what keeps adjudication from degenerating into an all-pairs LLM
    sweep of every same-country event, so lowering it must not admit the unrelated."""
    from src.core.storyline import lexical_kinship
    from src.pipeline.pass_d_score import STORYLINE_ADJUDICATION_LEXICAL_FLOOR

    assert lexical_kinship(
        "balochistan ispr terrorist killed",
        "karachi airport flight diverted weather",
    ) < STORYLINE_ADJUDICATION_LEXICAL_FLOOR


def test_adjudication_logs_llm_telemetry_when_given_a_connection(monkeypatch):
    """Adjudication is the biggest LLM consumer (~180 calls/day vs ~120 for
    classification) and logged none of it, so system_telemetry understated real usage
    by more than half."""
    from unittest.mock import MagicMock
    from src.core import storyline_adjudicator as sa

    db = MagicMock()
    router = MagicMock()
    calls = []
    monkeypatch.setattr(sa, "log_llm_telemetry",
                        lambda conn, res, r, success, purpose: calls.append((success, purpose)))

    event = {"storyline_hint": "balochistan ispr terrorist killed", "country_iso": "PK",
             "occurred_at_est": datetime(2026, 8, 6, 12, 0)}
    recent = [{"storyline_id": "sid-1", "storyline_hint": "pakistan balochistan militant killings",
               "country_iso": "PK", "occurred_at_est": datetime(2026, 8, 6, 11, 0)}]

    sa.adjudicate_storyline(
        event, recent, router, db_conn=db,
        call_llm_fn=lambda *a, **k: {"content": '{"match": null}'},
    )
    assert calls == [(True, "storyline_adjudication")]
