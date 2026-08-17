"""Two ingest/prescreen defects found by reading live output, 2026-08-17.

Both are recall/precision failures that every existing test passed straight through,
because nothing asserted on what the pipeline THREW AWAY.
"""

from src.pipeline.ingest_filters import is_content_farm
from src.pipeline.pass_c_classify import deterministic_relevance

PRESCREEN_FLOOR = 15  # classification.deterministic_skip_floor


def score(title: str) -> int:
    return deterministic_relevance(title, "")["score"]


class TestContentFarmRejected:
    """mshale.com republished scraped YouTube titles as news.

    136 of its 138 events carried the signature and ALL 20 of its alerts did. The
    content included attacks that never happened, paging at severity 100.
    """

    def test_named_domain_is_rejected(self):
        assert is_content_farm("Anything at all", "https://mshale.com/x", "mshale.com")

    def test_scraped_video_title_signature(self):
        assert is_content_farm(
            "Iranian Drone Attack Hits Airport In Kuwait Atlético Madrid (mDxFTHcvRK) - Mshale",
            None, None,
        )

    def test_random_hash_article_path(self):
        assert is_content_farm(None, "https://example.com/bd22c76b/dfce4942-Ff1aw6k0wM", None)

    def test_fabricated_alert_headline_is_caught(self):
        """The exact headline that paged at severity 100."""
        assert is_content_farm(
            "Iran Missile Strikes Rock Dubai, Abu Dhabi Airports After Khamenei's "
            "Death | Firstpost LIVE | N (SeBYeL4lGO) - Mshale",
            "https://mshale.com/e2408db9/0d8eee03MoMWinq", "mshale.com",
        )

    def test_real_publishers_pass(self):
        assert not is_content_farm(
            "Russian strikes kill two at Kryvyi Rih steel plant",
            "https://www.reuters.com/world/europe/kryvyi-rih-strike", "reuters.com")
        assert not is_content_farm(
            "Drone attack on Kyiv - The Guardian",
            "https://www.theguardian.com/world/2026/aug/17/kyiv-drone", "theguardian.com")


class TestCasualtyVerbRecall:
    """The prescreen archived real mass-casualty attacks at score 0.

    "kill" — the most common verb in conflict reporting — was absent from the hostile
    vocabulary in every form, as were injure/wound/down. Measured over 14 days: 89
    unambiguous attacks archived without an LLM ever seeing them, ~6 a day.
    """

    def test_drone_kills_with_toll(self):
        assert score("Ukrainian drone kills 12, injured 39 in Russia's Tatarstan") >= PRESCREEN_FLOOR

    def test_strikes_kill(self):
        assert score("Russian strikes kill two at Kryvyi Rih steel plant") >= PRESCREEN_FLOOR

    def test_present_tense_attack_plus_casualties(self):
        assert score("Terrorists attack Benue communities, kill, injure residents") >= PRESCREEN_FLOOR

    def test_air_defence_downs_drone(self):
        assert score("NATO Jet Downs Drone Over Romania as Russian Strikes Kill Woman") >= PRESCREEN_FLOOR

    def test_killed_with_actor(self):
        assert score("Militants killed 12 in Balochistan ambush") >= PRESCREEN_FLOOR


class TestPluralAttackPhrases:
    """`drone strikes?` carried the plural; the three neighbouring phrases did not."""

    def test_drone_attacks_plural(self):
        assert score("Russia says overnight Ukrainian drone attacks kill at least 10") >= PRESCREEN_FLOOR

    def test_missile_attacks_plural(self):
        assert score("Missile attacks reported across the region") >= PRESCREEN_FLOOR

    def test_terrorist_attacks_plural(self):
        assert score("Terrorist attacks reported in the capital") >= PRESCREEN_FLOOR


class TestMetaphorsStillRejected:
    """The casualty frames are anchored to a weapon/actor subject or a civilian object.

    A bare "kills" is where the metaphors live, and widening the vocabulary must not
    reopen that door — the whole point of the prescreen is to be cheap and quiet.
    """

    def test_box_office(self):
        assert score("The film bombed at the box office") < PRESCREEN_FLOOR

    def test_markets(self):
        assert score("Stocks attack record highs as inflation cools") < PRESCREEN_FLOOR

    def test_policy_kills_jobs(self):
        assert score("New tax deal kills jobs, say analysts") < PRESCREEN_FLOOR

    def test_sport(self):
        assert score("Manchester United kills off Chelsea comeback") < PRESCREEN_FLOOR
