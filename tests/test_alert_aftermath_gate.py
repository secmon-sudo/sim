"""
Aftermath reports must not page.

Measured over the 724 alerts SIM raised in the 6 days to 2026-08-10: severity comes
from the subject, not the article, and "fresh" degrades to "published today" because
occurred_at_est falls back to published_at (identical to the millisecond in 93% of
"previous_day" events). So the SEVERITY_ALERT_FLOOR path — which produced 61% of that
window's ALERTs — happily promoted:

    "Canyon Park West Shooting: one week later"        sev 95, conf 0.33
    "Iran War Timeline: Key Moments and Attacks"       sev 90, conf 0.35
    "War update: 233 combat clashes on frontline"      sev 90, conf 0.43
    "Nigerian army says it safely rescued 33 people"   sev 90, conf 0.41
    "IDF to demolish home of terrorist who killed …"   sev 100, conf 0.41

Confidence cannot separate these: real incidents in the same window sat at the same
values (Colombia car bomb 0.41, Sumy-Kyiv train strike 0.41), so a confidence guard on
the floor would have cut both. The signal is the headline's shape.

The precision half of this file is the point. The first version of the pattern matched
`(^|[ :|-])(live|latest|updates?)([ :|-]|$)` and caught mostly REAL incidents —
publisher taglines ("… | Shafaq News | Latest breaking news") and ordinary adjectives
("At least 7 killed in latest strikes"). Every string in TestPrecision is a real
headline from that corpus that the loose version wrongly suppressed.
"""

import pytest

from src.core.alerts import aftermath_kind, evaluate_alert_tier

# Real headlines, verbatim from the 2026-08-04..10 alert corpus.
AFTERMATH = [
    ("Canyon Park West Shooting: one week later - KMVT", "retrospective"),
    ("2 Years After Kursk Incursion, Russia Raises Civilian Death Toll to More Than 600",
     "retrospective"),
    ("Gaza holds funeral for 112 victims recovered nearly three years after 2023 airstrike",
     "retrospective"),
    ("Iran-US war latest: Trump military chief seeks 'off-ramp' from conflict", "desk_label"),
    ("Ukraine war latest: Ukrainian-made ballistic missiles could hit Russia in the fall",
     "desk_label"),
    ("Ukraine Latest: Russia Makes Slow Gains as Missile Attacks Intensify - georgiatoday.ge",
     "desk_label"),
    ("Ukraine live: Zelensky says interceptors 'could have saved lives'", "desk_label"),
    ("US-Israel-Iran War Latest Live News: UAE Vessel Targeted by Missile", "desk_label"),
    ("Russia-Ukraine War Latest News: Russia Strikes Odesa Port Fuel Facilities", "desk_label"),
    ("War update: 233 combat clashes on frontline over past day", "desk_label"),
    ("Iran War Timeline: Key Moments and Attacks In U.S. and Israel's Campaign", "desk_label"),
    # Running war diary — the marker is a day counter, not a desk word. This one paged
    # as ALERT on 2026-08-10 (run 20260810T105738) before the pattern existed.
    ("Ukraine Invasion Day 1,628: 135/151 RU drones neutralized - Daily Kos", "war_diary"),
    ("Israel-Hamas war, day 700: what we know", "war_diary"),
    ("Day 45 of the siege: aid convoys still blocked", "war_diary"),
    ("At least 4 people took heroic actions during In-N-Out shooting. Here's what we know",
     "explainer"),
    ("IDF expands probe into abnormally strong explosion that killed two reservists",
     "probe"),
    ("Ecuador charges ex-minister over presidential candidate assassination case", "judicial"),
    ("Woman charged after four men injured in central London stabbings", "judicial"),
    ("3 charged in shooting that killed innocent mother getting slushies for her kids",
     "judicial"),
    ("Nigerian army says it safely rescued 33 people kidnapped by gunmen", "resolution"),
    ("Dozens of Nigerians freed after 6 months in Jihadi captivity", "resolution"),
    ("In-N-Out reopens after horrific shooting left 3 dead", "resolution"),
    ("IDF to demolish home of terrorist who killed 18-year-old Yehuda Sherman in March",
     "resolution"),
]

# Real headlines the loose first draft wrongly matched. Every one is breaking news.
REAL_NEWS = [
    # "Latest"/"Live" inside the publisher's tagline, after the separator.
    "Houthi drones and missiles strike Yemen's Marib - Shafaq News | Latest breaking news in Iraq",
    "Yemen's Houthis hit Saudi oil tanker off Yanbu coast - Shafaq News | Latest breaking news",
    "Kharkiv strike toll rises as 5-year-old among 27 injured in Russian attack - Latest news from Azerbaijan",
    # "latest" as an ordinary adjective, mid-sentence, no colon.
    "At least 7 killed in latest Russia-Ukraine tit-for-tat strikes",
    "Houthis claim drone strike on Saudi oil refinery in latest Red Sea flare-up",
    "Ukraine Drone Strike Hits Russia's Yaroslavl Oil Refinery in Latest Attack on Energy Infrastructure",
    "Wildberries Warehouse Damaged in Latest Ukrainian Drone Strike",
    # Colon present, but the marker sits AFTER it — not a desk label.
    "Zelenskyy: 17 killed and 44 injured in Russia's latest massive attack on Kyiv",
    # "live" describing the event itself, not a live blog.
    "Mexican Influencer Shot to Death By Motorcycle Gunman On Live Stream - TMZ",
    "10 Photos of Valeria Marquez: Beauty Influencer Shot Dead on TikTok Live",
    # All-caps tabloid flourish before the colon — "HORROR" is not a desk word.
    "Mexican LIVE HORROR: Influencer Shot Dead As Thousands Watch In REAL-TIME",
    # Plain incidents.
    "'Massive' Russian strikes kill 17 in Kyiv, Zelenskyy says",
    "Car bomb hits Colombia highway day after new president sworn in",
    "Russian drone strikes passenger train travelling from Sumy to Kyiv",
    "Explosion heard in Saudi Arabia's Jubail as fires reported at energy facilities",
    "Fires seen in Russia's Belgorod amid reported Ukrainian drone attack",
    # A day count that is part of the sentence, not a label — no colon follows it.
    "Two killed on day 2 of protests in Nairobi",
    "Car bomb hits Colombia highway day after new president sworn in",
    "Day care centre shelled in Kharkiv, three children injured",
    # "charges" as a demolition term, not a court filing.
    "Troops planted explosive charges under the bridge before withdrawing",
    "Shaped charges used in attack on Kerch bridge, investigators say",
]


class TestRecall:
    @pytest.mark.parametrize("title,kind", AFTERMATH)
    def test_aftermath_headlines_are_named(self, title, kind):
        assert aftermath_kind(title) == kind


class TestPrecision:
    @pytest.mark.parametrize("title", REAL_NEWS)
    def test_real_incidents_are_not_gated(self, title):
        assert aftermath_kind(title) is None

    def test_empty_and_missing_titles_are_not_gated(self):
        assert aftermath_kind(None) is None
        assert aftermath_kind("") is None


def _event(**kw):
    base = {
        "severity_score": 95, "system_confidence": 0.40,
        "anchor_confidence": "LOW", "time_certainty": "previous_day",
        "event_type": "security_incident", "anchor_name_norm": None, "latitude": None,
    }
    base.update(kw)
    return base


class TestTierIntegration:
    def test_floor_promoted_roundup_is_withheld(self):
        """The exact production path: severity floor promotes, gate vetoes."""
        incident = _event(severity_score=95, source_title="Car bomb hits Colombia highway")
        roundup = _event(severity_score=95, source_title="Ukraine war latest: slow gains")
        assert evaluate_alert_tier(incident) == "ALERT"
        assert evaluate_alert_tier(roundup) is None

    def test_ladder_qualified_aftermath_is_also_withheld(self):
        """The gate is about the article, so it vetoes the ladder too, not just the floor."""
        ev = _event(severity_score=70, system_confidence=0.55, latitude=50.4,
                    source_title="Woman charged after four men injured in London stabbings")
        assert evaluate_alert_tier(dict(ev, source_title="Four men injured in London stabbings")) == "ALERT"
        assert evaluate_alert_tier(ev) is None

    def test_critical_is_exempt(self):
        """A roundup is sometimes the only carrier of a major development."""
        ev = _event(severity_score=100, system_confidence=0.80, latitude=46.4,
                    time_certainty="same_day",
                    source_title="Russia-Ukraine War Latest News: Odesa Port Struck")
        assert evaluate_alert_tier(ev) == "CRITICAL"

    def test_watch_is_gated_too(self):
        ev = _event(severity_score=50, system_confidence=0.45, time_certainty="same_day")
        assert evaluate_alert_tier(dict(ev, source_title="Blast reported in Kyiv")) == "WATCH"
        assert evaluate_alert_tier(
            dict(ev, source_title="Kyiv blast: one week later")) is None

    def test_missing_title_leaves_the_tier_alone(self):
        # Events without a headline must not be silently un-alerted.
        assert evaluate_alert_tier(_event(severity_score=95)) == "ALERT"

    def test_advisories_bypass_the_gate(self):
        # The advisory path returns before the gate; a standing advisory is not an
        # article about an old incident.
        ev = _event(severity_score=90, event_type="travel_advisory",
                    source_title="Updates: US raises Nigeria travel advisory to Level 4")
        assert evaluate_alert_tier(ev) == "ALERT"
