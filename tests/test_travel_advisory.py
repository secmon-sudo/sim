"""Tests for multi-agency travel advisory ingestion + alerting.

- _parse_advisory_level / _is_advisory_worth_ingesting understand US "Level N" AND the
  phrase-based wording of UK/CA/AU/NZ.
- evaluate_alert_tier routes country-level advisories to Telegram despite no anchor.
- travel_advisory is no longer capped by the generic-umbrella incident gate.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import src.pipeline.pass_a_ingest as pa
from src.core.alerts import evaluate_alert_tier
from src.pipeline.pass_a_ingest import (
    _is_advisory_worth_ingesting,
    _parse_advisory_level,
    fetch_travel_advisories,
)
from src.pipeline.pass_d_score import compute_severity


class TestParseAdvisoryLevel:
    def test_us_numeric(self):
        assert _parse_advisory_level("Somewhere - Level 4: Do Not Travel") == 4
        assert _parse_advisory_level("Placeland - Level 2: Exercise Caution") == 2

    def test_uk_phrases(self):
        assert _parse_advisory_level("Ukraine travel advice",
                                     "FCDO advise against all travel to Ukraine") == 4
        assert _parse_advisory_level("Egypt travel advice",
                                     "advise against all but essential travel to North Sinai") == 3

    def test_canada_australia_phrases(self):
        assert _parse_advisory_level("", "Avoid all travel to this country") == 4
        assert _parse_advisory_level("", "Reconsider your need to travel") == 3
        assert _parse_advisory_level("", "avoid non-essential travel") == 3

    def test_max_of_numeric_and_phrase(self):
        assert _parse_advisory_level("Level 2 update", "now advise against all travel") == 4

    def test_none(self):
        assert _parse_advisory_level("General travel news", "some update") == 0


class TestWorthIngesting:
    def test_high_level_ingested(self):
        assert _is_advisory_worth_ingesting("Country - Level 4: Do Not Travel", "") is True

    def test_uk_phrase_ingested(self):
        assert _is_advisory_worth_ingesting(
            "Iran travel advice", "The FCDO advise against all travel to Iran.") is True

    def test_no_change_skipped(self):
        assert _is_advisory_worth_ingesting(
            "Country - Level 4", "There are no changes to the advisory level.") is False

    def test_downgrade_skipped(self):
        assert _is_advisory_worth_ingesting(
            "Country - Level 3", "The advisory was downgraded from Level 4.") is False

    def test_low_level_no_upgrade_skipped(self):
        assert _is_advisory_worth_ingesting(
            "Country - Level 1: Exercise Normal Precautions", "routine update") is False


class TestAdvisoryAlertTier:
    def _adv(self, sev, etype="travel_advisory"):
        # No airport anchor, standing-advisory time — would fail normal gates.
        return {"severity_score": sev, "system_confidence": 0.6,
                "anchor_confidence": "LOW", "time_certainty": "this_week",
                "event_type": etype}

    def test_advisory_alerts_despite_no_anchor(self):
        assert evaluate_alert_tier(self._adv(60)) == "ALERT"

    def test_travel_ban_alerts(self):
        assert evaluate_alert_tier(self._adv(60, "travel_ban")) == "ALERT"

    def test_low_severity_advisory_is_watch(self):
        assert evaluate_alert_tier(self._adv(50)) == "WATCH"

    def test_non_advisory_unaffected(self):
        # A normal event with no anchor + this_week still gets nothing (gates intact).
        ev = self._adv(60, "geopolitical_conflict")
        assert evaluate_alert_tier(ev) is None


class TestAdvisoryNotGated:
    def _db(self, base):
        db = MagicMock()
        cur = MagicMock()
        cur.fetchone.return_value = (base,)
        db.execute.return_value = cur
        return db

    def test_travel_advisory_not_capped_by_incident_gate(self):
        # travel_advisory has no airport anchor + no casualties, but must NOT be capped
        # (it is not in GENERIC_UMBRELLA_TYPES) so its base severity survives.
        sev = compute_severity("travel_advisory", {"confidence": 0.0}, self._db(60), {})
        assert sev == 60


class TestFetchUkAtomFeed:
    def _atom(self, when):
        stamp = when.strftime("%Y-%m-%dT%H:%M:%SZ")
        return (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom">'
            '<title>Travel Advice Summary</title>'
            '<entry><title>Ukraine travel advice</title>'
            '<link rel="alternate" href="https://www.gov.uk/foreign-travel-advice/ukraine"/>'
            f'<updated>{stamp}</updated>'
            '<summary></summary></entry>'
            '</feed>'
        )

    def test_uk_atom_curated_ingest_and_dated_link(self, monkeypatch):
        # Empty-summary UK entry (curated high-risk country) must still ingest, parse the
        # Atom fields, and carry a date-stamped link so future updates aren't URL-deduped.
        now = datetime.now(timezone.utc)
        monkeypatch.setattr(pa, "SETTINGS",
                            {"sources": {"travel_advisory_feeds":
                                         ["https://www.gov.uk/foreign-travel-advice/ukraine.atom"]}})
        resp = MagicMock()
        resp.text = self._atom(now)
        monkeypatch.setattr(pa, "_http_get_with_retry", lambda *a, **k: resp)

        items = fetch_travel_advisories(stats={"age_filtered": 0, "queries_executed": 0})
        assert len(items) == 1
        it = items[0]
        assert it["source"] == "travel_advisory"
        assert it["title"] == "Ukraine travel advice"
        assert it["link"].startswith("https://www.gov.uk/foreign-travel-advice/ukraine#adv-")

    def test_us_style_still_gated_by_level(self, monkeypatch):
        # A non-UK (RSS) feed keeps the level gate: a Level 1 routine item is dropped.
        now = datetime.now(timezone.utc)
        rss = (
            '<?xml version="1.0"?><rss><channel>'
            '<item><title>Somewhere - Level 1: Exercise Normal Precautions</title>'
            '<link>https://travel.state.gov/x</link>'
            f'<pubDate>{now.strftime("%a, %d %b %Y %H:%M:%S +0000")}</pubDate>'
            '<description>routine update</description></item>'
            '</channel></rss>'
        )
        monkeypatch.setattr(pa, "SETTINGS",
                            {"sources": {"travel_advisory_feeds":
                                         ["https://travel.state.gov/_res/rss/TAsTWs.xml"]}})
        resp = MagicMock()
        resp.text = rss
        monkeypatch.setattr(pa, "_http_get_with_retry", lambda *a, **k: resp)

        items = fetch_travel_advisories(stats={"age_filtered": 0, "queries_executed": 0})
        assert items == []  # Level 1 routine -> filtered out
