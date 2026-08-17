"""Pass D must not let an anchor overrule the classifier's country.

Regression for the 2026-08-17 run: "Russia targets Danube port after one of Ukraine's
largest aerial attacks" resolved to LNZ at MEDIUM — Linz is genuinely a Danube port —
and Pass D stored the event as country_iso='AT' with anchor LNZ. Pass E rejected the
identical anchor moments later ("country disagrees with classifier (UA)"), so the two
passes were enforcing different rules and the one that writes first won: the row landed
in the geo distribution as Austria and would have gone into Austria's SITREP.

Measured before the change: over the 14 days to 2026-08-17, 1 of 142 anchored events
disagreed with its classifier country, and none of them had paged — so the veto costs
nothing that was previously right.
"""

import src.pipeline.pass_d_score as pd


class AnchorMasterConn:
    """Returns one anchor_master row: (czib_flag, latitude, longitude, country_iso)."""

    def __init__(self, row):
        self.row = row
        self.queried = False

    def execute(self, sql, params=None):
        self.queried = True
        return AnchorMasterCursor(self.row)


class AnchorMasterCursor:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class TestCountryDisagreementRejectsAnchor:
    def test_contradicting_anchor_is_dropped(self, monkeypatch):
        monkeypatch.setattr(pd, "normalize_anchor", lambda raw, db: ("LNZ", 0.65))
        conn = AnchorMasterConn((False, 48.23, 14.19, "AT"))

        anchor = pd.resolve_anchor_for_event(conn, {
            "anchor_name_raw": "Danube port", "country_iso": "UA",
        })

        assert anchor["norm"] is None
        assert anchor["country_iso"] == "UA"

    def test_rejected_anchor_leaves_no_residue(self, monkeypatch):
        """Its coordinates and CZIB flag must go with it, not just its country.

        A kept IATA code would carry the bad match into the suppression key, the CZIB
        flag and the airspace-impact block, and stray coordinates would place the event
        on the wrong continent while the row still read UA.
        """
        monkeypatch.setattr(pd, "normalize_anchor", lambda raw, db: ("LNZ", 0.65))
        conn = AnchorMasterConn((True, 48.23, 14.19, "AT"))

        anchor = pd.resolve_anchor_for_event(conn, {
            "anchor_name_raw": "Danube port", "country_iso": "UA",
        })

        assert anchor["czib_flag"] is False
        assert anchor["latitude"] is None and anchor["longitude"] is None
        assert anchor["confidence"] == 0.0
        assert anchor["level"] == "LOW"

    def test_agreeing_anchor_is_kept_with_its_data(self, monkeypatch):
        monkeypatch.setattr(pd, "normalize_anchor", lambda raw, db: ("KBP", 0.9))
        conn = AnchorMasterConn((True, 50.34, 30.89, "UA"))

        anchor = pd.resolve_anchor_for_event(conn, {
            "anchor_name_raw": "Boryspil International Airport", "country_iso": "UA",
        })

        assert anchor["norm"] == "KBP"
        assert anchor["czib_flag"] is True
        assert anchor["latitude"] == 50.34
        assert anchor["country_iso"] == "UA"

    def test_anchor_still_supplies_country_when_classifier_has_none(self, monkeypatch):
        """No classifier country means nothing to contradict — anchor_master is curated."""
        monkeypatch.setattr(pd, "normalize_anchor", lambda raw, db: ("SAH", 0.8))
        conn = AnchorMasterConn((False, 15.48, 44.22, "YE"))

        anchor = pd.resolve_anchor_for_event(conn, {
            "anchor_name_raw": "Sanaa airport", "country_iso": None,
        })

        assert anchor["norm"] == "SAH"
        assert anchor["country_iso"] == "YE"

    def test_rejected_anchor_falls_through_to_gazetteer(self, monkeypatch):
        """The event keeps the location it would have had without the fuzzy match.

        Coordinates come from the city gazetteer under the classifier's own country, so
        dropping the anchor loses the wrong airport, not the event's place on the map.
        """
        monkeypatch.setattr(pd, "normalize_anchor", lambda raw, db: ("VAR", 0.6))
        conn = AnchorMasterConn((False, 43.23, 27.83, "BG"))

        anchor = pd.resolve_anchor_for_event(conn, {
            "anchor_name_raw": "Kharkiv", "country_iso": "UA",
        })

        assert anchor["norm"] is None
        assert anchor["country_iso"] == "UA"
        assert anchor["latitude"] is not None and anchor["longitude"] is not None
