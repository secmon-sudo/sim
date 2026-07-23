"""Tests for the EASA CZIB (Conflict Zone Information Bulletin) sync.

CZIB is the one authoritative feed in the pipeline that says "this airspace is
dangerous" in an aviation regulator's own words, and a matching zone adds
CZIB_BONUS to an event's severity. Two failure modes matter: a country string
EASA phrases differently than our map silently drops the zone (the event then
scores as if no conflict zone existed), and one malformed row aborting the whole
batch would take every zone down with it.
"""

from unittest.mock import MagicMock, patch

import httpx

from src.services.czib_client import (
    EASA_CZIB_URL,
    _parse_countries,
    fetch_czib_data,
    sync_czib_to_db,
)


def _response(status: int, **kwargs) -> httpx.Response:
    """A Response with its request attached — raise_for_status() needs it."""
    return httpx.Response(status, request=httpx.Request("GET", EASA_CZIB_URL), **kwargs)


class TestCountryParsing:
    def test_single_country(self):
        assert _parse_countries("Iran") == ["IR"]

    def test_comma_separated_list(self):
        assert _parse_countries("Iran, Iraq, Syria") == ["IR", "IQ", "SY"]

    def test_case_and_whitespace_insensitive(self):
        assert _parse_countries("  IRAN ,\tiraq ") == ["IR", "IQ"]

    def test_russia_aliases_both_map(self):
        # EASA has used both spellings across bulletins.
        assert _parse_countries("Russia") == ["RU"]
        assert _parse_countries("Russian Federation") == ["RU"]

    def test_unknown_country_is_dropped_not_guessed(self):
        # A wrong ISO code would attach the bonus to the wrong country's events.
        assert _parse_countries("Atlantis") == []

    def test_known_and_unknown_mixed_keeps_the_known(self):
        assert _parse_countries("Iran, Atlantis, Yemen") == ["IR", "YE"]

    def test_empty_and_none(self):
        assert _parse_countries("") == []
        assert _parse_countries(None) == []

    def test_gulf_and_conflict_countries_are_covered(self):
        # The countries SIM reports on most; a gap here is a silent scoring miss.
        for name, iso in [("Iran", "IR"), ("Iraq", "IQ"), ("Israel", "IL"),
                          ("Yemen", "YE"), ("Lebanon", "LB"), ("Ukraine", "UA"),
                          ("Libya", "LY"), ("Sudan", "SD"), ("Afghanistan", "AF"),
                          ("Pakistan", "PK"), ("Mali", "ML")]:
            assert _parse_countries(name) == [iso], f"{name} not mapped"


class TestFetch:
    def test_returns_conflict_zones_array(self):
        payload = {"conflict_zones": [{"Nid": "1", "name": "Zone A"}]}
        with patch("src.services.czib_client.httpx.get",
                   return_value=_response(200, json=payload)):
            assert fetch_czib_data() == [{"Nid": "1", "name": "Zone A"}]

    def test_http_error_returns_empty_not_raises(self):
        # EASA being down must never take the pipeline with it.
        with patch("src.services.czib_client.httpx.get",
                   return_value=_response(503, text="down")):
            assert fetch_czib_data() == []

    def test_network_error_returns_empty(self):
        with patch("src.services.czib_client.httpx.get",
                   side_effect=httpx.ConnectError("no route")):
            assert fetch_czib_data() == []

    def test_unexpected_shape_returns_empty(self):
        with patch("src.services.czib_client.httpx.get",
                   return_value=_response(200, json={"unexpected": True})):
            assert fetch_czib_data() == []


def _db():
    db = MagicMock()
    db.transaction.return_value.__enter__ = lambda s: None
    db.transaction.return_value.__exit__ = lambda s, *a: False
    return db


class TestSync:
    def test_empty_fetch_writes_nothing(self):
        db = _db()
        with patch("src.services.czib_client.fetch_czib_data", return_value=[]):
            stats = sync_czib_to_db(db)
        assert stats == {"fetched": 0, "inserted": 0, "updated": 0}
        db.execute.assert_not_called()

    def test_zone_is_upserted_with_parsed_countries(self):
        db = _db()
        db.execute.return_value.fetchone.return_value = ("inserted",)
        zones = [{
            "Nid": "77", "name": "Iran FIR", "status": "Active",
            "country": "Iran, Iraq", "coordinates": "35N 51E",
            "issued_date": "2026-07-01T00:00:00+0300",
            "valid_until_date": "2026-12-31", "field_easa_valid_until_descr": "until further notice",
        }]
        with patch("src.services.czib_client.fetch_czib_data", return_value=zones):
            stats = sync_czib_to_db(db)
        assert stats["fetched"] == 1
        params = db.execute.call_args.args[1]
        assert "77" in params and ["IR", "IQ"] in params

    def test_one_bad_row_does_not_abort_the_batch(self):
        # Each upsert runs in its own transaction precisely so a single
        # malformed bulletin cannot wipe out the rest of the sync.
        db = _db()
        db.transaction.side_effect = [
            RuntimeError("bad row"),
            MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: False),
        ]
        db.execute.return_value.fetchone.return_value = ("inserted",)
        zones = [
            {"Nid": "1", "name": "Broken", "country": "Iran"},
            {"Nid": "2", "name": "Good", "country": "Yemen"},
        ]
        with patch("src.services.czib_client.fetch_czib_data", return_value=zones):
            stats = sync_czib_to_db(db)
        assert stats["fetched"] == 2
        assert stats["inserted"] + stats["updated"] >= 1

    def test_aborted_pool_connection_is_rolled_back_first(self):
        # Pool connections can come back in an aborted transaction state; the
        # sync clears it before issuing statements.
        db = _db()
        with patch("src.services.czib_client.fetch_czib_data", return_value=[]):
            sync_czib_to_db(db)
        db.rollback.assert_called_once()

    def test_rollback_failure_does_not_stop_the_sync(self):
        db = _db()
        db.rollback.side_effect = RuntimeError("already closed")
        with patch("src.services.czib_client.fetch_czib_data", return_value=[]):
            assert sync_czib_to_db(db)["fetched"] == 0

    def test_unparseable_issued_date_does_not_drop_the_zone(self):
        # The bulletin still marks dangerous airspace even if its timestamp is
        # in a format we don't recognise.
        db = _db()
        db.execute.return_value.fetchone.return_value = ("inserted",)
        zones = [{"Nid": "9", "name": "Z", "country": "Iran", "issued_date": "not-a-date"}]
        with patch("src.services.czib_client.fetch_czib_data", return_value=zones):
            stats = sync_czib_to_db(db)
        assert stats["fetched"] == 1
        db.execute.assert_called()
