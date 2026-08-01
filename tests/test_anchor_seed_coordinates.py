"""
db/anchors.json is the anchor seed — the coordinates every proximity claim in the
product ultimately rests on. It was built by geocoding city names, and six
airports landed in the wrong country by doing exactly what that method does
wrong: Athens in Georgia, Birmingham in Alabama, Brussels in Ontario, Panama City
in Florida, Alexandria in Alessandria, Santiago in Brazil.

The FIR reference (config/airspace.json) gives us an independent way to catch the
whole class: an airport must sit inside a flight information region of its own
country.
"""

import json
from pathlib import Path

import pytest

from src.core.airspace import fir_for_point

_ANCHORS = json.loads((Path(__file__).resolve().parents[1] / "db" / "anchors.json")
                      .read_text(encoding="utf-8"))

# Anchors whose coordinates are right but whose FIR box belongs to a neighbour.
# Two distinct reasons, both benign:
#   - the reference has no FIR for that country at all (Canada), so the nearest
#     box wins or nothing does;
#   - FIR boxes are rectangles over irregular borders, and Luxembourg genuinely
#     lies inside Brussels FIR.
_KNOWN_BORDER_CASES = {
    "YUL", "YVR", "YYZ",              # Canada — no Canadian FIR in the reference
    "LUX",                            # really is inside Brussels FIR
    "SXB", "PRN", "OUA", "COO",       # border-hugging airports
    "ASF", "OVB",                     # Russian airports inside the Kazakh box
}


def _located_anchors():
    for a in _ANCHORS:
        iso = (a.get("country") or "").strip().upper()
        if iso and iso != "XX" and a.get("lat") is not None:
            yield a, iso


@pytest.mark.parametrize("iata", ["ATH", "BHX", "BRU", "PTY", "HBE", "SCL"])
def test_previously_corrupt_anchors_are_in_their_own_country(iata):
    """The six that were wrong, pinned by name so a bad re-seed is loud."""
    anchor = next(a for a in _ANCHORS
                  if a.get("iata") == iata and a.get("type") == "Airport")
    fir = fir_for_point(anchor["lat"], anchor["lon"], anchor["country"])
    assert fir is not None and anchor["country"] in fir["countries"], (
        f"{iata} ({anchor['lat']},{anchor['lon']}) is not in {anchor['country']} airspace")


def test_no_new_anchor_lands_in_a_foreign_country():
    offenders = []
    for anchor, iso in _located_anchors():
        if anchor.get("iata") in _KNOWN_BORDER_CASES:
            continue
        fir = fir_for_point(anchor["lat"], anchor["lon"], iso)
        if fir is None or iso not in fir["countries"]:
            where = fir["icao"] if fir else "no FIR"
            offenders.append(
                f"{anchor.get('iata')} ({anchor.get('city')}, {iso}) at "
                f"{anchor['lat']},{anchor['lon']} → {where}"
            )
    assert not offenders, ("anchors outside their own country's airspace: "
                           + "; ".join(offenders))


def test_coordinates_are_within_range():
    for anchor, _ in _located_anchors():
        assert -90 <= anchor["lat"] <= 90, anchor.get("iata")
        assert -180 <= anchor["lon"] <= 180, anchor.get("iata")
