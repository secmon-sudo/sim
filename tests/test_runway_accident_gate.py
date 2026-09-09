"""A runway event is a security event only when someone was there who should not be.

SIM is a security monitor, not a safety one. The catalog has exactly one runway code,
so overruns, excursions, ATC near-misses and genuine intrusions all classify as
'runway_incursion' — and the type is not in SAFETY_EVENT_TYPES, so nothing ever capped
it. Measured across all 37 events carrying it since 2026-06-26: 18 paged at a mean
severity of 83.8, thirteen of those were accidents (one Miami cargo overrun produced
SEVEN cards), four were operational near-misses, and one was a security event.

The set's clearest security case never paged at all: a trespasser shut a UK runway and
drew a mayday call, and it sat at tier NULL while seven cards went out about an
aircraft leaving the tarmac by itself.
"""

import pytest

from src.pipeline.pass_d_score import (
    RUNWAY_ACCIDENT_SEVERITY_CAP,
    ALERT_SEVERITY_MIN,
    apply_runway_accident_downrank,
)


def _cap(title: str, severity: int = 100) -> int:
    return apply_runway_accident_downrank("runway_incursion", severity,
                                          {"source_title": title})


# The two the lexicon is FOR — both real, both from the measured set.
@pytest.mark.parametrize("title", [
    "TUI plane issued mayday call after 'trespasser' shut runway at UK airport",
    "Frontier Airlines jet strikes person on runway at Denver International Airport",
])
def test_an_intrusion_keeps_its_severity(title):
    assert _cap(title) == 100


# The thirteen it is against, plus the operational four. All real headlines.
@pytest.mark.parametrize("title", [
    "NTSB says Amazon cargo jet hit van, car after runway overrun",
    "Amazon cargo plane overruns Miami airport runway and strikes 'multiple' vehicles",
    "Major US airport closed after Amazon cargo plane overshoots runway hitting vehicles",
    "All Five Killed in Miami Runway Crash Were on the Ground, Officials Say",
    "Plane Skids Off Runway at Mashhad Airport",
    "Vietnam Airlines 787 overrun triggers runway closure in Munich",
    "Enugu Air confirms safe evacuation after aircraft veers off Benin Airport runway",
    "Cirrus Vision Jet Strikes Construction Equipment During Takeoff Attempt at Idaho Airport",
    "Bristol Airport Suspends All Flight Operations Due to Critical Runway Defect",
    "Bristol Airport Resumes Flights Following Hours of Runway Repairs",
    "Gatwick runway closure leaves flights grounded after aircraft incident",
    # Operational: a real incursion, but nobody trespassed — ATC and pilots did this.
    "American 308 Miami Runway Incursion: A Call-Sign Mix-Up",
    "Fourth Runway Incursion at Sydney Airport Triggers Fresh ATSB Safety Investigation",
    "ATSB Investigates Qantas Runway Incursion at Sydney Airport",
    "Runway Incursion at Sao Paulo Guarulhos: American and Delta Jets Cross Path",
    "Transport watchdog investigates another near miss at Sydney airport",
])
def test_an_accident_is_capped_below_the_alert_floor(title):
    capped = _cap(title)
    assert capped == RUNWAY_ACCIDENT_SEVERITY_CAP
    assert capped < ALERT_SEVERITY_MIN, "a capped accident must not be able to page"


def test_the_word_incursion_alone_is_not_an_intrusion():
    """The trap this gate exists to avoid. 'Runway incursion' is the ICAO term for an
    aircraft or vehicle being on a runway it was not cleared for — usually a controller
    or a pilot getting it wrong. Matching the type name would cap nothing."""
    assert _cap("ATSB Launches Probe into Third Sydney Airport Runway Incursion") \
        == RUNWAY_ACCIDENT_SEVERITY_CAP


def test_other_types_are_untouched():
    for event_type in ("missile_strike", "drone_airport_attack", "airspace_closure"):
        assert apply_runway_accident_downrank(
            event_type, 100, {"source_title": "overrun"}) == 100


def test_a_low_score_is_not_raised():
    assert _cap("Plane Skids Off Runway at Mashhad Airport", severity=20) == 20


def test_a_missing_title_caps_rather_than_pages():
    # No headline is no evidence of intrusion, and the common case is the accident.
    assert apply_runway_accident_downrank("runway_incursion", 100, {}) \
        == RUNWAY_ACCIDENT_SEVERITY_CAP


def test_pass_e_applies_the_same_cap():
    """Pass E recomputes severity and overwrites Pass D's, so a cap Pass D applies and
    Pass E does not is a cap that silently lifts on reconcile."""
    import inspect
    import src.pipeline.pass_e_reconcile as pe
    assert "apply_runway_accident_downrank" in inspect.getsource(pe.reconcile_single_event)
