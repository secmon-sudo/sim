"""
SIM — Airspace Impact Analysis

Answers the question a SITREP could never answer before: a drone hits eastern
Poland — *which airspace is that, and which commercial airports sit near it?*

The pipeline already catches aviation NEWS (ingest_filters._is_flight_disruption,
the aviation nexus bonus, the regional-disruption block). What was missing is the
link between an event's GEOGRAPHY and the airspace structure around it. This
module supplies exactly that, deterministically:

    event location  ->  containing FIR
                    ->  neighbouring FIRs (+ active EASA CZIB restrictions)
                    ->  commercial airports within a radius, by distance

Everything here is geometry over a vendored reference table
(config/airspace.json). Nothing is inferred, so the LLM narrating the SITREP can
be handed these facts and forbidden from inventing any others. The output is a
PROXIMITY assessment and is labelled as such everywhere it surfaces — it says
"this event happened inside Warszawa FIR, 28 km from Lublin airport", never
"flights were suspended". Confirmed disruptions come from the news side of the
pipeline, not from here.

FIR extents are axis-aligned bounding boxes (~0.1 degree). Real FIR boundaries
are polygons, so boxes overlap along borders; `fir_for_point` resolves that with
the event's country first and box size second. Good enough to name an airspace,
deliberately not good enough to navigate by.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.geo import geo_coords, haversine_km

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

with open(_CONFIG_DIR / "airspace.json", encoding="utf-8") as _f:
    _DATA = json.load(_f)

FIRS: List[Dict[str, Any]] = _DATA["firs"]
AIRPORTS: List[Dict[str, Any]] = _DATA["airports"]

_FIR_BY_ICAO: Dict[str, Dict[str, Any]] = {f["icao"]: f for f in FIRS}
_FIRS_BY_COUNTRY: Dict[str, List[Dict[str, Any]]] = {}
for _fir in FIRS:
    for _iso in _fir["countries"]:
        _FIRS_BY_COUNTRY.setdefault(_iso, []).append(_fir)

_AIRPORTS_BY_COUNTRY: Dict[str, List[Dict[str, Any]]] = {}
for _ap in AIRPORTS:
    _AIRPORTS_BY_COUNTRY.setdefault(_ap["country"], []).append(_ap)

# Airport tiers, most significant first — used to pick which airports represent a
# country when the event has no coordinate to measure distance from.
_TIER_RANK = {"hub": 0, "major": 1, "regional": 2}

with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as _f:
    _SITREP_CFG = json.load(_f).get("sitrep", {})

AIRSPACE_ENABLED = bool(_SITREP_CFG.get("airspace_enabled", True))
AIRSPACE_RADIUS_KM = float(_SITREP_CFG.get("airspace_radius_km", 300))
AIRSPACE_MAX_CLUSTERS = int(_SITREP_CFG.get("airspace_max_clusters", 5))
AIRSPACE_MAX_AIRPORTS = int(_SITREP_CFG.get("airspace_max_airports", 6))

# Event types whose geography plausibly bears on airspace: anything that puts
# ordnance, drones or debris in the air, plus the aviation-specific codes. A
# cluster outside this set still qualifies if its text trips the production
# flight-disruption gate (see `is_airspace_relevant`), so a mislabelled but
# genuinely aviation-related event is not lost.
AIRSPACE_THREAT_EVENT_TYPES = {
    "drone_airport_attack",
    "drone_attack_critical_infra",
    "drone_energy_attack",
    "drone_military_base_attack",
    "drone_port_attack",
    "missile_strike",
    "military_action",
    "war_escalation",
    "ceasefire_violation",
    "geopolitical_conflict",
    "insurgency_attack",
    "air_traffic_controller_threat",
    "aviation_personnel_attack",
    "mass_casualty_event",
}


def _bbox_area(fir: Dict[str, Any]) -> float:
    min_lat, min_lon, max_lat, max_lon = fir["bbox"]
    return (max_lat - min_lat) * (max_lon - min_lon)


def _in_bbox(lat: float, lon: float, bbox: List[float]) -> bool:
    return bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]


def fir_for_point(lat: float, lon: float,
                  country_iso: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The FIR containing a point, or None if outside the reference footprint.

    Bounding boxes overlap along shared borders, so a border point can sit in
    two or three boxes. The event's own country decides first — an event
    attributed to Poland belongs in Polish airspace even when the Lviv FIR box
    reaches over the border — and the tightest box breaks any remaining tie.
    """
    candidates = [f for f in FIRS if _in_bbox(lat, lon, f["bbox"])]
    if not candidates:
        return None
    iso = (country_iso or "").strip().upper()
    candidates.sort(key=lambda f: (iso not in f["countries"] if iso else False,
                                   _bbox_area(f)))
    return candidates[0]


def firs_for_country(country_iso: str) -> List[Dict[str, Any]]:
    """Every FIR covering a country's territory (a FIR may span several states)."""
    return list(_FIRS_BY_COUNTRY.get((country_iso or "").strip().upper(), []))


def neighbor_firs(fir: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [_FIR_BY_ICAO[n] for n in fir.get("neighbors", []) if n in _FIR_BY_ICAO]


def nearby_airports(lat: float, lon: float,
                    radius_km: float = AIRSPACE_RADIUS_KM,
                    limit: int = AIRSPACE_MAX_AIRPORTS) -> List[Dict[str, Any]]:
    """Commercial airports within `radius_km` of a point, nearest first."""
    hits = []
    for ap in AIRPORTS:
        dist = haversine_km(lat, lon, ap["lat"], ap["lon"])
        if dist <= radius_km:
            hits.append((dist, ap))
    hits.sort(key=lambda h: (h[0], _TIER_RANK.get(h[1]["tier"], 3)))
    return [
        {"iata": ap["iata"], "icao": ap["icao"], "name": ap["name_tr"],
         "city": ap["city"], "country": ap["country"], "tier": ap["tier"],
         "distance_km": round(dist)}
        for dist, ap in hits[:limit]
    ]


def country_airports(country_iso: str,
                     limit: int = AIRSPACE_MAX_AIRPORTS) -> List[Dict[str, Any]]:
    """A country's most significant airports, for events with no coordinate.

    No distance field: without a point there is nothing to measure from, and a
    fabricated distance is exactly the kind of detail this module exists to
    prevent.
    """
    rows = sorted(_AIRPORTS_BY_COUNTRY.get((country_iso or "").strip().upper(), []),
                  key=lambda a: (_TIER_RANK.get(a["tier"], 3), a["iata"]))
    return [
        {"iata": ap["iata"], "icao": ap["icao"], "name": ap["name_tr"],
         "city": ap["city"], "country": ap["country"], "tier": ap["tier"]}
        for ap in rows[:limit]
    ]


def _airport_by_location_name(text: str,
                              country_iso: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Match an anchor string like 'Rzeszów–Jasionka Airport' to a known airport."""
    low = (text or "").strip().lower()
    if len(low) < 3:
        return None
    iso = (country_iso or "").strip().upper()
    pool = _AIRPORTS_BY_COUNTRY.get(iso, []) if iso else AIRPORTS
    for ap in pool:
        for candidate in (ap["city"], ap["name"], ap["name_tr"]):
            cand = (candidate or "").strip().lower()
            if cand and (cand in low or low in cand):
                return ap
    return None


def resolve_cluster_point(cluster: Dict[str, Any]) -> Optional[Tuple[float, float, str]]:
    """Best available (lat, lon, source) for a SITREP cluster, or None.

    Ordered by trustworthiness: the coordinate Pass D/E already resolved, then
    the curated city gazetteer, then an airport name embedded in the anchor
    string. `source` is carried through to the report so a reader can see how
    precise the placement is.
    """
    iso = (cluster.get("country_iso") or "").strip().upper() or None

    lat, lon = cluster.get("latitude"), cluster.get("longitude")
    if lat is not None and lon is not None:
        try:
            return (float(lat), float(lon), "event")
        except (TypeError, ValueError):
            logger.debug("Cluster carried unusable coordinates: %r / %r", lat, lon)

    location = cluster.get("location") or ""
    coords = geo_coords(location, iso)
    if coords:
        return (coords[0], coords[1], "gazetteer")

    ap = _airport_by_location_name(location, iso)
    if ap:
        return (ap["lat"], ap["lon"], "airport")

    return None


def is_airspace_relevant(cluster: Dict[str, Any], flight_disruption_check=None) -> bool:
    """Whether a cluster's geography is worth an airspace assessment.

    `flight_disruption_check` is injected (rather than imported) to keep this
    core module free of a dependency on the ingest pipeline; callers pass
    ingest_filters._is_flight_disruption so the definition of "aviation event"
    stays identical to the one used at ingest.
    """
    if (cluster.get("event_type") or "") in AIRSPACE_THREAT_EVENT_TYPES:
        return True
    if flight_disruption_check is None:
        return False
    blob = f"{cluster.get('snippet') or ''} {cluster.get('location') or ''}"
    return bool(flight_disruption_check(blob))


def _neighbor_views(firs: List[Dict[str, Any]],
                    czib_by_iso: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Neighbour FIRs with the EASA-restricted ones first — a reader scanning the
    block cares about the restricted borders, not the alphabet."""
    views = [_fir_view(f, czib_by_iso) for f in firs]
    views.sort(key=lambda v: (not v["czib_active"], v["icao"]))
    return views


def _fir_view(fir: Dict[str, Any], czib_by_iso: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Shape a FIR for the report, flagging any active EASA restriction over it.

    CZIB bulletins are issued per state, so a FIR is flagged when ANY state it
    covers has an active bulletin. Unlike everything else here this is not a
    proximity inference — it is an authoritative EASA advisory — and the report
    presents it as such.
    """
    czibs = []
    for iso in fir["countries"]:
        for z in czib_by_iso.get(iso, []):
            if z not in czibs:
                czibs.append(z)
    return {
        "icao": fir["icao"],
        "name": fir["name_tr"],
        "countries": fir["countries"],
        "czib_active": bool(czibs),
        "czib": [
            {"name": z.get("name"), "valid_until": z.get("valid_until")}
            for z in czibs[:3]
        ],
    }


def assess_cluster(cluster: Dict[str, Any],
                   czib_by_iso: Dict[str, List[Dict[str, Any]]],
                   country_iso: Optional[str] = None,
                   radius_km: float = AIRSPACE_RADIUS_KM,
                   max_airports: int = AIRSPACE_MAX_AIRPORTS) -> Optional[Dict[str, Any]]:
    """Airspace exposure for one cluster, or None when it cannot be placed."""
    iso = (cluster.get("country_iso") or country_iso or "").strip().upper()
    point = resolve_cluster_point(cluster)

    if point:
        lat, lon, source = point
        fir = fir_for_point(lat, lon, iso)
        if fir is None:
            logger.debug("No FIR covers %.3f/%.3f — falling back to country scope", lat, lon)
        else:
            view = _fir_view(fir, czib_by_iso)
            return {
                "location": cluster.get("location"),
                "event_type": cluster.get("event_type"),
                "severity": cluster.get("severity"),
                "scope": "point",
                "point": {"lat": round(lat, 4), "lon": round(lon, 4), "source": source},
                "fir": view,
                "firs": [view],
                "neighbor_firs": _neighbor_views(neighbor_firs(fir), czib_by_iso),
                "airports": nearby_airports(lat, lon, radius_km, max_airports),
                "radius_km": round(radius_km),
            }

    country_firs = firs_for_country(iso)
    if not country_firs:
        return None
    # A country is not one airspace. Russia has six FIRs and India four, so
    # naming a single one for an event we could not place is a guess presented
    # as a fact — and the guess was whichever FIR sorted first (a Kashmir event
    # came out as "Mumbai FIR"). List them all instead: less precise, but true.
    views = [_fir_view(f, czib_by_iso) for f in country_firs]
    views.sort(key=lambda v: (not v["czib_active"], v["icao"]))
    # Neighbours of the whole country = every bordering FIR that is not itself
    # one of the country's own.
    own = {f["icao"] for f in country_firs}
    outside = [n for f in country_firs for n in neighbor_firs(f) if n["icao"] not in own]
    seen, unique = set(), []
    for n in outside:
        if n["icao"] not in seen:
            seen.add(n["icao"])
            unique.append(n)
    return {
        "location": cluster.get("location"),
        "event_type": cluster.get("event_type"),
        "severity": cluster.get("severity"),
        "scope": "country",
        "point": None,
        # `fir` stays the lead entry for consumers that show one line; `firs`
        # carries the honest full set.
        "fir": views[0],
        "firs": views,
        "neighbor_firs": _neighbor_views(unique, czib_by_iso),
        "airports": country_airports(iso, max_airports),
        "radius_km": None,
    }


def build_airspace_assessment(clusters: List[Dict[str, Any]], country_iso: str,
                              czib_by_iso: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                              max_clusters: int = AIRSPACE_MAX_CLUSTERS
                              ) -> Optional[Dict[str, Any]]:
    """Airspace exposure for a country's most severe airspace-relevant clusters.

    Returns None when the feature is off or nothing qualifies, so every consumer
    can treat "no assessment" as a single falsy case.
    """
    if not AIRSPACE_ENABLED:
        return None

    from src.pipeline.ingest_filters import _is_flight_disruption

    czib_by_iso = czib_by_iso or {}
    iso = (country_iso or "").strip().upper()

    relevant = [c for c in clusters if is_airspace_relevant(c, _is_flight_disruption)]
    relevant.sort(key=lambda c: -(c.get("severity") or 0))

    assessments = []
    seen = set()
    seen_firs = set()
    for cluster in relevant:
        assessment = assess_cluster(cluster, czib_by_iso, iso)
        if not assessment:
            continue
        icao = assessment["fir"]["icao"]
        # A country-scope card adds nothing once a located event already put a
        # card on that same FIR — it would repeat the airspace with less detail.
        if assessment["scope"] == "country" and icao in seen_firs:
            continue
        # One card per place: repeated shelling of the same city is one airspace
        # picture, and five identical FIR/airport lists would bury the report.
        key = (icao, (assessment.get("location") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        seen_firs.add(icao)
        assessments.append(assessment)
        if len(assessments) >= max_clusters:
            break

    if not assessments:
        return None
    return {"country_iso": iso, "assessments": assessments}


def compact_for_prompt(assessment: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Strip the assessment down to what the narrator actually needs.

    The full object is shaped for the HTML block, where every field earns its
    place. In the prompt it is mostly repetition: each card restates the same
    nine-entry neighbour list, and a neighbour with no restriction contributes
    nothing but its code. Measured on a full 25-cluster day this is ~40% of the
    airspace payload for zero added meaning — and the SITREP prompt already sits
    above the Groq request ceiling, so the remaining headroom belongs to events.

    Only the prompt uses this; `render_sitrep_html` keeps the rich object.
    """
    if not assessment or not assessment.get("assessments"):
        return None

    def _fir(view: Dict[str, Any]) -> Dict[str, Any]:
        out = {"icao": view["icao"], "name": view["name"]}
        if view.get("czib_active"):
            out["easa_czib_aktif"] = True
            out["czib"] = view.get("czib")
        return out

    compact = []
    for item in assessment["assessments"]:
        neighbors = item.get("neighbor_firs") or []
        entry = {
            "yer": item.get("location"),
            "kapsam": item["scope"],
            "fir": _fir(item["fir"]),
            # Restricted neighbours keep their bulletin; the rest are just codes.
            "kisitlamali_komsu_firlar": [_fir(n) for n in neighbors if n.get("czib_active")],
            "diger_komsu_firlar": [n["icao"] for n in neighbors if not n.get("czib_active")],
            "havalimanlari": [
                {k: v for k, v in a.items() if k in ("iata", "name", "distance_km")}
                for a in (item.get("airports") or [])
            ],
        }
        if item.get("radius_km"):
            entry["yaricap_km"] = item["radius_km"]
        compact.append(entry)
    return {"country_iso": assessment.get("country_iso"), "assessments": compact}


def summarize_assessment(assessment: Optional[Dict[str, Any]]) -> str:
    """One Turkish line for the cross-country digest, or '' when there is nothing."""
    if not assessment or not assessment.get("assessments"):
        return ""
    first = assessment["assessments"][0]
    fir = first["fir"]
    firs = first.get("firs") or [fir]
    if first.get("scope") == "country" and len(firs) > 1:
        parts = ["ülkenin hava sahaları: " + ", ".join(f["icao"] for f in firs)]
    else:
        parts = [f"{fir['name']} ({fir['icao']})"]

    restricted = [f["icao"] for f in first.get("neighbor_firs", []) if f.get("czib_active")]
    if fir.get("czib_active"):
        parts.append("bu FIR için aktif EASA CZIB kısıtlaması var")
    elif restricted:
        parts.append("komşu " + "/".join(restricted[:3]) + " FIR'larında aktif EASA CZIB kısıtlaması")

    airports = first.get("airports") or []
    if airports and first.get("scope") == "point":
        listed = ", ".join(
            f"{a['iata']} {a['name']} {a['distance_km']} km" for a in airports[:3]
        )
        parts.append(f"{first['radius_km']} km yarıçapta {listed}")
    elif airports:
        parts.append("başlıca havalimanları: "
                     + ", ".join(f"{a['iata']} {a['name']}" for a in airports[:3]))

    return " · ".join(parts)
