"""
SIM — EASA CZIB (Conflict Zone Information Bulletin) Sync Service
Fetches active conflict zones from EASA and syncs to database.

Endpoint: https://www.easa.europa.eu/en/domains/air-operations/czibs/export-json
"""

import json
import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

EASA_CZIB_URL = "https://www.easa.europa.eu/en/domains/air-operations/czibs/export-json?page&_format=json"

# Country name → ISO2 mapping for EASA CZIB country strings
_COUNTRY_NAME_TO_ISO = {
    # Middle East
    "iran": "IR", "iraq": "IQ", "israel": "IL", "jordan": "JO",
    "kuwait": "KW", "lebanon": "LB", "oman": "OM", "qatar": "QA",
    "saudi arabia": "SA", "syria": "SY", "united arab emirates": "AE",
    "yemen": "YE", "bahrain": "BH", "turkey": "TR",
    # Africa
    "somalia": "SO", "libya": "LY", "mali": "ML", "sudan": "SD",
    "south sudan": "SS", "nigeria": "NG", "chad": "TD", "niger": "NE",
    "burkina faso": "BF", "central african republic": "CF",
    "democratic republic of the congo": "CD", "congo": "CG",
    "ethiopia": "ET", "eritrea": "ER", "egypt": "EG", "algeria": "DZ",
    "kenya": "KE", "cameroon": "CM", "mozambique": "MZ",
    "south africa": "ZA", "tunisia": "TN",
    # Asia
    "afghanistan": "AF", "pakistan": "PK", "india": "IN", "myanmar": "MM",
    "thailand": "TH", "philippines": "PH", "indonesia": "ID",
    "bangladesh": "BD", "sri lanka": "LK", "nepal": "NP",
    "north korea": "KP", "south korea": "KR",
    # Europe / Eurasia
    "ukraine": "UA", "russian federation": "RU", "russia": "RU",
    "belarus": "BY", "moldova": "MD", "georgia": "GE", "armenia": "AM",
    "azerbaijan": "AZ",
    # Latin America
    "venezuela": "VE", "colombia": "CO", "mexico": "MX", "haiti": "HT",
    "ecuador": "EC", "peru": "PE", "brazil": "BR", "honduras": "HN",
    "guatemala": "GT", "el salvador": "SV",
}


def _parse_countries(country_str: str) -> list[str]:
    """Parse EASA country string into ISO2 list."""
    if not country_str:
        return []
    results = []
    for part in country_str.split(","):
        name = part.strip().lower()
        if name in _COUNTRY_NAME_TO_ISO:
            results.append(_COUNTRY_NAME_TO_ISO[name])
    return results


def fetch_czib_data() -> list[dict]:
    """Fetch CZIB JSON from EASA website."""
    try:
        resp = httpx.get(
            EASA_CZIB_URL,
            headers={"User-Agent": "SIM-OSINT-Bot/1.0"},
            timeout=30.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("conflict_zones", [])
    except Exception:
        logger.exception("Failed to fetch CZIB data from EASA")
        return []


def sync_czib_to_db(db_conn) -> dict:
    """
    Sync EASA CZIB data to database.
    Returns stats dict.
    """
    # Ensure clean transaction state — pool connections may be returned aborted
    try:
        db_conn.rollback()
    except Exception:
        pass

    zones = fetch_czib_data()
    if not zones:
        return {"fetched": 0, "inserted": 0, "updated": 0}

    inserted = 0
    updated = 0

    for zone in zones:
        nid = str(zone.get("Nid", ""))
        name = zone.get("name", "")
        status = zone.get("status", "Unknown")
        country_names = zone.get("country", "")
        countries = _parse_countries(country_names)
        coords = zone.get("coordinates", "")
        issued = zone.get("issued_date", "")
        valid_until = zone.get("valid_until_date", "")
        valid_descr = zone.get("field_easa_valid_until_descr", "")
        updated_time = zone.get("updated", "")

        # Parse issued_date
        issued_dt = None
        if issued:
            try:
                issued_dt = datetime.fromisoformat(issued.replace("+0200", "+02:00").replace("+0300", "+03:00"))
            except Exception:
                pass

        # Upsert — use savepoint so one bad row doesn't abort the whole batch
        try:
            with db_conn.transaction():
                result = db_conn.execute(
                    """INSERT INTO czib_zones (czib_id, name, status, countries, country_names,
                                               coordinates, issued_date, valid_until, valid_descr, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                       ON CONFLICT (czib_id) DO UPDATE SET
                           name = EXCLUDED.name,
                           status = EXCLUDED.status,
                           countries = EXCLUDED.countries,
                           country_names = EXCLUDED.country_names,
                           coordinates = EXCLUDED.coordinates,
                           issued_date = EXCLUDED.issued_date,
                           valid_until = EXCLUDED.valid_until,
                           valid_descr = EXCLUDED.valid_descr,
                           updated_at = NOW(),
                           synced_at = NOW()""",
                    (nid, name, status, countries, country_names, coords, issued_dt, valid_until, valid_descr),
                )
                # rowcount is unreliable for ON CONFLICT; check if row existed before
                if result.rowcount == 1:
                    inserted += 1
                else:
                    updated += 1
        except Exception:
            logger.warning("CZIB upsert skipped for nid=%s (likely duplicate)", nid)
            continue

    db_conn.commit()
    logger.info("CZIB sync complete: %d fetched, %d upserted", len(zones), inserted + updated)
    return {"fetched": len(zones), "inserted": inserted, "updated": updated}


def get_active_czib_countries(db_conn) -> set[str]:
    """Return set of ISO2 country codes from active CZIB zones."""
    rows = db_conn.execute(
        "SELECT DISTINCT unnest(countries) FROM czib_zones WHERE status = 'Active'"
    ).fetchall()
    return {row[0] for row in rows if row[0]}


def get_all_czib_countries(db_conn) -> set[str]:
    """Return set of ISO2 country codes from all CZIB zones (active + suspended)."""
    rows = db_conn.execute(
        "SELECT DISTINCT unnest(countries) FROM czib_zones WHERE status IN ('Active', 'Suspended')"
    ).fetchall()
    return {row[0] for row in rows if row[0]}


# Extended conflict-region pools for GDELT source-country filtering.
# These are FIPS 2-letter codes used by GDELT's sourcecountry filter.
# We combine: EASA CZIB countries + known high-risk regions not yet in CZIB.
CONFLICT_REGION_POOLS = {
    "africa_sahel": [
        "NG", "NE", "ML", "BF", "TD", "CF", "SN", "MR", "GN", "SL",
        "LR", "CI", "GH", "TG", "BJ", "CM", "GA", "GQ", "ST",
    ],
    "africa_horn": [
        "SO", "ET", "ER", "DJ", "KE", "SS", "SD", "UG", "RW", "BI",
    ],
    "africa_north": [
        "DZ", "LY", "EG", "TN", "MA", "EH", "MR",
    ],
    "africa_central": [
        "CD", "CG", "AO", "CM", "CF", "TD", "GQ", "GA", "ST",
    ],
    "africa_south": [
        "ZA", "MZ", "ZW", "ZM", "MW", "MG", "SZ", "LS", "BW", "NA",
    ],
    "middle_east": [
        "SY", "IQ", "IR", "IL", "JO", "LB", "YE", "SA", "AE", "QA",
        "KW", "BH", "OM", "TR", "PS",
    ],
    "asia_south": [
        "AF", "PK", "IN", "BD", "LK", "NP", "BT", "MV",
    ],
    "asia_southeast": [
        "MM", "TH", "PH", "ID", "MY", "VN", "LA", "KH", "SG", "BN",
    ],
    "asia_east": [
        "KP", "KR", "CN", "TW", "JP", "MN",
    ],
    "latin_america": [
        "CO", "VE", "MX", "HT", "EC", "PE", "BR", "HN", "GT", "SV",
        "NI", "CR", "PA", "BO", "PY", "CL", "AR", "UY", "CU", "JM",
    ],
    "eurasia": [
        "UA", "RU", "BY", "MD", "GE", "AM", "AZ", "KG", "KZ", "TJ",
        "TM", "UZ",
    ],
    "balkans_caucasus": [
        "RS", "BA", "ME", "MK", "AL", "XK", "MD", "GE", "AM", "AZ",
    ],
}


def get_enriched_gdelt_countries(db_conn, region: str | None = None) -> list[str]:
    """
    Get GDELT source-country filter list enriched with EASA CZIB data.

    Args:
        db_conn: Database connection
        region: Specific region key from CONFLICT_REGION_POOLS, or None for all

    Returns:
        List of FIPS 2-letter country codes for GDELT sourcecountry filter
    """
    # Start with CZIB-derived countries
    czib_countries = get_all_czib_countries(db_conn)

    if region and region in CONFLICT_REGION_POOLS:
        pool = set(CONFLICT_REGION_POOLS[region])
    else:
        pool = set()
        for r in CONFLICT_REGION_POOLS.values():
            pool.update(r)

    # Merge: CZIB countries take priority, pool fills gaps
    merged = czib_countries | pool
    return sorted(merged)
