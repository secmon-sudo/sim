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
                           synced_at = NOW()
                       RETURNING (xmax = 0) AS was_inserted""",
                    (nid, name, status, countries, country_names, coords, issued_dt, valid_until, valid_descr),
                )
                # rowcount is 1 for both branches of ON CONFLICT; xmax=0 is the
                # standard Postgres trick to distinguish a fresh insert from an update.
                row = result.fetchone()
                if row and row[0]:
                    inserted += 1
                else:
                    updated += 1
        except Exception:
            logger.warning("CZIB upsert skipped for nid=%s (likely duplicate)", nid)
            continue

    db_conn.commit()
    logger.info("CZIB sync complete: %d fetched, %d upserted", len(zones), inserted + updated)
    return {"fetched": len(zones), "inserted": inserted, "updated": updated}
