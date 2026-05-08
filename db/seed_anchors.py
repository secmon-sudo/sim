"""
SIM — Anchor Master Seed Script
Loads anchor data (airports, hotels, military bases, ports) from JSON into anchor_master table.

Usage:
    python db/seed_anchors.py --file db/anchors.json --dry-run
    python db/seed_anchors.py --file db/anchors.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import psycopg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# City → country fallback (only used when JSON has no "country" field)
CITY_TO_COUNTRY = {
    "LUXEMBOURG": "LU", "RIGA": "LV", "LONDON": "GB", "PARIS": "FR",
    "BERLIN": "DE", "MUNICH": "DE", "FRANKFURT": "DE",
    "AMSTERDAM": "NL", "BRUSSELS": "BE", "MADRID": "ES",
    "ROME": "IT", "MILAN": "IT", "VIENNA": "AT", "ZURICH": "CH",
    "LISBON": "PT", "DUBLIN": "IE", "OSLO": "NO", "STOCKHOLM": "SE",
    "HELSINKI": "FI", "COPENHAGEN": "DK", "WARSAW": "PL", "PRAGUE": "CZ",
    "BUDAPEST": "HU", "BUCHAREST": "RO", "SOFIA": "BG", "ATHENS": "GR",
    "ISTANBUL": "TR", "ANKARA": "TR", "IZMIR": "TR", "ANTALYA": "TR",
    "BODRUM": "TR", "TRABZON": "TR", "ADANA": "TR", "GAZIANTEP": "TR",
    "DUBAI": "AE", "ABU DHABI": "AE", "DOHA": "QA", "RIYADH": "SA",
    "JEDDAH": "SA", "MUSCAT": "OM", "KUWAIT": "KW",
    "TEL AVIV": "IL", "AMMAN": "JO", "BEIRUT": "LB", "BAGHDAD": "IQ",
    "TEHRAN": "IR", "CAIRO": "EG", "CASABLANCA": "MA", "TUNIS": "TN",
    "TRIPOLI": "LY", "BENGHAZI": "LY", "KHARTOUM": "SD",
    "NAIROBI": "KE", "LAGOS": "NG", "JOHANNESBURG": "ZA",
    "MOGADISHU": "SO", "BAMAKO": "ML", "ZANZIBAR": "TZ",
    "DAR ES SALAAM": "TZ", "ADDIS ABABA": "ET",
    "TOKYO": "JP", "BEIJING": "CN", "SHANGHAI": "CN", "SEOUL": "KR",
    "SINGAPORE": "SG", "BANGKOK": "TH", "KUALA LUMPUR": "MY",
    "NEW YORK": "US", "LOS ANGELES": "US", "CHICAGO": "US", "MIAMI": "US",
    "WASHINGTON": "US", "TORONTO": "CA", "MEXICO CITY": "MX",
    "SAO PAULO": "BR", "BUENOS AIRES": "AR", "BOGOTA": "CO",
    "SYDNEY": "AU", "MELBOURNE": "AU", "MOSCOW": "RU",
    "AL QASSIM": "SA", "MEDINA": "SA", "TABUK": "SA", "ABHA": "SA",
}


def detect_country(city: str) -> str:
    """Detect country from city name."""
    city_upper = city.upper().strip()
    if city_upper in CITY_TO_COUNTRY:
        return CITY_TO_COUNTRY[city_upper]
    for key, code in CITY_TO_COUNTRY.items():
        if key in city_upper or city_upper in key:
            return code
    return "XX"


def generate_anchor_code(name: str, anchor_type: str, index: int) -> str:
    """Generate a unique short code for non-airport anchors (hotels, bases, ports)."""
    prefix = {"Hotel": "H", "Military Base": "M", "Port": "P", "Resort": "R"}.get(anchor_type, "X")
    consonants = [c.upper() for c in name if c.isalpha() and c.lower() not in "aeiou"]
    suffix = "".join(consonants[:2]) if len(consonants) >= 2 else name[:2].upper()
    return f"{prefix}{suffix}{index % 100:02d}"


def load_anchors(file_path: str, db_url: str, dry_run: bool = False):
    """Load anchor data from JSON file into anchor_master table."""

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded %d anchors from %s", len(data), file_path)

    if not dry_run:
        conn = psycopg.connect(db_url, autocommit=False)

    inserted = 0
    skipped = 0
    errors = 0
    non_iata_idx = 0

    for item in data:
        name = item.get("name", "").strip()
        anchor_type = item.get("type", "Airport").strip()
        city = item.get("city", "").strip()
        lat = item.get("lat")
        lon = item.get("lon")

        if not name:
            skipped += 1
            continue

        # Country
        country = item.get("country", "")
        if not country:
            country = detect_country(city)
        country = country.upper()[:2]

        # IATA / ICAO
        iata = item.get("iata", "")
        icao = item.get("icao", "")

        if not iata:
            non_iata_idx += 1
            iata = generate_anchor_code(name, anchor_type, non_iata_idx)

        iata = iata.upper()[:4]

        # CZIB
        czib = item.get("czib", False)

        # Aliases — include city as alias
        aliases = item.get("aliases", [])
        if city and city not in aliases:
            aliases.append(city)

        # Map type
        type_map = {
            "airport": "airport",
            "hotel": "hotel_chain",
            "military base": "military_base",
            "military": "military_base",
            "port": "port",
            "resort": "hotel_chain",
        }
        db_type = type_map.get(anchor_type.lower(), "airport")

        if dry_run:
            logger.info("[DRY] %4s | %-50s | %-14s | %s | %9.4f, %9.4f | czib=%s",
                        iata, name[:50], db_type, country, lat or 0, lon or 0, czib)
            inserted += 1
            continue

        try:
            with conn.transaction():
                conn.execute(
                    """INSERT INTO anchor_master
                       (iata_code, icao_code, anchor_type, canonical_name,
                        aliases, country_iso, latitude, longitude, czib_flag)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (iata_code) DO UPDATE SET
                           canonical_name = EXCLUDED.canonical_name,
                           aliases = EXCLUDED.aliases,
                           latitude = EXCLUDED.latitude,
                           longitude = EXCLUDED.longitude,
                           czib_flag = EXCLUDED.czib_flag,
                           updated_at = NOW()""",
                    (iata, icao or None, db_type, name, json.dumps(aliases),
                     country, lat, lon, czib),
                )
                inserted += 1
        except Exception as e:
            logger.error("Error inserting %s (%s): %s", name[:40], iata, e)
            errors += 1

    if not dry_run:
        conn.close()

    logger.info("=== Seed Complete ===")
    logger.info("  Inserted/Updated: %d", inserted)
    logger.info("  Skipped: %d", skipped)
    logger.info("  Errors: %d", errors)
    return {"inserted": inserted, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Seed anchor_master from JSON")
    parser.add_argument("--file", required=True, help="Path to anchors JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Preview without inserting")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"), help="Database URL")
    args = parser.parse_args()

    if not args.dry_run and not args.db_url:
        logger.error("DATABASE_URL not set. Use --db-url or set env var.")
        sys.exit(1)

    if not Path(args.file).exists():
        logger.error("File not found: %s", args.file)
        sys.exit(1)

    load_anchors(args.file, args.db_url or "", dry_run=args.dry_run)


if __name__ == "__main__":
    main()
