"""SIM — monthly recall audit: measure the events this pipeline never saw.

Why this exists
---------------
Everything SIM measures about itself is measured on what it produced. The
feedback buttons ask "was the card we sent any good"; the vocabulary audit asks
"was the rejection right"; the dedup replay asks "did the verdict move". None of
them can answer the question that matters most to a monitor — *what happened that
we never heard about at all* — because a story that never entered the corpus
leaves no trace in it.

This script answers it with an outside reference: the UCDP Candidate Events
Dataset, a researcher-coded record of organized violence published monthly by the
Uppsala Conflict Data Program. It is a month behind, which is exactly why it can
never be an ingest source (SIM pages within hours) and exactly why it makes a
clean benchmark: it is compiled independently, after the fact, by people reading
sources SIM does not read.

First run, 2026-09-11, against the July 2026 release — 110 events of 10 deaths or
more, on 85 distinct country-days:

    SIM had something from that country in the window   71   84%
    SIM actually paged a card                           34   40%
    SIM had nothing at all                              14   16%

and in the breakdown, the finding that prompted this file: Ethiopia 5 of 5 blind,
Burkina Faso 3 of 4. Across the whole of July, SIM ingested 2 Ethiopian events and
1 Burkinabè one, while UCDP recorded 9 and 5 mass-casualty events there. The
Sahel and the Horn were effectively outside the source pool and nothing in the
pipeline could have told us.

What the numbers mean, and do not
---------------------------------
A match here is a country-day, not an event. "SIM had something" means some row
from that country landed in the window, not that it was the same incident — so
84% is an UPPER bound on recall and 16% blind is a LOWER bound on what was
missed. That asymmetry is deliberate: a loose match makes the blind count
conservative, and the blind count is the one that should drive work.

Usage
-----
    python -m scripts.recall_audit                     # latest release, writes telemetry
    python -m scripts.recall_audit --dry-run           # print only
    python -m scripts.recall_audit --version 26.0.7 --min-deaths 25

Attribution (required by the licence)
-------------------------------------
UCDP data are licensed CC BY 4.0. Sundberg, Ralph and Erik Melander (2013),
"Introducing the UCDP Georeferenced Event Dataset", Journal of Peace Research
50(4); Hegre, Håvard, Mihai Croicu, Kristine Eck and Stina Högbladh (2020),
"Introducing the UCDP Candidate Events Dataset", Research & Politics 7(3).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from collections import Counter
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from src.services.supabase_client import get_connection, put_connection

logger = logging.getLogger(__name__)

# The keyless bulk download. The API in front of the same data now requires an
# access token (x-ucdp-access-token, requested by email from the maintainer), and
# an audit that needs a human to renew a credential is an audit that stops
# running. These CSVs need nothing — verified 2026-09-11, HTTP 200 with no header.
CANDIDATE_URL = "https://ucdp.uu.se/downloads/candidateged/GEDEvent_v{yy}_0_{m}.csv"

# How many months back to look when no version is given. UCDP publishes with up
# to a month's lag, so the current month is usually absent and the previous one
# is the newest that exists; six gives room for a slipped release without
# hammering the server.
_DISCOVERY_MONTHS = 6

# The event has to have landed in SIM's corpus somewhere in here. Wider forward
# than backward because UCDP records the day the violence happened while SIM
# records the day it ingested the report, and a Sahel massacre routinely reaches
# an English-language wire two days late. One day back covers the reverse case, a
# wire filing before UCDP's date_start.
_WINDOW_BACK_DAYS = 1
_WINDOW_FORWARD_DAYS = 3

# UCDP country names → ISO-3166-1 alpha-2, which is what events.country_iso holds.
# UCDP carries historical names in parentheses ("DR Congo (Zaire)") and its own
# spellings, so this cannot be a generic ISO library lookup. Names that are not in
# here are COUNTED AND PRINTED rather than skipped silently: a missing mapping
# shrinks the reference set, which would make SIM look better than it is.
UCDP_COUNTRY_ISO: Dict[str, str] = {
    "Afghanistan": "AF", "Algeria": "DZ", "Angola": "AO", "Argentina": "AR",
    "Armenia": "AM", "Azerbaijan": "AZ", "Bahrain": "BH", "Bangladesh": "BD",
    "Belarus": "BY", "Belize": "BZ", "Benin": "BJ", "Bolivia": "BO",
    "Bosnia-Herzegovina": "BA", "Botswana": "BW", "Brazil": "BR",
    "Burkina Faso": "BF", "Burundi": "BI", "Cambodia (Kampuchea)": "KH",
    "Cameroon": "CM", "Canada": "CA", "Central African Republic": "CF",
    "Chad": "TD", "Chile": "CL", "China": "CN", "Colombia": "CO",
    "Congo": "CG", "Costa Rica": "CR", "Cuba": "CU", "Cyprus": "CY",
    "DR Congo (Zaire)": "CD", "Djibouti": "DJ", "Dominican Republic": "DO",
    "Ecuador": "EC", "Egypt": "EG", "El Salvador": "SV", "Eritrea": "ER",
    "Ethiopia": "ET", "France": "FR", "Georgia": "GE", "Germany": "DE",
    "Ghana": "GH", "Greece": "GR", "Guatemala": "GT", "Guinea": "GN",
    "Guinea-Bissau": "GW", "Guyana": "GY", "Haiti": "HT", "Honduras": "HN",
    "India": "IN", "Indonesia": "ID", "Iran": "IR", "Iraq": "IQ",
    "Israel": "IL", "Italy": "IT", "Ivory Coast": "CI", "Jamaica": "JM",
    "Jordan": "JO", "Kazakhstan": "KZ", "Kenya": "KE", "Kuwait": "KW",
    "Kyrgyzstan": "KG", "Laos": "LA", "Lebanon": "LB", "Lesotho": "LS",
    "Liberia": "LR", "Libya": "LY", "Madagascar (Malagasy)": "MG",
    "Malawi": "MW", "Malaysia": "MY", "Mali": "ML", "Mauritania": "MR",
    "Mexico": "MX", "Moldova": "MD", "Morocco": "MA", "Mozambique": "MZ",
    "Myanmar (Burma)": "MM", "Namibia": "NA", "Nepal": "NP",
    "Nicaragua": "NI", "Niger": "NE", "Nigeria": "NG",
    "North Korea": "KP", "North Macedonia": "MK", "Oman": "OM",
    "Pakistan": "PK", "Panama": "PA", "Papua New Guinea": "PG",
    "Paraguay": "PY", "Peru": "PE", "Philippines": "PH", "Qatar": "QA",
    "Russia (Soviet Union)": "RU", "Rwanda": "RW", "Saudi Arabia": "SA",
    "Senegal": "SN", "Serbia (Yugoslavia)": "RS", "Sierra Leone": "SL",
    "Solomon Islands": "SB", "Somalia": "SO", "South Africa": "ZA",
    "South Korea": "KR", "South Sudan": "SS", "Spain": "ES",
    "Sri Lanka": "LK", "Sudan": "SD", "Sweden": "SE", "Syria": "SY",
    "Tajikistan": "TJ", "Tanzania": "TZ", "Thailand": "TH",
    "Togo": "TG", "Trinidad and Tobago": "TT", "Tunisia": "TN",
    "Turkey": "TR", "Turkmenistan": "TM", "Uganda": "UG", "Ukraine": "UA",
    "United Arab Emirates": "AE", "United Kingdom": "GB",
    "United States of America": "US", "Uruguay": "UY", "Uzbekistan": "UZ",
    "Venezuela": "VE", "Vietnam (South Vietnam)": "VN",
    "Yemen (North Yemen)": "YE", "Zambia": "ZM", "Zimbabwe (Rhodesia)": "ZW",
}


def candidate_url(version: str) -> str:
    """Monthly release id ("26.0.7") → its CSV URL.

    The id is also the API's version string, so the same number names the same
    data in both places and a result can be reproduced from either.
    """
    parts = version.strip().split(".")
    if len(parts) != 3 or parts[1] != "0":
        raise ValueError(
            f"not a monthly candidate version: {version!r} (expected e.g. 26.0.7)")
    return CANDIDATE_URL.format(yy=parts[0], m=parts[2])


def _release_month(version: str) -> Tuple[int, int]:
    """The (year, month) a monthly release is named for. 26.0.7 → (2026, 7)."""
    parts = version.strip().split(".")
    return 2000 + int(parts[0]), int(parts[2])


def discover_latest_version(today: Optional[date] = None,
                            fetch=None) -> Optional[str]:
    """Walk back from this month until a release answers.

    UCDP publishes a month behind, so the current month is normally missing. This
    asks rather than assumes, because the release cadence has slipped before and a
    hard-coded offset would silently audit the wrong month.
    """
    today = today or datetime.now(timezone.utc).date()
    fetch = fetch or _head_ok
    year, month = today.year, today.month
    for _ in range(_DISCOVERY_MONTHS):
        version = f"{year % 100}.0.{month}"
        if fetch(candidate_url(version)):
            return version
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return None


def _head_ok(url: str) -> bool:
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            return client.head(url).status_code == 200
    except httpx.HTTPError:
        return False


def fetch_candidate_csv(version: str) -> str:
    """Download one monthly release. ~1.4 MB for July 2026."""
    url = candidate_url(version)
    logger.info("Fetching UCDP candidate release %s from %s", version, url)
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def reference_events(csv_text: str, year: int, month: int,
                     min_deaths: int) -> Tuple[List[Dict[str, Any]], Counter]:
    """The mass-casualty events of one month, as (iso, day, deaths) records.

    Restricted to the release's OWN month even though the file carries revisions
    reaching months back: those older rows are corrections to data SIM was already
    audited against, and counting them again would score the same miss twice.

    Returns the records and a counter of country names with no ISO mapping.
    """
    rows: List[Dict[str, Any]] = []
    unmapped: Counter = Counter()
    for row in csv.DictReader(io.StringIO(csv_text)):
        try:
            deaths = int(row.get("best") or 0)
        except ValueError:
            continue
        if deaths < min_deaths:
            continue
        day = (row.get("date_start") or "")[:10]
        if not day.startswith(f"{year:04d}-{month:02d}"):
            continue
        name = (row.get("country") or "").strip()
        iso = UCDP_COUNTRY_ISO.get(name)
        if not iso:
            unmapped[name] += 1
            continue
        rows.append({"iso": iso, "day": day, "deaths": deaths,
                     "place": (row.get("where_coordinates") or "").strip()})
    return rows, unmapped


def corpus_start(conn) -> Optional[date]:
    """The oldest row SIM holds. Reference days before it are not misses."""
    row = conn.execute("SELECT min(ingested_at)::date FROM events").fetchone()
    return row[0] if row and row[0] else None


def measure(conn, records: Sequence[Dict[str, Any]],
            start: Optional[date]) -> Dict[str, Any]:
    """For each reference country-day, did anything from that country land?

    One query per distinct country-day rather than a join against a temp table:
    the reference set is under a hundred rows a month, and a TEMP TABLE does not
    survive the pooled connection this runs on (measured 2026-08-17).
    """
    pairs = sorted({(r["iso"], r["day"]) for r in records})
    per_country: Dict[str, Counter] = {}
    covered = paged = blind = skipped = 0
    blind_days: List[Tuple[str, str]] = []
    for iso, day in pairs:
        as_date = datetime.strptime(day, "%Y-%m-%d").date()
        if start and as_date < start:
            skipped += 1
            continue
        row = conn.execute(
            """SELECT count(*),
                      count(*) FILTER (WHERE alert_tier IS NOT NULL
                                         AND alert_tier <> 'NONE')
                 FROM events
                WHERE country_iso = %s
                  AND ingested_at >= %s::date - make_interval(days => %s)
                  AND ingested_at <  %s::date + make_interval(days => %s)""",
            (iso, day, _WINDOW_BACK_DAYS, day, _WINDOW_FORWARD_DAYS),
        ).fetchone()
        in_corpus, tiered = (row or (0, 0))
        bucket = per_country.setdefault(iso, Counter())
        bucket["days"] += 1
        if in_corpus:
            covered += 1
        else:
            blind += 1
            bucket["blind"] += 1
            blind_days.append((iso, day))
        if tiered:
            paged += 1
            bucket["paged"] += 1
    return {
        "reference_events": len(records),
        "country_days": len(pairs),
        "scored_days": len(pairs) - skipped,
        "skipped_before_corpus": skipped,
        "covered": covered,
        "paged": paged,
        "blind": blind,
        "per_country": {iso: dict(c) for iso, c in per_country.items()},
        "blind_days": blind_days,
    }


def _print_report(version: str, min_deaths: int, result: Dict[str, Any],
                  unmapped: Counter) -> None:
    scored = result["scored_days"] or 1
    print(f"\n=== Recall audit — UCDP candidate {version}, "
          f"events of {min_deaths}+ deaths ===\n")
    print(f"  reference events        {result['reference_events']}")
    print(f"  distinct country-days   {result['country_days']}"
          f" ({result['skipped_before_corpus']} before SIM's corpus, not scored)")
    print(f"  SIM had something       {result['covered']:>4}"
          f"  {result['covered'] / scored:>6.0%}   (upper bound on recall)")
    print(f"  SIM paged a card        {result['paged']:>4}"
          f"  {result['paged'] / scored:>6.0%}")
    print(f"  SIM saw nothing         {result['blind']:>4}"
          f"  {result['blind'] / scored:>6.0%}   (lower bound on misses)")
    if result["per_country"]:
        print("\n  country   days  blind  paged")
        ordered = sorted(result["per_country"].items(),
                         key=lambda kv: (-kv[1].get("blind", 0), -kv[1]["days"]))
        for iso, counts in ordered:
            print(f"  {iso:<9} {counts['days']:>4}  {counts.get('blind', 0):>5}"
                  f"  {counts.get('paged', 0):>5}")
    if unmapped:
        print("\n  UNMAPPED country names (add to UCDP_COUNTRY_ISO — these events\n"
              "  left the reference set and make SIM look better than it is):")
        for name, n in unmapped.most_common():
            print(f"    {name!r}: {n}")
    print("\n  Source: UCDP Candidate Events Dataset, CC BY 4.0 "
          "(Hegre et al. 2020; Sundberg & Melander 2013)\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="monthly release id, e.g. 26.0.7 "
                                          "(default: the newest that answers)")
    parser.add_argument("--min-deaths", type=int, default=10,
                        help="reference threshold, best estimate (default 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the report, write no telemetry row")
    args = parser.parse_args(argv)

    version = args.version or discover_latest_version()
    if not version:
        print("No UCDP candidate release answered in the last "
              f"{_DISCOVERY_MONTHS} months — check the download URL before "
              "reading anything into this.", file=sys.stderr)
        return 1

    year, month = _release_month(version)
    csv_text = fetch_candidate_csv(version)
    records, unmapped = reference_events(csv_text, year, month, args.min_deaths)
    if not records:
        print(f"Release {version} carries no {args.min_deaths}+ death events "
              f"dated {year}-{month:02d} — nothing to audit.", file=sys.stderr)
        return 1

    conn = get_connection()
    try:
        result = measure(conn, records, corpus_start(conn))
        _print_report(version, args.min_deaths, result, unmapped)
        if not args.dry_run:
            payload = dict(result)
            payload.pop("blind_days", None)
            payload.update({"version": version, "month": f"{year}-{month:02d}",
                            "min_deaths": args.min_deaths,
                            "unmapped_countries": dict(unmapped)})
            conn.execute(
                "INSERT INTO system_telemetry(event_type, value_json) "
                "VALUES ('recall_audit', %s)", (json.dumps(payload),))
            logger.info("Recall audit telemetry written for %s", version)
    finally:
        put_connection(conn)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
