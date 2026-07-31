"""
SIM — Airspace reference builder (regenerates config/airspace.json).

The airspace impact block in the SITREP (src/core/airspace.py) resolves an event
to a FIR and to the commercial airports around it. That needs reference data the
repo never had, and this script is where it is maintained: edit the FIRS table or
the SUPPLEMENT below, re-run, commit the regenerated JSON.

The FIR table is hand-curated (ICAO designator, extent, adjacency). The airport
list is seeded from db/anchors.json — which already carries lat/lon/IATA/country
for ~300 airports — and topped up with a curated supplement for the countries the
anchor gazetteer barely covers (Poland had exactly one entry; the Baltics,
Nordics, Balkans, Turkey and the Ukraine/Russia border regions were similar).

Two invariants are enforced here rather than left to review:
  * adjacency is symmetrised, so "who borders me" is complete from either side;
  * every airport must fall inside some FIR extent. That check earned its keep
    immediately — it caught three corrupt anchor records whose coordinates point
    at the wrong continent (BRU → Ontario, BHX → Alabama, PTY → Florida), which
    are still wrong in db/anchors.json and affect Pass D proximity scoring.

Usage:  python scripts/build_airspace_reference.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# FIR table: (icao, name, name_tr, country, bbox[min_lat,min_lon,max_lat,max_lon], neighbors)
# bbox values are the published FIR extents rounded to ~0.1 degree.
# ---------------------------------------------------------------------------
FIRS = [
    # --- Central & Eastern Europe -----------------------------------------
    ("EPWW", "Warszawa FIR", "Varşova FIR", "PL", [48.9, 14.1, 55.0, 24.2],
     ["UMKK", "UMMV", "UKLV", "EYVL", "LZBB", "LKAA", "EDMM", "EDWW", "ESAA"]),
    ("LKAA", "Praha FIR", "Prag FIR", "CZ", [48.5, 12.0, 51.1, 18.9],
     ["EPWW", "LZBB", "LOVV", "EDMM", "EDGG"]),
    ("LZBB", "Bratislava FIR", "Bratislava FIR", "SK", [47.7, 16.8, 49.7, 22.6],
     ["EPWW", "LKAA", "LOVV", "LHCC", "UKLV"]),
    ("LHCC", "Budapest FIR", "Budapeşte FIR", "HU", [45.7, 16.1, 48.6, 22.9],
     ["LZBB", "LOVV", "LDZO", "LYBA", "LRBB", "UKLV"]),
    ("LOVV", "Wien FIR", "Viyana FIR", "AT", [46.3, 9.5, 49.1, 17.2],
     ["LKAA", "LZBB", "LHCC", "LDZO", "LIMM", "LSAS", "EDMM"]),
    ("LSAS", "Switzerland FIR", "İsviçre FIR", "CH", [45.8, 5.9, 47.9, 10.5],
     ["LOVV", "LIMM", "LFMM", "LFFF", "EDGG"]),
    ("EDWW", "Bremen FIR", "Bremen FIR", "DE", [51.3, 6.0, 55.1, 14.1],
     ["EPWW", "EKDK", "EHAA", "EDGG", "EDMM", "LKAA"]),
    ("EDGG", "Langen FIR", "Langen FIR", "DE", [47.5, 5.9, 51.6, 10.6],
     ["EDWW", "EDMM", "EHAA", "EBBU", "LFFF", "LSAS"]),
    ("EDMM", "München FIR", "Münih FIR", "DE", [47.2, 9.0, 51.6, 13.9],
     ["EDWW", "EDGG", "LKAA", "LOVV", "LSAS", "EPWW"]),
    # --- Baltics, Belarus, Kaliningrad ------------------------------------
    ("EYVL", "Vilnius FIR", "Vilnius FIR", "LT", [53.8, 20.9, 56.5, 26.9],
     ["EPWW", "UMKK", "EVRR", "UMMV"]),
    ("EVRR", "Riga FIR", "Riga FIR", "LV", [55.6, 20.5, 58.1, 28.3],
     ["EYVL", "EETT", "UMMV", "ULLL", "ESAA"]),
    ("EETT", "Tallinn FIR", "Tallinn FIR", "EE", [57.5, 21.7, 59.8, 28.2],
     ["EVRR", "ULLL", "EFIN", "ESAA"]),
    ("UMKK", "Kaliningrad FIR", "Kaliningrad FIR", "RU", [54.2, 19.4, 55.4, 22.9],
     ["EPWW", "EYVL", "ESAA"]),
    ("UMMV", "Minsk FIR", "Minsk FIR", "BY", [51.2, 23.1, 56.2, 32.8],
     ["EPWW", "EYVL", "EVRR", "UKBV", "UKLV", "UUWV", "ULLL"]),
    # --- Nordics -----------------------------------------------------------
    ("EKDK", "København FIR", "Kopenhag FIR", "DK", [54.4, 7.7, 58.0, 15.2],
     ["EDWW", "ESAA", "ENOR", "EHAA"]),
    ("ESAA", "Sweden FIR", "İsveç FIR", "SE", [54.9, 10.5, 69.1, 24.2],
     ["EKDK", "ENOR", "EFIN", "EPWW", "UMKK", "EVRR", "EETT"]),
    ("ENOR", "Norway FIR", "Norveç FIR", "NO", [57.0, 4.0, 71.5, 31.5],
     ["ESAA", "EFIN", "EKDK", "ULLL"]),
    ("EFIN", "Finland FIR", "Finlandiya FIR", "FI", [59.4, 19.0, 70.1, 31.6],
     ["ESAA", "ENOR", "EETT", "ULLL"]),
    # --- Western Europe ----------------------------------------------------
    ("EGTT", "London FIR", "Londra FIR", "GB", [49.9, -6.5, 55.0, 2.1],
     ["EGPX", "EISN", "EHAA", "EBBU", "LFFF", "LFRR"]),
    ("EGPX", "Scottish FIR", "İskoçya FIR", "GB", [55.0, -9.0, 61.0, 1.5],
     ["EGTT", "EISN", "ENOR"]),
    ("EISN", "Shannon FIR", "Shannon FIR", "IE", [51.0, -11.0, 55.5, -5.4],
     ["EGTT", "EGPX"]),
    ("EHAA", "Amsterdam FIR", "Amsterdam FIR", "NL", [50.7, 2.0, 55.0, 7.3],
     ["EBBU", "EDWW", "EDGG", "EGTT", "EKDK"]),
    ("EBBU", "Brussels FIR", "Brüksel FIR", "BE", [49.4, 2.4, 51.6, 6.4],
     ["EHAA", "EDGG", "LFFF", "EGTT"]),
    ("LFFF", "Paris FIR", "Paris FIR", "FR", [46.0, 0.9, 51.1, 5.6],
     ["EBBU", "EDGG", "LSAS", "LFMM", "LFBB", "LFRR", "EGTT"]),
    ("LFRR", "Brest FIR", "Brest FIR", "FR", [43.0, -8.0, 49.5, 0.9],
     ["LFFF", "LFBB", "EGTT"]),
    ("LFBB", "Bordeaux FIR", "Bordeaux FIR", "FR", [42.3, -2.0, 46.5, 3.1],
     ["LFFF", "LFRR", "LFMM", "LECM"]),
    ("LFMM", "Marseille FIR", "Marsilya FIR", "FR", [41.3, 2.9, 46.5, 9.6],
     ["LFFF", "LFBB", "LSAS", "LIMM", "LIRR", "LECB"]),
    ("LECM", "Madrid FIR", "Madrid FIR", "ES", [35.6, -9.6, 43.8, 0.6],
     ["LECB", "LPPC", "LFBB", "GMMM"]),
    ("LECB", "Barcelona FIR", "Barselona FIR", "ES", [38.4, -0.1, 43.0, 4.6],
     ["LECM", "LFMM", "LFBB"]),
    ("LPPC", "Lisboa FIR", "Lizbon FIR", "PT", [36.0, -9.7, 42.2, -6.1],
     ["LECM", "GMMM"]),
    ("LIMM", "Milano FIR", "Milano FIR", "IT", [43.4, 6.6, 47.1, 13.6],
     ["LIRR", "LFMM", "LSAS", "LOVV", "LDZO"]),
    ("LIRR", "Roma FIR", "Roma FIR", "IT", [36.6, 6.6, 43.6, 18.6],
     ["LIMM", "LIBB", "LFMM", "LMMM"]),
    ("LIBB", "Brindisi FIR", "Brindisi FIR", "IT", [36.5, 12.0, 42.6, 20.0],
     ["LIRR", "LGGG", "LYBA", "LAAA", "LMMM", "LYPG"]),
    ("LMMM", "Malta FIR", "Malta FIR", "MT", [33.0, 12.0, 36.6, 18.0],
     ["LIRR", "LIBB", "HLLL", "DTTC", "LGGG"]),
    # --- Balkans -----------------------------------------------------------
    ("LJLA", "Ljubljana FIR", "Ljubljana FIR", "SI", [45.4, 13.3, 46.9, 16.6],
     ["LOVV", "LDZO", "LIMM", "LHCC"]),
    ("LDZO", "Zagreb FIR", "Zagreb FIR", "HR", [42.3, 13.0, 46.6, 19.5],
     ["LOVV", "LHCC", "LYBA", "LQSB", "LIMM", "LYPG", "LJLA"]),
    ("LQSB", "Sarajevo FIR", "Saraybosna FIR", "BA", [42.5, 15.7, 45.3, 19.7],
     ["LDZO", "LYBA", "LYPG"]),
    ("LYBA", "Beograd FIR", "Belgrad FIR", "RS", [42.2, 18.8, 46.2, 23.0],
     ["LHCC", "LDZO", "LQSB", "LRBB", "LBSR", "LWSS", "LYPG"]),
    ("LYPG", "Podgorica FIR", "Podgorica FIR", "ME", [41.7, 18.3, 43.6, 20.5],
     ["LYBA", "LQSB", "LDZO", "LAAA", "LIBB"]),
    ("LWSS", "Skopje FIR", "Üsküp FIR", "MK", [40.8, 20.4, 42.4, 23.1],
     ["LYBA", "LBSR", "LGGG", "LAAA"]),
    ("LAAA", "Tirana FIR", "Tiran FIR", "AL", [39.6, 19.2, 42.7, 21.1],
     ["LYPG", "LWSS", "LGGG", "LIBB"]),
    ("LBSR", "Sofia FIR", "Sofya FIR", "BG", [41.2, 22.3, 44.3, 28.7],
     ["LYBA", "LRBB", "LWSS", "LGGG", "LTBB"]),
    ("LRBB", "Bucuresti FIR", "Bükreş FIR", "RO", [43.4, 20.2, 48.3, 29.7],
     ["LHCC", "LYBA", "LBSR", "LUUU", "UKLV", "UKOV", "LTBB"]),
    ("LUUU", "Chisinau FIR", "Kişinev FIR", "MD", [45.4, 26.6, 48.5, 30.2],
     ["LRBB", "UKLV", "UKOV"]),
    ("LGGG", "Athinai FIR", "Atina FIR", "GR", [33.9, 19.3, 41.8, 29.8],
     ["LBSR", "LWSS", "LAAA", "LIBB", "LTBB", "LCCC", "LMMM"]),
    # --- Turkey, Cyprus, Caucasus -----------------------------------------
    ("LTBB", "Istanbul FIR", "İstanbul FIR", "TR", [35.8, 25.6, 42.2, 30.0],
     ["LGGG", "LBSR", "LTAA", "LCCC", "UKOV", "UKFV"]),
    ("LTAA", "Ankara FIR", "Ankara FIR", "TR", [34.5, 30.0, 42.2, 45.0],
     ["LTBB", "LCCC", "UGGG", "UDDD", "OSTT", "ORBB", "OIIX", "UBBA"]),
    ("LCCC", "Nicosia FIR", "Lefkoşa FIR", "CY", [33.6, 31.9, 35.9, 35.2],
     ["LTAA", "LTBB", "LGGG", "OSTT", "OLBB", "LLLL", "HECC"]),
    ("UGGG", "Tbilisi FIR", "Tiflis FIR", "GE", [41.0, 40.0, 43.7, 46.8],
     ["LTAA", "URRV", "UDDD", "UBBA"]),
    ("UDDD", "Yerevan FIR", "Erivan FIR", "AM", [38.8, 43.4, 41.4, 46.7],
     ["LTAA", "UGGG", "UBBA", "OIIX"]),
    ("UBBA", "Baku FIR", "Bakü FIR", "AZ", [38.3, 44.7, 42.1, 50.8],
     ["UGGG", "UDDD", "OIIX", "URRV", "UTAA"]),
    # --- Ukraine & western Russia -----------------------------------------
    ("UKLV", "Lviv FIR", "Lviv FIR", "UA", [47.8, 22.6, 51.6, 27.6],
     ["EPWW", "LZBB", "LHCC", "LRBB", "LUUU", "UMMV", "UKBV", "UKOV"]),
    ("UKBV", "Kyiv FIR", "Kiev FIR", "UA", [48.5, 26.5, 52.4, 35.6],
     ["UKLV", "UKOV", "UKDV", "UMMV", "UUWV"]),
    ("UKOV", "Odesa FIR", "Odesa FIR", "UA", [44.5, 28.2, 48.6, 33.8],
     ["UKLV", "UKBV", "UKDV", "UKFV", "LUUU", "LRBB", "LTBB"]),
    ("UKDV", "Dnipro FIR", "Dnipro FIR", "UA", [46.2, 33.5, 50.5, 40.3],
     ["UKBV", "UKOV", "UKFV", "UUWV", "URRV"]),
    ("UKFV", "Simferopol FIR", "Simferopol FIR", "UA", [43.3, 32.4, 46.3, 36.8],
     ["UKOV", "UKDV", "URRV", "LTBB"]),
    ("UUWV", "Moscow FIR", "Moskova FIR", "RU", [52.5, 30.0, 60.0, 47.0],
     ["UMMV", "UKBV", "UKDV", "URRV", "ULLL", "UWWW"]),
    ("ULLL", "St Petersburg FIR", "St. Petersburg FIR", "RU", [56.0, 27.0, 70.5, 47.0],
     ["UMMV", "EVRR", "EETT", "EFIN", "ENOR", "UUWV"]),
    ("URRV", "Rostov FIR", "Rostov FIR", "RU", [43.0, 36.0, 52.5, 47.6],
     ["UKDV", "UKFV", "UUWV", "UGGG", "UWWW", "UAAX"]),
    ("UWWW", "Samara FIR", "Samara FIR", "RU", [49.5, 45.0, 58.5, 60.5],
     ["UUWV", "URRV", "UAAX", "USSV"]),
    ("USSV", "Yekaterinburg FIR", "Yekaterinburg FIR", "RU", [51.5, 57.0, 66.5, 75.5],
     ["UWWW", "UAAX"]),
    # --- Middle East -------------------------------------------------------
    ("OSTT", "Damascus FIR", "Şam FIR", "SY", [32.3, 35.6, 37.4, 42.4],
     ["LTAA", "LCCC", "OLBB", "LLLL", "OJAC", "ORBB"]),
    ("OLBB", "Beirut FIR", "Beyrut FIR", "LB", [33.0, 34.0, 34.8, 36.7],
     ["OSTT", "LLLL", "LCCC"]),
    ("LLLL", "Tel Aviv FIR", "Tel Aviv FIR", "IL", [29.3, 33.9, 33.4, 35.9],
     ["OLBB", "OSTT", "OJAC", "HECC", "LCCC"]),
    ("OJAC", "Amman FIR", "Amman FIR", "JO", [29.1, 34.9, 33.4, 39.4],
     ["LLLL", "OSTT", "ORBB", "OEJD", "HECC"]),
    ("ORBB", "Baghdad FIR", "Bağdat FIR", "IQ", [29.0, 38.7, 37.4, 48.7],
     ["LTAA", "OSTT", "OJAC", "OEJD", "OKAC", "OIIX"]),
    ("OIIX", "Tehran FIR", "Tahran FIR", "IR", [25.0, 44.0, 39.8, 63.4],
     ["LTAA", "UDDD", "UBBA", "ORBB", "OKAC", "OBBB", "OMAE", "OOMM",
      "UTAA", "OAKX", "OPKR"]),
    ("OKAC", "Kuwait FIR", "Kuveyt FIR", "KW", [28.4, 46.4, 30.2, 48.6],
     ["ORBB", "OIIX", "OEJD", "OBBB"]),
    ("OBBB", "Bahrain FIR", "Bahreyn FIR", "BH", [23.4, 48.9, 27.7, 53.7],
     ["OKAC", "OIIX", "OEJD", "OTDF", "OMAE"]),
    ("OTDF", "Doha FIR", "Doha FIR", "QA", [24.4, 50.4, 26.6, 52.1],
     ["OBBB", "OEJD", "OMAE"]),
    ("OMAE", "Emirates FIR", "BAE FIR", "AE", [22.4, 51.0, 26.6, 57.1],
     ["OBBB", "OTDF", "OEJD", "OOMM", "OIIX"]),
    ("OOMM", "Muscat FIR", "Maskat FIR", "OM", [15.9, 52.0, 26.6, 60.5],
     ["OMAE", "OEJD", "OYSC", "OIIX", "OPKR"]),
    ("OEJD", "Jeddah FIR", "Cidde FIR", "SA", [15.9, 34.0, 32.3, 55.8],
     ["OJAC", "ORBB", "OKAC", "OBBB", "OTDF", "OMAE", "OOMM", "OYSC", "HECC"]),
    ("OYSC", "Sanaa FIR", "Sana FIR", "YE", [11.9, 41.9, 19.1, 54.1],
     ["OEJD", "OOMM", "HCSM", "HAAA"]),
    # --- North Africa & Sahel ---------------------------------------------
    ("HECC", "Cairo FIR", "Kahire FIR", "EG", [21.7, 24.6, 32.0, 37.1],
     ["LLLL", "OJAC", "OEJD", "HLLL", "HSSS", "LCCC", "LGGG"]),
    ("HLLL", "Tripoli FIR", "Trablus FIR", "LY", [19.4, 9.3, 33.3, 25.2],
     ["HECC", "DTTC", "DAAA", "FTTT", "HSSS", "DRRR", "LMMM"]),
    ("DTTC", "Tunis FIR", "Tunus FIR", "TN", [30.1, 7.5, 38.1, 12.1],
     ["HLLL", "DAAA", "LMMM"]),
    ("DAAA", "Alger FIR", "Cezayir FIR", "DZ", [18.9, -8.7, 37.2, 12.0],
     ["DTTC", "HLLL", "GMMM", "GOOO", "DRRR", "LECM"]),
    ("GMMM", "Casablanca FIR", "Kazablanka FIR", "MA", [27.5, -13.3, 36.0, -0.9],
     ["DAAA", "GOOO", "LECM", "LPPC"]),
    ("GOOO", "Dakar FIR", "Dakar FIR", "SN", [9.0, -20.0, 26.5, 4.6],
     ["DAAA", "DRRR", "GLRB", "DGAC"]),
    ("DRRR", "Niamey FIR", "Niamey FIR", "NE", [8.5, -0.5, 23.7, 16.1],
     ["GOOO", "DAAA", "HLLL", "FTTT", "DNKK", "DGAC"]),
    ("FTTT", "N'Djamena FIR", "Encemine FIR", "TD", [5.5, 13.3, 23.6, 24.1],
     ["DRRR", "HLLL", "HSSS", "DNKK", "FZZA"]),
    ("DNKK", "Kano FIR", "Kano FIR", "NG", [3.5, 2.5, 14.1, 14.8],
     ["DRRR", "FTTT", "DGAC", "FZZA"]),
    ("DGAC", "Accra FIR", "Akra FIR", "GH", [1.0, -4.0, 11.5, 3.0],
     ["DRRR", "GLRB", "GOOO", "DNKK"]),
    ("GLRB", "Roberts FIR", "Roberts FIR", "LR", [2.0, -14.0, 13.0, -4.0],
     ["GOOO", "DGAC"]),
    # --- Horn of Africa & East Africa -------------------------------------
    ("HSSS", "Khartoum FIR", "Hartum FIR", "SD", [8.5, 21.7, 22.3, 38.7],
     ["HECC", "HLLL", "FTTT", "HJJJ", "HAAA", "HHAA"]),
    ("HJJJ", "Juba FIR", "Cuba FIR", "SS", [3.3, 24.0, 12.3, 36.0],
     ["HSSS", "HAAA", "HKNA", "FZZA", "FTTT"]),
    ("HHAA", "Asmara FIR", "Asmara FIR", "ER", [12.3, 36.4, 18.1, 43.3],
     ["HSSS", "HAAA", "OYSC"]),
    ("HAAA", "Addis Ababa FIR", "Addis Ababa FIR", "ET", [3.3, 32.9, 15.1, 48.1],
     ["HSSS", "HHAA", "HJJJ", "HKNA", "HCSM", "OYSC"]),
    ("HCSM", "Mogadishu FIR", "Mogadişu FIR", "SO", [-1.8, 40.8, 12.1, 51.6],
     ["HAAA", "HKNA", "OYSC"]),
    ("HKNA", "Nairobi FIR", "Nairobi FIR", "KE", [-4.8, 33.8, 5.1, 42.0],
     ["HAAA", "HJJJ", "HCSM", "HTDC", "FZZA"]),
    ("HTDC", "Dar es Salaam FIR", "Darüsselam FIR", "TZ", [-11.8, 29.3, -0.9, 40.5],
     ["HKNA", "FZZA", "FQBE"]),
    ("FZZA", "Kinshasa FIR", "Kinşasa FIR", "CD", [-13.6, 11.9, 5.5, 31.5],
     ["FTTT", "DNKK", "HJJJ", "HKNA", "HTDC", "FQBE"]),
    ("FQBE", "Beira FIR", "Beira FIR", "MZ", [-27.0, 30.1, -10.3, 41.1],
     ["HTDC", "FZZA", "FAJA"]),
    ("FAJA", "Johannesburg FIR", "Johannesburg FIR", "ZA", [-35.0, 16.3, -22.0, 33.1],
     ["FQBE", "FZZA"]),
    # --- Central, South & East Asia ---------------------------------------
    ("UTAA", "Ashgabat FIR", "Aşkabat FIR", "TM", [35.1, 52.4, 42.9, 66.8],
     ["OIIX", "UBBA", "UAAX", "UTTR", "OAKX"]),
    ("UTTR", "Tashkent FIR", "Taşkent FIR", "UZ", [37.1, 55.9, 45.7, 73.3],
     ["UTAA", "UAAX", "UTDD", "UCFM", "OAKX"]),
    ("UTDD", "Dushanbe FIR", "Duşanbe FIR", "TJ", [36.6, 67.3, 41.1, 75.3],
     ["UTTR", "UCFM", "OAKX", "ZWUQ"]),
    ("UCFM", "Bishkek FIR", "Bişkek FIR", "KG", [39.1, 69.2, 43.4, 80.4],
     ["UTTR", "UTDD", "UAAX", "ZWUQ"]),
    ("UAAX", "Almaty FIR", "Almatı FIR", "KZ", [40.5, 46.4, 55.6, 87.5],
     ["URRV", "UWWW", "USSV", "UTAA", "UTTR", "UCFM", "ZWUQ"]),
    ("OAKX", "Kabul FIR", "Kabil FIR", "AF", [29.3, 60.4, 38.6, 75.0],
     ["OIIX", "UTAA", "UTTR", "UTDD", "OPKR", "OPLR"]),
    ("OPKR", "Karachi FIR", "Karaçi FIR", "PK", [23.5, 60.8, 29.6, 71.2],
     ["OIIX", "OAKX", "OPLR", "OOMM", "VABF", "VIDF"]),
    ("OPLR", "Lahore FIR", "Lahor FIR", "PK", [27.4, 68.0, 37.2, 77.1],
     ["OPKR", "OAKX", "VIDF", "ZWUQ"]),
    ("VIDF", "Delhi FIR", "Delhi FIR", "IN", [22.9, 68.0, 35.7, 82.1],
     ["OPLR", "OPKR", "VABF", "VECF", "VNSM", "ZWUQ"]),
    ("VABF", "Mumbai FIR", "Mumbai FIR", "IN", [7.9, 60.0, 24.6, 77.1],
     ["VIDF", "VOMF", "OPKR", "OOMM", "VCCF"]),
    ("VOMF", "Chennai FIR", "Chennai FIR", "IN", [4.9, 74.0, 19.1, 85.1],
     ["VABF", "VIDF", "VECF", "VCCF"]),
    ("VECF", "Kolkata FIR", "Kalküta FIR", "IN", [16.9, 80.0, 28.6, 92.6],
     ["VIDF", "VOMF", "VGFR", "VNSM", "VYYF"]),
    ("VNSM", "Kathmandu FIR", "Katmandu FIR", "NP", [26.3, 80.0, 30.5, 88.3],
     ["VIDF", "VECF"]),
    ("VGFR", "Dhaka FIR", "Dakka FIR", "BD", [20.4, 88.0, 26.8, 92.8],
     ["VECF", "VYYF"]),
    ("VCCF", "Colombo FIR", "Kolombo FIR", "LK", [1.9, 73.0, 10.1, 84.1],
     ["VOMF", "VABF"]),
    ("VYYF", "Yangon FIR", "Yangon FIR", "MM", [9.4, 91.9, 28.7, 101.3],
     ["VECF", "VGFR", "VTBB", "ZGZU"]),
    ("VTBB", "Bangkok FIR", "Bangkok FIR", "TH", [5.4, 96.9, 20.6, 106.1],
     ["VYYF", "WMFC", "ZGZU"]),
    ("WMFC", "Kuala Lumpur FIR", "Kuala Lumpur FIR", "MY", [0.9, 99.0, 7.6, 105.1],
     ["VTBB", "WSJC", "WIIF"]),
    ("WSJC", "Singapore FIR", "Singapur FIR", "SG", [-0.1, 102.0, 7.1, 109.1],
     ["WMFC", "WIIF", "RPHI"]),
    ("WIIF", "Jakarta FIR", "Cakarta FIR", "ID", [-9.1, 95.0, 3.6, 119.1],
     ["WMFC", "WSJC", "WAAF"]),
    ("WAAF", "Ujung Pandang FIR", "Ujung Pandang FIR", "ID", [-11.1, 115.0, 5.1, 141.1],
     ["WIIF", "RPHI"]),
    ("RPHI", "Manila FIR", "Manila FIR", "PH", [4.9, 116.0, 21.1, 135.1],
     ["WSJC", "WAAF", "RCAA", "ZGZU"]),
    ("ZWUQ", "Urumqi FIR", "Urumçi FIR", "CN", [34.0, 73.0, 49.2, 96.5],
     ["UAAX", "UCFM", "UTDD", "OPLR", "VIDF", "ZBPE", "ZLHW", "ZMUB"]),
    ("ZLHW", "Lanzhou FIR", "Lanzhou FIR", "CN", [29.9, 96.4, 42.1, 111.1],
     ["ZWUQ", "ZBPE", "ZPKM", "ZGZU", "ZMUB"]),
    ("ZPKM", "Kunming FIR", "Kunming FIR", "CN", [20.9, 96.9, 30.0, 106.1],
     ["ZLHW", "ZGZU", "VYYF", "VVHN", "VTBB"]),
    ("ZBPE", "Beijing FIR", "Pekin FIR", "CN", [34.9, 104.9, 45.1, 127.1],
     ["ZWUQ", "ZSHA", "ZGZU", "RKRR"]),
    ("ZSHA", "Shanghai FIR", "Şanghay FIR", "CN", [26.9, 116.0, 35.1, 125.1],
     ["ZBPE", "ZGZU", "RKRR", "RJJJ"]),
    ("ZGZU", "Guangzhou FIR", "Guangzhou FIR", "CN", [16.9, 104.9, 27.1, 120.1],
     ["ZSHA", "ZBPE", "ZWUQ", "VTBB", "VYYF", "RPHI", "RCAA"]),
    ("RCAA", "Taipei FIR", "Taipei FIR", "TW", [19.9, 117.4, 26.6, 124.1],
     ["ZGZU", "ZSHA", "RPHI", "RJJJ"]),
    ("RKRR", "Incheon FIR", "İncheon FIR", "KR", [32.0, 123.9, 39.1, 132.1],
     ["ZBPE", "ZSHA", "RJJJ"]),
    ("RJJJ", "Fukuoka FIR", "Fukuoka FIR", "JP", [23.9, 122.9, 46.1, 146.1],
     ["RKRR", "RCAA", "ZSHA"]),
    # --- Latin America (feed coverage: HT, VE, MX, EC, CO, Central America) -
    ("MMFR", "Mexico FIR", "Meksika FIR", "MX", [14.0, -118.1, 32.9, -85.9],
     ["MHTG"]),
    ("MHTG", "Central America FIR", "Orta Amerika FIR", "HN", [12.4, -92.6, 18.6, -82.9],
     ["MMFR", "MTEG", "SKED", "MPZL"]),
    ("MPZL", "Panama FIR", "Panama FIR", "PA", [4.9, -83.7, 12.6, -76.8],
     ["MHTG", "SKED"]),
    ("MTEG", "Port-au-Prince FIR", "Port-au-Prince FIR", "HT", [17.4, -75.1, 20.6, -71.4],
     ["MHTG", "SKED"]),
    ("SKED", "Bogota FIR", "Bogota FIR", "CO", [-4.6, -79.6, 13.6, -66.4],
     ["MHTG", "MTEG", "SVZM", "SEFG"]),
    ("SVZM", "Maiquetia FIR", "Maiquetia FIR", "VE", [0.4, -73.6, 13.1, -59.4],
     ["SKED"]),
    ("SEFG", "Guayaquil FIR", "Guayaquil FIR", "EC", [-5.6, -84.1, 2.1, -74.9],
     ["SKED"]),
    ("SBBS", "Brasilia FIR", "Brezilya FIR", "BR", [-25.1, -60.1, -1.9, -40.9],
     ["SKED"]),
    ("SPIM", "Lima FIR", "Lima FIR", "PE", [-19.1, -84.1, 0.6, -68.0],
     ["SKED", "SEFG", "SCEZ", "SBBS"]),
    ("SCEZ", "Santiago FIR", "Santiago FIR", "CL", [-46.1, -80.1, -17.4, -66.0],
     ["SPIM", "SAEF"]),
    ("SAEF", "Ezeiza FIR", "Ezeiza FIR", "AR", [-42.1, -73.6, -21.4, -52.9],
     ["SCEZ", "SBBS"]),
    ("MUFH", "Habana FIR", "Havana FIR", "CU", [19.0, -85.6, 24.6, -73.4],
     ["MTEG", "MHTG", "MMFR"]),
    # --- Sub-Saharan Africa (delegated blocks) -----------------------------
    ("FCCC", "Brazzaville FIR", "Brazzaville FIR", "CG", [-6.1, 7.9, 8.1, 19.6],
     ["DNKK", "FTTT", "FZZA", "FNAN"]),
    ("FNAN", "Luanda FIR", "Luanda FIR", "AO", [-18.6, 8.0, -3.9, 24.6],
     ["FCCC", "FZZA", "FLFI", "FAJA"]),
    ("FLFI", "Lusaka FIR", "Lusaka FIR", "ZM", [-18.6, 21.4, -7.9, 34.1],
     ["FZZA", "FNAN", "FAJA", "FQBE", "HTDC"]),
    ("HUEC", "Kampala FIR", "Kampala FIR", "UG", [-2.6, 28.9, 4.6, 35.6],
     ["HKNA", "HJJJ", "FZZA", "HTDC"]),
    ("FMMM", "Antananarivo FIR", "Antananarivo FIR", "MG", [-27.1, 39.9, -5.9, 60.1],
     ["FQBE", "FIMM"]),
    ("FIMM", "Mauritius FIR", "Mauritius FIR", "MU", [-25.1, 51.9, -4.9, 75.1],
     ["FMMM", "FSSS"]),
    ("FSSS", "Seychelles FIR", "Seyşeller FIR", "SC", [-11.1, 45.9, -2.9, 60.1],
     ["FIMM", "FMMM", "HCSM", "HKNA"]),
    # --- Remaining Asia-Pacific -------------------------------------------
    ("VVTS", "Ho Chi Minh FIR", "Ho Chi Minh FIR", "VN", [6.9, 101.9, 17.1, 114.1],
     ["VVHN", "VTBB", "ZGZU", "RPHI", "WSJC"]),
    ("VVHN", "Hanoi FIR", "Hanoi FIR", "VN", [17.0, 101.9, 23.6, 110.1],
     ["VVTS", "VTBB", "ZGZU", "VYYF"]),
    ("VHHK", "Hong Kong FIR", "Hong Kong FIR", "HK", [17.9, 109.9, 23.6, 117.6],
     ["ZGZU", "RPHI", "RCAA", "VVTS"]),
    ("ZMUB", "Ulaanbaatar FIR", "Ulanbator FIR", "MN", [41.4, 87.4, 52.3, 120.1],
     ["ZWUQ", "ZBPE", "USSV", "UAAX"]),
    ("VRMF", "Male FIR", "Male FIR", "MV", [-1.1, 70.9, 8.1, 76.1],
     ["VCCF", "VABF"]),
    ("YBBB", "Brisbane FIR", "Brisbane FIR", "AU", [-45.1, 109.9, -8.9, 163.1],
     ["WAAF", "WIIF"]),
]

# A FIR is not a country: several are delegated blocks covering many states
# (Dakar covers Mali/Mauritania/Gambia…, Niamey covers Burkina/Benin/Togo…).
# The tuple above carries the primary state; this maps the full membership so
# country-level lookups resolve correctly.
MULTI_COUNTRY = {
    "GOOO": ["SN", "ML", "MR", "GM", "GW", "CV"],
    "DRRR": ["NE", "BF", "BJ", "TG"],
    "FTTT": ["TD", "CF"],
    "DGAC": ["GH", "CI"],
    "GLRB": ["LR", "SL", "GN"],
    "MHTG": ["HN", "GT", "SV", "NI", "CR", "BZ"],
    "FIMM": ["MU"],
    "HCSM": ["SO", "DJ"],
    "HAAA": ["ET", "DJ"],
    "FAJA": ["ZA", "BW", "NA", "LS", "SZ"],
    "FCCC": ["CG", "GA", "CM", "GQ"],
    "FLFI": ["ZM", "ZW", "MW"],
    "HUEC": ["UG", "RW", "BI"],
    "FMMM": ["MG", "KM"],
    "VVTS": ["VN", "KH"],
    "LTAA": ["TR"],
}

# ---------------------------------------------------------------------------
# Curated airport supplement, keyed by country. Fills the gaps the anchor
# gazetteer leaves — most European countries have a single anchor entry, and a
# "nearest commercial airports" list needs regional fields, not just the hub.
# (iata, icao, name, name_tr, city, lat, lon, tier)
# ---------------------------------------------------------------------------
SUPPLEMENT = {
    "PL": [
        ("WAW", "EPWA", "Warsaw Chopin", "Varşova Chopin", "Warsaw", 52.166, 20.967, "hub"),
        ("WMI", "EPMO", "Warsaw Modlin", "Varşova Modlin", "Warsaw", 52.451, 20.652, "regional"),
        ("KRK", "EPKK", "Kraków John Paul II", "Krakov", "Kraków", 50.078, 19.785, "major"),
        ("KTW", "EPKT", "Katowice", "Katoviçe", "Katowice", 50.474, 19.080, "major"),
        ("GDN", "EPGD", "Gdańsk Lech Wałęsa", "Gdansk", "Gdańsk", 54.378, 18.466, "major"),
        ("WRO", "EPWR", "Wrocław Copernicus", "Vroclav", "Wrocław", 51.103, 16.886, "major"),
        ("POZ", "EPPO", "Poznań Ławica", "Poznan", "Poznań", 52.421, 16.826, "major"),
        ("RZE", "EPRZ", "Rzeszów–Jasionka", "Rzeszow–Jasionka", "Rzeszów", 50.110, 22.019, "regional"),
        ("LUZ", "EPLB", "Lublin", "Lublin", "Lublin", 51.240, 22.714, "regional"),
        ("SZZ", "EPSC", "Szczecin–Goleniów", "Szczecin", "Szczecin", 53.585, 14.902, "regional"),
        ("BZG", "EPBY", "Bydgoszcz", "Bydgoszcz", "Bydgoszcz", 53.097, 17.978, "regional"),
        ("LCJ", "EPLL", "Łódź", "Lodz", "Łódź", 51.722, 19.398, "regional"),
    ],
    "UA": [
        ("KBP", "UKBB", "Kyiv Boryspil", "Kiev Boryspil", "Kyiv", 50.345, 30.895, "hub"),
        ("IEV", "UKKK", "Kyiv Zhuliany", "Kiev Juliani", "Kyiv", 50.402, 30.452, "major"),
        ("LWO", "UKLL", "Lviv Danylo Halytskyi", "Lviv", "Lviv", 49.813, 23.956, "major"),
        ("ODS", "UKOO", "Odesa", "Odesa", "Odesa", 46.427, 30.677, "major"),
        ("DNK", "UKDD", "Dnipro", "Dnipro", "Dnipro", 48.357, 35.100, "regional"),
        ("HRK", "UKHH", "Kharkiv", "Harkiv", "Kharkiv", 49.925, 36.290, "regional"),
        ("OZH", "UKDE", "Zaporizhzhia", "Zaporijya", "Zaporizhzhia", 47.867, 35.316, "regional"),
        ("SIP", "UKFF", "Simferopol", "Simferopol", "Simferopol", 45.052, 33.975, "regional"),
    ],
    "RU": [
        ("SVO", "UUEE", "Moscow Sheremetyevo", "Moskova Şeremetyevo", "Moscow", 55.973, 37.415, "hub"),
        ("DME", "UUDD", "Moscow Domodedovo", "Moskova Domodedovo", "Moscow", 55.408, 37.906, "hub"),
        ("VKO", "UUWW", "Moscow Vnukovo", "Moskova Vnukovo", "Moscow", 55.596, 37.261, "major"),
        ("LED", "ULLI", "St Petersburg Pulkovo", "St. Petersburg Pulkovo", "St Petersburg", 59.800, 30.263, "hub"),
        ("KGD", "UMKK", "Kaliningrad Khrabrovo", "Kaliningrad", "Kaliningrad", 54.890, 20.593, "regional"),
        ("ROV", "URRP", "Rostov-on-Don Platov", "Rostov Platov", "Rostov-on-Don", 47.494, 39.925, "major"),
        ("AER", "URSS", "Sochi", "Soçi", "Sochi", 43.450, 39.957, "major"),
        ("KRR", "URKK", "Krasnodar", "Krasnodar", "Krasnodar", 45.035, 39.171, "major"),
        ("MRV", "URMM", "Mineralnye Vody", "Mineralnye Vody", "Mineralnye Vody", 44.225, 43.082, "regional"),
        ("VOG", "URWW", "Volgograd", "Volgograd", "Volgograd", 48.782, 44.346, "regional"),
        ("KZN", "UWKD", "Kazan", "Kazan", "Kazan", 55.606, 49.279, "major"),
        ("SVX", "USSS", "Yekaterinburg Koltsovo", "Yekaterinburg", "Yekaterinburg", 56.743, 60.803, "major"),
        ("BZK", "UUBP", "Bryansk", "Bryansk", "Bryansk", 53.214, 34.176, "regional"),
        ("URS", "UUOK", "Kursk Vostochny", "Kursk", "Kursk", 51.751, 36.296, "regional"),
        ("EGO", "UUOB", "Belgorod", "Belgorod", "Belgorod", 50.644, 36.590, "regional"),
    ],
    "BY": [
        ("MSQ", "UMMS", "Minsk National", "Minsk", "Minsk", 53.885, 28.031, "major"),
        ("GME", "UMGG", "Gomel", "Gomel", "Gomel", 52.527, 31.017, "regional"),
        ("BQT", "UMBB", "Brest", "Brest", "Brest", 52.108, 23.898, "regional"),
    ],
    "LT": [
        ("VNO", "EYVI", "Vilnius", "Vilnius", "Vilnius", 54.634, 25.286, "major"),
        ("KUN", "EYKA", "Kaunas", "Kaunas", "Kaunas", 54.964, 24.085, "regional"),
        ("PLQ", "EYPA", "Palanga", "Palanga", "Palanga", 55.973, 21.094, "regional"),
    ],
    "LV": [("RIX", "EVRA", "Riga", "Riga", "Riga", 56.924, 23.971, "major")],
    "EE": [
        ("TLL", "EETN", "Tallinn Lennart Meri", "Tallinn", "Tallinn", 59.413, 24.833, "major"),
        ("TAY", "EETU", "Tartu", "Tartu", "Tartu", 58.307, 26.690, "regional"),
    ],
    "FI": [
        ("HEL", "EFHK", "Helsinki-Vantaa", "Helsinki-Vantaa", "Helsinki", 60.317, 24.963, "hub"),
        ("RVN", "EFRO", "Rovaniemi", "Rovaniemi", "Rovaniemi", 66.565, 25.830, "regional"),
    ],
    "SE": [
        ("ARN", "ESSA", "Stockholm Arlanda", "Stockholm Arlanda", "Stockholm", 59.652, 17.919, "hub"),
        ("GOT", "ESGG", "Gothenburg Landvetter", "Göteborg", "Gothenburg", 57.663, 12.280, "major"),
        ("MMX", "ESMS", "Malmö", "Malmö", "Malmö", 55.530, 13.372, "regional"),
    ],
    "NO": [
        ("OSL", "ENGM", "Oslo Gardermoen", "Oslo Gardermoen", "Oslo", 60.194, 11.100, "hub"),
        ("BGO", "ENBR", "Bergen Flesland", "Bergen", "Bergen", 60.293, 5.218, "major"),
        ("TOS", "ENTC", "Tromsø", "Tromsø", "Tromsø", 69.683, 18.919, "regional"),
    ],
    "DK": [
        ("CPH", "EKCH", "Copenhagen Kastrup", "Kopenhag", "Copenhagen", 55.618, 12.656, "hub"),
        ("BLL", "EKBI", "Billund", "Billund", "Billund", 55.740, 9.152, "regional"),
    ],
    "RO": [
        ("OTP", "LROP", "Bucharest Henri Coandă", "Bükreş Otopeni", "Bucharest", 44.572, 26.102, "hub"),
        ("CLJ", "LRCL", "Cluj-Napoca", "Kluj", "Cluj-Napoca", 46.785, 23.686, "major"),
        ("TSR", "LRTR", "Timișoara", "Timişoara", "Timișoara", 45.810, 21.338, "regional"),
        ("CND", "LRCK", "Constanța Mihail Kogălniceanu", "Köstence", "Constanța", 44.362, 28.488, "regional"),
        ("IAS", "LRIA", "Iași", "Yaş", "Iași", 47.179, 27.621, "regional"),
        ("SCV", "LRSV", "Suceava", "Suceava", "Suceava", 47.687, 26.354, "regional"),
    ],
    "MD": [("KIV", "LUKK", "Chișinău", "Kişinev", "Chișinău", 46.928, 28.931, "major")],
    "SK": [
        ("BTS", "LZIB", "Bratislava", "Bratislava", "Bratislava", 48.170, 17.213, "major"),
        ("KSC", "LZKZ", "Košice", "Koşice", "Košice", 48.663, 21.241, "regional"),
    ],
    "CZ": [
        ("PRG", "LKPR", "Prague Václav Havel", "Prag", "Prague", 50.101, 14.260, "hub"),
        ("BRQ", "LKTB", "Brno-Tuřany", "Brno", "Brno", 49.151, 16.694, "regional"),
        ("OSR", "LKMT", "Ostrava", "Ostrava", "Ostrava", 49.696, 18.111, "regional"),
    ],
    "HU": [
        ("BUD", "LHBP", "Budapest Ferenc Liszt", "Budapeşte", "Budapest", 47.437, 19.256, "hub"),
        ("DEB", "LHDC", "Debrecen", "Debrecen", "Debrecen", 47.489, 21.615, "regional"),
    ],
    "AT": [
        ("VIE", "LOWW", "Vienna Schwechat", "Viyana", "Vienna", 48.110, 16.570, "hub"),
        ("SZG", "LOWS", "Salzburg", "Salzburg", "Salzburg", 47.794, 13.004, "regional"),
        ("INN", "LOWI", "Innsbruck", "Innsbruck", "Innsbruck", 47.260, 11.344, "regional"),
    ],
    "BG": [
        ("SOF", "LBSF", "Sofia", "Sofya", "Sofia", 42.696, 23.411, "major"),
        ("VAR", "LBWN", "Varna", "Varna", "Varna", 43.232, 27.825, "regional"),
        ("BOJ", "LBBG", "Burgas", "Burgaz", "Burgas", 42.570, 27.515, "regional"),
    ],
    "RS": [
        ("BEG", "LYBE", "Belgrade Nikola Tesla", "Belgrad", "Belgrade", 44.818, 20.309, "major"),
        ("INI", "LYNI", "Niš", "Niş", "Niš", 43.337, 21.854, "regional"),
    ],
    "HR": [
        ("ZAG", "LDZA", "Zagreb Franjo Tuđman", "Zagreb", "Zagreb", 45.743, 16.069, "major"),
        ("SPU", "LDSP", "Split", "Split", "Split", 43.539, 16.298, "regional"),
        ("DBV", "LDDU", "Dubrovnik", "Dubrovnik", "Dubrovnik", 42.561, 18.268, "regional"),
    ],
    "BA": [("SJJ", "LQSA", "Sarajevo", "Saraybosna", "Sarajevo", 43.825, 18.331, "regional")],
    "MK": [("SKP", "LWSK", "Skopje", "Üsküp", "Skopje", 41.962, 21.621, "regional")],
    "AL": [("TIA", "LATI", "Tirana Nënë Tereza", "Tiran", "Tirana", 41.415, 19.720, "regional")],
    "ME": [
        ("TGD", "LYPG", "Podgorica", "Podgorica", "Podgorica", 42.359, 19.252, "regional"),
        ("TIV", "LYTV", "Tivat", "Tivat", "Tivat", 42.404, 18.723, "regional"),
    ],
    "SI": [("LJU", "LJLJ", "Ljubljana Jože Pučnik", "Ljubljana", "Ljubljana", 46.224, 14.458, "regional")],
    "GR": [
        ("ATH", "LGAV", "Athens Eleftherios Venizelos", "Atina", "Athens", 37.936, 23.947, "hub"),
        ("SKG", "LGTS", "Thessaloniki", "Selanik", "Thessaloniki", 40.520, 22.971, "major"),
        ("HER", "LGIR", "Heraklion", "Heraklion", "Heraklion", 35.340, 25.180, "major"),
        ("RHO", "LGRP", "Rhodes", "Rodos", "Rhodes", 36.405, 28.086, "regional"),
        ("CFU", "LGKR", "Corfu", "Korfu", "Corfu", 39.602, 19.912, "regional"),
    ],
    "CY": [
        ("LCA", "LCLK", "Larnaca", "Larnaka", "Larnaca", 34.875, 33.625, "major"),
        ("PFO", "LCPH", "Paphos", "Baf", "Paphos", 34.718, 32.486, "regional"),
        ("ECN", "LCEN", "Ercan", "Ercan", "Nicosia", 35.155, 33.500, "regional"),
    ],
    "TR": [
        ("IST", "LTFM", "Istanbul Airport", "İstanbul Havalimanı", "Istanbul", 41.262, 28.742, "hub"),
        ("SAW", "LTFJ", "Istanbul Sabiha Gökçen", "İstanbul Sabiha Gökçen", "Istanbul", 40.899, 29.309, "hub"),
        ("ESB", "LTAC", "Ankara Esenboğa", "Ankara Esenboğa", "Ankara", 40.128, 32.995, "major"),
        ("AYT", "LTAI", "Antalya", "Antalya", "Antalya", 36.899, 30.800, "hub"),
        ("ADB", "LTBJ", "İzmir Adnan Menderes", "İzmir Adnan Menderes", "İzmir", 38.292, 27.157, "major"),
        ("ADA", "LTAF", "Adana Şakirpaşa", "Adana Şakirpaşa", "Adana", 36.982, 35.280, "major"),
        ("GZT", "LTAJ", "Gaziantep", "Gaziantep", "Gaziantep", 36.947, 37.479, "regional"),
        ("TZX", "LTCG", "Trabzon", "Trabzon", "Trabzon", 40.995, 39.790, "regional"),
        ("DIY", "LTCC", "Diyarbakır", "Diyarbakır", "Diyarbakır", 37.894, 40.201, "regional"),
        ("VAN", "LTCI", "Van Ferit Melen", "Van Ferit Melen", "Van", 38.468, 43.332, "regional"),
        ("BJV", "LTFE", "Bodrum Milas", "Bodrum Milas", "Bodrum", 37.251, 27.664, "regional"),
        ("DLM", "LTBS", "Dalaman", "Dalaman", "Dalaman", 36.713, 28.793, "regional"),
        ("HTY", "LTDA", "Hatay", "Hatay", "Hatay", 36.363, 36.282, "regional"),
        ("SXZ", "LTCM", "Siirt", "Siirt", "Siirt", 37.978, 41.840, "regional"),
    ],
    "GE": [
        ("TBS", "UGTB", "Tbilisi", "Tiflis", "Tbilisi", 41.669, 44.955, "major"),
        ("BUS", "UGSB", "Batumi", "Batum", "Batumi", 41.610, 41.600, "regional"),
        ("KUT", "UGKO", "Kutaisi", "Kutaisi", "Kutaisi", 42.177, 42.483, "regional"),
    ],
    "AM": [("EVN", "UDYZ", "Yerevan Zvartnots", "Erivan Zvartnots", "Yerevan", 40.147, 44.396, "major")],
    "AZ": [
        ("GYD", "UBBB", "Baku Heydar Aliyev", "Bakü Haydar Aliyev", "Baku", 40.467, 50.047, "major"),
        ("GNJ", "UBBG", "Ganja", "Gence", "Ganja", 40.738, 46.317, "regional"),
    ],
    "DE": [
        ("FRA", "EDDF", "Frankfurt", "Frankfurt", "Frankfurt", 50.033, 8.571, "hub"),
        ("MUC", "EDDM", "Munich", "Münih", "Munich", 48.354, 11.786, "hub"),
        ("BER", "EDDB", "Berlin Brandenburg", "Berlin Brandenburg", "Berlin", 52.362, 13.501, "hub"),
        ("DUS", "EDDL", "Düsseldorf", "Düsseldorf", "Düsseldorf", 51.289, 6.767, "major"),
        ("HAM", "EDDH", "Hamburg", "Hamburg", "Hamburg", 53.630, 9.988, "major"),
        ("CGN", "EDDK", "Cologne Bonn", "Köln Bonn", "Cologne", 50.866, 7.143, "major"),
        ("STR", "EDDS", "Stuttgart", "Stuttgart", "Stuttgart", 48.690, 9.222, "major"),
        ("LEJ", "EDDP", "Leipzig/Halle", "Leipzig", "Leipzig", 51.424, 12.236, "regional"),
    ],
    # GB/BE entries are hand-set because the anchor gazetteer geocodes BHX to
    # Birmingham, Alabama and BRU to Ontario, Canada (verified 2026-07-31).
    "GB": [
        ("LHR", "EGLL", "London Heathrow", "Londra Heathrow", "London", 51.470, -0.454, "hub"),
        ("LGW", "EGKK", "London Gatwick", "Londra Gatwick", "London", 51.148, -0.190, "major"),
        ("STN", "EGSS", "London Stansted", "Londra Stansted", "London", 51.885, 0.235, "major"),
        ("LTN", "EGGW", "London Luton", "Londra Luton", "London", 51.875, -0.368, "regional"),
        ("MAN", "EGCC", "Manchester", "Manchester", "Manchester", 53.365, -2.273, "major"),
        ("BHX", "EGBB", "Birmingham", "Birmingham", "Birmingham", 52.454, -1.748, "major"),
        ("EDI", "EGPH", "Edinburgh", "Edinburgh", "Edinburgh", 55.950, -3.372, "major"),
        ("GLA", "EGPF", "Glasgow", "Glasgow", "Glasgow", 55.872, -4.433, "regional"),
        ("BFS", "EGAA", "Belfast International", "Belfast", "Belfast", 54.658, -6.216, "regional"),
    ],
    "BE": [
        ("BRU", "EBBR", "Brussels", "Brüksel", "Brussels", 50.901, 4.484, "hub"),
        ("CRL", "EBCI", "Brussels South Charleroi", "Charleroi", "Charleroi", 50.459, 4.454, "regional"),
        ("LGG", "EBLG", "Liège", "Liège", "Liège", 50.637, 5.443, "regional"),
        ("OST", "EBOS", "Ostend-Bruges", "Ostend", "Ostend", 51.199, 2.863, "regional"),
    ],
    "PA": [("PTY", "MPTO", "Panama Tocumen", "Panama Tocumen", "Panama City", 9.071, -79.383, "hub")],
    "IL": [
        ("TLV", "LLBG", "Tel Aviv Ben Gurion", "Tel Aviv Ben Gurion", "Tel Aviv", 32.011, 34.887, "hub"),
        ("ETM", "LLER", "Ramon", "Ramon", "Eilat", 29.723, 35.011, "regional"),
        ("HFA", "LLHA", "Haifa", "Hayfa", "Haifa", 32.809, 35.043, "regional"),
    ],
    "JO": [
        ("AMM", "OJAI", "Amman Queen Alia", "Amman Kraliçe Alia", "Amman", 31.723, 35.993, "major"),
        ("AQJ", "OJAQ", "Aqaba King Hussein", "Akabe", "Aqaba", 29.612, 35.018, "regional"),
    ],
    "LB": [("BEY", "OLBA", "Beirut Rafic Hariri", "Beyrut Rafik Hariri", "Beirut", 33.821, 35.488, "major")],
    "SY": [
        ("DAM", "OSDI", "Damascus", "Şam", "Damascus", 33.411, 36.516, "major"),
        ("ALP", "OSAP", "Aleppo", "Halep", "Aleppo", 36.181, 37.224, "regional"),
        ("LTK", "OSLK", "Latakia Bassel al-Assad", "Lazkiye", "Latakia", 35.401, 35.949, "regional"),
    ],
    "IQ": [
        ("BGW", "ORBI", "Baghdad", "Bağdat", "Baghdad", 33.263, 44.235, "major"),
        ("EBL", "ORER", "Erbil", "Erbil", "Erbil", 36.238, 43.963, "major"),
        ("ISU", "ORSU", "Sulaymaniyah", "Süleymaniye", "Sulaymaniyah", 35.562, 45.317, "regional"),
        ("BSR", "ORMM", "Basra", "Basra", "Basra", 30.549, 47.662, "regional"),
        ("NJF", "ORNI", "Najaf", "Necef", "Najaf", 31.990, 44.404, "regional"),
    ],
    "IR": [
        ("IKA", "OIIE", "Tehran Imam Khomeini", "Tahran İmam Humeyni", "Tehran", 35.416, 51.152, "hub"),
        ("THR", "OIII", "Tehran Mehrabad", "Tahran Mehrabad", "Tehran", 35.689, 51.313, "major"),
        ("MHD", "OIMM", "Mashhad", "Meşhed", "Mashhad", 36.235, 59.641, "major"),
        ("IFN", "OIFM", "Isfahan", "İsfahan", "Isfahan", 32.751, 51.861, "major"),
        ("SYZ", "OISS", "Shiraz", "Şiraz", "Shiraz", 29.539, 52.590, "regional"),
        ("TBZ", "OITT", "Tabriz", "Tebriz", "Tabriz", 38.134, 46.235, "regional"),
        ("BND", "OIKB", "Bandar Abbas", "Bender Abbas", "Bandar Abbas", 27.218, 56.378, "regional"),
        ("AWZ", "OIAW", "Ahvaz", "Ahvaz", "Ahvaz", 31.337, 48.760, "regional"),
    ],
    "SA": [
        ("RUH", "OERK", "Riyadh King Khalid", "Riyad Kral Halid", "Riyadh", 24.958, 46.700, "hub"),
        ("JED", "OEJN", "Jeddah King Abdulaziz", "Cidde Kral Abdülaziz", "Jeddah", 21.680, 39.157, "hub"),
        ("DMM", "OEDF", "Dammam King Fahd", "Dammam Kral Fahd", "Dammam", 26.471, 49.798, "major"),
        ("MED", "OEMA", "Medina", "Medine", "Medina", 24.554, 39.705, "major"),
        ("AHB", "OEAB", "Abha", "Ebha", "Abha", 18.240, 42.657, "regional"),
    ],
    "AE": [
        ("DXB", "OMDB", "Dubai International", "Dubai", "Dubai", 25.253, 55.365, "hub"),
        ("DWC", "OMDW", "Dubai Al Maktoum", "Dubai Al Maktoum", "Dubai", 24.896, 55.161, "major"),
        ("AUH", "OMAA", "Abu Dhabi Zayed", "Abu Dabi Zayed", "Abu Dhabi", 24.433, 54.651, "hub"),
        ("SHJ", "OMSJ", "Sharjah", "Şarika", "Sharjah", 25.328, 55.517, "major"),
    ],
    "QA": [("DOH", "OTHH", "Doha Hamad", "Doha Hamad", "Doha", 25.273, 51.608, "hub")],
    "KW": [("KWI", "OKKK", "Kuwait International", "Kuveyt", "Kuwait City", 29.227, 47.969, "major")],
    "BH": [("BAH", "OBBI", "Bahrain International", "Bahreyn", "Manama", 26.271, 50.634, "major")],
    "OM": [
        ("MCT", "OOMS", "Muscat", "Maskat", "Muscat", 23.593, 58.284, "major"),
        ("SLL", "OOSA", "Salalah", "Selale", "Salalah", 17.039, 54.091, "regional"),
    ],
    "YE": [
        ("SAH", "OYSN", "Sanaa", "Sana", "Sanaa", 15.476, 44.220, "regional"),
        ("ADE", "OYAA", "Aden", "Aden", "Aden", 12.830, 45.029, "regional"),
    ],
    "EG": [
        ("CAI", "HECA", "Cairo", "Kahire", "Cairo", 30.112, 31.400, "hub"),
        ("HRG", "HEGN", "Hurghada", "Hurgada", "Hurghada", 27.178, 33.799, "major"),
        ("SSH", "HESH", "Sharm el-Sheikh", "Şarm El Şeyh", "Sharm el-Sheikh", 27.977, 34.395, "major"),
        ("ATZ", "HEAT", "Assiut", "Asyut", "Assiut", 27.046, 31.012, "regional"),
    ],
    "LY": [
        ("TIP", "HLLT", "Tripoli Mitiga", "Trablus Mitiga", "Tripoli", 32.894, 13.276, "regional"),
        ("BEN", "HLLB", "Benghazi Benina", "Bingazi Benina", "Benghazi", 32.097, 20.269, "regional"),
        ("MRA", "HLMS", "Misrata", "Misrata", "Misrata", 32.325, 15.061, "regional"),
    ],
    "AF": [
        ("KBL", "OAKB", "Kabul Hamid Karzai", "Kabil", "Kabul", 34.566, 69.212, "major"),
        ("KDH", "OAKN", "Kandahar", "Kandahar", "Kandahar", 31.506, 65.848, "regional"),
        ("MZR", "OAMS", "Mazar-i-Sharif", "Mezar-ı Şerif", "Mazar-i-Sharif", 36.707, 67.209, "regional"),
    ],
    "PK": [
        ("ISB", "OPIS", "Islamabad", "İslamabad", "Islamabad", 33.549, 72.826, "major"),
        ("KHI", "OPKC", "Karachi Jinnah", "Karaçi Cinnah", "Karachi", 24.907, 67.161, "hub"),
        ("LHE", "OPLA", "Lahore Allama Iqbal", "Lahor", "Lahore", 31.522, 74.404, "major"),
        ("PEW", "OPPS", "Peshawar Bacha Khan", "Peşaver", "Peshawar", 33.994, 71.515, "regional"),
        ("UET", "OPQT", "Quetta", "Quetta", "Quetta", 30.251, 66.938, "regional"),
    ],
    "SD": [
        ("KRT", "HSSS", "Khartoum", "Hartum", "Khartoum", 15.590, 32.553, "regional"),
        ("PZU", "HSPN", "Port Sudan", "Port Sudan", "Port Sudan", 19.434, 37.234, "regional"),
    ],
    "SO": [
        ("MGQ", "HCMM", "Mogadishu Aden Adde", "Mogadişu", "Mogadishu", 2.014, 45.305, "regional"),
        ("HGA", "HCMH", "Hargeisa", "Hargeysa", "Hargeisa", 9.518, 44.089, "regional"),
    ],
    "ML": [
        ("BKO", "GABS", "Bamako Modibo Keïta", "Bamako", "Bamako", 12.534, -7.950, "regional"),
        ("GAQ", "GAGO", "Gao", "Gao", "Gao", 16.248, -0.005, "regional"),
    ],
    "NE": [("NIM", "DRRN", "Niamey Diori Hamani", "Niamey", "Niamey", 13.481, 2.184, "regional")],
    "BF": [("OUA", "DFFD", "Ouagadougou", "Ouagadougou", "Ouagadougou", 12.353, -1.512, "regional")],
    "NG": [
        ("LOS", "DNMM", "Lagos Murtala Muhammed", "Lagos", "Lagos", 6.577, 3.321, "hub"),
        ("ABV", "DNAA", "Abuja Nnamdi Azikiwe", "Abuja", "Abuja", 9.007, 7.263, "major"),
        ("KAN", "DNKN", "Kano Mallam Aminu", "Kano", "Kano", 12.048, 8.524, "regional"),
        ("MIU", "DNMA", "Maiduguri", "Maiduguri", "Maiduguri", 11.855, 13.081, "regional"),
    ],
    "ET": [("ADD", "HAAB", "Addis Ababa Bole", "Addis Ababa Bole", "Addis Ababa", 8.978, 38.799, "hub")],
    "KE": [
        ("NBO", "HKJK", "Nairobi Jomo Kenyatta", "Nairobi", "Nairobi", -1.319, 36.928, "hub"),
        ("MBA", "HKMO", "Mombasa Moi", "Mombasa", "Mombasa", -4.035, 39.594, "regional"),
    ],
    "IN": [
        ("DEL", "VIDP", "Delhi Indira Gandhi", "Delhi", "Delhi", 28.556, 77.100, "hub"),
        ("BOM", "VABB", "Mumbai Chhatrapati Shivaji", "Mumbai", "Mumbai", 19.089, 72.868, "hub"),
        ("MAA", "VOMM", "Chennai", "Chennai", "Chennai", 12.994, 80.180, "major"),
        ("CCU", "VECC", "Kolkata Netaji Subhas", "Kalküta", "Kolkata", 22.655, 88.447, "major"),
        ("SXR", "VISR", "Srinagar", "Srinagar", "Srinagar", 33.987, 74.774, "regional"),
    ],
    "TW": [("TPE", "RCTP", "Taipei Taoyuan", "Taipei Taoyuan", "Taipei", 25.078, 121.233, "hub")],
    "CN": [
        ("PEK", "ZBAA", "Beijing Capital", "Pekin", "Beijing", 40.080, 116.585, "hub"),
        ("PKX", "ZBAD", "Beijing Daxing", "Pekin Daxing", "Beijing", 39.510, 116.411, "hub"),
        ("PVG", "ZSPD", "Shanghai Pudong", "Şanghay Pudong", "Shanghai", 31.143, 121.805, "hub"),
        ("CAN", "ZGGG", "Guangzhou Baiyun", "Guangzhou", "Guangzhou", 23.392, 113.299, "hub"),
        ("URC", "ZWWW", "Urumqi Diwopu", "Urumçi", "Urumqi", 43.907, 87.474, "major"),
        ("XIY", "ZLXY", "Xi'an Xianyang", "Xi'an Xianyang", "Xi'an", 34.447, 108.752, "major"),
    ],
    "MM": [("RGN", "VYYY", "Yangon", "Yangon", "Yangon", 16.907, 96.133, "regional")],
    "HT": [("PAP", "MTPP", "Port-au-Prince Toussaint Louverture", "Port-au-Prince", "Port-au-Prince", 18.580, -72.293, "regional")],
    "VE": [("CCS", "SVMI", "Caracas Simón Bolívar", "Karakas", "Caracas", 10.601, -66.991, "major")],
    "EC": [
        ("UIO", "SEQM", "Quito Mariscal Sucre", "Quito", "Quito", -0.129, -78.358, "major"),
        ("GYE", "SEGU", "Guayaquil José Joaquín de Olmedo", "Guayaquil", "Guayaquil", -2.157, -79.884, "major"),
    ],
    "MX": [
        ("MEX", "MMMX", "Mexico City Benito Juárez", "Meksiko", "Mexico City", 19.436, -99.072, "hub"),
        ("CUN", "MMUN", "Cancún", "Cancun", "Cancún", 21.037, -86.877, "major"),
    ],
    "CO": [("BOG", "SKBO", "Bogotá El Dorado", "Bogota", "Bogotá", 4.702, -74.147, "hub")],
}


def _in_bbox(lat: float, lon: float, bbox: list) -> bool:
    return bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]


def main() -> None:
    firs = [
        {"icao": icao, "name": name, "name_tr": name_tr,
         "countries": MULTI_COUNTRY.get(icao, [country]),
         "bbox": bbox, "neighbors": neighbors}
        for icao, name, name_tr, country, bbox, neighbors in FIRS
    ]

    # Adjacency is a symmetric relation, but the hand-written table declares each
    # border from whichever side it was easiest to think about. Close it here so
    # "who borders me" is complete no matter which FIR you ask from.
    by_icao = {f["icao"]: f for f in firs}
    for f in firs:
        for n in list(f["neighbors"]):
            if n not in by_icao:
                raise SystemExit(f"{f['icao']} lists unknown neighbor {n}")
            back = by_icao[n]["neighbors"]
            if f["icao"] not in back:
                back.append(f["icao"])
    for f in firs:
        f["neighbors"] = sorted(set(f["neighbors"]))

    def covered(lat: float, lon: float) -> bool:
        return any(_in_bbox(lat, lon, f["bbox"]) for f in firs)

    airports: dict[str, dict] = {}

    # 1. Curated supplement wins on conflict — it carries ICAO codes and Turkish names.
    for country, rows in SUPPLEMENT.items():
        for iata, icao, name, name_tr, city, lat, lon, tier in rows:
            if not covered(lat, lon):
                raise SystemExit(f"supplement airport {iata} ({lat},{lon}) is outside every FIR bbox")
            airports[iata] = {
                "iata": iata, "icao": icao, "name": name, "name_tr": name_tr,
                "city": city, "country": country, "lat": round(lat, 4),
                "lon": round(lon, 4), "tier": tier,
            }

    # 2. Fill the rest from the anchor gazetteer (lat/lon/IATA/country already there).
    with open(REPO / "db" / "anchors.json", encoding="utf-8") as fh:
        anchors = json.load(fh)
    seeded = 0
    dropped: dict[str, int] = {}
    for a in anchors:
        if a.get("type") != "Airport":
            continue
        iata, country = a.get("iata"), a.get("country")
        if not iata or iata in airports:
            continue
        lat, lon = a.get("lat"), a.get("lon")
        if lat is None or lon is None:
            continue
        if not covered(float(lat), float(lon)):
            # Outside the FIR footprint — an airport we could never place in an
            # airspace would be an orphan record in a FIR-keyed dataset.
            dropped[country or "??"] = dropped.get(country or "??", 0) + 1
            continue
        name = a["name"]
        airports[iata] = {
            "iata": iata, "icao": None, "name": name, "name_tr": name,
            "city": (a.get("city") or "").title(), "country": country,
            "lat": round(float(lat), 4), "lon": round(float(lon), 4),
            "tier": "regional",
        }
        seeded += 1

    out = {
        "version": "2026-07-31",
        "note": "FIR bbox extents are approximate (~0.1 deg) and airport tiers are "
                "editorial. Used for proximity analysis only, never for navigation.",
        "firs": sorted(firs, key=lambda f: f["icao"]),
        "airports": sorted(airports.values(), key=lambda a: (a["country"], a["iata"])),
    }
    dest = REPO / "config" / "airspace.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")

    fir_countries = {c for f in firs for c in f["countries"]}
    print(f"wrote {dest}")
    print(f"  FIRs      : {len(firs)} covering {len(fir_countries)} countries")
    print(f"  airports  : {len(airports)} ({seeded} seeded from anchors.json)")
    print(f"  dropped (outside every FIR bbox): {sorted(dropped.items())}")
    print("  FIR countries with no airport: "
          f"{sorted(fir_countries - {a['country'] for a in airports.values()})}")


if __name__ == "__main__":
    main()
