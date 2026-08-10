"""
SIM — SITREP Verification Labeling
Rule-based "Doğruluk Durumu" labels for daily country SITREPs.

Labels are computed deterministically from source domains — the LLM never
decides or upgrades them. Canonical labels (Turkish, verbatim in reports):

    Onaylandı (Resmî)          — at least one official/state source in the cluster
    Onaylandı (Çoklu kaynak)   — ≥2 independent (registrable-domain) sources
    Doğrulanmamış (Tek kaynak) — single source, not official
"""

from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

LABEL_OFFICIAL = "Onaylandı (Resmî)"
LABEL_MULTI = "Onaylandı (Çoklu kaynak)"
LABEL_SINGLE = "Doğrulanmamış (Tek kaynak)"

CANONICAL_LABELS = (LABEL_OFFICIAL, LABEL_MULTI, LABEL_SINGLE)

# Intergovernmental TLD, matched as a suffix ("nato.int", "reliefweb.int").
OFFICIAL_TLD_SUFFIXES = (".int",)

# Government / military domain LABELS. This used to be a suffix test against
# (".gov", ".mil", ".int") whose comment claimed "centcom.mil and mod.gov.ua both
# hit" — mod.gov.ua did not, because it ends in ".ua". Only US-style .gov/.mil plus
# the hard-coded gov.uk/gov.il qualified, so Ukraine's MoD, India's MHA and South
# Africa's police were never official while (until 2026-08-10) TASS was official
# everywhere. Matching on the label instead covers every country's gov.XX form.
#
# A label test, not a substring one: "mygovnews.com" has the single label
# "mygovnews" and stays unofficial.
OFFICIAL_DOMAIN_LABELS = frozenset({"gov", "mil"})

# Suffix match also covers subdomains and country variants (e.g. "travel.state.gov",
# "gov.uk", "gov.il"). These confer "official" wherever the event happened: a
# multinational body has no adversary, and a government portal publishing an advisory
# or a statement is officially speaking for itself whatever country it is about.
OFFICIAL_DOMAINS = (
    # multi-national / NGO-official
    "un.org",
    "nato.int",
    "reliefweb.int",
    "europa.eu",
    "gdacs.org",
    # national portals that don't use .gov/.mil
    "gov.uk",
    "gov.il",
)

# State news agencies, mapped to the country whose state they speak for.
#
# v1 treated these as official everywhere, reasoning that "the state said it happened"
# is the confirmation signal even when the state is a party to the conflict. Measuring
# it (14 days to 2026-08-10) showed the rule fires almost entirely in the case where it
# is wrong: Anadolu carried 143 events of which 2 were about Turkey and 81 about other
# countries, TASS 126 of which 21 were about Russia and 55 about others — overwhelmingly
# Ukraine. So the 2026-08-09 Ukraine SITREP told its reader, as "Onaylandı (Resmî)",
# that Russian forces had struck "military warehouses storing electronic warfare
# equipment" in Odesa port — sourced to TASS quoting the Russian MoD. That is a
# belligerent's targeting claim about the adversary's territory presented as verified
# fact, which is the one thing a verification label must never do.
#
# A state agency reporting its OWN country is still the strongest available
# confirmation and keeps the official label. Reporting anyone else it is an ordinary
# source: it still counts toward multi-source corroboration, it just cannot confer
# "officially confirmed" on its own.
STATE_MEDIA_HOME_ISO = {
    "irna.ir": "IR",
    "mehrnews.com": "IR",
    "tasnimnews.com": "IR",
    "farsnews.ir": "IR",
    "iribnews.ir": "IR",
    "aa.com.tr": "TR",
    "sana.sy": "SY",
    "tass.com": "RU",
    "kuna.net.kw": "KW",
    "wam.ae": "AE",
    # These four are the reason the state-media check must run BEFORE the generic
    # government-label rule: they are wire services sitting on gov domains, and they
    # report abroad far more than at home. Over the 14 days to 2026-08-10
    # newsonair.gov.in filed 2 events about India against 7 about Iran, 3 about
    # Pakistan and 3 about Saudi Arabia; ddnews.gov.in filed none about India at all.
    "spa.gov.sa": "SA",
    "petra.gov.jo": "JO",
    "newsonair.gov.in": "IN",
    "ddnews.gov.in": "IN",
    "bna.bh": "BH",
    "ina.iq": "IQ",
    "saba.ye": "YE",
}

# Known second-level public suffixes so registrable_domain("news.gov.uk") returns
# "gov.uk"-anchored hosts correctly. Not a full PSL — covers the feeds SIM ingests.
_SECOND_LEVEL_SUFFIXES = {
    "co.uk", "gov.uk", "ac.uk", "org.uk",
    "com.tr", "gov.tr", "org.tr", "net.tr",
    "com.au", "gov.au", "co.il", "gov.il",
    "co.jp", "go.jp", "com.br", "gov.br",
    "co.in", "gov.in", "com.pk", "gov.pk",
    "gov.jo", "gov.sa", "gov.ae", "net.kw",
    "gov.ua", "com.ua", "gov.za", "co.za",
}


def registrable_domain(domain_or_url: str) -> str:
    """Reduce a hostname or URL to its registrable domain (heuristic eTLD+1)."""
    if not domain_or_url:
        return ""
    host = domain_or_url.strip().lower()
    if "//" in host:
        host = urlparse(host).netloc or host
    host = host.split("@")[-1].split(":")[0].strip(".")
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    last_two = ".".join(parts[-2:])
    if last_two in _SECOND_LEVEL_SUFFIXES:
        return ".".join(parts[-3:])
    return last_two


def state_media_home_iso(domain: str) -> str | None:
    """The ISO2 whose state this domain's news agency speaks for, if it is one."""
    reg = registrable_domain(domain)
    if not reg:
        return None
    for agency, iso in STATE_MEDIA_HOME_ISO.items():
        if reg == agency or reg.endswith("." + agency):
            return iso
    return None


def is_official_domain(domain: str, event_isos: Iterable[str] = ()) -> bool:
    """True if the domain counts as an official source FOR THESE EVENTS.

    `event_isos` is the country (or countries) the reporting is about. It only
    matters for state news agencies, which speak officially for their own country
    and are ordinary — often interested — sources about anyone else; see
    STATE_MEDIA_HOME_ISO. Callers that genuinely have no country context pass
    nothing, in which case a state agency does not qualify: an unverifiable
    "official" is the failure mode this guard exists to prevent.
    """
    reg = registrable_domain(domain)
    if not reg:
        return False
    # State media is checked FIRST and decides on its own: SPA and Petra are news
    # agencies that happen to sit on gov.sa / gov.jo, so the generic government-label
    # rule below would otherwise hand them the very cross-border authority this
    # function exists to withhold.
    home = state_media_home_iso(reg)
    if home is not None:
        return home in {iso for iso in event_isos if iso}
    if reg.endswith(OFFICIAL_TLD_SUFFIXES):
        return True
    if OFFICIAL_DOMAIN_LABELS & set(reg.split(".")):
        return True
    return any(reg == d or reg.endswith("." + d) for d in OFFICIAL_DOMAINS)


def label_cluster(
    events: List[Dict[str, Any]],
    penalized_domains: Optional[Iterable[str]] = None,
) -> str:
    """
    Compute the verification label for one event cluster (same real-world event
    reported by 1..n sources). `penalized_domains` (from the domain_penalties
    table) are excluded from both the official check and the independence count;
    if every source is penalized the cluster stays unverified.
    """
    penalized = {registrable_domain(d) for d in (penalized_domains or ())}
    domains = []
    for ev in events:
        reg = registrable_domain(ev.get("source_domain") or ev.get("source_url") or "")
        if reg and reg not in penalized:
            domains.append(reg)

    # Which country the cluster is about — a state agency is only "official" at home.
    event_isos = {(ev.get("country_iso") or "").strip().upper() for ev in events}

    if any(is_official_domain(d, event_isos) for d in domains):
        return LABEL_OFFICIAL
    if len(set(domains)) >= 2:
        return LABEL_MULTI
    return LABEL_SINGLE


def fallback_cluster_key(event: Dict[str, Any]) -> tuple:
    """
    Grouping key for events without a storyline_id: same location + event type
    + calendar day is treated as one incident for corroboration purposes.
    """
    occurred = event.get("occurred_at_est") or event.get("published_at")
    day = str(occurred)[:10] if occurred else ""
    location = (event.get("anchor_name_norm") or event.get("anchor_name_raw") or "").strip().lower()
    return (location, (event.get("event_type") or "").strip().lower(), day)
