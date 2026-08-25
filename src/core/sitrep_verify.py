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


def registrable_domain(domain_or_url: str) -> str:
    """Reduce a hostname or URL to its registrable domain (eTLD+1).

    The suffix decision is delegated to ingest's extract_domain(), i.e. to tldextract
    and the real public suffix list. It used to be a hand-maintained set of 28
    second-level suffixes, which silently ate the publisher's name on every ccTLD not on
    the list: nst.com.my -> "com.my", abc.net.au -> "net.au", nhk.or.jp -> "or.jp", and
    mk.co.kr / asiae.co.kr / koreatimes.co.kr all -> "co.kr". Measured 2026-08-13, 202
    events across 65 domains over 7 days.

    That was not cosmetic. Three decisions read this function:
      * label_cluster() counts len(set(domains)) for "Onaylandı (Çoklu kaynak)" — three
        distinct Korean outlets collapsed to one, publishing a two-source cluster as
        "Tek kaynak";
      * _record_corroboration() treats an equal registrable domain as an outlet
        republishing itself and records nothing, which suppresses system_confidence —
        the number the ALERT (0.50) and CRITICAL (0.62) gates read;
      * state_media_home_iso()/is_official_domain() inherit the same collapse.

    Imported lazily, mirroring airspace.py: it keeps src.core importable without
    src.pipeline (ingest_filters loads keywords.json/settings.json at import) and there
    is exactly one eTLD+1 implementation in the codebase, so the report layer and the
    ingest layer cannot disagree about what one publisher is.
    """
    if not domain_or_url:
        return ""
    host = domain_or_url.strip().lower()
    if "//" in host:
        host = urlparse(host).netloc or host
    # Kept: tldextract reads hostnames, not userinfo/port syntax.
    host = host.split("@")[-1].split(":")[0].strip(".")
    if not host:
        return ""
    from src.pipeline.ingest_filters import extract_domain
    return extract_domain(host) or host


# Carriers: domains that redistribute another newsroom's copy instead of reporting.
# They are not independent publishers, so they must never turn a single-source story
# into a corroborated one — the verification label is the thing a reader trusts most.
#
# Measured over the 7 days to 2026-08-25 on live corroborating_sources: yahoo.com was
# the single most frequent corroborator in the whole corpus (93 credits), ahead of
# Al Jazeera, with reddit.com at 43 and aol.com + aol.co.uk at 46. Yahoo and AOL are
# one syndication feed, so an event carrying both looked doubly confirmed by a single
# wire. The clearest case: the Haiti church attack was credited to inbox.lv,
# modernghana.com and yahoo.com under the BYTE-IDENTICAL headline "Haiti gang members
# kill around 40 displaced people in church attack near capital" — one agency report
# counted as three independent outlets.
#
# Brands are matched on the FIRST LABEL of the registrable domain, the same technique
# OFFICIAL_DOMAIN_LABELS uses, so ccTLD editions (yahoo.co.jp, aol.co.uk) are covered
# without listing every country. Short or generic names that no label test can safely
# claim ("t.me", "x.com") are matched exactly instead.
#
# Deliberately NOT listed: outlets with a real newsroom that ALSO republish wire copy
# (modernghana.com, streamlinefeed.co.ke). Blanket-listing them would discard their
# original reporting; the identical-headline signal is the honest way to catch a
# syndicated filing, and that is a separate change.
NON_INDEPENDENT_LABELS = frozenset({
    "yahoo", "aol", "msn", "reddit", "facebook", "instagram", "threads",
    "linkedin", "youtube", "flipboard", "newsbreak", "biztoc",
    # Google News collapses to "google.com" under the public suffix list, so the
    # entry has to be the brand, not the news.google.com hostname it arrives as.
    "google",
})
NON_INDEPENDENT_DOMAINS = frozenset({
    "t.me", "x.com", "twitter.com", "inbox.lv",
})


def is_independent_publisher(domain: str) -> bool:
    """True when this domain can count as an outlet of its own.

    False for carriers (see NON_INDEPENDENT_*) and for anything that does not resolve
    to a registrable domain — an unattributable source cannot corroborate anything.
    """
    reg = registrable_domain(domain)
    if not reg or reg in NON_INDEPENDENT_DOMAINS:
        return False
    return reg.split(".")[0] not in NON_INDEPENDENT_LABELS


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
    table) and carriers (see is_independent_publisher) are excluded from both the
    official check and the independence count; if every source is excluded the
    cluster stays unverified.
    """
    penalized = {registrable_domain(d) for d in (penalized_domains or ())}
    domains = []
    for ev in events:
        reg = registrable_domain(ev.get("source_domain") or ev.get("source_url") or "")
        # Carriers are dropped for the same reason penalized domains are: the count
        # below is "how many newsrooms saw this", and a syndication portal or a
        # crosspost adds a copy, not a witness.
        if reg and reg not in penalized and is_independent_publisher(reg):
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
