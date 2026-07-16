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

# Government / military / intergovernmental TLD suffixes. Matched against the
# end of the registrable host, so "centcom.mil" and "mod.gov.ua" both hit.
OFFICIAL_TLD_SUFFIXES = (".gov", ".mil", ".int")

# Suffix match also covers subdomains and country variants (e.g. "travel.state.gov",
# "gov.uk", "gov.il"). State news agencies are treated as official in v1 — for the
# SITREP use case "the state said it happened" is the confirmation signal, even
# when the state is a party to the conflict.
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
    "petra.gov.jo",
    # state news agencies
    "irna.ir",
    "mehrnews.com",
    "tasnimnews.com",
    "farsnews.ir",
    "iribnews.ir",
    "aa.com.tr",
    "sana.sy",
    "tass.com",
    "kuna.net.kw",
    "wam.ae",
    "spa.gov.sa",
    "bna.bh",
    "ina.iq",
    "saba.ye",
)

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


def is_official_domain(domain: str) -> bool:
    """True if the domain (or its registrable parent) is an official/state source."""
    reg = registrable_domain(domain)
    if not reg:
        return False
    if reg.endswith(OFFICIAL_TLD_SUFFIXES):
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

    if any(is_official_domain(d) for d in domains):
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
