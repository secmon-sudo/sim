"""
SIM — Pass A ingest: text filters & dedup primitives (pure, no network/DB)

Noise/keyword filtering, canonicalization, URL/domain helpers and the
title/content similarity machinery used for ingest-time dedup.
Split out of pass_a_ingest.py on 2026-07-16 (was a 1.9K-line monolith).
"""

import difflib
import hashlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import tldextract

from src.core.geo import places_disagree

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
with open(_CONFIG_DIR / "keywords.json", encoding="utf-8") as f:
    KEYWORDS_CONFIG = json.load(f)
with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as f:
    SETTINGS = json.load(f)

_DEDUP = SETTINGS.get("dedup", {})
_TITLE_SIM_THRESHOLD = _DEDUP.get("title_similarity_threshold", 0.78)
_TITLE_TOKEN_THRESHOLD = _DEDUP.get("title_token_jaccard_threshold", 0.72)
_CONTENT_SHINGLE_THRESHOLD = _DEDUP.get("content_shingle_threshold", 0.40)

# Prompt injection patterns to strip before LLM classification
PROMPT_INJECTION_PATTERNS = re.compile(
    r"\[INST\]|<\|system\|>|<\|user\|>|<\|assistant\|>|IGNORE PREVIOUS INSTRUCTIONS|"
    r"FORGET ALL PRIOR|YOU ARE NOW|SYSTEM OVERRIDE",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Noise filters
# ---------------------------------------------------------------------------

def _compile_noise_patterns() -> list[re.Pattern]:
    """Compile noise filters with word boundaries to reduce false positives."""
    patterns = []
    for pattern in KEYWORDS_CONFIG.get("noise_filters", []):
        escaped = re.escape(pattern)
        try:
            patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
        except re.error:
            patterns.append(re.compile(re.escape(pattern), re.IGNORECASE))
    return patterns


NOISE_PATTERNS = _compile_noise_patterns()


# Additional hard-coded title-level sports/entertainment blockers.
# These are compiled once and applied *before* the config-based filters.
_SPORTS_ENT_BLOCKERS = [
    re.compile(r"\btransfer\s+(window|deal|rumor|gossip|news)\b", re.IGNORECASE),
    re.compile(r"\b(hijack|hijacked)\s+(deal|transfer|move|signing)\b", re.IGNORECASE),
    re.compile(r"\b(football|soccer|premier\s+league|la\s+liga|bundesliga|serie\s+a|champions\s+league|fifa|uefa|world\s+cup)\b", re.IGNORECASE),
    re.compile(r"\b(liverpool|tottenham|manchester\s+(united|city)|chelsea|arsenal|barcelona|real\s+madrid|bayern|juventus|ac\s+milan|inter\s+milan|psg|borussia)\b", re.IGNORECASE),
    re.compile(r"\b(match|score|goal|fixture|kick\s*off|half[-\s]time|full[-\s]time)\b", re.IGNORECASE),
    re.compile(r"\b(netflix|disney\+|hulu|amazon\s+prime|streaming|season\s+\d+|episode\s+\d+|doctor\s+who|tv\s+series|tv\s+show|movie\s+review|box\s+office)\b", re.IGNORECASE),
    re.compile(r"\b(celebrity|gossip|rumour|rumor|speculation|insider)\b", re.IGNORECASE),
    re.compile(r"\b(bitcoin|crypto|nft|blockchain|stock\s+market|shares\s+rise|shares\s+fall|ipo|earnings)\b", re.IGNORECASE),
]


# Military/security context patterns — if any of these match, the article
# should NOT be discarded by noise filters even if a noise keyword is present.
# e.g. "military training exercise near border" is real news, not noise.
_MILITARY_CONTEXT_BYPASS = re.compile(
    r"\b(military|army|troops|soldiers|combat|battlefield|frontline|"
    r"war zone|airbase|naval|marines|special forces|regiment|battalion|"
    r"armed forces|defense ministry|ministry of defence|pentagon|"
    r"NATO|Wagner|militia|insurgent|guerrilla|paramilitary|"
    r"airstrike|missile|bombing|shelling|casualties|killed in|"
    r"drone strike|offensive|ceasefire|blockade|siege|ambush)\b",
    re.IGNORECASE,
)


# Bypass cancellers — even with military vocabulary, these markers indicate the
# article is ANALYSIS/RECAP/MEDIA about conflict, not a live incident. They cancel
# the military bypass so the normal noise filters apply (e.g. "documentary about the
# missile strike", "investigation into the bombing", "opinion: why the war drags on").
_BYPASS_CANCEL_PATTERN = re.compile(
    r"\b(documentary|docuseries|investigation into|investigates|"
    r"opinion|op-?ed|editorial|analysis|explainer|explained|"
    r"what we know|here's what|the story of|how the|why the|"
    r"podcast|book review|new book|film about|movie about|"
    r"retrospective|in pictures|in photos|photo essay|timeline of)\b",
    re.IGNORECASE,
)


# Retrospective / anniversary patterns — these indicate an article ABOUT a past
# event (recap, memorial, "N years ago"), not a current incident. They override the
# military-context bypass: "10th anniversary of the airstrike" is stale news, not a
# live event, even though it mentions "airstrike".
_RETROSPECTIVE_PATTERN = re.compile(
    r"\b\d+\s*(?:st|nd|rd|th)?\s*anniversary\b"
    r"|\banniversary of\b"
    r"|\b\d+\s+years?\s+(?:ago|since|on)\b"
    r"|\bon this day\b"
    r"|\byears ago today\b"
    r"|\blooking back\b"
    r"|\bremember(?:ing|ed)?\s+the\b"
    r"|\bthrowback\b"
    r"|\b(?:a\s+)?(?:decade|decades)\s+(?:ago|since)\b"
    r"|\bback in (?:19|20)\d\d\b"
    r"|\bmarks?\s+\d+\s+years\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Content-farm rejection
# ---------------------------------------------------------------------------
#
# Measured 2026-08-17: mshale.com produced 136 of 138 events carrying the signature
# below, and ALL 20 of its alerts. It republishes scraped YouTube and TV titles as
# news, so its "reporting" included fabricated attacks that paged at severity 100 —
# "Iran Missile Strikes Rock Dubai, Abu Dhabi Airports After Khamenei's Death" and
# "One Killed And 63 Injured In Kuwait Airport Drone Attack", neither of which
# happened. For a security product that is the worst possible output, and the volume
# was accelerating (70 events in the final 7 days).
#
# The domain list alone would not be worth writing — the next farm has a different
# name. The signature is what generalises, and it is specific enough to be safe:
# a trailing 9-12 character YouTube-style id in parentheses, or the publisher's own
# name appended as a suffix, combined with an unrelated trending token. Scanned over
# 60 days, no legitimate source in the corpus matches it.
_CONTENT_FARM_DOMAINS = {"mshale.com"}

# "... Some Trending Phrase (dQw4w9WgXcQ) - Mshale" / "... (1HA4RMekh7) - mshale.com"
_SCRAPED_VIDEO_TITLE_RE = re.compile(
    r"\([A-Za-z0-9_\-]{9,12}\)\s*[-–—]\s*[\w.\- ]{2,30}$"
)
# Random-hash article paths: /bd22c76b/dfce4942-Ff1aw6k0wM
_HASH_PATH_RE = re.compile(r"/[0-9a-f]{8}/[0-9a-f]{8}[A-Za-z0-9_\-]*/?$")


def is_content_farm(title: str | None, url: str | None = None,
                    domain: str | None = None) -> bool:
    """True when the item comes from a scraped-content farm rather than a publisher.

    Checked at ingest, before any keyword or severity logic runs: this content must
    never reach the classifier, because its headlines are shaped exactly like real
    breaking-news wire copy and score accordingly.
    """
    if domain and domain.lower().lstrip("www.") in _CONTENT_FARM_DOMAINS:
        return True
    if title and _SCRAPED_VIDEO_TITLE_RE.search(title.strip()):
        return True
    if url and _HASH_PATH_RE.search(url.split("?")[0]):
        return True
    return False


# ---------------------------------------------------------------------------
# Social platforms are carriers, not publishers
# ---------------------------------------------------------------------------
#
# Google News indexes publishers' own social posts, and its <source url=> then names
# the PLATFORM: 127 events over 30 days to 2026-08-19 were filed under "facebook.com".
# The content is genuine — DW News, the New York Times, the Washington Post and
# ABS-CBN posting their own stories — so this is a misattribution problem, not a junk
# problem, and the items are worth keeping.
#
# It is not cosmetic, because source_domain is an IDENTITY that three decisions read:
#   * _record_corroboration() refuses a duplicate whose registrable domain equals the
#     survivor's, so an NYT post and a DW post both filed as facebook.com looked like
#     one outlet republishing itself and their corroboration was never recorded;
#   * the same collapse runs the other way — dw.com plus facebook.com/deutschewellenews
#     counted as TWO independent domains for label_cluster(), inflating a single
#     outlet into "Onaylandı (Çoklu kaynak)";
#   * domain_penalties accrue against the platform as one undifferentiated blob.
#
# The publisher is recoverable: it is the page slug in the post URL. Pages that are
# not in the map keep the platform domain — that is the behaviour they already had,
# so an unrecognised page costs nothing, while a mapped one is repaired.
_SOCIAL_PLATFORM_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com", "t.me", "threads.net",
}

# Facebook page slug → the publisher's real registrable domain. Built from the pages
# actually seen in the corpus; extend it as new ones appear rather than guessing.
# Deliberately NOT populated with commentators and aggregators (officialbenshapiro,
# lonewolfnewsandmedia): they are not outlets, and mapping them would manufacture a
# publisher identity the corroboration count would then trust.
_FACEBOOK_PAGE_PUBLISHERS = {
    "deutschewellenews": "dw.com",
    "nytimes": "nytimes.com",
    "washingtonpost": "washingtonpost.com",
    "manchestereveningnews": "manchestereveningnews.co.uk",
    "theliverpoolecho": "liverpoolecho.co.uk",
    "abscbnnews": "abs-cbn.com",
    "rapplerdotcom": "rappler.com",
    "sunstardavaonews": "sunstar.com.ph",
    "dailyguardianph": "dailyguardian.com.ph",
    "detroitfreepress": "freep.com",
    "bbcsurrey": "bbc.co.uk",
    "addisstandardeng": "addisstandard.com",
}

# facebook.com/<page>/posts/... , /videos/... , /photos/... — the slug is the first
# path segment. Profile-id URLs (/profile.php?id=) carry no slug and stay unmapped.
_FACEBOOK_PAGE_RE = re.compile(r"facebook\.com/([A-Za-z0-9._-]+)/", re.IGNORECASE)


def is_social_platform(domain: str | None) -> bool:
    """True when the domain names a carrier rather than a publisher."""
    if not domain:
        return False
    return domain.strip().lower().removeprefix("www.") in _SOCIAL_PLATFORM_DOMAINS


def social_publisher_domain(url: str | None) -> str | None:
    """The real publisher behind a social post URL, or None if unrecognised."""
    if not url:
        return None
    m = _FACEBOOK_PAGE_RE.search(url)
    if not m:
        return None
    return _FACEBOOK_PAGE_PUBLISHERS.get(m.group(1).lower())


def is_noise(text: str) -> bool:
    """Check if text matches known noise patterns using word boundaries.

    Military/security context overrides noise filters — an article about
    'military training exercise near border' is real news, not simulator noise.
    EXCEPTION: retrospective/anniversary content is always noise (it describes a
    past event, not a current incident) and overrides the military bypass.
    """
    text_lower = text.lower()

    # Retrospectives are stale by definition — filtered even with military context
    if _RETROSPECTIVE_PATTERN.search(text_lower):
        return True

    # Military/security context normally overrides noise filters — but NOT when the
    # article is analysis/recap/media about conflict rather than a live incident.
    if _MILITARY_CONTEXT_BYPASS.search(text_lower) and not _BYPASS_CANCEL_PATTERN.search(text_lower):
        return False

    for pattern in NOISE_PATTERNS:
        if pattern.search(text_lower):
            return True
    for pattern in _SPORTS_ENT_BLOCKERS:
        if pattern.search(text_lower):
            return True
    return False


# Standalone high-signal terms that should ALWAYS match from static feeds,
# even without compound context like "airport attack" or "hotel bombing".
# These are words/phrases that almost always indicate a real security event.
_HIGH_SIGNAL_TERMS = {
    "explosion", "explosions", "bombing", "bombings", "shelling",
    "airstrike", "airstrikes", "air strike", "air strikes",
    "missile", "missiles", "missile strike", "missile attack",
    "gunfire", "gunshots", "shooting",
    "assassination", "assassinated", "massacre", "massacred",
    "invasion", "invaded", "coup", "overthrow", "overthrown",
    "ceasefire", "blockade", "siege", "ambush", "offensive",
    "casualties", "fatalities", "killed", "wounded", "dead",
    # Present tense and gerunds of the verbs above. Headlines default to the
    # present ("Strike KILLS 16"), not the past, so listing only the participle
    # dropped the most canonical event shape this pipeline exists to catch:
    # measured 2026-08-24, "Russian strike on Kyiv kills 16" was rejected here
    # while "...killed 16" passed. Bare "injured"/"injuries" were absent too —
    # config carried only compounds ("multiple injured").
    # NB: bare "kills"/"killing" are deliberately NOT here. They belong to the
    # anchored _CASUALTY_CLAIM_PATTERN below — pass_c's prescreen scores off this
    # same lexicon, and a bare casualty verb defeats the anchoring it relies on to
    # reject "the tax deal kills jobs". Nouns and participles are safe here; verbs
    # are not, and listing a verb in both places double-counts it — "Blast injures
    # 18" outscored "18 injured in blast" until "injures" was taken back out.
    "injured", "injuries",
    "abducts", "abducting", "kidnaps", "kidnapping",
    "assassinates", "assassinating", "invades", "invading", "massacres",
    "artillery", "mortar", "rocket", "rockets",
    "drone attack", "drone strike", "drone strikes",
    "war", "warfare", "conflict", "clashes",
    "evacuated", "evacuation",
    "military operation", "ground offensive",
    "nuclear", "chemical weapon", "biological weapon",
    "terror attack", "terrorist attack", "terrorist",
    "hostage", "hostages", "kidnapped", "abducted",
    "insurgent", "insurgents", "insurgency",
    "militia", "paramilitary",
    "sanctions", "embargo",
    "refugee", "refugees", "displaced",
    "humanitarian crisis", "famine",
    "large-scale attack", "major attack", "massive attack",
    "suicide bomb", "suicide bomber", "car bomb", "truck bomb",
    "IED", "improvised explosive",
    "incursion", "retaliation", "retaliatory",
}


def _compile_security_keyword_pattern() -> re.Pattern:
    """Compile high-signal terms + all config keywords into one word-boundary regex.

    Word boundaries (\\b) prevent substring false positives that plain
    `keyword in text` produced — e.g. "war" matching "Warsaw"/"forward",
    "coup" matching "couple", "riot" matching "patriot", "dead" matching
    "deadline". \\b uses Unicode \\w, so it also works for Arabic/Hebrew/Cyrillic
    keywords (boundaries between word chars and spaces/punctuation).
    """
    terms: set[str] = set(_HIGH_SIGNAL_TERMS)
    for keyword_group in ("emergency_keywords", "geopolitical_keywords"):
        for keywords in KEYWORDS_CONFIG.get(keyword_group, {}).values():
            terms.update(kw.lower() for kw in keywords)

    parts = []
    for term in terms:
        term = term.strip()
        if not term:
            continue
        try:
            re.compile(rf"\b{re.escape(term)}\b")
            parts.append(rf"\b{re.escape(term)}\b")
        except re.error:
            parts.append(re.escape(term))
    return re.compile("|".join(parts), re.IGNORECASE)


_SECURITY_KEYWORD_PATTERN = _compile_security_keyword_pattern()

# Flight disruption resists the flat keyword list. Headlines phrase it a dozen
# ways — "cancel Kuwait flights", "Airport temporarily suspends operations",
# "airlines suspending flights" — and every fixed phrase added to cover one of
# them either misses the next variant or, if shortened to "suspends operations",
# admits every mine, factory and telco that halts business (measured: 6 of 9
# non-security control headlines passed on such phrases).
#
# A conjunction is the honest rule: an aviation noun AND a disruption verb in
# the same text. Weather and maintenance cases satisfy it too, but they are
# already caught downstream by is_noise(), which runs on every ingested item —
# so the two filters together express "aviation stopped flying, and not because
# of weather", which is exactly the security scope of the SITREPs.
_AVIATION_CONTEXT_PATTERN = re.compile(
    r"\b(airport|airports|airline|airlines|airspace|flight|flights|"
    r"carrier|carriers|aviation|terminal)\b",
    re.IGNORECASE,
)
_DISRUPTION_PATTERN = re.compile(
    r"\b(suspend|suspends|suspended|suspending|suspension|suspensions|"
    r"halt|halts|halted|cancel|cancels|cancelled|canceled|cancelling|"
    r"canceling|cancellation|cancellations|grounded|reroute|reroutes|"
    r"rerouted|closure|closures|disruption|disruptions)\b",
    re.IGNORECASE,
)


# The vocabulary above is noun-shaped: it has "disruption" but not "disrupted",
# and no "delayed", "diverted" or "stranded" at all — so "Flights diverted after
# Manchester Airport security breach" was not an aviation disruption to this
# pipeline. Measured 2026-08-23 over 7 days: 660 events carry an aviation noun, the
# strict gate above claims 120, and these verbs are in another 77.
#
# They cannot simply join the list. Read over a whole article those 77 are 45%
# junk: a war roundup mentions an airport in one paragraph and a delay in another,
# and co-occurrence in 4,000 characters means nothing. Two conditions fix that:
#
#   * the verb and the aviation noun must share the HEADLINE, which is short
#     enough that co-occurrence implies they are about each other, and
#   * the article must carry a security nexus somewhere — this pipeline is not
#     interested in a technical snag, a snowstorm, or the bearded vulture that
#     delayed a flight at Heraklion.
#
# So gated, the same window yields 19 additions and every one of them is a real
# security-caused disruption: the Manchester airfield breach (5 filings), the
# Houston Hobby bomb threat (3), Moscow's airports closing under drone attack (4),
# Moldovan airspace closed by a cruise missile, an unauthorised aircraft at Fort
# Lauderdale. Precision 19/19 against 21/38 for the ungated form.
_AVIATION_HEADLINE_PATTERN = re.compile(
    r"\b(airport|airports|airline|airlines|airspace|flight|flights|"
    r"carrier|carriers|aviation|terminal|aircraft|runway|airfield|tarmac)\b",
    re.IGNORECASE,
)
_WEAK_DISRUPTION_PATTERN = re.compile(
    r"\b(disrupt|disrupts|disrupted|disrupting|delay|delays|delayed|"
    r"divert|diverts|diverted|diversion|diversions|stranded|grounding|"
    r"shutdown|closed)\b",
    re.IGNORECASE,
)
_SECURITY_NEXUS_PATTERN = re.compile(
    r"\b(bomb|bombs|explosive|explosives|drone|drones|uav|missile|missiles|"
    r"rocket|rockets|attack|attacks|attacked|shooting|gunman|gunmen|hijack|"
    r"hijacked|hijacking|breach|breached|incursion|intrusion|unauthorized|"
    r"unauthorised|security|evacuated|evacuation|terror|terrorist|militant|"
    r"militants|shelling|sabotage|threat|threats)\b",
    re.IGNORECASE,
)


def _is_flight_disruption(text: str, title: str | None = None) -> bool:
    """Aviation noun + disruption verb in the same text (see block comment).

    `title` opens the second path: a weak disruption verb counts when it shares the
    headline with an aviation noun AND the article carries a security nexus. Absent
    a title only the strict path runs, so a caller that has no headline to offer
    keeps exactly the old behaviour.
    """
    if _AVIATION_CONTEXT_PATTERN.search(text) and _DISRUPTION_PATTERN.search(text):
        return True
    if not title:
        return False
    return bool(_AVIATION_HEADLINE_PATTERN.search(title)
                and _WEAK_DISRUPTION_PATTERN.search(title)
                and _SECURITY_NEXUS_PATTERN.search(text))


# A prohibited item carried THROUGH passenger screening — ammunition in a bag, a
# pistol at the checkpoint, a live round found on board. Aviation security in the
# most literal sense, and until 2026-08-27 the pipeline could not see it at all.
#
# Measured that day on the headline that prompted this ("Businessman flies to Delhi
# with 31 live rounds after passing through Dhaka airport security"): priority_score
# 0, deterministic_relevance score 0 with has_security False, and no row in the
# database at all. Three gates, three zeroes. Of 1485 configured keyword terms only
# seven touch weapons — "ammunition depot", "weapons cache", "gun battle", "gun
# attack", "knife attack", "weapon", "airport knife" — and not one of them describes
# an item getting past a checkpoint. None of the four aviation news queries reaches
# it either; they ask about attacks and about flight disruption.
#
# The flat lexicon is the wrong instrument here for the reason the disruption block
# above gives: the class is written as a verb ("flies with", "passing through",
# "found aboard"), and every fixed phrase that covers one variant misses the next.
# A conjunction is the honest rule — a prohibited item AND an aviation noun AND a
# screening-or-carriage word, all three sharing the HEADLINE, which is short enough
# that co-occurrence implies the words are about each other.
#
# Measured over 15,429 unique headlines (14 days, every status):
#
#   item + aviation                    46 matches, 36 of them off-class
#   item + aviation + screening        16 matches, 15 on-class + 1 explainer
#
# The 15 are the Varanasi checkpoint discharge, the Denver live-round-on-board
# evacuation, and the Hyderabad baggage bomb scare — 6 of which the prescreen had
# archived unread, including "Live bullet found aboard United flight". The one
# non-incident is "What are the rules for carrying firearms on flights?", admitted
# by the carriage verbs and archived one LLM call later by the report_kind gate.
#
# Two deliberate exclusions, both measured: "blade" is out because engine fan blades
# collide with it ("Passenger Partially Sucked Out of Plane After Fan Blade Shatters
# Window"), and "found" is out because it drags in ten Leipzig drone-explosive
# filings — real security events, but a different class that is already covered, and
# admitting them buys exactly one on-class headline whose incident three sibling
# filings already carry.
_PROHIBITED_ITEM_PATTERN = re.compile(
    r"\b(live rounds?|live ammunition|ammunition|ammo|bullets?|cartridges?|"
    r"firearms?|handguns?|pistols?|revolvers?|guns?|grenades?|explosives?|"
    r"detonators?|knives|knife|weapons?)\b",
    re.IGNORECASE,
)
_SCREENING_CONTEXT_PATTERN = re.compile(
    r"\b(security|screening|screened|screeners?|checkpoint|scanner|scanners|"
    r"x-ray|luggage|baggage|carry-on|suitcase|boarding|board|aboard|onboard|"
    r"deplane|cabin|passengers?|check|checks|checked|carry|carrying|carried|"
    r"carries|confiscat\w*|seiz\w*|smuggl\w*)\b",
    re.IGNORECASE,
)


# The aviation-security classes that are neither an attack nor a flight
# disruption: a bomb threat called in, a runway incursion, a drone over the
# approach path, GNSS jamming, a laser in the cockpit, a stowaway in the wheel
# well. Each is an incident this pipeline exists to report, and none of them
# carries a word the flat lexicon recognises.
#
# Measured 2026-08-27 over 15,429 unique headlines (14 days, every status): an
# aviation noun plus one of these class terms matches 49 headlines and every one
# of them is on-class — no junk to trade away. Ten were sitting in the archive,
# prescreen-archived at score 0 without an LLM ever reading them:
#
#   "African stowaway found frozen to death in plane's wheel compartment at Gatwick"
#   "Sydney Airport faces safety scrutiny after third ground near-miss in three weeks"
#   "Qantas Jets Avoids Collision at Sydney Airport Marks 4th Near Miss Incident"
#   "Police Didn't Notify Public Of G7 Bomb Scare At Calgary Airport"
#
# The Sydney runway-incursion series is the clearest symptom: some filings of the
# same investigation were scored and others archived, on nothing but wording.
#
# The other 39 already passed, but every one of them scored priority 1 — the
# median inserted item, which is to say the coin-flip line. So the class was not
# only being archived, it was permanently first in the queue to be dropped
# whenever the insert budget tightened, which it always does.
_AVIATION_INCIDENT_PATTERN = re.compile(
    r"\b(lasers?|lasered|laser strike|laser attack|"
    r"gnss|gps jamming|gps spoofing|jamming|spoofing|"
    r"drone sighting|drones? spotted|drones? sighted|uav sighting|"
    r"cockpit|stowaways?|runway incursion|near miss|near-miss|"
    r"bomb threat|bomb scare|bomb hoax|hoax call)\b",
    re.IGNORECASE,
)


def _is_aviation_security_incident(title: str) -> bool:
    """Aviation noun + an aviation-security class term, both in the headline."""
    if not title:
        return False
    return bool(_AVIATION_HEADLINE_PATTERN.search(title)
                and _AVIATION_INCIDENT_PATTERN.search(title))


def _is_screening_breach(title: str) -> bool:
    """Prohibited item + aviation noun + screening/carriage word, all in the headline.

    Headline-only by design (see block comment): over a whole article the three
    vocabularies co-occur for reasons that have nothing to do with each other.
    """
    if not title:
        return False
    return bool(_PROHIBITED_ITEM_PATTERN.search(title)
                and _AVIATION_HEADLINE_PATTERN.search(title)
                and _SCREENING_CONTEXT_PATTERN.search(title))


# Counts, shared by the ingest gate below and the priority scorer further down.
_CASUALTY_NUM = (r"(?:\d{1,4}|dozens?|scores|one|two|three|four|five|six|seven|"
                 r"eight|nine|ten|eleven|twelve)")

# A casualty VERB earns entry only when something anchors it, for the reason
# pass_c's HOSTILE_ACT_PATTERN gives: bare "kills" is where the metaphors live
# ("the deal kills jobs", "United kills off the comeback"). The three anchors are
# an armed subject, an explicitly human object, or a stated count — none of which
# a metaphor carries. Added 2026-08-24, when the flat lexicon was found to hold
# "killed" but not "kills", and so dropped "Strike kills 16" outright.
_CASUALTY_CLAIM_PATTERN = re.compile(
    r"\b(?:drones?|missiles?|rockets?|uavs?|strikes?|airstrikes?|shelling|"
    r"bombardment|blasts?|explosions?|troops|forces|militants?|gunmen|rebels?|"
    r"insurgents?|terrorists?|jets?|warplanes?|raids?|attacks?|gangs?|police|"
    r"soldiers|army|militia)"
    r"['\u2019\"]?\s+(?:[\w'\u2019-]+\s+){0,3}"
    r"(?:kills?|killing|injur(?:e|es|ing)|wounds?|wounding)\b"
    r"|\b(?:kills?|killing|injur(?:e|es|ing)|wounds?|wounding)\s+"
    r"(?:[\w'\u2019-]+\s+){0,2}"
    r"(?:civilians?|people|residents?|children|child|women|worshippers|passengers|"
    r"pilgrims|troops|soldiers|officers)\b"
    r"|\b(?:kills?|killing|injur(?:e|es|ing)|wounds?|wounding)\s+"
    r"(?:at\s+least\s+|more\s+than\s+|nearly\s+|up\s+to\s+|some\s+)?"
    + _CASUALTY_NUM + r"\b",
    re.IGNORECASE,
)


def _matches_security_keywords(title: str, description: str) -> bool:
    """Check if article title/description contains at least one security keyword.

    Used as a post-filter for general RSS feeds (reddit, aljazeera, reuters)
    that aren't pre-filtered by search query. Matches on word boundaries to
    avoid substring false positives. Covers high-signal standalone terms plus
    config emergency/geopolitical keywords across all languages (en, ar, tr, fr),
    plus the aviation-disruption and screening-breach conjunctions.
    """
    text = f"{title} {description}"
    return (bool(_SECURITY_KEYWORD_PATTERN.search(text))
            or bool(_CASUALTY_CLAIM_PATTERN.search(text))
            or _is_flight_disruption(text, title)
            # Same reason the disruption conjunction is here: this class carries no
            # standalone security keyword, so on a general feed it was rejected before
            # the priority scorer ever ranked it. Measured 2026-08-27, the Dhaka
            # live-rounds headline returned False from this function — the earliest of
            # the four zeroes, and the one that leaves no database row to notice.
            or _is_screening_breach(title)
            # Same reason again: measured 2026-08-27, "ATSB Probes Third Runway Near
            # Miss at Sydney Airport" and "African stowaway found frozen to death in
            # plane's wheel compartment at Gatwick Airport" both scored 0 here.
            or _is_aviation_security_incident(title))


# ---------------------------------------------------------------------------
# Ingest priority scoring
# ---------------------------------------------------------------------------
# The per-run insert budget and per-domain caps are CONTENT-BLIND: without a
# ranking, whichever items happen to sit at the top of a feed claim a domain's
# slots, and a capped-out high-severity item loses to a routine one. This cheap
# (no LLM, no network) scorer orders candidates so budget cuts fall on the
# least valuable items. It is a triage heuristic, NOT a severity score — Pass D
# owns real scoring; nothing downstream may read _priority.

# Terms that almost always indicate a major, actionable incident. Multi-language
# (en/tr/ar) because scoring runs BEFORE translation.
_CRITICAL_PRIORITY_TERMS = [
    # mass-casualty / mass-violence
    "mass casualty", "mass shooting", "massacre", "dozens killed", "scores killed",
    "suicide bombing", "suicide bomber", "car bomb", "truck bomb",
    # strikes & strategic weapons
    "airstrike", "air strike", "missile strike", "missile attack", "drone strike",
    "ballistic missile", "cruise missile", "shot down", "intercepted",
    # WMD / CBRN
    "nuclear", "chemical attack", "chemical weapon", "nerve agent", "radiological",
    "dirty bomb",
    # state-level ruptures
    "invasion", "coup", "assassination", "assassinated", "declaration of war",
    "martial law", "state of emergency",
    # critical infrastructure & transport hubs
    "airport attack", "airport explosion", "aircraft shot down", "plane shot down",
    "refinery", "pipeline", "power plant", "power grid", "desalination",
    "port attack", "tanker attack", "warship",
    # captivity
    "hostage", "hijack",
    # tr
    "hava saldırısı", "füze saldırısı", "intihar saldırısı", "katliam",
    "çok sayıda ölü", "suikast", "darbe", "işgal", "rehine", "nükleer",
    # ar
    "غارة جوية", "ضربة صاروخية", "تفجير انتحاري", "مجزرة", "اغتيال",
    "انقلاب", "غزو", "رهينة", "نووي",
]

_CRITICAL_PRIORITY_PATTERN = re.compile(
    "|".join(rf"\b{re.escape(t)}\b" for t in _CRITICAL_PRIORITY_TERMS),
    re.IGNORECASE,
)

# "N killed/dead/wounded…" — a concrete casualty count is the strongest cheap
# signal that an article reports a real, current incident.
# The count may be a digit or a word, and the clause runs in either order:
# "14 killed in the barrage" and "barrage kills 14" report the same event. Only
# the first was matched until 2026-08-24, so identical events scored 6 or 0 on
# word order alone — and because the per-run insert budget always binds, a 0
# here is not a lower rank but an item dropped.
# Hebrew was the one feed language with no casualty vocabulary here, so its
# incident reports could never earn the casualty bonus and sat permanently below
# their English equivalents in the budget ranking — visible the moment the
# English side was strengthened on 2026-08-24.
_CASUALTY_PARTICIPLE = (r"(?:killed|dead|deaths|injured|wounded|casualties|fatalities|"
                        r"ölü|yaralı|قتيل|قتلى|جريح|"
                        r"הרוגים|הרוג|נהרגו|נהרג|פצועים|פצוע|נפצעו|נפצע|חללים)")
_CASUALTY_VERB = r"(?:kills?|killing|injures?|injuring|wounds?|wounding)"

_CASUALTY_COUNT_PATTERN = re.compile(
    rf"\b(?P<n1>{_CASUALTY_NUM})\s+(?:people\s+|civilians\s+|soldiers\s+)?"
    rf"{_CASUALTY_PARTICIPLE}\b"
    rf"|\b{_CASUALTY_VERB}\s+"
    rf"(?:at\s+least\s+|more\s+than\s+|nearly\s+|up\s+to\s+|some\s+)?"
    rf"(?P<n2>{_CASUALTY_NUM})\b",
    re.IGNORECASE,
)

# Word-form counts, so the ">= 10" escalation is not silently forfeited whenever
# a headline spells the number out. "dozens"/"scores" are read at their
# conventional floor, which is all the >= 10 test needs.
_CASUALTY_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "dozen": 12, "dozens": 24, "scores": 40,
}


def _casualty_count(match: re.Match) -> int:
    """Numeric value of whichever count group matched, 0 if unreadable."""
    raw = (match.group("n1") or match.group("n2") or "").lower()
    if raw.isdigit():
        return int(raw)
    return _CASUALTY_NUM_WORDS.get(raw, 0)

_BREAKING_PATTERN = re.compile(
    r"\b(breaking|urgent|just in|son dakika|عاجل)\b", re.IGNORECASE
)


def priority_score(title: str, description: str) -> int:
    """Cheap ingest-triage priority for one candidate item. Higher = insert first.

    Components (all word-boundary regex, multi-language):
      +4 per distinct critical-term hit (capped at 3 hits)
      +1 per security-keyword hit (capped at 5)
      +3 if a concrete casualty count is stated (+2 more if >= 10),
         or +2 for an anchored casualty claim that states no number
      +1 for breaking/urgent markers
      +4 if the headline describes a prohibited item through aviation screening
    """
    text = f"{title} {description}"
    score = 0

    critical_hits = {m.group(0).lower() for m in _CRITICAL_PRIORITY_PATTERN.finditer(text)}
    score += 4 * min(len(critical_hits), 3)

    keyword_hits = {m.group(0).lower() for m in _SECURITY_KEYWORD_PATTERN.finditer(text)}
    claim = _CASUALTY_CLAIM_PATTERN.search(text)
    if claim:
        # Stands in for the keyword hit the flat lexicon cannot safely carry:
        # "killed" is a listed term, a bare "kills" can never be one. Without this
        # the same event scored a point lower for being written in the present
        # tense, which is the tense wire copy actually uses.
        keyword_hits.add("__casualty_claim__")
    score += min(len(keyword_hits), 5)

    casualty = _CASUALTY_COUNT_PATTERN.search(text)
    if casualty:
        score += 3
        if _casualty_count(casualty) >= 10:
            score += 2
    elif claim:
        # An anchored casualty claim that states no number — "strike kills child",
        # "gunmen kill worshippers" — still reports an incident. Without this it
        # cleared the gate and then scored 0, which under a permanently binding
        # insert budget means dropped: admitted in name only.
        score += 2

    if _BREAKING_PATTERN.search(text):
        score += 1

    if _is_screening_breach(title):
        # Weighted like a critical-term hit rather than a keyword, because a keyword's
        # +1 would not change this class's fate. Measured 2026-08-27: it scores 0 on
        # every other component here, and the insert budget always binds — run #1740
        # fetched 1629 candidates and inserted 100, where the median inserted item
        # scored 1 and the highest DROPPED item scored 3. At +1 the class sits on the
        # coin-flip line; the whole point of the gate is that it stops being dropped.
        # Title only: see _is_screening_breach on why the conjunction needs a headline.
        score += 4

    if _is_aviation_security_incident(title):
        # Weighted the same and for the same measured reason: of the 49 headlines this
        # matches, the 39 that already passed all scored exactly 1 — the median inserted
        # item. A class that is always on the coin-flip line is a class that disappears
        # the moment the budget tightens, and the budget is never slack.
        score += 4

    return score


# ---------------------------------------------------------------------------
# Domain / URL helpers
# ---------------------------------------------------------------------------

def extract_domain(url: str) -> str:
    """Extract eTLD+1 domain from URL."""
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def compute_url_hash(url: str) -> str:
    """SHA-256 hash of normalized URL for deduplication."""
    normalized = url.strip().lower()
    # Strip query params and fragments — for Google News redirect URLs the
    # article ID is in the path and params are tracking; other sources likewise.
    normalized = normalized.split("?")[0].split("#")[0]
    return hashlib.sha256(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def canonicalize_text(raw_text: str) -> str:
    """Clean and normalize raw article text."""
    # Strip HTML tags — require tag to start with letter or '/' to avoid
    # false positives on math expressions like "3 < 5 > 2"
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", raw_text)
    text = PROMPT_INJECTION_PATTERNS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# The dedup loop asks the same questions of the same stored strings over and over:
# every candidate is compared against the whole recent-events window, so one run
# normalizes the same ~2,000 stored titles and bodies ~1,000 times each. These are
# pure functions of the string, so caching them changes nothing but the bill —
# profiled 2026-08-25, normalization was 8 s and shingle-building 4 s of the 64 s
# spent in find_content_duplicate. Same reasoning as _place_keys_cached in core.geo.
@lru_cache(maxsize=16384)
def normalize_title(title: str) -> str:
    """Normalize title for deduplication comparison."""
    text = title.lower()
    # Strip trailing source attribution BEFORE removing punctuation
    # Heuristic: if the part after the last dash/pipe is short, it's likely a source name
    for sep in (" - ", " | ", " — ", " – "):
        if sep in text:
            parts = text.rsplit(sep, 1)
            if len(parts) == 2 and len(parts[1].strip()) <= 45:
                text = parts[0].strip()
                break
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_similarity(title_a: str, title_b: str, min_ratio: float = 0.0) -> float:
    """Compute similarity between two normalized titles.

    `min_ratio` is a caller's accept threshold, and it turns on difflib's own cheap
    upper bounds: real_quick_ratio() (length only) and quick_ratio() (multiset of
    characters) are both documented to be >= ratio(), so a bound below the threshold
    proves ratio() is below it too and the O(n*m) matcher can be skipped. Profiled
    2026-08-25 on 200 real candidates against 600 stored events: SequenceMatcher was
    50 s of the 71 s spent in find_content_duplicate, and almost all of those pairs
    are unrelated headlines that no threshold would ever accept.

    THE SHORT-CIRCUIT RETURN IS AN UPPER BOUND, NOT THE RATIO. It is only meaningful
    as "below min_ratio" — pass min_ratio only when comparing against it, and leave
    it at 0.0 (every bound passes) when the number itself is the answer.
    """
    norm_a = normalize_title(title_a)
    norm_b = normalize_title(title_b)
    if not norm_a or not norm_b:
        return 0.0
    matcher = difflib.SequenceMatcher(None, norm_a, norm_b)
    if min_ratio > 0.0:
        bound = matcher.real_quick_ratio()
        if bound < min_ratio:
            return bound
        bound = matcher.quick_ratio()
        if bound < min_ratio:
            return bound
    return matcher.ratio()


@lru_cache(maxsize=16384)
def _word_set_cached(text: str) -> frozenset[str]:
    return frozenset(normalize_title(text).split())


def _word_set(text: str) -> set[str]:
    """Normalized word set of a title (lowercased, punctuation-stripped)."""
    return set(_word_set_cached(text))


@lru_cache(maxsize=8192)
def _shingles_cached(text: str, n: int = 4) -> frozenset[str]:
    words = normalize_title(text).split()
    if len(words) < n:
        return frozenset(words)
    return frozenset(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


def _shingles(text: str, n: int = 4) -> set[str]:
    """Word n-grams (shingles) of canonical text — robust to reordering/truncation."""
    return set(_shingles_cached(text, n))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def title_token_similarity(title_a: str, title_b: str) -> float:
    """Word-set Jaccard of two titles.

    Catches cross-source rephrasing that SequenceMatcher misses (reordered words,
    different source suffixes, inserted words) — e.g. two outlets covering the same
    incident with differently worded headlines.
    """
    return _jaccard(_word_set(title_a), _word_set(title_b))


def find_content_duplicate(recent_events: list[tuple], title: str,
                           canonical_text: str) -> int | None:
    """
    Return the index of the first similar article in recent_events, else None.

    Three complementary signals (any one triggers a dedup):
      1. Title SequenceMatcher  — near-identical headlines (incl. source suffix).
      2. Title word-set Jaccard — cross-source rephrasing / reordered headlines
         that SequenceMatcher's char-ratio misses.
      3. Content word-shingle Jaccard — same body reported by different outlets;
         replaces the old O(N*M) full-text SequenceMatcher (faster, truncation-robust).

    Returning the INDEX (not a bool) lets Pass A credit the surviving event with
    the duplicate's source as corroboration instead of discarding the signal.

    `recent_events` entries are (title, canonical_text) or, preferred,
    (title, canonical_text, place_hint) — the stored event's anchor, which names
    the place its headline often leaves out ("...strike on Ukrainian town",
    anchor "Pechenihy"). The incoming item has no anchor: Pass C has not seen it
    yet, so only the stored side can contribute one.
    """
    # Read the cached frozensets directly rather than the set-copying wrappers: this
    # runs once per stored event per candidate, and copying a body's several-hundred
    # shingles each time would hand back exactly what the cache saves. Nothing here
    # mutates them.
    title_tokens = _word_set_cached(title)
    text_shingles = _shingles_cached(canonical_text) if len(canonical_text) > 100 else None

    for idx, entry in enumerate(recent_events):
        existing_title, existing_text = entry[0], entry[1]
        existing_place = entry[2] if len(entry) > 2 else ""
        # Veto: two headlines that name DIFFERENT known places are not the same
        # incident, whatever the letters say. Wire headlines about one war share a
        # scaffolding ("Russian missile strike on X kills N") that the char-ratio
        # matcher scores on its own: measured 2026-08-20, "Russian missile strike on
        # Kharkiv region kills ten" scored 0.667 against "Russian Missile Attack on
        # Kyiv Kills 12" — over the configured 0.65 — so the Kharkiv massacre was
        # dropped as a duplicate AND filed as corroborating evidence for the Kyiv
        # strike, which is how a Kharkiv link ended up cited under a Kyiv cluster in
        # that day's SITREP. Only mutual, disjoint place claims veto (see
        # places_disagree); one side naming no city stays a duplicate candidate.
        if places_disagree(title, f"{existing_title} {existing_place}".strip()):
            continue

        # Signal 1: char-ratio title similarity (primary)
        if title_similarity(title, existing_title, _TITLE_SIM_THRESHOLD) >= _TITLE_SIM_THRESHOLD:
            return idx

        # Signal 2: token-set title similarity (cross-source rephrasing)
        if _jaccard(title_tokens, _word_set_cached(existing_title)) >= _TITLE_TOKEN_THRESHOLD:
            return idx

        # Signal 3: content shingle similarity for longer texts
        if text_shingles is not None and len(existing_text) > 100:
            if _jaccard(text_shingles, _shingles_cached(existing_text)) >= _CONTENT_SHINGLE_THRESHOLD:
                return idx
    return None


def check_content_duplicate(recent_events: list[tuple[str, str]], title: str, canonical_text: str) -> bool:
    """Boolean wrapper around find_content_duplicate (historical API)."""
    return find_content_duplicate(recent_events, title, canonical_text) is not None

