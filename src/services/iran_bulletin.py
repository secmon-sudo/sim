"""SIM — Iran theatre bulletin: who struck whom, and on whose word.

A second report alongside the SITREP, for the duration of the Iran war. The SITREP
is organised by country; this one is organised by DIRECTION, because that is the
question the theatre actually poses:

    1. strikes ON Iranian soil
    2. strikes FROM Iran on its neighbours
    3. regional and strategic moves (airspace, shipping, diplomacy)

Three measurements shaped this module, all taken 3 Sep 2026 over the live corpus.

**Direction is derivable, but not from what SIM already stores.** Pass C's schema
carries event_type, anchor_name, country_iso, occurred_at, casualties, report_kind
— where something happened, never who did it. So the actor has to be extracted.
The information IS there to extract: 411 of 474 theatre headlines (87%) name at
least one actor by name. And country_iso already supplies the other half of the
pair, because it records where the event landed — Jordan for a strike on a Jordanian
base, IQ for Erbil, IR for a strike on Iran.

**But naming an actor is not the same as knowing the direction.** "Iran fires
missiles in response to US strikes" names both sides; "Military denies Iran's
claims that it struck a US base in Jordan" names both AND negates. Subject-verb-
object is what separates them, which is why this runs an LLM rather than a regex.

**The hard field is not direction, it is standing.** report_kind already removes
the wrong KIND of article for free — of 474 theatre events, 38 were commentary,
25 followup and 8 roundup. Inside the 403 that remain, 59 (14.6%) carry claim
language and 7 are outright denials. Those are not defects to filter out: a
one-sided claim is a real event with a real provenance, and the report this module
reproduces states exactly that in its "Kaynak ve Durum" line. So claim standing is
extracted as a FIELD, not used as a veto.

Deliberately NOT reusing Pass C for this. Its prompt is the most fragile surface in
the pipeline — batch size is bounded by TPM, the JSON truncates under pressure, and
every model migration has to be re-proved against it. Extraction here is a separate,
bulk-router call over a few dozen events, so a bad day for the bulletin cannot cost
the pipeline a classification.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core import counters
from src.core.llm_client import call_llm, log_llm_telemetry
from src.core.llm_router import LLMRouter

logger = logging.getLogger(__name__)

# The theatre. Iran plus every country that has taken or hosted a strike in this
# war — measured from the corpus rather than drawn on a map, so a country that
# stops appearing simply contributes nothing instead of forcing a code change.
THEATRE_ISO = ("IR", "IQ", "KW", "JO", "SA", "AE", "QA", "BH", "OM", "IL", "LB", "YE")

# Turkish names for the theatre, as a constant. get_country_name() would need a
# database round trip per event — 182 of them in the first real run — and returns
# whatever anchor_master holds, which is English. The set is fixed and small, so
# the table is both cheaper and correct for a Turkish report.
THEATRE_NAMES = {
    "IR": "İran", "IQ": "Irak", "KW": "Kuveyt", "JO": "Ürdün",
    "SA": "Suudi Arabistan", "AE": "BAE", "QA": "Katar", "BH": "Bahreyn",
    "OM": "Umman", "IL": "İsrail", "LB": "Lübnan", "YE": "Yemen",
}

# Which side an actor belongs to. The bulletin's sections need a SIDE, not a name:
# "IRGC", "Revolutionary Guards" and "Iran" all place an event in section 2.
IRAN_SIDE = "iran"
US_SIDE = "us_coalition"
OTHER_SIDE = "other"
UNATTRIBUTED = "unattributed"

# How well the actor attribution stands up. Straight from the source report's own
# "Kaynak ve Durum" field, which is the honest way to carry a one-sided claim.
STANDING_CONFIRMED = "confirmed"      # both sides or an independent party say so
STANDING_CLAIMED = "claimed"          # one belligerent asserts it, unconfirmed
STANDING_DENIED = "denied"            # asserted and explicitly denied
STANDING_UNKNOWN = "unknown"

# Whether the headline is about the war at all. The theatre is defined by COUNTRY,
# so everything that happens in twelve countries lands in the fetch — and on
# 2026-09-04 the bulletin duly reported a Nazareth homicide, two sisters killed in
# Sharjah, an Iranian professors' pay dispute and an IndiGo flight that diverted
# because its pilot fell ill. None of those is a defect in the pipeline; they are
# real events, correctly classified, and they belong in the country SITREP. They
# are simply not this report's subject.
#
# Asked as a FIELD rather than filtered by event_type, because the type does not
# separate them: "Two Palestinians killed by IDF fire" and "Three men shot dead in
# Nazareth" are both civilian_casualties in Israel, and only one of them is the war.
WAR_RELATED = "war_related"

# Where an Iran-side actor is at HOME rather than reaching across a border.
#
# Section 2 is "İran'dan komşu ülkelere" — an attack that CROSSES into another
# country. The Houthis are Iran-side by the actor table and they are also one
# belligerent in Yemen's own civil war, so "Houthi shelling of displaced people in
# Yemen's Marib" was arriving in section 2 as an Iranian strike on a neighbour.
# Measured across the 6-9 Sep bulletins: 4-5 rows a day, every day, plus Hezbollah
# inside Lebanon.
#
# Keyed on the TARGET country and never on the filing country, because the filing
# country does not separate the two cases: the same window filed BOTH the Marib
# shelling AND "Houthi strikes injure 73 in Saudi Arabia" under YE. The first is
# the Yemeni war, the second is section-2 material, and only the target tells them
# apart.
#
# IQ was measured and deliberately left out. Iran does strike Iraq — the Kurdish
# opposition in the north, the Surdash camp, US positions near Erbil — so an
# Iran-side actor with target_country=IQ is exactly what section 2 is for.
IRAN_SIDE_HOME_ISO = frozenset({"YE", "LB"})

UNKNOWN_COUNTRY = "unknown"

SECTION_ON_IRAN = "on_iran"
SECTION_FROM_IRAN = "from_iran"
SECTION_REGIONAL = "regional"

# Only a bare two-letter code counts as an answer to target_country.
_ISO2_RE = re.compile(r"[A-Z]{2}")

_EXTRACTION_SYSTEM_PROMPT = (
    "You read security news headlines and report WHO ACTED, not what you believe "
    "happened. You never infer an actor that the text does not name, and you never "
    "upgrade a one-sided claim into a fact. Answer with JSON only."
)


# Accidental occurrences: an engine failure, a bird strike, a diversion for a sick
# pilot. The country SITREP has excluded these from its narrative since day one
# (sitrep_generator.SAFETY_ONLY_EVENT_TYPES) because the report is about hostile
# acts; this one is about a war, so the argument is only stronger. Imported rather
# than copied — sitrep_generator already keeps the canonical list, and pass_d_score
# keeps the other; a third copy is how the two in source_credibility.py started.
#
# It is a cheap cut and it is made BEFORE the model sees the batch, so an IndiGo
# diversion in Muscat does not also cost a direction extraction.
try:  # pragma: no cover - exercised via the real module in production
    from src.services.sitrep_generator import SAFETY_ONLY_EVENT_TYPES
except Exception:  # a missing optional dep must not take the bulletin down
    SAFETY_ONLY_EVENT_TYPES = {
        "bird_strike", "engine_failure", "emergency_landing", "depressurization",
        "fire_on_board", "unruly_passenger", "runway_incursion",
    }


def fetch_theatre_events(db_conn, window_start: datetime,
                         window_end: datetime) -> List[Dict[str, Any]]:
    """Theatre events that are reports of a new incident, newest first.

    report_kind does the first cut and it does it for free: commentary, followup
    and roundup articles are 71 of every 474 theatre events, and none of them is a
    strike. Events classified before report_kind existed (11 Aug 2026) carry NULL,
    and those are kept — excluding them would silently shorten the window rather
    than filter it.

    Safety-only event types are the second cut, and a deterministic one.
    """
    rows = db_conn.execute(
        """SELECT id, source_title, source_domain, source_url, country_iso,
                  severity_score, occurred_at_est, time_certainty, event_type,
                  storyline_id, corroborating_sources,
                  llm_parsed_output->>'report_kind' AS report_kind
             FROM events
            WHERE country_iso = ANY(%s)
              AND created_at >= %s AND created_at < %s
              AND (llm_parsed_output->>'report_kind' IS NULL
                   OR llm_parsed_output->>'report_kind' = 'new_incident')
              AND (event_type IS NULL OR NOT (event_type = ANY(%s)))
            ORDER BY created_at DESC""",
        (list(THEATRE_ISO), window_start, window_end,
         sorted(SAFETY_ONLY_EVENT_TYPES)),
    ).fetchall()
    return [
        {
            "id": r[0], "title": r[1], "domain": r[2], "url": r[3],
            "country_iso": r[4], "severity": r[5], "occurred_at": r[6],
            "time_certainty": r[7], "event_type": r[8], "storyline_id": r[9],
            "corroborating_sources": r[10] or [], "report_kind": r[11],
        }
        for r in rows
    ]


# One row per STORY, not per outlet. Measured on the 9 Sep window: 202 theatre
# events carried 53 storylines, and the two biggest — the same Houthi attack on
# southern Saudi Arabia, split across two storylines by the linker — accounted for
# 105 of them. Un-collapsed, that single story was 52% of the bulletin: 52% of the
# appendix the operator reads as the day's record, and 52% of the narrator's
# payload, where an article count is the only weight a headline has.
#
# The country SITREP has always collapsed (SA: 138 events into 22 clusters). The
# bulletin was the one report reading raw event rows, which is also why its
# "from Iran" count went 29 -> 138 overnight and read as a five-fold escalation
# when the escalation was real but the multiplier was syndication.
#
# Collapsing BEFORE the extraction is deliberate: direction is a property of the
# story, not of the outlet that filed it, and it cuts the LLM call count by the
# same 74% — fewer batches is directly fewer chances to hit the burst ceiling
# that produced the 5 Sep actorless bulletin.
def _representative_key(event: Dict[str, Any]) -> tuple:
    """Sort key that puts the best-INFORMED filing of a story first.

    Corroboration leads, then the casualty figure, then severity, then recency.
    Severity is never first anywhere in SIM for the same reason it is not first
    here — Pass D saturates at 100, so every member of a mass-casualty story ties
    on it and the ordering silently collapses onto whatever comes next.

    The SITREP puts the casualty figure first and this deliberately does not,
    because the two are grouping different things. A SITREP cluster IS one
    incident, so its fullest toll describes the incident. A storyline is a
    THREAD, and its fullest toll can belong to another strand of it: on 9 Sep the
    79-filing Houthi/Saudi story was represented by "Saudi Arabia vows retaliation
    as Houthi attacks set oil facilities ablaze, Yemen clashes kill 300" — a real
    toll, from the Yemeni front, standing in for an attack on Saudi Arabia.
    Ranking on corroboration instead picks "Houthi strikes in Saudi Arabia wound
    73: Riyadh-led coalition". Measured over that window it changes 3 of 53
    stories, and all three are the three biggest.
    """
    try:  # pragma: no cover - exercised via the real module in production
        from src.services.sitrep_generator import _casualty_magnitude
        deaths, casualties = _casualty_magnitude({"source_title": event.get("title") or ""})
    except Exception:
        deaths = casualties = 0
    occurred = event.get("occurred_at")
    return (
        -len(event.get("corroborating_sources") or []),
        -deaths,
        -casualties,
        -(event.get("severity") or 0),
        occurred is None,
        -(occurred.timestamp() if isinstance(occurred, datetime) else 0),
    )


def collapse_by_storyline(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One representative per storyline, carrying the outlets it stood in for.

    Nothing is thrown away that the report was showing: the other members become
    `sibling_sources`, so the appendix row still links every outlet that carried
    the story and still says how many there were. What goes away is the same
    headline printed twenty times.

    An event with no storyline_id is its own story. That is not a hypothetical
    branch kept for tidiness — it is the honest reading, since a NULL there means
    the linker never placed it, not that it belongs with anything else.
    """
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    order: List[Any] = []
    for index, event in enumerate(events):
        key = event.get("storyline_id") or ("__unlinked__", index)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(event)

    collapsed = []
    for key in order:
        members = sorted(groups[key], key=_representative_key)
        rep = dict(members[0])
        rep["outlet_count"] = len(members)
        rep["sibling_sources"] = [
            {"name": m.get("domain"), "url": m.get("url"), "title": m.get("title")}
            for m in members[1:]
        ]
        collapsed.append(rep)
    return collapsed


def _extraction_prompt(events: List[Dict[str, Any]]) -> str:
    lines = [
        "For each numbered headline, name the actor that CARRIED OUT the action and "
        "say how well that attribution stands up.",
        "",
        f'actor: "{IRAN_SIDE}" (the Iranian state or its forces — Iran, IRGC, '
        f'Revolutionary Guards, Tehran, or an Iran-aligned armed group such as '
        f'Hezbollah or the Houthis), "{US_SIDE}" (the United States, CENTCOM, or '
        f'a force acting WITH the US against Iran in this war), '
        f'"{OTHER_SIDE}" (any other named actor — including Israel acting on its '
        f'own, a national army, a police force, or a civilian body), or '
        f'"{UNATTRIBUTED}" when the text names no actor at all.',
        "",
        f'target: who or what was ON THE RECEIVING END, with the same four values. '
        f'"{IRAN_SIDE}" when Iran or Iranian territory was hit, "{US_SIDE}" when US '
        f'forces or their bases were hit, "{OTHER_SIDE}" for anyone else, '
        f'"{UNATTRIBUTED}" when the text names no target.',
        "",
        f'target_country: the ISO 3166 alpha-2 code of the country the action '
        f'LANDED IN — "SA" for a strike on Saudi Arabia, "IR" for a strike on '
        f'Iranian soil, "YE" for shelling inside Yemen — or "{UNKNOWN_COUNTRY}" '
        f'when the headline does not say where it landed. This is where it '
        f'landed, not where the story was filed and not the attacker\'s country: '
        f'"Houthi shelling of displaced people in Yemen\'s Marib" is '
        f'target_country=YE, "Houthi strikes injure 73 in Saudi Arabia" is '
        f'target_country=SA, and both have actor=iran.',
        "",
        'war_related: true when the headline is about ARMED CONFLICT or a '
        'military/security operation — a strike, shelling, an interception, air '
        'defence, a raid, a blockade, a seizure or release across a front line, '
        'airspace or shipping disruption, an evacuation, a military threat, '
        'diplomacy over the fighting, or casualties from any of it. false when '
        'the event is not military at all and merely HAPPENED in one of these '
        'countries: ordinary crime and policing, a road or aviation accident, a '
        'labour or pay dispute, an unrelated domestic political story.',
        "",
        f'standing: "{STANDING_CONFIRMED}" when the headline reports the action as '
        f'having happened, "{STANDING_CLAIMED}" when one side claims/alleges/says it '
        f'without confirmation, "{STANDING_DENIED}" when the headline reports it as '
        f'denied or rejected, "{STANDING_UNKNOWN}" when it cannot be told.',
        "",
        "Rules that matter more than fluency:",
        '- The actor is the SUBJECT of the action, not whoever is mentioned first. '
        '"Iran fires missiles in response to US strikes" is actor=iran.',
        '- A threat, a vow or a warning is not an action: standing=unknown, and the '
        'actor is the party making the threat.',
        '- "X says N killed in Y strikes" reports Y as the actor. Naming who SAID it '
        'sets the standing, never the actor.',
        '- Never guess an actor from context. If the headline says a tanker was '
        '"struck by unidentified projectiles", the actor is unattributed.',
        '- In "A denies B\'s claim that it struck C", the actor is B — the party '
        'said to have acted — never A, the one issuing the denial. standing=denied.',
        '- An actor named as an ADJECTIVE or a possessive is still named. '
        '"Kuwait\'s air defences intercept Iranian missiles" is actor=iran, and so '
        'is \'"Iranian aggression": Kuwait responds to missile and UAV attacks\'. '
        'Do not answer unattributed because the attacker is not the subject.',
        '- But the adjective has to name a STATE or an ARMED FORCE. "Iranian '
        'missiles" and "Iranian forces" are iran; "Iranian professors", "Iranian '
        f'media" and "Iranian shipping" are civilians and belong to "{OTHER_SIDE}". '
        'A nationality is not a belligerent.',
        f'- Israel is "{OTHER_SIDE}", never "{US_SIDE}". An Israeli strike in '
        f'Lebanon or the West Bank is Israel acting on its own; "{US_SIDE}" means '
        'the United States and whoever is fighting Iran alongside it.',
        '- An interception, a shoot-down, or a defensive response is an attack seen '
        'from the receiving end: the actor is whoever FIRED, not whoever intercepted. '
        'If the text does not say whose missiles they were, the actor is '
        f'"{UNATTRIBUTED}" — that is the honest answer, not a reason to guess.',
        '- That rule is ONLY about headlines that name nobody, and it never '
        'overrides the ones above. "Iran launches strikes targeting Kuwait" names '
        f'the attacker as the SUBJECT: actor="{IRAN_SIDE}". "Kuwait intercepts '
        f'Iranian missiles" names it as an ADJECTIVE: actor="{IRAN_SIDE}". Both '
        'are named. What you must not do is supply an attacker the text never '
        'mentions in any form, in any position.',
        "",
        '- The target is not the country the story is filed under. "Iran strikes '
        'bases in Bahrain, Iraq and Jordan" is actor=iran, target=us_coalition.',
        "",
        'Reply with JSON only: '
        '{"items":[{"n":1,"actor":"...","target":"...","target_country":"..",'
        '"standing":"...","war_related":true}]}',
        "",
    ]
    for i, ev in enumerate(events, 1):
        lines.append(f'{i}. {ev["title"]}')
    return "\n".join(lines)


def _parse_extraction(content: str, expected: int) -> List[Dict[str, Any]]:
    """Parse the batch reply, tolerating a model that wraps or pads its JSON.

    Two shapes have to survive, and one span cannot cover both. A preamble
    followed by one object ("Here is my analysis. {...}") needs the widest span
    from the first brace to the last; a model that emits a SECOND object after
    the answer ({"items":[...]}\n{"note":"..."}) makes that same span invalid
    JSON and throws "Extra data". So: decode greedily from the first brace, and
    if that fails, take the first complete object there instead.

    Measured, not imagined — gemini-3.5-flash-lite failed the direction probe
    twice in a row on 2026-09-04 with "Extra data: line 1 column 87" and
    "Extra data: line 7 column 4", having answered every row correctly. A
    trailing object was costing a whole batch its extraction.
    """
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in extraction reply")
    try:
        parsed = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        parsed, _ = json.JSONDecoder().raw_decode(content[start:])
    items = parsed.get("items", [])
    # war_related defaults TRUE on every failure path below. The field decides
    # whether an event is dropped from the report entirely, and a parse failure is
    # not evidence that an event is off-topic — defaulting it false would let one
    # malformed batch silently delete a day's strikes.
    out: List[Dict[str, Any]] = [
        {"actor": UNATTRIBUTED, "target": UNATTRIBUTED,
         "target_country": UNKNOWN_COUNTRY,
         "standing": STANDING_UNKNOWN, WAR_RELATED: True}
        for _ in range(expected)
    ]
    valid_actors = {IRAN_SIDE, US_SIDE, OTHER_SIDE, UNATTRIBUTED}
    valid_standing = {STANDING_CONFIRMED, STANDING_CLAIMED,
                      STANDING_DENIED, STANDING_UNKNOWN}
    recovered = 0
    for item in items:
        try:
            idx = int(item["n"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= idx < expected:
            continue
        actor = str(item.get("actor", "")).strip().lower()
        target = str(item.get("target", "")).strip().lower()
        standing = str(item.get("standing", "")).strip().lower()
        # An ISO code or nothing. Anything else — a country name, a region, a
        # sentence — is read as absent rather than half-trusted, on the same rule
        # the labels above follow: a value that decides a section is either one of
        # the values we asked for or it is missing.
        iso = str(item.get("target_country", "")).strip().upper()
        target_country = iso if _ISO2_RE.fullmatch(iso) else UNKNOWN_COUNTRY
        # An unrecognised value is treated as absent rather than trusted. The
        # bulletin's sections are built from these, so a hallucinated label would
        # move a real strike into the wrong half of the war.
        out[idx] = {
            "actor": actor if actor in valid_actors else UNATTRIBUTED,
            "target": target if target in valid_actors else UNATTRIBUTED,
            "target_country": target_country,
            "standing": standing if standing in valid_standing else STANDING_UNKNOWN,
            # Only an explicit false drops an event; a missing or unreadable value
            # keeps it. Same asymmetry as above, for the same reason.
            WAR_RELATED: item.get(WAR_RELATED) is not False,
        }
        recovered += 1
    # A reply that carries fewer items than the batch had is the failure that hid
    # the 5 Sep collapse: every missing item silently takes the unattributed
    # default, the sections still render, and the only symptom is that the report
    # has quietly stopped saying which way anything was going. On that day
    # gemini-3.5-flash-lite averaged 442 completion tokens where twelve items need
    # about 600, and 51 of 73 events came back with no actor. Fail open, but never
    # fail silent.
    if recovered < expected:
        counters.bump(counters.BULLETIN_DIRECTION_SHORT_REPLY, expected - recovered)
        logger.warning(
            "Direction extraction recovered %d of %d items; the other %d keep the "
            "unattributed default", recovered, expected, expected - recovered)
    return out


# Groq's free tier enforces 1,000 OUTPUT tokens per minute per model, and the
# request is rejected before it runs if the asked-for max_tokens exceeds it —
# "Request too large ... Requested 1019", logged in full on 5 Sep now that first
# 429 bodies are visible. max_tokens here is 50*batch + 512, so 12 asked for
# 1,112 and qwen, the primary and by far the most accurate slot, could not serve
# a single batch of the bulletin. 8 asks for 912 and fits with room.
#
# Same shape as the Pass C fix of 2026-08-11, and the same lesson: a batch size
# is not a throughput choice, it is a promise about the reply's size.
DIRECTION_BATCH_SIZE = 8


def extract_direction(router: LLMRouter, events: List[Dict[str, Any]],
                      db_conn=None,
                      batch_size: int = DIRECTION_BATCH_SIZE) -> List[Dict[str, Any]]:
    """Attach actor and standing to each event, in place, and return the list.

    A failed batch does not fail the bulletin: those events keep the unattributed
    default and fall to the regional section, which is the honest place for an
    event whose direction we could not establish.

    ``db_conn`` is only for spend attribution. It is optional so the extraction can
    be exercised without a database, but a stage that never logs is a stage that
    looks free in the rollup — the pipeline passes one.
    """
    for start in range(0, len(events), batch_size):
        chunk = events[start:start + batch_size]
        try:
            result = call_llm(
                router=router,
                prompt=_extraction_prompt(chunk),
                system_prompt=_EXTRACTION_SYSTEM_PROMPT,
                # ~50 tokens per item for {"n":N,"actor":"...","standing":"..."},
                # plus headroom for a reasoning preamble the bulk slots may emit.
                max_tokens=50 * len(chunk) + 512,
            )
            if db_conn is not None:
                log_llm_telemetry(db_conn, result, router, success=True,
                                  purpose="bulletin_direction")
            parsed = _parse_extraction(result.get("content", ""), len(chunk))
        except Exception:
            counters.bump(counters.BULLETIN_DIRECTION_BATCH_FAILED)
            logger.warning("Direction extraction failed for a batch of %d; those "
                           "events stay unattributed", len(chunk), exc_info=True)
            parsed = [{"actor": UNATTRIBUTED, "target": UNATTRIBUTED,
                       "target_country": UNKNOWN_COUNTRY,
                       "standing": STANDING_UNKNOWN, WAR_RELATED: True}
                      for _ in chunk]
        for ev, fields in zip(chunk, parsed):
            ev.update(fields)
    return events


def assign_section(event: Dict[str, Any]) -> str:
    """Which of the bulletin's three sections this event belongs in.

    Direction is the pair (actor, target), both read from the headline. country_iso
    is a FALLBACK and nothing more.

    It used to be the primary signal, on the assumption that it records where an
    event landed. Measured against the first real bulletin, that assumption is
    false: Pass C files "Iran strikes bases in Bahrain, Iraq and Jordan" under IR,
    because Iran is the dominant country in the text, not because Iran was hit. 29
    of one window's 74 "regional" events were Iranian strikes on neighbours sitting
    in the wrong section — 16% of the bulletin, all of it the section-2 material the
    report exists to show.

    Two asymmetries stay, for the same reasons as before:
      * an exchange needs two different sides — Iran acting on Iran is an internal
        security incident, not part of the war, and falls to regional;
      * an event whose actor could not be established never enters a directional
        section, because putting it there would assert the very thing that could
        not be read.

    A third joined them on 9 Sep 2026: section 2 needs the action to CROSS a
    border. An Iran-side actor striking inside the country it operates from is
    that country's own war — see IRAN_SIDE_HOME_ISO.
    """
    actor = event.get("actor", UNATTRIBUTED)
    if actor in (UNATTRIBUTED, OTHER_SIDE):
        return SECTION_REGIONAL

    if actor == IRAN_SIDE and event.get("target_country") in IRAN_SIDE_HOME_ISO:
        # The Iran-side actor is at home: the Houthis inside Yemen, Hezbollah
        # inside Lebanon. A local war is a regional development, not a strike
        # from Iran on a neighbour. See IRAN_SIDE_HOME_ISO.
        return SECTION_REGIONAL

    target = event.get("target", UNATTRIBUTED)
    if target == UNATTRIBUTED:
        # Nothing was named on the receiving end, so fall back to where the event
        # was filed. This is the old rule, kept only for the case it was right for.
        target = IRAN_SIDE if event.get("country_iso") == "IR" else OTHER_SIDE

    if actor == target:
        # One side acting on itself is not an exchange: air defence over its own
        # territory, an internal incident, a domestic announcement.
        return SECTION_REGIONAL
    if actor == US_SIDE and target == IRAN_SIDE:
        return SECTION_ON_IRAN
    if actor == IRAN_SIDE:
        return SECTION_FROM_IRAN
    return SECTION_REGIONAL


# The heading a bullet sits under when no country can be named for it. Measured
# 2026-09-10: 14 of the 27 section-2 events had target_country "unknown" — every
# Hormuz shipping filing among them — and the narrator, given no place for them,
# printed them under the heading of the country above. Four bullets about the
# Strait of Hormuz shipped under "Ürdün".
UNPLACED_LABEL = "Belirsiz hedef"


def bulletin_place(event: Dict[str, Any], section: str) -> str:
    """The Turkish place heading a bullet belongs under.

    Which field answers "where" depends on the section, and country_iso is the
    wrong answer twice over in section 2: Pass C files "Iran strikes ships outside
    Hormuz" under IR because Iran is the dominant country in the text, and section
    2 is precisely about events that LAND somewhere else. So the target leads and
    the filing country is the fallback — the same order assign_section already
    uses to decide the section itself.
    """
    target = str(event.get("target_country") or "").upper()
    iso = str(event.get("country_iso") or "").upper()
    if section == SECTION_ON_IRAN:
        return THEATRE_NAMES["IR"]
    if section == SECTION_FROM_IRAN:
        for candidate in (target, iso):
            if candidate in THEATRE_NAMES and candidate != "IR":
                return THEATRE_NAMES[candidate]
        return UNPLACED_LABEL
    for candidate in (iso, target):
        if candidate in THEATRE_NAMES:
            return THEATRE_NAMES[candidate]
    return UNPLACED_LABEL


def group_into_sections(events: List[Dict[str, Any]]
                        ) -> Dict[str, List[Dict[str, Any]]]:
    """The bulletin's three buckets, each ordered by place then severity.

    Ordered by PLACE first since 2026-09-10. The narrative groups its bullets under
    place headings, so handing it a severity-ordered list asks it to do the
    grouping itself out of an interleaved sequence — and it did it wrong, gluing
    placeless Hormuz bullets under the previous country's heading. Sorting here
    makes the grouping the model has to perform a matter of reading the list in
    order. Severity still decides the order inside a place, and the placeless
    bucket sorts last because it is the one heading a reader skips.
    """
    sections: Dict[str, List[Dict[str, Any]]] = {
        SECTION_ON_IRAN: [], SECTION_FROM_IRAN: [], SECTION_REGIONAL: [],
    }
    for event in events:
        section = assign_section(event)
        event["place"] = bulletin_place(event, section)
        sections[section].append(event)
    for bucket in sections.values():
        bucket.sort(key=lambda e: (e.get("place") == UNPLACED_LABEL,
                                   e.get("place") or "",
                                   -(e.get("severity") or 0)))
    return sections

# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------
#
# Rendered by sitrep_html.render_sitrep_html, which is shape-driven rather than
# SITREP-specific: an ALL-CAPS line becomes a section header, a short line that
# does not end in punctuation becomes a place sub-heading, and a bulleted line
# becomes a bullet. That is exactly the shape of the report this bulletin
# reproduces — section, place, facts — so the narrative is written to it and no
# second renderer is needed.

SECTION_TITLES = {
    SECTION_ON_IRAN: "İRAN TOPRAKLARINA YÖNELİK SALDIRILAR",
    SECTION_FROM_IRAN: "İRAN'DAN KOMŞU ÜLKELERE YÖNELİK SALDIRILAR",
    SECTION_REGIONAL: "BÖLGESEL GELİŞMELER VE STRATEJİK HAMLELER",
}

# What each standing is called in the report's own source line. The vocabulary is
# the SITREP's, so a reader moving between the two reports is not asked to learn a
# second one — and "Doğrulanmamış" is never dressed up as anything stronger.
STANDING_LABELS = {
    STANDING_CONFIRMED: "Doğrulandı",
    STANDING_CLAIMED: "Tek taraflı iddia",
    STANDING_DENIED: "İddia edildi, yalanlandı",
    STANDING_UNKNOWN: "Durum belirsiz",
}

# What each actor is CALLED in the report. STANDING_LABELS has always existed and
# the standing reads correctly in Turkish because of it; the actor had no such
# table, so `fail: "us_coalition"` went into the narrator's payload raw and came
# back out in the prose — eight times in the 4 Sep bulletin, in sentences like
# "us_coalition tarafından gerçekleştirilen saldırılarda". A slug is not a name.
#
# OTHER_SIDE has no name to give: it means "an actor SIM did not classify", and
# the only place its identity exists is the headline, which the narrator is
# already reading. So the label is an instruction to go and read it, paired with
# the rule below — a fixed word like "diğer" would produce "diğer tarafından
# gerçekleştirilen saldırı", which names nobody while sounding like it does.
ACTOR_LABELS = {
    IRAN_SIDE: "İran",
    US_SIDE: "ABD/koalisyon güçleri",
    OTHER_SIDE: "başlıkta adı geçen taraf",
    UNATTRIBUTED: "belirsiz",
}

_NARRATIVE_SYSTEM_PROMPT = (
    "Sen bir güvenlik analistisin. Türkçe, düz ve devrik olmayan cümlelerle "
    "yazarsın. Sana verilen veride olmayan hiçbir olayı, sayıyı, yeri veya "
    "tarihi yazmazsın; eksik bilgiyi uydurmak yerine eksik bırakırsın."
)


def _narrative_prompt(sections: Dict[str, List[Dict[str, Any]]],
                      window_start: datetime, window_end: datetime) -> str:
    payload = {}
    for key, title in SECTION_TITLES.items():
        payload[title] = [
            {
                "baslik": ev.get("title", ""),
                "yer": ev.get("place") or UNPLACED_LABEL,
                "fail": ACTOR_LABELS.get(ev.get("actor"), "belirsiz"),
                "durum": STANDING_LABELS.get(ev.get("standing"), "Durum belirsiz"),
                "siddet": ev.get("severity"),
                "yayinci": ev.get("domain"),
                "bagimsiz_kaynak": len(ev.get("corroborating_sources") or []),
            }
            for ev in sections.get(key, [])
        ]
    return "\n".join([
        f"Aşağıdaki veriden {window_start:%d.%m.%Y %H:%M} — {window_end:%d.%m.%Y %H:%M} "
        "UTC penceresi için bölgesel askeri gelişmeler bültenini yaz.",
        "",
        "BİÇİM (renderer bu şekle göre çalışır, birebir uy):",
        "- Rapor İKİ DÜZEYLİDİR. Önce bölüm başlığı: verideki bölüm adını TAMAMI "
        "BÜYÜK HARF, tek satır, birebir yaz. Altına o bölümün yer başlıklarını ve "
        "maddelerini koy. Veride dolu olan HER bölüm raporda kendi başlığıyla yer "
        "almalı; bölüm başlıklarını atlayıp doğrudan yer başlıklarına geçme.",
        "- İlk bölüm YÖNETİCİ ÖZETİ olsun: 2-3 paragraf, madde işareti yok.",
        "- Maddeleri verideki 'yer' alanına göre grupla ve her grubun başına o "
        "yeri kısa bir satır olarak yaz (nokta ile bitmesin). Yer başlığını "
        "verideki değerden birebir al, kendin ülke atama; veri sırası zaten "
        "yere göre gruplanmıştır.",
        "- Ayrıntıları '- ' ile başlayan maddeler halinde yaz.",
        "- HER maddenin sonuna ' — Durum: X' ekle. X, o olayın verideki 'durum' "
        "alanının BİREBİR kopyasıdır: Doğrulandı / Tek taraflı iddia / "
        "İddia edildi, yalanlandı / Durum belirsiz. Bir maddede birden çok olayı "
        "birleştirdiysen EN ZAYIF durumu yaz (yalanlandı < iddia < belirsiz < "
        "doğrulandı). Durumu kendin yükseltme.",
        "",
        "KURALLAR:",
        "- Saat verme. Elimizde olayların saati YOK; 'akşam saatlerinde' gibi "
        "ifadeler de uydurmadır. Yalnız verilen pencereye atıf yap.",
        "- Her olayın failini ve durumunu yaz. 'Tek taraflı iddia' veya "
        "'İddia edildi, yalanlandı' olan bir olayı ASLA gerçekleşmiş gibi anlatma; "
        "kimin iddia ettiğini söyle. 'doğrulanmıştır', 'teyit edilmiştir' gibi "
        "ifadeleri YALNIZ durumu 'Doğrulandı' olan olaylar için kullan.",
        "- Fail alanı 'başlıkta adı geçen taraf' olan olaylarda faili başlıktan "
        "oku ve adıyla yaz.",
        "- Fail alanı 'belirsiz' olan olaylarda kimseye fail atfetme; olayı "
        "failsiz anlat.",
        "- Veri alanlarını olduğu gibi cümleye kopyalama; hepsi Türkçe "
        "yazılacak.",
        "- Verideki sayıları değiştirme, yuvarlama, TOPLAMA. Sayıları rakamla "
        "yaz (beş değil 5). Veride olmayan bir sayıyı yazma.",
        "- Aynı olay birden çok başlıkta geçebilir; bunlar AYRI olaylar DEĞİL, "
        "aynı olayın farklı yayıncılardaki halleridir. Onları tek maddede birleştir; "
        "'bir daha', 'ikinci kez', 'toplam N' gibi ifadelerle ikinci bir olay "
        "uydurma.",
        "",
        "VERİ:",
        json.dumps(payload, ensure_ascii=False, indent=1),
    ])


# Every value the narrator's payload is keyed on. If one of these appears in the
# finished Turkish prose, the model copied a field name out of the data instead of
# writing it — which is exactly what shipped on 4 Sep, eight times, as
# "us_coalition tarafından gerçekleştirilen saldırılarda".
#
# ACTOR_LABELS closed the hole that let it happen. This closes the class: any
# future field added to the payload with a slug for a value gets caught here
# whether or not anyone remembers to write a label table for it.
_INTERNAL_TOKENS = (
    IRAN_SIDE, US_SIDE, OTHER_SIDE, UNATTRIBUTED,
    STANDING_CONFIRMED, STANDING_CLAIMED, STANDING_DENIED, STANDING_UNKNOWN,
    WAR_RELATED, SECTION_ON_IRAN, SECTION_FROM_IRAN, SECTION_REGIONAL,
)


def narrative_is_usable(text: str) -> bool:
    """Two things a bulletin narrative must be, both machine-checkable.

    It must not print an internal identifier at the reader, and it must carry at
    least one ALL-CAPS section line — the renderer is shape-driven and a narrative
    without one collapses into a single undifferentiated block.

    Note "iran" is a token here and also a perfectly ordinary English word the
    prose will not contain (the report is Turkish, where it is "İran"), which is
    why the match is on word boundaries and case-sensitive: the slug is lowercase
    ASCII in every payload it comes from.
    """
    if not text or not text.strip():
        return False
    for token in _INTERNAL_TOKENS:
        if re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", text):
            logger.warning("Bulletin narrative leaked the internal token %r", token)
            return False
    for line in text.splitlines():
        letters = [c for c in line.strip() if c.isalpha()]
        if len(letters) >= 4 and all(c == c.upper() for c in letters):
            return True
    logger.warning("Bulletin narrative carries no ALL-CAPS section line")
    return False


# A bullet line, and the status tag the format rule requires at the end of one.
_BULLET_PREFIX_RE = re.compile(r"^\s*[-•]\s+")
_STATUS_TAG_RE = re.compile(r"—\s*Durum:\s*([^—]+?)\s*$")


def narrative_standing_is_honest(text: str,
                                 sections: Dict[str, List[Dict[str, Any]]]) -> bool:
    """Every bullet declares a standing, and confirmation is not invented.

    Measured on the 10 Sep 2026 bulletin: of 47 events the extractor marked 24
    "claimed", 2 "denied" and 5 "unknown" — and the narrative ended all nineteen of
    its bullets with "doğrulanmıştır". IRGC claims about US destroyers, an Aramco
    fire and a strike on a Jordanian airbase all reached the reader as confirmed
    fact, and the one event the US had publicly DENIED was narrated as confirmed
    too. The standing was in the payload and the prompt already forbade this; the
    narrator simply collapsed the 7 near-duplicate Jordan filings into one bullet
    and dropped the weakest label on the floor.

    So the label stops being advice and becomes format. Two checks, both cheap:

    1. Every bullet ends in one of the four STANDING_LABELS. A model that will not
       tag its bullets is a model whose prose cannot be audited at all.
    2. The number of bullets claiming "Doğrulandı" cannot exceed the number of
       events that actually carry that standing. This is the systemic case — 19
       confirmations from 16 confirmed events is arithmetic, not judgement — and it
       is what the 10 Sep bulletin would have failed on.

    It does NOT catch a single upgraded bullet inside a compliant report (probed
    2026-09-10 on the real payload: 16/16 bullets tagged, one claim upgraded). That
    needs bullet-to-event attribution, which the narrative does not carry.
    """
    bullets = [line.strip() for line in text.splitlines()
               if _BULLET_PREFIX_RE.match(line)]
    if not bullets:
        logger.warning("Bulletin narrative carries no bullet lines")
        return False
    allowed = set(STANDING_LABELS.values())
    tags: List[str] = []
    for bullet in bullets:
        match = _STATUS_TAG_RE.search(bullet)
        label = match.group(1).strip() if match else ""
        if label not in allowed:
            logger.warning("Bulletin bullet carries no usable standing tag: %.80s",
                           bullet)
            return False
        tags.append(label)
    confirmed_label = STANDING_LABELS[STANDING_CONFIRMED]
    claimed_confirmed = sum(1 for tag in tags if tag == confirmed_label)
    have_confirmed = sum(1 for events in sections.values() for event in events
                         if event.get("standing") == STANDING_CONFIRMED)
    if claimed_confirmed > have_confirmed:
        logger.warning(
            "Bulletin narrative confirms %d events; only %d are confirmed",
            claimed_confirmed, have_confirmed)
        return False
    return True


# Digits, once thousand separators are out of the way ("1.054" is one number, not
# two). English number words are mapped as well because the payload's headlines are
# English: "US strikes five tankers" is where a Turkish "5" legitimately comes from.
_NUMBER_TOKEN_RE = re.compile(r"\d+")
_THOUSANDS_SEP_RE = re.compile(r"(?<=\d)[.,](?=\d{3}\b)")
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}


def _numbers_in(text: str) -> set:
    """Every number the text states, digits and English number words alike."""
    flat = _THOUSANDS_SEP_RE.sub("", text or "")
    found = {int(token) for token in _NUMBER_TOKEN_RE.findall(flat)}
    lowered = flat.lower()
    for word, value in _WORD_NUMBERS.items():
        if re.search(rf"(?<![\w-]){word}(?![\w-])", lowered):
            found.add(value)
    return found


def narrative_numbers_are_sourced(text: str, prompt: str) -> bool:
    """No number in the prose that was not in the data the model was handed.

    The prompt has said "do not change, round or SUM the numbers" since the report
    existed, and on 10 Sep 2026 the narrative opened with "beş İran petrol tankeri
    vurulmuştur" followed by "beş İran petrol tankeri DAHA vurulmuştur" — two
    filings of one strike read as two strikes. Re-probed on the real payload the
    same model summed them instead: "toplam 10 adet petrol tankeri". Ten tankers
    were never struck and the figure appears nowhere in the data.

    A casualty or asset count is the part of an intelligence bulletin a reader
    acts on, so the arithmetic is checked rather than requested. Everything the
    prompt carries counts as sourced — the payload, the window stamp, the format
    examples — because that is exactly the set the model was allowed to read.

    1 is exempt: Turkish "bir" is also the indefinite article, and a rule that
    fires on "bir tanker" would reject every honest narrative ever written. Numbers
    written as Turkish words go unchecked, which is why the prompt now asks for
    digits — this fails OPEN on an unchecked spelling and never on an honest one.
    """
    allowed = _numbers_in(prompt)
    stray = sorted(n for n in _numbers_in(text) if n > 1 and n not in allowed)
    if stray:
        logger.warning("Bulletin narrative states numbers absent from the data: %s",
                       stray[:8])
        return False
    return True


def narrative_covers_sections(text: str,
                              sections: Dict[str, List[Dict[str, Any]]]) -> bool:
    """Every section that has events appears under its own heading.

    The report's whole claim is directional — what was done TO Iran, what was done
    FROM it, what happened around it — and that claim lives entirely in the three
    section headings. Probed 2026-09-10: told firmly enough to group its bullets by
    place, the narrator dropped all three section headings and ran the places
    together, turning a directional bulletin into a list of countries. The
    ALL-CAPS check above still passed, because "YÖNETİCİ ÖZETİ" is ALL-CAPS too.
    """
    for key, title in SECTION_TITLES.items():
        if sections.get(key) and title not in text:
            logger.warning("Bulletin narrative is missing the section heading %r",
                           title)
            return False
    return True


def build_bulletin(db_conn, router: LLMRouter, window_start: datetime,
                   window_end: datetime, max_tokens: int = 6000,
                   narrative_router: Optional[LLMRouter] = None) -> Dict[str, Any]:
    """Fetch, attribute, group and narrate the theatre bulletin.

    Returns the narrative plus the grouped sections, so the caller can render and
    dispatch without re-deriving either.

    Two routers, because these are two different jobs and conflating them cost
    the 6 Sep bulletin entirely. Direction extraction wants slots MEASURED for
    direction accuracy and sends small batches; the narrative wants prose and
    sends the whole day — 11,064 tokens that morning. Running both on the
    direction router meant that narrowing the direction slots to one Groq model
    also handed Groq's per-request size ceiling the narrative, which it refused,
    and the report did not publish at all.

    ``narrative_router`` falls back to ``router`` so every existing caller and
    test keeps working; production passes the quality cascade.
    """
    events = fetch_theatre_events(db_conn, window_start, window_end)
    if not events:
        logger.info("Iran bulletin: no theatre events in window")
        return {"events": [], "sections": group_into_sections([]), "narrative": "",
                "status": "empty"}

    filed = len(events)
    events = collapse_by_storyline(events)
    if len(events) < filed:
        logger.info("Iran bulletin: %d filings collapsed into %d stories",
                    filed, len(events))

    extract_direction(router, events, db_conn=db_conn)

    # Off-topic events leave the report here rather than at the fetch, because
    # only the extraction can tell them apart — see WAR_RELATED. An event the
    # extractor put a BELLIGERENT behind is never dropped, whatever it answered
    # to this field: "US strike kills key broker in Houthi arms alliance" is the
    # war by any reading, and a single mislabelled field should not be able to
    # delete a strike from the record.
    kept, dropped = [], []
    for ev in events:
        belligerent = ev.get("actor") in (IRAN_SIDE, US_SIDE)
        (kept if belligerent or ev.get(WAR_RELATED, True) else dropped).append(ev)
    if dropped:
        logger.info("Iran bulletin: %d of %d events dropped as off-topic (e.g. %s)",
                    len(dropped), len(events),
                    "; ".join((e.get("title") or "")[:60] for e in dropped[:3]))
    events = kept
    sections = group_into_sections(events)
    logger.info(
        "Iran bulletin: %d events — on Iran %d, from Iran %d, regional %d",
        len(events), len(sections[SECTION_ON_IRAN]),
        len(sections[SECTION_FROM_IRAN]), len(sections[SECTION_REGIONAL]),
    )

    prompt = _narrative_prompt(sections, window_start, window_end)
    result = call_llm(
        router=narrative_router or router,
        prompt=prompt,
        system_prompt=_NARRATIVE_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        # Prose, never JSON: a reasoning model asked for JSON here returns the
        # narrative wrapped in a string field and the renderer sees one long line.
        json_mode=False,
        accept=lambda text: (narrative_is_usable(text)
                             and narrative_covers_sections(text, sections)
                             and narrative_standing_is_honest(text, sections)
                             and narrative_numbers_are_sourced(text, prompt)),
    )
    if db_conn is not None:
        log_llm_telemetry(db_conn, result, router, success=True,
                          purpose="bulletin_narrative")
    return {"events": events, "sections": sections,
            "narrative": result.get("content", ""), "status": "ok",
            "model": result.get("model")}
