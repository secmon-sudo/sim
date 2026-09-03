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
from datetime import datetime
from typing import Any, Dict, List

from src.core.llm_client import call_llm, log_llm_telemetry
from src.core.llm_router import LLMRouter

logger = logging.getLogger(__name__)

# The theatre. Iran plus every country that has taken or hosted a strike in this
# war — measured from the corpus rather than drawn on a map, so a country that
# stops appearing simply contributes nothing instead of forcing a code change.
THEATRE_ISO = ("IR", "IQ", "KW", "JO", "SA", "AE", "QA", "BH", "OM", "IL", "LB", "YE")

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

SECTION_ON_IRAN = "on_iran"
SECTION_FROM_IRAN = "from_iran"
SECTION_REGIONAL = "regional"

_EXTRACTION_SYSTEM_PROMPT = (
    "You read security news headlines and report WHO ACTED, not what you believe "
    "happened. You never infer an actor that the text does not name, and you never "
    "upgrade a one-sided claim into a fact. Answer with JSON only."
)


def fetch_theatre_events(db_conn, window_start: datetime,
                         window_end: datetime) -> List[Dict[str, Any]]:
    """Theatre events that are reports of a new incident, newest first.

    report_kind does the first cut and it does it for free: commentary, followup
    and roundup articles are 71 of every 474 theatre events, and none of them is a
    strike. Events classified before report_kind existed (11 Aug 2026) carry NULL,
    and those are kept — excluding them would silently shorten the window rather
    than filter it.
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
            ORDER BY created_at DESC""",
        (list(THEATRE_ISO), window_start, window_end),
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


def _extraction_prompt(events: List[Dict[str, Any]]) -> str:
    lines = [
        "For each numbered headline, name the actor that CARRIED OUT the action and "
        "say how well that attribution stands up.",
        "",
        f'actor: "{IRAN_SIDE}" (Iran, IRGC, Revolutionary Guards, Tehran, or an '
        f'Iran-aligned group), "{US_SIDE}" (United States, CENTCOM, coalition, or '
        f'an allied military), "{OTHER_SIDE}" (any other named actor), or '
        f'"{UNATTRIBUTED}" when the text names no actor at all.',
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
        "",
        'Reply with JSON only: {"items":[{"n":1,"actor":"...","standing":"..."}]}',
        "",
    ]
    for i, ev in enumerate(events, 1):
        lines.append(f'{i}. {ev["title"]}')
    return "\n".join(lines)


def _parse_extraction(content: str, expected: int) -> List[Dict[str, str]]:
    """Parse the batch reply, tolerating a model that wraps or pads its JSON."""
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in extraction reply")
    items = json.loads(content[start:end + 1]).get("items", [])
    out: List[Dict[str, str]] = [
        {"actor": UNATTRIBUTED, "standing": STANDING_UNKNOWN} for _ in range(expected)
    ]
    valid_actors = {IRAN_SIDE, US_SIDE, OTHER_SIDE, UNATTRIBUTED}
    valid_standing = {STANDING_CONFIRMED, STANDING_CLAIMED,
                      STANDING_DENIED, STANDING_UNKNOWN}
    for item in items:
        try:
            idx = int(item["n"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= idx < expected:
            continue
        actor = str(item.get("actor", "")).strip().lower()
        standing = str(item.get("standing", "")).strip().lower()
        # An unrecognised value is treated as absent rather than trusted. The
        # bulletin's sections are built from these, so a hallucinated label would
        # move a real strike into the wrong half of the war.
        out[idx] = {
            "actor": actor if actor in valid_actors else UNATTRIBUTED,
            "standing": standing if standing in valid_standing else STANDING_UNKNOWN,
        }
    return out


def extract_direction(router: LLMRouter, events: List[Dict[str, Any]],
                      db_conn=None, batch_size: int = 12) -> List[Dict[str, Any]]:
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
                # ~40 tokens per item for {"n":N,"actor":"...","standing":"..."},
                # plus headroom for a reasoning preamble the bulk slots may emit.
                max_tokens=40 * len(chunk) + 512,
            )
            if db_conn is not None:
                log_llm_telemetry(db_conn, result, router, success=True,
                                  purpose="bulletin_direction")
            parsed = _parse_extraction(result.get("content", ""), len(chunk))
        except Exception:
            logger.warning("Direction extraction failed for a batch of %d; those "
                           "events stay unattributed", len(chunk), exc_info=True)
            parsed = [{"actor": UNATTRIBUTED, "standing": STANDING_UNKNOWN}
                      for _ in chunk]
        for ev, fields in zip(chunk, parsed):
            ev.update(fields)
    return events


def assign_section(event: Dict[str, Any]) -> str:
    """Which of the bulletin's three sections this event belongs in.

    country_iso says where it landed and the extracted actor says who acted, and
    the pair is what the section headings mean. The two asymmetries are deliberate:

      * a strike on Iranian soil BY Iran is not section 1 — internal security
        incidents are not part of the war's exchange, so they fall to regional;
      * an event whose actor could not be established never enters a directional
        section, because putting it there would assert the very thing that could
        not be read.
    """
    actor = event.get("actor", UNATTRIBUTED)
    if actor in (UNATTRIBUTED, OTHER_SIDE):
        return SECTION_REGIONAL
    if event.get("country_iso") == "IR":
        return SECTION_ON_IRAN if actor == US_SIDE else SECTION_REGIONAL
    return SECTION_FROM_IRAN if actor == IRAN_SIDE else SECTION_REGIONAL


def group_into_sections(events: List[Dict[str, Any]]
                        ) -> Dict[str, List[Dict[str, Any]]]:
    """The bulletin's three buckets, each ordered by severity then recency."""
    sections: Dict[str, List[Dict[str, Any]]] = {
        SECTION_ON_IRAN: [], SECTION_FROM_IRAN: [], SECTION_REGIONAL: [],
    }
    for event in events:
        sections[assign_section(event)].append(event)
    for bucket in sections.values():
        bucket.sort(key=lambda e: (e.get("severity") or 0), reverse=True)
    return sections
