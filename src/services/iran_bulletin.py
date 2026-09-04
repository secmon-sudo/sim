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

SECTION_ON_IRAN = "on_iran"
SECTION_FROM_IRAN = "from_iran"
SECTION_REGIONAL = "regional"

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
        f'war_related: true when the headline is about ARMED CONFLICT or a '
        f'military/security operation — a strike, shelling, an interception, air '
        f'defence, a raid, a blockade, a seizure or release across a front line, '
        f'airspace or shipping disruption, an evacuation, a military threat, '
        f'diplomacy over the fighting, or casualties from any of it. false when '
        f'the event is not military at all and merely HAPPENED in one of these '
        f'countries: ordinary crime and policing, a road or aviation accident, a '
        f'labour or pay dispute, an unrelated domestic political story.',
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
        "",
        '- The target is not the country the story is filed under. "Iran strikes '
        'bases in Bahrain, Iraq and Jordan" is actor=iran, target=us_coalition.',
        "",
        'Reply with JSON only: '
        '{"items":[{"n":1,"actor":"...","target":"...","standing":"...",'
        '"war_related":true}]}',
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
    # war_related defaults TRUE on every failure path below. The field decides
    # whether an event is dropped from the report entirely, and a parse failure is
    # not evidence that an event is off-topic — defaulting it false would let one
    # malformed batch silently delete a day's strikes.
    out: List[Dict[str, Any]] = [
        {"actor": UNATTRIBUTED, "target": UNATTRIBUTED,
         "standing": STANDING_UNKNOWN, WAR_RELATED: True}
        for _ in range(expected)
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
        target = str(item.get("target", "")).strip().lower()
        standing = str(item.get("standing", "")).strip().lower()
        # An unrecognised value is treated as absent rather than trusted. The
        # bulletin's sections are built from these, so a hallucinated label would
        # move a real strike into the wrong half of the war.
        out[idx] = {
            "actor": actor if actor in valid_actors else UNATTRIBUTED,
            "target": target if target in valid_actors else UNATTRIBUTED,
            "standing": standing if standing in valid_standing else STANDING_UNKNOWN,
            # Only an explicit false drops an event; a missing or unreadable value
            # keeps it. Same asymmetry as above, for the same reason.
            WAR_RELATED: item.get(WAR_RELATED) is not False,
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
                # ~50 tokens per item for {"n":N,"actor":"...","standing":"..."},
                # plus headroom for a reasoning preamble the bulk slots may emit.
                max_tokens=50 * len(chunk) + 512,
            )
            if db_conn is not None:
                log_llm_telemetry(db_conn, result, router, success=True,
                                  purpose="bulletin_direction")
            parsed = _parse_extraction(result.get("content", ""), len(chunk))
        except Exception:
            logger.warning("Direction extraction failed for a batch of %d; those "
                           "events stay unattributed", len(chunk), exc_info=True)
            parsed = [{"actor": UNATTRIBUTED, "target": UNATTRIBUTED,
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
    """
    actor = event.get("actor", UNATTRIBUTED)
    if actor in (UNATTRIBUTED, OTHER_SIDE):
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
                "ulke": ev.get("country_iso"),
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
        "- Her bölüm başlığını TAMAMI BÜYÜK HARF tek satır olarak yaz.",
        "- İlk bölüm YÖNETİCİ ÖZETİ olsun: 2-3 paragraf, madde işareti yok.",
        "- Bir olay kümesinin yerini kısa bir satır olarak yaz (nokta ile bitmesin).",
        "- Ayrıntıları '- ' ile başlayan maddeler halinde yaz.",
        "",
        "KURALLAR:",
        "- Saat verme. Elimizde olayların saati YOK; 'akşam saatlerinde' gibi "
        "ifadeler de uydurmadır. Yalnız verilen pencereye atıf yap.",
        "- Her olayın failini ve durumunu yaz. 'Tek taraflı iddia' veya "
        "'İddia edildi, yalanlandı' olan bir olayı ASLA gerçekleşmiş gibi anlatma; "
        "kimin iddia ettiğini söyle.",
        "- Fail alanı 'başlıkta adı geçen taraf' olan olaylarda faili başlıktan "
        "oku ve adıyla yaz.",
        "- Fail alanı 'belirsiz' olan olaylarda kimseye fail atfetme; olayı "
        "failsiz anlat.",
        "- Veri alanlarını olduğu gibi cümleye kopyalama; hepsi Türkçe "
        "yazılacak.",
        "- Verideki sayıları değiştirme, yuvarlama, toplama.",
        "",
        "VERİ:",
        json.dumps(payload, ensure_ascii=False, indent=1),
    ])


def build_bulletin(db_conn, router: LLMRouter, window_start: datetime,
                   window_end: datetime, max_tokens: int = 6000) -> Dict[str, Any]:
    """Fetch, attribute, group and narrate the theatre bulletin.

    Returns the narrative plus the grouped sections, so the caller can render and
    dispatch without re-deriving either.
    """
    events = fetch_theatre_events(db_conn, window_start, window_end)
    if not events:
        logger.info("Iran bulletin: no theatre events in window")
        return {"events": [], "sections": group_into_sections([]), "narrative": "",
                "status": "empty"}

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

    result = call_llm(
        router=router,
        prompt=_narrative_prompt(sections, window_start, window_end),
        system_prompt=_NARRATIVE_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        # Prose, never JSON: a reasoning model asked for JSON here returns the
        # narrative wrapped in a string field and the renderer sees one long line.
        json_mode=False,
    )
    if db_conn is not None:
        log_llm_telemetry(db_conn, result, router, success=True,
                          purpose="bulletin_narrative")
    return {"events": events, "sections": sections,
            "narrative": result.get("content", ""), "status": "ok",
            "model": result.get("model")}
