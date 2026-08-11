"""
SIM — Alert Engine
Blueprint V20.1 §5.2 + §5.3

3-tier alert evaluation (WATCH / ALERT / CRITICAL) with
composite suppression key to prevent duplicate notifications.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.core.geo import geo_key

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as _f:
    _SETTINGS = json.load(_f)


@dataclass
class AlertTier:
    name: str
    color: str
    notify_channels: list[str]


TIERS = {
    "CRITICAL": AlertTier("CRITICAL", "#DC2626", ["telegram"]),
    "ALERT":    AlertTier("ALERT",    "#EA580C", ["telegram"]),
    "WATCH":    AlertTier("WATCH",    "#CA8A04", ["telegram"]),
}

# Official government travel advisories are actionable at the COUNTRY level, so they
# never carry an airport anchor and the standard anchor/time gates would drop them.
# They are pre-filtered upstream (Level 3-4 / "do not travel", curated high-risk
# countries) so gate them on severity alone.
ADVISORY_EVENT_TYPES = {"travel_advisory", "travel_ban"}

# A standing country-level advisory is not a breaking incident, so it must not outrank
# one. At the old floor of 55 every single advisory reached ALERT (88 of 88 over 7 days)
# while 108 of 252 confirmed mass-casualty-grade fresh events sat at WATCH — a Level-2
# Belgium advisory literally outranking a suicide bombing that killed 14.
#
# Advisory severity is effectively three discrete values (60 / 75 / 90), so 75 is the
# natural seam: it keeps the Level-3-4 "do not travel" advisories the docstring above
# describes at ALERT and drops the routine Level-2 bulk to WATCH.
ADVISORY_ALERT_SEVERITY_MIN = _SETTINGS.get("alert", {}).get(
    "advisory_alert_severity_min", 75
)

# Severity floor: a confirmed high-severity event that is happening NOW cannot rank
# below WATCH, whatever the corroboration signals say. system_confidence measures how
# well-sourced a report is, which is a fair reason to rank one incident above another
# but not a fair reason to rank a mass-casualty attack below a travel advisory.
# Deliberately does NOT reach CRITICAL — that tier still has to be earned on location
# and confidence.
SEVERITY_ALERT_FLOOR = _SETTINGS.get("alert", {}).get("severity_alert_floor", 90)

# A time_certainty that pins the event to roughly "now" — the only values that make
# an event newsworthy as a page rather than as background.
FRESH_TIME_CERTAINTY = ["same_day", "previous_day"]

# Tier thresholds live in config/settings.json -> alert.tiers. They used to be
# duplicated here as literals, which meant editing the config changed nothing —
# a silent trap for anyone tuning alert volume.
#
# Each tier is evaluated in order and takes the first match:
#   severity_min / confidence_min   — inclusive floors
#   require_location                — event must resolve to a place (IATA anchor or coords)
#   require_location_or_fresh       — ...or, failing that, carry a fresh time_certainty
#   time_certainty_include          — require time_certainty to be in this list
#   time_certainty_exclude          — reject when time_certainty is in this list
#   anchor_confidence               — allowed values (omit to accept any)
#
# Calibration note (7 Aug 2026). The original V19 gates keyed ALERT/CRITICAL on
# anchor_confidence and on time_certainty != unknown. Measured against a week of
# real corpus, both signals turned out to be near-constant and the conjunction was
# unsatisfiable — 1455 of 1469 alerted events were anchor LOW (the anchor level
# really means "an IATA airport was resolved", which conflict reporting rarely
# offers) and 86% of all scored events carry time_certainty='unknown'. So the
# gates never fired and a fallback in dispatch_alert force-labelled everything
# ALERT, which is why CRITICAL had never once fired. The gates below are set from
# the observed distributions (confidence p50=0.41, p90=0.62, p95=0.66, max=0.78)
# and treat "we know WHERE it happened" as a first-class signal alongside "we know
# WHEN": ALERT needs one of the two, CRITICAL needs both.
_DEFAULT_TIER_RULES = {
    "CRITICAL": {"severity_min": 80, "confidence_min": 0.62,
                 "require_location": True,
                 "time_certainty_include": FRESH_TIME_CERTAINTY},
    "ALERT": {"severity_min": 65, "confidence_min": 0.50,
              "require_location_or_fresh": True},
    "WATCH": {"severity_min": 45, "confidence_min": 0.40,
              "time_certainty_include": FRESH_TIME_CERTAINTY},
}

# Severity descends across tiers, so evaluation order is fixed rather than taken
# from dict order in the config — a reordered config file must not silently make
# every CRITICAL event fire as WATCH.
TIER_ORDER = ("CRITICAL", "ALERT", "WATCH")

_TIER_RANK = {name: i for i, name in enumerate(reversed(TIER_ORDER), start=1)}


def tier_rank(tier: str | None) -> int:
    """Order tiers so escalations are comparable. None/unrecognised sorts below WATCH."""
    return _TIER_RANK.get(tier or "", 0)


def _tier_rules() -> dict:
    """Merge configured tier rules over the built-in defaults, per tier."""
    configured = _SETTINGS.get("alert", {}).get("tiers", {})
    rules = {}
    for name in TIER_ORDER:
        merged = dict(_DEFAULT_TIER_RULES[name])
        merged.update(configured.get(name) or {})
        rules[name] = merged
    return rules


TIER_RULES = _tier_rules()


# ── Aftermath reports ──────────────────────────────────────────────────────
#
# A page is a claim that something is happening NOW. Neither of the two gates that
# are supposed to enforce that can tell a fresh incident from a report ABOUT an old
# one:
#
#   - severity comes from the underlying subject, so "Canyon Park West Shooting: one
#     week later" scores 95 exactly like the shooting did;
#   - time_certainty ends up "same_day"/"previous_day" because occurred_at_est falls
#     back to publication time — measured 2026-08-10, occurred_at_est equals
#     published_at to the millisecond in 93% of "previous_day" events. So "fresh"
#     really means "published today", which a week-old retrospective satisfies.
#
# So severity ≥ 90 + fresh — the SEVERITY_ALERT_FLOOR path that produced 61% of the
# ALERTs in the 2 days to 2026-08-10 — promotes live-blog roundups, anniversary
# pieces, prosecutions and rescue-completed stories into pages.
#
# Confidence cannot separate them: over that corpus the junk (0.33-0.41) sits in the
# same band as real incidents (Colombia car bomb 0.41, Sumy train strike 0.41), so a
# confidence guard on the floor would cut both. The distinguishing signal is in the
# headline's SHAPE, and these patterns were chosen by measuring precision against 724
# real alerts rather than by intuition — an earlier, looser version matched publisher
# taglines ("… | Shafaq News | Latest breaking news") and ordinary adjectives ("killed
# in latest strikes"), i.e. mostly real incidents.
#
# Gated events are still scored, stored, clustered and reported — only the page is
# withheld. And CRITICAL is exempt (see evaluate_alert_tier): when a roundup is the
# only carrier of a genuinely major development, losing the page is worse than the
# noise.

# The classifier's verdict on what the ARTICLE is (pass_c's report_kind). These three
# classes report on something already in the news rather than something happening now,
# so they cannot carry a page below CRITICAL.
#
# Only these three veto: `new_incident` is also what an absent, unparseable or invented
# value resolves to (see _safe_report_kind), which keeps the failure direction right —
# a degraded classification lets alerts through rather than silencing them. That is why
# the check is written as membership in this set and never as `!= "new_incident"`.
REPORT_KIND_NOT_NEWS = frozenset({"followup", "roundup", "commentary"})

# Publisher/section suffix: "Headline - Reuters", "Headline | Shafaq News | Latest…".
# Stripped before the desk-label test so a publisher's tagline cannot trip it.
_PUBLISHER_SUFFIX_RE = re.compile(r"\s+[-|–—]\s+[^-|–—]*$")

_AFTERMATH_PATTERNS = (
    # Desk label: the marker sits in LABEL position — at the head of the headline and
    # immediately before the colon, optionally trailed by desk words. This is what
    # separates "Ukraine war latest: …" (a running roundup) from "At least 7 killed in
    # latest Russia-Ukraine strikes" (an incident that happens to use the word).
    # Applied to the publisher-stripped headline.
    ("desk_label",
     re.compile(r"^[^:]{0,70}\b(latest|live|updates?|timeline|recap|roundup)\b"
                r"(\s+(news|live|updates?|blog|coverage|briefing))*\s*:", re.I)),
    # Running war diary: "Ukraine Invasion Day 1,628: 135/151 RU drones neutralized",
    # "Israel-Hamas war, day 700: what we know". Same roundup shape as desk_label but
    # the marker is a day counter rather than a desk word, so the pattern above misses
    # it — measured 2026-08-10, one such headline paged as ALERT. The colon is
    # required: it is what separates the label from "Two killed on day 2 of protests".
    ("war_diary",
     re.compile(r"^[^:]{0,50}\bday\s+\d[\d,]{0,6}\b[^:]{0,20}:", re.I)),
    # "one week later", "2 Years After Kursk Incursion", "three years after".
    ("retrospective",
     re.compile(r"\b(one|two|three|four|five|\d+)\s+"
                r"(week|month|year|day)s?\s+(later|after|since)\b", re.I)),
    ("explainer",
     re.compile(r"\b(what we know|as it happened|key moments|explainer)\b", re.I)),
    ("probe",
     re.compile(r"(expands? probe|probe into|investigation into|inquiry into)", re.I)),
    # The news is the prosecution, not the attack. "charges" needs its object and
    # preposition ("Ecuador charges ex-minister OVER …") because the bare noun is a
    # demolition term — "troops planted charges" is an incident, not a court filing.
    ("judicial",
     re.compile(r"\b(charged|sentenced|convicted|indicted|pleads? guilty|on trial)\b"
                r"|\bcharges\s+(\w+[- ]){1,3}(over|with|in connection)\b", re.I)),
    # The news is that it ENDED — a rescue completed, a hostage freed, a site reopened.
    ("resolution",
     re.compile(r"(safely rescued|released hostage|freed after|reopens? after|to demolish)", re.I)),
)


# Patterns whose marker must sit in LABEL position (at the head, before a colon).
# They are tested against the publisher-stripped headline.
_LABEL_POSITION_PATTERNS = {"desk_label", "war_diary"}


def aftermath_kind(title: str | None) -> str | None:
    """Name the aftermath pattern this headline matches, or None if it reads as news.

    Returns the pattern name (not just a bool) so the suppression is auditable in the
    logs — a silent gate on the alert path is how the last one went unnoticed.
    """
    if not title:
        return None
    stripped = _PUBLISHER_SUFFIX_RE.sub("", title).strip()
    for name, pattern in _AFTERMATH_PATTERNS:
        # Label-position patterns read the publisher-stripped headline so a tagline
        # cannot supply the colon they key on; the rest read the headline as written.
        target = stripped if name in _LABEL_POSITION_PATTERNS else title
        if pattern.search(target):
            return name
    return None


def is_located(event: dict) -> bool:
    """True when the event resolved to a real place.

    Either an IATA anchor (aviation events) or coordinates from the city gazetteer
    (everything else). Pass D resolves both, so this is the honest "we know where
    this happened" test — unlike anchor_confidence, which only ever means "an IATA
    airport matched" and is LOW for ~99% of the corpus.
    """
    return bool(event.get("anchor_name_norm")) or event.get("latitude") is not None


def _matches_tier(rule: dict, sev, conf, anc: str, time_: str, located: bool) -> bool:
    if sev < rule["severity_min"] or conf < rule["confidence_min"]:
        return False
    fresh = time_ in FRESH_TIME_CERTAINTY
    if rule.get("require_location") and not located:
        return False
    if rule.get("require_location_or_fresh") and not (located or fresh):
        return False
    allowed = rule.get("anchor_confidence")
    if allowed is not None and anc not in allowed:
        return False
    excluded = rule.get("time_certainty_exclude")
    if excluded and time_ in excluded:
        return False
    required = rule.get("time_certainty_include")
    if required and time_ not in required:
        return False
    return True


def evaluate_alert_tier(event: dict) -> str | None:
    """
    Evaluate which alert tier an event qualifies for.

    Returns: 'CRITICAL', 'ALERT', 'WATCH', or None
    """
    return evaluate_alert_tier_verbose(event)[0]


def evaluate_alert_tier_verbose(event: dict) -> tuple[str | None, str | None]:
    """Tier plus the name of the gate that vetoed it, or (tier, None).

    The veto reason exists so the article-shape gates are measurable. Both of them turn
    a tier into None, which is indistinguishable in the stored data from an event that
    simply never qualified — so without this, the only evidence either gate does
    anything is a log line, and the only way to tune them is to guess.
    """
    sev = event.get("severity_score", 0)
    conf = event.get("system_confidence", 0.0)
    anc = event.get("anchor_confidence", "LOW")
    time_ = event.get("time_certainty") or "unknown"
    located = is_located(event)

    # Travel advisory path — country-level official warning, no airport anchor and its
    # "time" is the standing advisory date, so bypass the anchor/time gates and key on
    # severity only (already pre-filtered to Level 3-4 / "do not travel" upstream).
    if event.get("event_type") in ADVISORY_EVENT_TYPES:
        return ("ALERT" if sev >= ADVISORY_ALERT_SEVERITY_MIN else "WATCH"), None

    tier = None
    for name in TIER_ORDER:
        if _matches_tier(TIER_RULES[name], sev, conf, anc, time_, located):
            tier = name
            break

    # Severity floor — see SEVERITY_ALERT_FLOOR. Applied after the ladder so it can
    # only ever raise a tier to ALERT, never lower one or manufacture a CRITICAL.
    if (sev >= SEVERITY_ALERT_FLOOR
            and time_ in FRESH_TIME_CERTAINTY
            and tier_rank(tier) < tier_rank("ALERT")):
        tier = "ALERT"

    # Article-shape gates — see _AFTERMATH_PATTERNS and REPORT_KIND_NOT_NEWS. Last, so
    # they can veto both the ladder and the floor: those two decide whether the SUBJECT
    # is alert-worthy, and these decide whether the ARTICLE is reporting it as news.
    # CRITICAL is deliberately exempt from both — a roundup is sometimes the only carrier
    # of a genuinely major development, and missing that costs more than the noise it
    # lets through.
    #
    # Two gates rather than one because they read different evidence and fail
    # differently. The title patterns are free and catch desk labels the classifier
    # never sees the shape of; report_kind is the classifier's judgement of the whole
    # article, which is the only thing that separates "Police release video of the mass
    # shooting" from the shooting. Measured 2026-08-11 over 7 days: the severity floor
    # produced 296 of 561 ALERT-tier events with no confidence requirement at all, and
    # the junk among them was overwhelmingly this shape — arrests, charges, reopenings,
    # released footage, condemnations, "Day 1,625" war diaries.
    if tier is not None and tier != "CRITICAL":
        kind = aftermath_kind(event.get("source_title"))
        if kind:
            logger.info("Alert suppressed as %s report (would have been %s): %.80s",
                        kind, tier, event.get("source_title") or "")
            return None, "aftermath_title"
        report_kind = event.get("report_kind")
        if report_kind in REPORT_KIND_NOT_NEWS:
            logger.info("Alert suppressed as %s article (would have been %s): %.80s",
                        report_kind, tier, event.get("source_title") or "")
            return None, f"report_kind_{report_kind}"

    return tier, None


def build_suppression_key(event: dict) -> str:
    """
    Build composite key to prevent duplicate alerts.
    Uses IATA code (not raw text) to avoid key fragmentation.

    Deliberately carries NO severity component. It used to bucket severity by tens,
    which meant one storyline paged once per bucket its members happened to land in:
    measured 2026-08-10 (run 20260810T105738), storyline 37d41620 sent two cards
    eight seconds apart because one member scored 100 and another 90. Severity is a
    per-report estimate of the SAME incident, so it must not fragment the key.

    A storyline that genuinely gets worse still re-pages: suppression_blocks lets a
    card through when its tier outranks the tier the key last fired at. Tier is the
    honest "this changed" signal; a severity bucket is not.

    With no storyline_id there is no evidence two events are the same incident, so
    the key falls back to the event's own id rather than collapsing every unclustered
    alert into one shared "no_storyline|UNKNOWN" slot. (Measured over 7 days: 0 of
    1148 tiered events lacked a storyline, so this is a guard, not a live path.)
    """
    storyline_id = event.get("storyline_id")
    if not storyline_id:
        return f"no_storyline|{event.get('id') or id(event)}"
    return "|".join([
        str(storyline_id),
        event.get("anchor_name_norm") or "UNKNOWN",
    ])


def build_geo_suppression_key(event: dict) -> str | None:
    """Storyline-independent suppression fingerprint (safety net).

    The primary key keys off storyline_id, so when the same real-world event is split
    across several storyline_ids (paraphrased hints that fall below the Jaccard
    threshold), each fragment produces a different primary key and every source pages
    separately. This fingerprint drops storyline_id and keys off coarse geography
    instead — country + resolved location + severity bucket — so near-identical alerts
    within the suppression TTL collapse regardless of storyline fragmentation.

    Location resolution prefers the precise IATA anchor, then a coarse geo_key derived
    from the raw location text. Returns None when no usable location is known (so the
    net is never so broad that it mutes unrelated same-country alerts).

    Severity is excluded for the same reason as in build_suppression_key: two reports
    of one incident routinely differ by a bucket, and a net with a hole that shape
    catches nothing.
    """
    loc = event.get("anchor_name_norm") or geo_key(
        event.get("anchor_name_raw"), event.get("country_iso")
    )
    if not loc or loc == "UNKNOWN":
        return None
    return "|".join([
        "geofp",
        event.get("country_iso") or "??",
        loc,
    ])


def recent_paged_alerts(db_conn, country_iso: str | None, exclude_event_id: str | None,
                        limit: int = 6) -> list[dict]:
    """Alerts that already paged and are still inside their suppression window.

    The candidate set for the dispatch-time duplicate adjudicator (see
    adjudicate_duplicate_page). Scoped to live suppression claims, so it stays small
    enough to hand to a model whole: over the 11 Aug 2026 sample the busiest country held
    5 live claims.

    Scoped to one country when the event has one. When it does not — 57 of 922 tiered
    events over the week to 2026-08-11 carry no country_iso — the candidates are the
    most recent live claims regardless of country, because the alternative was returning
    nothing and skipping the duplicate check entirely for those 6%. A broader candidate
    list is safe here in a way a broader suppression KEY would not be: nothing is muted
    without the model affirming the incidents are the same, and it fails toward sending.

    An event holds one claim per suppression key (primary + geo), so rows are folded back
    to one row per event, carrying the HIGHEST tier any of its keys fired at. Ordering
    tier by array_position rather than by the tier string keeps 'CRITICAL' above 'WATCH';
    alphabetically it is not.
    """
    try:
        rows = db_conn.execute(
            """SELECT e.id,
                      (ARRAY['WATCH','ALERT','CRITICAL'])[
                          MAX(array_position(ARRAY['WATCH','ALERT','CRITICAL'],
                                             s.alert_tier))] AS alert_tier,
                      e.source_title, e.storyline_hint,
                      e.anchor_name_raw, e.anchor_name_norm,
                      MAX(s.expires_at) AS last_expiry
               FROM alert_suppression s
               JOIN events e ON e.id = s.event_id
               WHERE s.expires_at > NOW()
                 AND (%s::text IS NULL OR e.country_iso = %s)
                 AND (%s::uuid IS NULL OR e.id <> %s::uuid)
               GROUP BY e.id, e.source_title, e.storyline_hint,
                        e.anchor_name_raw, e.anchor_name_norm
               ORDER BY last_expiry DESC
               LIMIT %s""",
            (country_iso, country_iso, exclude_event_id, exclude_event_id, limit),
        ).fetchall()
    except Exception:
        logger.exception("Failed to fetch recently paged alerts; treating as none")
        return []
    return [
        {
            "id": str(r[0]),
            "alert_tier": r[1],
            "source_title": r[2],
            "storyline_hint": r[3],
            "anchor_name_raw": r[4],
            "anchor_name_norm": r[5],
        }
        for r in rows
        # A claim whose tier never resolved carries no rank to compare against, so it
        # cannot mute anything — leaving it in would only spend prompt space.
        if r[1]
    ]


def active_suppression_tier(db_conn, suppression_key: str) -> str | None:
    """The tier this key last fired at within the TTL, or None if it is free."""
    row = db_conn.execute(
        """SELECT alert_tier FROM alert_suppression
           WHERE suppression_key = %s AND expires_at > NOW()""",
        (suppression_key,),
    ).fetchone()
    return row[0] if row else None


def suppression_blocks(db_conn, suppression_key: str, tier: str) -> bool:
    """True when an existing claim on this key should mute a card at `tier`.

    A claim only mutes cards at or below the tier it fired at, so a storyline (or a
    location, via the geo fingerprint) that genuinely gets WORSE still pages once at
    the higher tier. That escalation allowance is the only reason the same key may
    fire twice — the keys themselves deliberately carry no severity component, because
    severity is a per-report estimate of one incident and bucketing it produced
    duplicate cards (see build_suppression_key).
    """
    fired_at = active_suppression_tier(db_conn, suppression_key)
    if fired_at is None:
        return False
    return tier_rank(tier) <= tier_rank(fired_at)


def record_suppression(db_conn, suppression_key: str, tier: str, event_id: str, ttl_hours: int = 4):
    """Record a suppression entry so future duplicates are muted.

    On conflict the stored tier is RAISED to the tier that just fired. Without that the
    row kept the tier of its first claim forever, so the one-off escalation allowance in
    suppression_blocks became unlimited: a storyline that paged once at ALERT then paged
    at CRITICAL on every subsequent run, because the stored tier never stopped being
    ALERT. Measured 2026-08-11 across runs 1453/1456/1458.

    The tier is only ever raised, never lowered. record_suppression is reached only when
    suppression_blocks let the card through, which already implies the new tier outranks
    the stored one — but keying the write off an explicit comparison rather than that
    invariant keeps a future caller from silently re-opening the same hole. Tier order
    matches tier_rank(); an unrecognised stored tier sorts below WATCH, so it is replaced.
    """
    db_conn.execute(
        """INSERT INTO alert_suppression (suppression_key, alert_tier, event_id, expires_at)
           VALUES (%s, %s, %s, NOW() + (%s * INTERVAL '1 hour'))
           ON CONFLICT (suppression_key) DO UPDATE SET
               expires_at = EXCLUDED.expires_at,
               alert_tier = CASE
                   WHEN COALESCE(array_position(ARRAY['WATCH','ALERT','CRITICAL'],
                                                EXCLUDED.alert_tier), 0)
                      > COALESCE(array_position(ARRAY['WATCH','ALERT','CRITICAL'],
                                                alert_suppression.alert_tier), 0)
                   THEN EXCLUDED.alert_tier
                   ELSE alert_suppression.alert_tier
               END,
               event_id = CASE
                   WHEN COALESCE(array_position(ARRAY['WATCH','ALERT','CRITICAL'],
                                                EXCLUDED.alert_tier), 0)
                      > COALESCE(array_position(ARRAY['WATCH','ALERT','CRITICAL'],
                                                alert_suppression.alert_tier), 0)
                   THEN EXCLUDED.event_id
                   ELSE alert_suppression.event_id
               END""",
        (suppression_key, tier, event_id, ttl_hours),
    )
    db_conn.commit()
