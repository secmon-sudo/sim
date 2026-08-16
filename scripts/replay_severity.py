"""Replay severity scoring under a proposed catalog rescale, and report the delta.

Why this exists
---------------
severity_score is saturated: over the 7 days to 2026-08-16, 1411 of ~2200 scored
events sat at EXACTLY 100 and another 517 at 95. Two event types account for most of
it — missile_strike has a catalog base of 100 (the ceiling itself) and
drone_attack_critical_infra 90, and all 948 of their events landed on 100. With bases
that high the +70 of available bonuses has nowhere to go, so min(score, 100) discards
the entire evidence contribution.

The consequence is not cosmetic. Measured over the same window:

  * 0 of 924 ALERTs failed severity_min (65). Severity never blocks anything.
  * 687 of 924 (74%) sit BELOW the ladder's confidence floor of 0.50 — median
    system_confidence 0.450 — and paged anyway via SEVERITY_ALERT_FLOOR (>=90 + fresh).
  * Only 194 (21%) satisfy the ALERT ladder on their own merits.

So the saturated floor is what actually admits three quarters of the alert volume, and
lowering the bases would take that away. That is a large production change, and its
magnitude cannot be derived from stored rows: severity is persisted post-clip and
anchor_data["confidence"] is never stored, so the inputs cannot be reconstructed after
the fact (an attempt reproduced only 1261 of 5469 stored values). This script recomputes
from the real inputs with the real function instead of guessing.

Usage
-----
    python -m scripts.replay_severity                  # 7-day window, default proposal
    python -m scripts.replay_severity --days 14 --base-cap 70

Read-only: it never writes to the database.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from src.core.alerts import evaluate_alert_tier
from src.pipeline.pass_d_score import (
    MAX_SEVERITY,
    apply_safety_downrank,
    compute_aviation_bonus,
    compute_severity,
)
from src.services.supabase_client import close_pool, get_connection, put_connection

# The proposal: no event type may start at or near the ceiling on its label alone.
# A cap leaves headroom for proximity (+30), CZIB (+20) and casualties (+20) to
# actually move the score, which is what the formula was always documented to do.
DEFAULT_BASE_CAP = 75


class _CatalogConn:
    """Serves a rescaled severity_base to compute_severity, delegating nothing else.

    compute_severity looks the base up through db_conn, so overriding the catalog is
    the whole of the intervention — the scoring logic under test stays untouched.
    """

    def __init__(self, bases: dict[str, int]):
        self.bases = bases

    def execute(self, sql, params=None):
        if "severity_base" in sql and params:
            return _Row(self.bases.get(params[0]))
        return _Row(None)


class _Row:
    def __init__(self, value):
        self.value = value

    def fetchone(self):
        return None if self.value is None else (self.value,)


def load_catalog(conn) -> dict[str, int]:
    rows = conn.execute("SELECT code, severity_base FROM event_type_catalog").fetchall()
    return {r[0]: r[1] for r in rows}


def load_events(conn, days: int) -> list[dict]:
    rows = conn.execute(
        """SELECT e.id, e.event_type, e.severity_score, e.system_confidence,
                  e.anchor_name_norm, e.anchor_confidence, e.latitude,
                  e.time_certainty, e.date_verified, e.alert_tier, e.source_title,
                  e.llm_parsed_output, COALESCE(a.czib_flag, false),
                  e.anchor_name_raw, e.storyline_hint
           FROM events e
           LEFT JOIN anchor_master a ON a.iata_code = e.anchor_name_norm
           WHERE e.ingested_at > NOW() - (%s * INTERVAL '1 day')
             AND e.severity_score IS NOT NULL
             AND e.status IN ('scored', 'reconciled', 'archived')""",
        (days,),
    ).fetchall()

    out = []
    for r in rows:
        parsed = r[11] if isinstance(r[11], dict) else json.loads(r[11] or "{}")
        out.append({
            "id": str(r[0]),
            "event_type": r[1],
            "stored_severity": r[2],
            "system_confidence": r[3] or 0.0,
            "anchor_name_norm": r[4],
            "anchor_confidence": r[5],
            "latitude": r[6],
            "time_certainty": r[7],
            "date_verified": r[8],
            "stored_tier": r[9],
            "source_title": r[10],
            "llm_parsed": parsed,
            "czib_flag": r[12],
            # Both feed compute_aviation_bonus's keyword blob.
            "anchor_name_raw": r[13],
            "storyline_hint": r[14],
        })
    return out


# anchor_data["confidence"] is not persisted, so it is rebuilt from the stored level.
# Only the >= 0.6 boundary matters to compute_severity (PROXIMITY_BONUS), and the
# level brackets it: HIGH is exact/alias (0.8-1.0), MEDIUM is the accepted fuzzy band
# (0.50-0.60), LOW never earned an anchor at all.
_LEVEL_TO_CONFIDENCE = {"HIGH": 0.9, "MEDIUM": 0.55, "LOW": 0.0}


def anchor_data(ev: dict) -> dict:
    return {
        "confidence": _LEVEL_TO_CONFIDENCE.get(ev["anchor_confidence"] or "LOW", 0.0),
        "czib_flag": ev["czib_flag"],
    }


def score(ev: dict, conn) -> int:
    """Mirror score_single_event exactly: catalog + bonuses, THEN aviation, THEN safety.

    The aviation nexus bonus is applied outside compute_severity (pass_d_score line
    824), so a replay that calls compute_severity alone silently under-scores every
    aviation event by up to 15 points.
    """
    anchor = anchor_data(ev)
    sev = compute_severity(ev["event_type"], anchor, conn, ev["llm_parsed"])
    sev = min(sev + compute_aviation_bonus(ev, anchor), MAX_SEVERITY)
    sev, _ = apply_safety_downrank(ev["event_type"], sev, ev["llm_parsed"])
    return sev


def tier_for(ev: dict, severity: int) -> str | None:
    return evaluate_alert_tier({
        "severity_score": severity,
        "system_confidence": ev["system_confidence"],
        "anchor_confidence": ev["anchor_confidence"] or "LOW",
        "time_certainty": ev["time_certainty"] or "unknown",
        "date_verified": ev["date_verified"],
        "event_type": ev["event_type"],
        "anchor_name_norm": ev["anchor_name_norm"],
        "latitude": ev["latitude"],
        "source_title": ev["source_title"],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--base-cap", type=int, default=DEFAULT_BASE_CAP)
    args = ap.parse_args()

    # Use the pipeline's pool: it auto-switches the Supabase pooler to Transaction
    # Mode, which a raw psycopg.connect does not (see backfill_snapshots).
    conn = get_connection()
    try:
        catalog = load_catalog(conn)
        events = load_events(conn, args.days)
    finally:
        put_connection(conn)
        close_pool()

    current = _CatalogConn(catalog)
    proposed = _CatalogConn({k: min(v, args.base_cap) for k, v in catalog.items()})

    # Fidelity check first: if replaying the CURRENT catalog does not reproduce the
    # stored severities, the proposed numbers mean nothing. Report it rather than
    # quietly presenting a delta built on a broken reconstruction.
    reproduced = sum(1 for e in events if score(e, current) == e["stored_severity"])
    print(f"events replayed: {len(events)}  window: {args.days}d  base cap: {args.base_cap}")
    print(f"fidelity (current catalog reproduces stored severity): "
          f"{reproduced}/{len(events)} ({100 * reproduced / max(len(events), 1):.1f}%)\n")

    sev_now = Counter()
    sev_new = Counter()
    tier_now = Counter()
    tier_new = Counter()
    lost, gained = [], []

    for e in events:
        s_now, s_new = score(e, current), score(e, proposed)
        sev_now[s_now] += 1
        sev_new[s_new] += 1

        t_now, t_new = tier_for(e, s_now), tier_for(e, s_new)
        tier_now[t_now or "none"] += 1
        tier_new[t_new or "none"] += 1
        if t_now and not t_new:
            lost.append(e)
        elif t_new and not t_now:
            gained.append(e)

    print(f"severity == 100:  {sev_now[100]} -> {sev_new[100]}")
    print(f"severity >= 90:   {sum(n for s, n in sev_now.items() if s >= 90)} -> "
          f"{sum(n for s, n in sev_new.items() if s >= 90)}")
    print(f"distinct values:  {len(sev_now)} -> {len(sev_new)}\n")

    for tier in ("CRITICAL", "ALERT", "WATCH", "none"):
        print(f"{tier:9s} {tier_now[tier]:6d} -> {tier_new[tier]:6d}")

    print(f"\nlost a tier: {len(lost)}   gained a tier: {len(gained)}")
    print("\nsample of alerts that would STOP paging:")
    for e in lost[:15]:
        print(f"  [{e['stored_tier']}] sev={e['stored_severity']} "
              f"conf={e['system_confidence']:.2f} {(e['source_title'] or '')[:80]}")


if __name__ == "__main__":
    main()
