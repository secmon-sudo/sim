"""SIM — weekly vocabulary audit: measure what the semantic gates throw away.

Why this exists
---------------
The gates that decide what enters this pipeline are hand-maintained regex
lexicons, and every gap in them is invisible until somebody trips over it. The
record is not short:

  * the prescreen scored on noun phrases, so "drones ATTACKED" matched nothing
    and real attacks were archived without an LLM ever seeing them (2026-08-11);
  * "kill" was missing from the same vocabulary — 89 mass-casualty attacks in 14
    days scored 0 (2026-08-17);
  * the flight-disruption gate had "disruption" but not "disrupted", "delayed" or
    "diverted", so an airport security breach that grounded seven flights was not
    an aviation disruption at all (2026-08-23).

All three were found by accident, months apart, by someone reading output for an
unrelated reason. This script turns that into a measurement: sample what each gate
REJECTED, ask a cheap model whether the rejection was right, and record the miss
rate as telemetry so the next gap shows up as a number instead of a coincidence.

What it audits
--------------
  keyword_filter  _matches_security_keywords, over the configured publisher feeds
                  and standing news queries — exactly where Pass A applies it.
  noise_filter    is_noise, over everything Pass A fetched.
  prescreen       deterministic_relevance, sampled from events Pass C archived
                  without spending an LLM call on them.

The first two need a fetch, because a rejected item is never stored — that is the
whole problem. The fetch is the same one Pass A does (RSS, free) and nothing is
written to `events`: this script is read-only apart from its telemetry row.

What it does NOT audit: the age filter, which runs inside fetch_rss_feed and so
removes stale items before this script ever sees them, and the dedup gates, whose
rejections are a similarity judgement rather than a vocabulary one. This is a
vocabulary audit; those need their own measurement.

Usage
-----
    python -m scripts.vocab_audit                  # full audit, writes telemetry
    python -m scripts.vocab_audit --dry-run        # print only
    python -m scripts.vocab_audit --samples 20 --gate prescreen
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.core.llm_client import call_llm
from src.core.llm_router import LLMRouter, build_bulk_router
from src.pipeline.ingest_filters import canonicalize_text, is_noise
from src.pipeline.ingest_queries import build_search_queries
from src.pipeline.ingest_sources import fetch_rss_feed
from src.pipeline.pass_a_ingest import SETTINGS, _matches_security_keywords
from src.pipeline.pass_c_classify import deterministic_relevance
from src.services.ops_notifier import send_ops_alert
from src.services.supabase_client import close_pool, get_connection, put_connection

logger = logging.getLogger(__name__)

GATE_KEYWORD = "keyword_filter"
GATE_NOISE = "noise_filter"
GATE_PRESCREEN = "prescreen"
GATES = (GATE_KEYWORD, GATE_NOISE, GATE_PRESCREEN)

# Sampled per gate. Small on purpose: the judge is a model, the answer is a rate,
# and a rate from 40 draws is already enough to see a 20% miss rate. The cost of
# being wrong here is a week's delay, not a bad alert.
DEFAULT_SAMPLES = 40

# Miss rate that turns the audit into an ops page. Nothing rejects perfectly — the
# gates exist to be strict — so the question is whether the rate MOVED, and 10%
# of a 40-draw sample (4 real misses) is where a vocabulary gap starts costing
# real events every day.
DEFAULT_ALERT_THRESHOLD = 0.10

# Queries per audit, mirroring Pass A's own ceiling.
MAX_QUERIES = 50

_JUDGE_BATCH = 10

_JUDGE_SYSTEM = (
    "Sen bir güvenlik istihbaratı derleme sisteminin kapsam denetçisisin. Sana "
    "haber BAŞLIKLARI verilecek. Her biri için tek bir soruyu cevaplayacaksın: "
    "bu başlık, sivil havacılığı veya seyahat güvenliğini ilgilendirebilecek bir "
    "GÜVENLİK OLAYINI haber veriyor mu?\n\n"
    "KAPSAM İÇİ (evet): silahlı saldırı, çatışma, füze/İHA/roket saldırısı, "
    "bombalama, terör eylemi, rehin alma, uçak kaçırma, hava sahası ihlali, "
    "güvenlik kaynaklı havalimanı/uçuş kesintisi, askeri operasyon, ayaklanma ve "
    "şiddetli gösteri, toplu ölümlü olay, resmi seyahat uyarısı.\n\n"
    "KAPSAM DIŞI (hayır): spor, magazin, ekonomi/piyasa haberi, teknik arıza veya "
    "hava muhalefeti kaynaklı uçuş gecikmesi, sıradan trafik kazası, yorum/analiz "
    "yazısı, ürün tanıtımı, siyasi demeç, geçmiş bir olayın yıldönümü derlemesi.\n\n"
    "Emin değilsen HAYIR de: bu denetim, kapının kaçırdığı GERÇEK olayları saymak "
    "içindir, şüphelileri değil.\n\n"
    "ÇIKTI: yalnızca JSON dizisi, her öğe {\"i\": <sıra numarası>, \"kapsam\": "
    "true|false}. Açıklama, başka metin yok."
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _item_text(item: Dict[str, Any]) -> str:
    return f"{item.get('title') or ''} {item.get('description') or ''}".strip()


def collect_ingest_rejections(db_conn, max_queries: int = MAX_QUERIES) -> Dict[str, List[Dict[str, Any]]]:
    """Re-run Pass A's fetch and record what each ingest gate refused.

    Faithful to production in WHERE each gate applies: the keyword filter only
    guards the configured feeds (query results are pre-filtered by the query
    itself), while the noise filter sees everything. Auditing the keyword gate
    over query results would manufacture misses that production never makes.
    """
    rejected: Dict[str, List[Dict[str, Any]]] = {GATE_KEYWORD: [], GATE_NOISE: []}
    seen_urls: set[str] = set()
    # fetch_rss_feed increments counters on this without creating them — Pass A
    # hands it a fully seeded stats dict. A plain {} raises KeyError mid-parse and
    # the feed's whole item list is lost inside the caller's except: measured on
    # the first live run, BBC Middle East returned nothing for exactly this reason.
    stats: Dict[str, int] = defaultdict(int)

    def _note(item: Dict[str, Any], gate: str) -> None:
        url = item.get("link") or ""
        if url and url in seen_urls:
            return
        seen_urls.add(url)
        rejected[gate].append({"title": (item.get("title") or "").strip(),
                               "url": url, "gate": gate})

    query_items: List[Dict[str, Any]] = []
    for query_info in build_search_queries(db_conn)[:max_queries]:
        try:
            query_items.extend(fetch_rss_feed(query_info, is_direct_url=False, stats=stats))
        except Exception:
            logger.exception("Audit: query feed failed, continuing")

    configured = (SETTINGS.get("sources", {}).get("publisher_feeds", [])
                  + SETTINGS.get("sources", {}).get("news_queries", []))
    feed_items: List[Dict[str, Any]] = []
    for feed_url in configured:
        try:
            items = fetch_rss_feed(feed_url, is_direct_url=True, stats=stats)
        except Exception:
            logger.exception("Audit: configured feed failed, continuing")
            continue
        for item in items:
            if _matches_security_keywords(item.get("title", ""), item.get("description", "")):
                feed_items.append(item)
            else:
                _note(item, GATE_KEYWORD)

    for item in query_items + feed_items:
        if is_noise(canonicalize_text(_item_text(item))):
            _note(item, GATE_NOISE)

    logger.info("Audit: fetched %d query items + %d feed items; rejected %d keyword, %d noise",
                len(query_items), len(feed_items),
                len(rejected[GATE_KEYWORD]), len(rejected[GATE_NOISE]))
    return rejected


def collect_prescreen_rejections(db_conn, days: int = 7) -> List[Dict[str, Any]]:
    """Events Pass C archived on the prescreen's word alone, no LLM call spent.

    These ARE stored, so no fetch is needed — but the same blindness applies, and
    two of the three incidents in the module docstring were exactly here.

    The score is RE-COMPUTED with today's vocabulary rather than read from the
    stored prescreen verdict: a row archived last week under the old lexicon that
    would score today is a gap already closed, not one to report again.
    """
    # llm_raw_output IS NULL is the marker: no LLM response was ever received for
    # this row. NOT llm_parsed_output — the prescreen writes its own verdict there
    # ({"prescreen": {"score": 0, ...}}), so that column is never null and the
    # first version of this query returned zero rows every time, which would have
    # shown up as a permanently healthy gate.
    rows = db_conn.execute(
        """SELECT source_title, source_url, canonical_text
             FROM events
            WHERE status = 'archived'
              AND llm_raw_output IS NULL
              AND ingested_at > NOW() - (%s * INTERVAL '1 day')
              AND source_title IS NOT NULL""",
        (days,),
    ).fetchall()
    out = []
    for title, url, text in rows:
        # Confirm against the live function rather than trusting the status: a row
        # can be archived for reasons that have nothing to do with vocabulary.
        if deterministic_relevance(title or "", text or "").get("score", 0) <= 0:
            out.append({"title": (title or "").strip(), "url": url or "",
                        "gate": GATE_PRESCREEN})
    logger.info("Audit: %d archived rows, %d of them scored 0 by the prescreen",
                len(rows), len(out))
    return out


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def _parse_verdicts(content: str, expected: int) -> Dict[int, bool]:
    """Map index -> in-scope, from the model's JSON array."""
    match = _JSON_ARRAY_RE.search(content or "")
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return {}
    out: Dict[int, bool] = {}
    for entry in parsed if isinstance(parsed, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("i"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < expected:
            out[idx] = bool(entry.get("kapsam"))
    return out


def judge_headlines(router: LLMRouter, headlines: Sequence[str],
                    batch_size: int = _JUDGE_BATCH) -> List[Optional[bool]]:
    """One verdict per headline; None where the model did not answer.

    Unanswered stays None rather than defaulting either way — a parse failure is
    missing data, and folding it into "correctly rejected" would quietly flatter
    every gate.
    """
    verdicts: List[Optional[bool]] = [None] * len(headlines)
    for start in range(0, len(headlines), batch_size):
        chunk = list(headlines[start:start + batch_size])
        prompt = "\n".join(f"{i}. {h[:200]}" for i, h in enumerate(chunk))
        try:
            result = call_llm(router, prompt, _JUDGE_SYSTEM, max_tokens=512, json_mode=False)
        except Exception:
            logger.exception("Audit: judge call failed for batch at %d", start)
            continue
        for idx, verdict in _parse_verdicts(result.get("content") or "", len(chunk)).items():
            verdicts[start + idx] = verdict
    return verdicts


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def sample(items: Sequence[Dict[str, Any]], size: int, seed: str) -> List[Dict[str, Any]]:
    """Deterministic sample: the same week's audit is reproducible."""
    if len(items) <= size:
        return list(items)
    return random.Random(seed).sample(list(items), size)


def audit_gate(router: LLMRouter, gate: str, items: Sequence[Dict[str, Any]],
               samples: int, seed: str) -> Dict[str, Any]:
    drawn = sample(items, samples, f"{seed}:{gate}")
    verdicts = judge_headlines(router, [d["title"] for d in drawn])
    judged = [(d, v) for d, v in zip(drawn, verdicts) if v is not None]
    misses = [d for d, v in judged if v]
    return {
        "gate": gate,
        "rejected_total": len(items),
        "sampled": len(drawn),
        "judged": len(judged),
        "misses": len(misses),
        "miss_rate": round(len(misses) / len(judged), 3) if judged else None,
        "examples": [m["title"][:160] for m in misses[:8]],
        "example_urls": [m["url"] for m in misses[:8] if m.get("url")],
    }


def run_audit(db_conn, router: LLMRouter, samples: int = DEFAULT_SAMPLES,
              days: int = 7, gates: Sequence[str] = GATES) -> Dict[str, Any]:
    seed = datetime.now(timezone.utc).strftime("%G-W%V")
    results = []
    if GATE_KEYWORD in gates or GATE_NOISE in gates:
        rejected = collect_ingest_rejections(db_conn)
        for gate in (GATE_KEYWORD, GATE_NOISE):
            if gate in gates:
                results.append(audit_gate(router, gate, rejected[gate], samples, seed))
    if GATE_PRESCREEN in gates:
        results.append(audit_gate(router, GATE_PRESCREEN,
                                  collect_prescreen_rejections(db_conn, days),
                                  samples, seed))
    return {"week": seed, "days": days, "samples": samples, "gates": results}


def record_audit(db_conn, report: Dict[str, Any]) -> None:
    db_conn.execute(
        "INSERT INTO system_telemetry(event_type, value_json) VALUES ('vocab_audit', %s)",
        (json.dumps(report, ensure_ascii=False),),
    )


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"Sözlük denetimi {report['week']}"]
    for gate in report["gates"]:
        rate = gate["miss_rate"]
        rate_txt = "veri yok" if rate is None else f"%{rate * 100:.0f}"
        lines.append(
            f"• {gate['gate']}: {gate['misses']}/{gate['judged']} kaçırma ({rate_txt}) "
            f"— reddedilen toplam {gate['rejected_total']}")
        for example in gate["examples"][:3]:
            lines.append(f"    ↳ {example}")
    return "\n".join(lines)


def alerting_gates(report: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
    return [g for g in report["gates"]
            if g["miss_rate"] is not None and g["miss_rate"] >= threshold]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--days", type=int, default=7,
                        help="Window for the prescreen gate (stored rows)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_ALERT_THRESHOLD)
    parser.add_argument("--gate", action="append", choices=list(GATES),
                        help="Audit only this gate (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the report; write no telemetry and page nobody")
    args = parser.parse_args()

    conn = get_connection()
    try:
        report = run_audit(conn, build_bulk_router(), samples=args.samples,
                           days=args.days, gates=args.gate or GATES)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print()
        print(format_report(report))
        if args.dry_run:
            return 0
        record_audit(conn, report)
        breached = alerting_gates(report, args.threshold)
        if breached:
            send_ops_alert(
                format_report(report)
                + f"\n\nEşik %{args.threshold * 100:.0f} aşıldı: "
                + ", ".join(g["gate"] for g in breached)
                + "\nSözlüğü genişletmeden önce örnekleri oku — kapı sıkı olduğu "
                  "için var.",
                title="SIM SÖZLÜK DENETİMİ")
    finally:
        put_connection(conn)
        close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
