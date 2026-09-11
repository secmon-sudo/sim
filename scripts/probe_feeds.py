"""SIM — feed probe: test a candidate source where production actually fetches.

Why this exists
---------------
Source lists arrive as research: a table of publishers, each marked "verified,
valid XML, current items". On 2026-09-11 one such list of 36 feeds for the five
countries the recall audit found blind was checked against this pipeline's own
fetcher, and the marks did not survive:

  * four feeds answered 403 (Cloudflare) — including three of Ethiopia's four
    English papers, the country SIM is blindest in;
  * three refused the connection outright;
  * three returned valid XML whose newest item was months or years old
    (L'Observateur Paalga's most recent entry was dated 2013);
  * and of 218 items fetched across all of them, TEN passed the ingest keyword
    gate, four of those from a single English-language feed.

"Valid XML with items in it" is not the question. The question is how many
SECURITY items reach the corpus from where the pipeline runs, and only a probe
can answer that — the same lesson the model probe records for models: do not pick
from the catalogue description, run it.

Run it in Actions, not on a laptop: a 403 is usually about the IP, and the IP that
matters is the runner's.

Verdicts
--------
  ADD      fresh items, and at least one passed the keyword gate
  THIN     fresh items, none of them security — the feed works, the yield is zero
  STALE    items, but nothing inside max_article_age_days
  BLOCKED  403/401/429 — bot protection, not a content problem
  DEAD     no connection, or no items at all

Usage
-----
    python -m scripts.probe_feeds --urls "https://a/feed https://b/rss"
    python -m scripts.probe_feeds --from-config        # re-probe what is configured
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Sequence

import httpx

from src.pipeline.ingest_filters import _matches_security_keywords, is_noise
from src.pipeline.ingest_sources import SETTINGS, fetch_rss_feed

logger = logging.getLogger(__name__)

# The same header production sends. A probe that introduces its own User-Agent
# measures a different site than the one Pass A will meet.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# How old the newest item may be before the feed counts as abandoned. Longer than
# the ingest window on purpose: a weekly paper is thin, not dead, and the two
# deserve different words.
_STALE_AFTER_DAYS = 21

_DATE_RE = re.compile(r"<(?:pubDate|dc:date|updated|published)>([^<]+)<", re.IGNORECASE)


def _parse_date(raw: str) -> Optional[datetime]:
    raw = raw.strip()
    for parse in (parsedate_to_datetime,
                  lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))):
        try:
            parsed = parse(raw)
        except Exception:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def raw_check(url: str, client: Optional[httpx.Client] = None) -> Dict[str, object]:
    """HTTP status, item count and newest date, without the production filters.

    fetch_rss_feed answers "what would Pass A take", which is the number that
    matters — but it returns an empty list for a 403, a dead host and a feed whose
    items are all too old alike, and those need different work. This separates
    them.
    """
    owned = client is None
    client = client or httpx.Client(timeout=30.0, follow_redirects=True,
                                    headers={"User-Agent": _UA})
    try:
        response = client.get(url)
        body = response.text
        status = response.status_code
    except httpx.HTTPError as exc:
        return {"status": 0, "items": 0, "newest": None, "error": str(exc)[:80]}
    finally:
        if owned:
            client.close()
    items = len(re.findall(r"<item[\s>]|<entry[\s>]", body, re.IGNORECASE))
    dates = [d for d in (_parse_date(m) for m in _DATE_RE.findall(body)) if d]
    return {"status": status, "items": items,
            "newest": max(dates) if dates else None, "error": ""}


def verdict(raw: Dict[str, object], fresh: int, security: int,
            now: Optional[datetime] = None) -> str:
    """One word per feed, ordered so the most actionable cause wins."""
    now = now or datetime.now(timezone.utc)
    status = int(raw.get("status") or 0)
    if status in (401, 403, 429):
        return "BLOCKED"
    if status == 0 or status >= 500 or not raw.get("items"):
        return "DEAD"
    newest = raw.get("newest")
    if isinstance(newest, datetime) and now - newest > timedelta(days=_STALE_AFTER_DAYS):
        return "STALE"
    if fresh == 0:
        return "STALE"
    return "ADD" if security else "THIN"


def probe(url: str, client: Optional[httpx.Client] = None) -> Dict[str, object]:
    """Fetch one feed both ways and score it."""
    raw = raw_check(url, client=client)
    try:
        items = fetch_rss_feed(url, is_direct_url=True) or []
    except Exception as exc:  # a probe must never die on one bad feed
        logger.warning("Production fetch raised for %s: %s", url, exc)
        items = []
    security = sum(
        1 for item in items
        if not is_noise(f"{item.get('title', '')} {item.get('description', '')}")
        and _matches_security_keywords(item.get("title", ""), item.get("description", ""))
    )
    return {"url": url, "raw": raw, "fresh": len(items), "security": security,
            "verdict": verdict(raw, len(items), security)}


def _print_table(results: Sequence[Dict[str, object]]) -> None:
    order = {"ADD": 0, "THIN": 1, "STALE": 2, "BLOCKED": 3, "DEAD": 4}
    print(f"\n{'verdict':<8} {'http':>4} {'raw':>4} {'fresh':>5} {'sec':>4}  "
          f"{'newest':<11} url")
    for result in sorted(results, key=lambda r: (order.get(r["verdict"], 9),
                                                 -int(r["security"]))):
        raw = result["raw"]
        newest = raw.get("newest")
        stamp = newest.strftime("%Y-%m-%d") if isinstance(newest, datetime) else "-"
        print(f"{result['verdict']:<8} {raw.get('status', 0):>4} "
              f"{raw.get('items', 0):>4} {result['fresh']:>5} "
              f"{result['security']:>4}  {stamp:<11} {result['url']}")
    counts: Dict[str, int] = {}
    for result in results:
        counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1
    print("\n  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("\n  ADD means: fresh items AND at least one of them cleared the ingest\n"
          "  keyword gate. THIN means the feed works and yields nothing — the\n"
          "  usual answer for a foreign-language source, because that gate is\n"
          "  English. Check the language before reading THIN as 'low quality'.\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urls", default="",
                        help="candidate feeds, separated by spaces, commas or newlines")
    parser.add_argument("--from-config", action="store_true",
                        help="probe the feeds already in publisher_feeds")
    args = parser.parse_args(argv)

    urls: List[str] = [u for u in re.split(r"[\s,]+", args.urls) if u.strip()]
    if args.from_config:
        urls += list(SETTINGS.get("sources", {}).get("publisher_feeds", []))
    if not urls:
        print("Nothing to probe: pass --urls or --from-config", file=sys.stderr)
        return 1

    seen: set = set()
    results = []
    with httpx.Client(timeout=30.0, follow_redirects=True,
                      headers={"User-Agent": _UA}) as client:
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            results.append(probe(url, client=client))
    _print_table(results)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
