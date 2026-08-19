"""
SIM — Google News URL resolver.

SITREP sources routinely arrive as news.google.com/rss/articles/… redirects,
which are useless as citations: they expose no publisher, and the verification
labels in sitrep_verify key off the registrable domain. This resolves them to
the real article URL — offline from the legacy base64 ID where possible, and via
Google's own batchexecute endpoint for the newer opaque IDs.

Split out of sitrep_web_enrich on 2026-08-01, when the Gemini grounding half of
that module was removed: grounding died with the retirement of the last
free-tier model that supported it (2026-07-09), but URL resolution never
depended on it and runs on every SITREP.
"""

import base64
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

_GOOGLE_NEWS_RE = re.compile(r"news\.google\.com/(?:rss/)?articles/([^?/]+)")
_URL_IN_BYTES_RE = re.compile(rb"https?://[\x21-\x7e]+")


def decode_google_news_url(url: str) -> Optional[str]:
    """
    Offline decode of legacy Google News article IDs (the base64 payload embeds
    the publisher URL). Returns None for the newer opaque AU_yq… IDs — those
    need the HTTP fallback in resolve_url().
    """
    m = _GOOGLE_NEWS_RE.search(url or "")
    if not m:
        return None
    token = m.group(1)
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
    except Exception:
        return None
    for match in _URL_IN_BYTES_RE.finditer(raw):
        candidate = match.group(0)
        # trim trailing protobuf length/control bytes that got glued on
        candidate = candidate.split(b"\xd2")[0].rstrip(b"\x01\x02\x03")
        try:
            decoded = candidate.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        if "news.google.com" not in decoded and "." in decoded[8:]:
            return decoded
    return None


def _batchexecute_decode(art_id: str, page_html: str, timeout: float) -> Optional[str]:
    """
    Decode a new-format (opaque) Google News article ID via the DotsSplashUi
    batchexecute endpoint, using the signature/timestamp the interstitial page
    embeds. Returns None on any failure — callers keep the original link.
    """
    sg = re.search(r'data-n-a-sg="([^"]+)"', page_html)
    ts = re.search(r'data-n-a-ts="([^"]+)"', page_html)
    if not sg or not ts:
        return None
    inner = json.dumps([
        "garturlreq",
        [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1,
          None, None, None, None, None, 0, 1],
         "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0],
        art_id, int(ts.group(1)), sg.group(1),
    ])
    try:
        resp = httpx.post(
            "https://news.google.com/_/DotsSplashUi/data/batchexecute",
            content="f.req=" + quote(json.dumps([[["Fbv4je", inner]]])),
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
            timeout=timeout,
        )
        resp.raise_for_status()
        chunk = resp.text.split("\n\n")[1]
        for entry in json.loads(chunk):
            if isinstance(entry, list) and len(entry) > 2 and entry[2]:
                decoded = json.loads(entry[2])[1]
                if isinstance(decoded, str) and decoded.startswith("http"):
                    return decoded
    except Exception as e:
        logger.debug("batchexecute decode failed for %s: %s", art_id[:40], str(e)[:120])
    return None


def resolve_url(url: str, timeout: float = 6.0) -> str:
    """Resolve a Google News redirect to the publisher URL; returns input on failure."""
    if "news.google.com" not in (url or ""):
        return url
    decoded = decode_google_news_url(url)
    if decoded:
        return decoded
    try:
        # SOCS cookie pre-accepts Google's EU cookie-consent wall, which would
        # otherwise swallow the redirect chain on European egress IPs.
        resp = httpx.get(url, follow_redirects=True, timeout=timeout,
                         cookies={"SOCS": "CAI"},
                         headers={"User-Agent": "Mozilla/5.0 (SIM-SITREP)"})
        final = str(resp.url)
        if "news.google.com" not in final:
            return final
        # new-format opaque ID: interstitial page carries the signature needed
        # for the batchexecute decode
        m_id = _GOOGLE_NEWS_RE.search(url)
        if m_id:
            decoded = _batchexecute_decode(m_id.group(1), resp.text, timeout)
            if decoded:
                return decoded
        # last resort: embedded target link on consent pages
        m = re.search(r'href="(https?://(?!.*google\.com)[^"]+)"', resp.text)
        if m:
            return m.group(1)
    except Exception:
        logger.debug("Google News redirect resolution failed for %s", url[:80])
    return url


# The old budget of 20 was set blind and it bound on the busiest countries, which
# are exactly the ones a reader opens: Lebanon on 16 Aug and Ukraine on 19 Aug both
# carried 96 sources and finished with 23 and 15 unresolved Google News links in the
# appendix. Resolution is cheap — 5 of 5 sampled links resolved on 2026-08-19 at a
# mean of 0.30s — so the budget, not the cost, was the constraint.
#
# It is raised, but NOT to unbounded, and a wall-clock deadline is added beside it:
# per-request timeouts do not bound total time when the slow path is many requests
# each landing just under the limit (the lesson from the Pass C wall-clock ceiling).
# 60 links at the observed 0.30s is ~18s; the deadline is what protects the SITREP
# step on a day when Google is slow instead.
DEFAULT_MAX_RESOLVE = 60
DEFAULT_RESOLVE_DEADLINE_S = 45.0


def resolve_cluster_urls(clusters: List[Dict[str, Any]],
                         max_resolve: int = DEFAULT_MAX_RESOLVE,
                         deadline_seconds: float = DEFAULT_RESOLVE_DEADLINE_S) -> None:
    """In-place: replace Google News redirect links in cluster sources.

    Stops on whichever limit is reached first. Unresolved links are left as they
    are — a Google News redirect is a poor citation, but it is a working one.
    """
    budget = max_resolve
    give_up_at = time.monotonic() + deadline_seconds
    attempted = resolved = 0
    for cluster in clusters:
        for source in cluster.get("sources", []):
            u = source.get("url") or ""
            if "news.google.com" not in u:
                continue
            if budget <= 0 or time.monotonic() >= give_up_at:
                logger.info(
                    "Google News resolution stopped early (%s): %d/%d resolved",
                    "budget" if budget <= 0 else "deadline", resolved, attempted,
                )
                return
            budget -= 1
            attempted += 1
            source["url"] = resolve_url(u)
            resolved += "news.google.com" not in source["url"]
    if attempted:
        logger.info("Google News resolution: %d/%d resolved", resolved, attempted)


