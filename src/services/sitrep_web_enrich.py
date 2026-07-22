"""
SIM — SITREP Web Enrichment
Two responsibilities:

1. Resolve Google News RSS redirect URLs (news.google.com/rss/articles/...) to
   the real publisher article URL so SITREP source links are usable.
2. Optional Gemini "Grounding with Google Search" enrichment (env-gated on
   GEMINI_API_KEY): expand top clusters with corroborated detail and fill the
   strategic section. Grounded source DOMAINS also feed the rule-based
   verification labels — the label logic itself stays in sitrep_verify.
"""

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from src.core.sitrep_verify import registrable_domain

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("SITREP_GEMINI_MODEL", "gemini-3.1-flash-lite")
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
with open(_CONFIG_DIR / "settings.json", encoding="utf-8") as _f:
    _SITREP_CFG = json.load(_f).get("sitrep", {})

# Seconds between grounded calls. 7.0 caps the rate at ~8.5/min, under
# gemini-2.5-flash-lite's free-tier 10 RPM ceiling. A paid tier would make this
# pacing irrelevant (RPM in the thousands), but grounding is billed per request
# there, so the project stays on the free tier and keeps the throttle.
WEB_ENRICH_COOLDOWN_S = float(_SITREP_CFG.get("web_enrich_cooldown_s", 7.0))

# Hard ceiling on grounded (Search-tool) calls per process.
#
# The binding constraint is NOT the Search-grounding quota (1.5K/day) but the
# grounding-capable MODEL's request-per-day limit: on the free tier
# gemini-2.5-flash-lite allows 20 RPD, and it is the only option — the Gemini 3
# family has 0 Search-grounding quota, so its far higher 500 RPD is unusable
# here. 20 calls/day is therefore the whole budget for grounding.
#
# The 20 RPD is PER PROJECT, and each configured key is its own project, so the
# real ceiling is 20 x len(_gemini_keys()) — rotation exists to reach it. With
# two keys that is 40; the budget sits under it at 36 and a full run needs
# 5 countries x (2 whole-country calls + 3 cluster enrichments) = 25.
# Exceeding the budget disables grounding for the rest of the run exactly like a
# spent quota does — which is what kept later countries from silently losing
# everything to the first ones.
# NB: if a key's project lacks the 2.5 model the rotation target 404s and the
# effective ceiling collapses back to one project's 20.
MAX_GROUNDED_CALLS_PER_RUN = int(_SITREP_CFG.get("max_grounded_calls_per_run", 36))
_grounded_calls = 0

# API keys appear in the query string, so raw httpx error strings (404/timeout)
# carry a live credential into the logs — and CI logs/artifacts of a public repo
# are world-readable. Never log an unredacted transport error.
_KEY_IN_URL_RE = re.compile(r"([?&]key=)[\w.\-]+")


def _redact(text: str) -> str:
    return _KEY_IN_URL_RE.sub(r"\1***", text or "")

_GOOGLE_NEWS_RE = re.compile(r"news\.google\.com/(?:rss/)?articles/([^?/]+)")
_URL_IN_BYTES_RE = re.compile(rb"https?://[\x21-\x7e]+")

# Circuit breaker + key rotation. Sustained 429s mean the CURRENT key's quota is
# gone (flash-lite RPD is shared with the main router's Gemini fallback slot and
# resets at Pacific midnight = 07:00 UTC): rotate to the next configured key
# (GEMINI_API_KEY → GEMINI_API_KEY_2) and only disable grounded calls for the
# rest of the process once every key is exhausted.
# A 429 whose quotaId contains "PerDay" rotates IMMEDIATELY, bypassing the
# counter: daily-quota 429s flap (sporadic 200s slip through and reset the
# counter), which kept rotation from ever firing (observed 2026-07-19).
_RATE_LIMIT_TRIP_AFTER = 3
_consecutive_429 = 0
_quota_exhausted = False
_key_idx = 0


def _gemini_keys() -> List[tuple]:
    """
    (api_key, model) per configured key. The backup key may need a DIFFERENT
    model: observed 2026-07-18 that a project can carry full model quota yet
    ZERO Search-grounding quota for the Gemini 3 family (grounded calls 429
    from the very first request while Gemini 2.5 grounding stays at 1.5K/day).
    SITREP_GEMINI_MODEL_2 (e.g. gemini-2.5-flash-lite) overrides the model for
    GEMINI_API_KEY_2; it defaults to the primary model.
    """
    pairs = []
    if os.environ.get("GEMINI_API_KEY"):
        pairs.append((os.environ["GEMINI_API_KEY"], GEMINI_MODEL))
    if os.environ.get("GEMINI_API_KEY_2"):
        pairs.append((os.environ["GEMINI_API_KEY_2"],
                      os.environ.get("SITREP_GEMINI_MODEL_2", GEMINI_MODEL)))
    return pairs


def _reset_gemini_state() -> None:
    """Reset breaker/rotation/budget state (test isolation only)."""
    global _consecutive_429, _quota_exhausted, _key_idx, _grounded_calls
    _consecutive_429 = 0
    _quota_exhausted = False
    _key_idx = 0
    _grounded_calls = 0


def _gemini_retry_delay(resp: httpx.Response, attempt: int) -> float:
    """Honor the RetryInfo delay Gemini returns on 429; fall back to backoff."""
    try:
        for detail in resp.json()["error"]["details"]:
            delay = detail.get("retryDelay")
            if delay:
                return min(float(delay.rstrip("s")), 60.0)
    except Exception:
        pass
    return 5.0 * (2 ** attempt)


def _gemini_429_reason(resp: httpx.Response) -> str:
    """Extract WHICH quota tripped from a 429 body (QuotaFailure.quotaId — e.g.
    ...PerDay... vs ...PerMinute...), so logs distinguish a spent daily quota
    from mere pacing. Returns the raw error message as fallback."""
    try:
        err = resp.json()["error"]
        for detail in err.get("details", []):
            for violation in detail.get("violations", []):
                if violation.get("quotaId"):
                    return violation["quotaId"]
        # No QuotaFailure violation in the body. Observed in prod (2026-07-18):
        # hard 429s while the model's RPD dashboard showed 16/500 used — so the
        # tripping limit is NOT model RPD (likely the google_search grounding
        # tool's own quota). Keep the raw details so the next occurrence is
        # diagnosable from logs alone.
        msg = (err.get("message") or "")[:120]
        details = json.dumps(err.get("details", []), ensure_ascii=False)[:400]
        return f"{msg} | details: {details}" if details not in ("[]", "") else msg
    except Exception:
        return "unparsable 429 body"


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


def resolve_cluster_urls(clusters: List[Dict[str, Any]], max_resolve: int = 20) -> None:
    """In-place: replace Google News redirect links in cluster sources."""
    budget = max_resolve
    for cluster in clusters:
        for source in cluster.get("sources", []):
            if budget <= 0:
                return
            u = source.get("url") or ""
            if "news.google.com" in u:
                budget -= 1
                source["url"] = resolve_url(u)


def _call_gemini(prompt: str, api_key: str, max_tokens: int = 1024) -> Optional[Dict[str, Any]]:
    """
    One grounded Gemini call. Returns None on failure, else:
    {"text", "sources": [{"name","url","title"}], "supports": [(segment_text, [chunk_idx])]}
    """
    global _consecutive_429, _quota_exhausted, _key_idx, _grounded_calls
    if _quota_exhausted:
        return None
    if _grounded_calls >= MAX_GROUNDED_CALLS_PER_RUN:
        _quota_exhausted = True
        logger.warning(
            "Grounded-call budget spent (%d calls this run) — disabling web "
            "enrichment for the rest of the run", _grounded_calls,
        )
        return None
    _grounded_calls += 1
    keys = _gemini_keys() or [(api_key, GEMINI_MODEL)]
    _key_idx = min(_key_idx, len(keys) - 1)
    active_key, active_model = keys[_key_idx]
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
    }
    try:
        resp = None
        for attempt in range(3):
            resp = httpx.post(
                _GEMINI_URL.format(model=active_model),
                params={"key": active_key},
                json=body,
                timeout=30.0,
            )
            if resp.status_code != 429:
                break
            reason = _gemini_429_reason(resp)
            if "PerDay" in reason:
                # Daily quota is spent until the 07:00 UTC reset — retrying
                # this key (or counting failures) only burns wall-clock time.
                if _key_idx + 1 < len(keys):
                    _key_idx += 1
                    _consecutive_429 = 0
                    active_key, active_model = keys[_key_idx]
                    logger.warning(
                        "Gemini key #%d daily quota spent (%s) — rotating to key #%d",
                        _key_idx, reason, _key_idx + 1,
                    )
                    continue
                _quota_exhausted = True
                logger.warning(
                    "All %d Gemini key(s) daily-exhausted (last: %s) — "
                    "disabling web enrichment for the rest of this run",
                    len(keys), reason,
                )
                return None
            if attempt < 2:
                delay = _gemini_retry_delay(resp, attempt)
                logger.info("Gemini 429 (%s), retrying in %.0fs", reason, delay)
                time.sleep(delay)
        if resp.status_code == 429:
            reason = _gemini_429_reason(resp)
            _consecutive_429 += 1
            if _consecutive_429 >= _RATE_LIMIT_TRIP_AFTER:
                if _key_idx + 1 < len(keys):
                    _key_idx += 1
                    _consecutive_429 = 0
                    logger.warning(
                        "Gemini key #%d quota exhausted (%s) — rotating to backup key #%d",
                        _key_idx, reason, _key_idx + 1,
                    )
                else:
                    _quota_exhausted = True
                    logger.warning(
                        "All %d Gemini key(s) exhausted (last: %s) — "
                        "disabling web enrichment for the rest of this run",
                        len(keys), reason,
                    )
            else:
                logger.warning("Gemini grounded call rate-limited (429: %s), giving up", reason)
            return None
        resp.raise_for_status()
        _consecutive_429 = 0
        data = resp.json()
        candidate = (data.get("candidates") or [{}])[0]
        text = "".join(
            p.get("text", "") for p in candidate.get("content", {}).get("parts", [])
        ).strip()
        meta = candidate.get("groundingMetadata", {})
        sources = []
        for chunk in meta.get("groundingChunks", []):
            web = chunk.get("web") or {}
            title = (web.get("title") or "").strip()
            uri = web.get("uri")
            if not uri:
                continue
            # grounding metadata's web.title is normally the publisher domain
            name = title if "." in title else registrable_domain(uri)
            sources.append({"name": name or "web", "url": uri, "title": title[:240]})
        supports = []
        for sup in meta.get("groundingSupports", []):
            seg_text = (sup.get("segment") or {}).get("text") or ""
            idxs = sup.get("groundingChunkIndices") or []
            if seg_text and idxs:
                supports.append((seg_text, idxs))
        if not text:
            return None
        return {"text": text, "sources": sources, "supports": supports}
    except Exception as e:
        logger.warning("Gemini grounded call failed: %s", _redact(str(e))[:200])
        return None


def enrich_cluster(cluster: Dict[str, Any], country_name: str, api_key: str) -> None:
    """
    In-place: add grounded `web_context` + extra `sources` to one cluster.
    Search snippets are treated strictly as data; the prompt forbids invention
    and the caller re-derives the verification label from the domain set.
    """
    titles = "; ".join(s.get("title") or "" for s in cluster.get("sources", [])[:2])
    prompt = (
        "You are verifying a security incident for an intelligence report. Search the web "
        "for this specific event and report ONLY facts found in actual search results.\n"
        f"Country: {country_name}\nLocation: {cluster.get('location')}\n"
        f"Event type: {cluster.get('event_type')}\nDate: {cluster.get('date')}\n"
        f"Known headlines: {titles}\n\n"
        "Look specifically for:\n"
        "- the exact facility/target hit (base, port, airport, plant, neighborhood)\n"
        "- weapon systems used (missile type, drone model, aircraft) if reported\n"
        "- casualty and damage figures, evacuations, infrastructure/utility outages — for "
        "every figure, state explicitly whether it covers THIS specific event or a wider "
        "cumulative toll (e.g. a nationwide total across multiple strikes); never present "
        "a cumulative toll as a single-event figure\n"
        "- official statements (CENTCOM, defense ministries, governors, state agencies)\n"
        "- the reported local time of the event, if any source states it explicitly\n"
        "- operational impact (airspace, ports, roads closed)\n\n"
        "Write 3-5 factual sentences IN TURKISH synthesizing what you found. Include the "
        "reported time ONLY if a source states it explicitly. If you find nothing about "
        "this specific event, reply exactly: EK_BILGI_YOK"
    )
    res = _call_gemini(prompt, api_key, max_tokens=768)
    if not res or res["text"].strip().startswith("EK_BILGI_YOK"):
        return
    cluster["web_context"] = res["text"][:1200]
    known_urls = {s.get("url") for s in cluster["sources"]}
    for s in res["sources"][:4]:
        if s["url"] not in known_urls:
            cluster["sources"].append(s)


def strategic_sweep(country_name: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    One grounded query for the strategic/political picture of the last 24h
    (airspace, advisories, sanctions, diplomacy) to feed BÖLÜM III.
    """
    prompt = (
        f"Search for strategic and political security developments about {country_name} "
        "in the LAST 24 HOURS only. Cover each of these areas if reported:\n"
        "- aviation impact of the security situation: WHICH named airlines suspended, "
        "cancelled, rerouted or resumed flights to/over the country, which airports were "
        "closed or attacked, which airspace was closed and to whom. Name the carrier and "
        "the route/airport explicitly (e.g. 'Emirates suspended Tehran flights until X'). "
        "Ignore routine safety/technical matters (weather delays, maintenance, NOTAMs "
        "unrelated to the security situation)\n"
        "- travel advisories and evacuation orders (State Dept, FCDO, other governments)\n"
        "- embassy/consulate closures, staff drawdowns\n"
        "- sanctions, UN/NATO/EU decisions, major diplomatic statements\n"
        "- official military statements about ongoing or planned operations\n"
        "- maritime/shipping impact (straits, ports, insurance rates)\n"
        "- market and currency reaction to the security situation\n"
        "Summarize IN TURKISH, one development per line starting with '• ', each with "
        "concrete detail (who, what, figures). Only facts found in search results — no "
        "speculation. If nothing significant, reply exactly: EK_BILGI_YOK"
    )
    res = _call_gemini(prompt, api_key, max_tokens=1024)
    if not res or res["text"].strip().startswith("EK_BILGI_YOK"):
        return None
    return {"text": res["text"][:2000], "sources": res["sources"][:6]}


def discover_incidents(country_name: str, api_key: str,
                       known_summaries: List[str],
                       max_incidents: int = 10) -> List[Dict[str, Any]]:
    """
    Grounded discovery sweep: find security incidents of the last 24h that the
    ingest pipeline MISSED, as extra SITREP clusters. Each discovered incident's
    sources come from grounding metadata (per-line groundingSupports mapping),
    and its verification label is derived from those real domains — a line no
    grounding chunk supports is dropped entirely.
    """
    known = "\n".join(f"- {s}" for s in known_summaries[:20]) or "- (yok)"
    prompt = (
        f"Search the web for security incidents in {country_name} in the LAST 24 HOURS:\n"
        "- airstrikes, missile and drone attacks, shelling, explosions\n"
        "- armed clashes, terror attacks, assassinations, IED attacks\n"
        "- attacks on military bases, ports, airports, energy and critical infrastructure\n"
        "- air-defense activations, interceptions, naval incidents\n"
        "- major unrest, curfews, mass evacuations\n"
        "Check official military sources (CENTCOM, defense ministries, state agencies), "
        "wire services (Reuters, AP, AFP), major outlets (BBC, CNN, Al Jazeera) and "
        "credible regional media. Cover ALL regions of the country, not just the capital.\n\n"
        "ALREADY KNOWN incidents (do NOT repeat these):\n"
        f"{known}\n\n"
        f"Output up to {max_incidents} NEW incidents, one per line, EXACTLY this format "
        "(no other text, no markdown):\n"
        "LOKASYON: <city or area> | SAAT: <local time ONLY if a source explicitly states "
        "it, otherwise 'belirsiz'> | OLAY: <2-4 sentence factual summary IN TURKISH: what "
        "was hit, weapons used, casualties, official statements — only facts from search "
        "results>\n"
        "Only include incidents you found in actual search results. "
        "If there are none, reply exactly: EK_BILGI_YOK"
    )
    res = _call_gemini(prompt, api_key, max_tokens=2048)
    if not res or res["text"].strip().startswith("EK_BILGI_YOK"):
        return []

    clusters: List[Dict[str, Any]] = []
    for line in res["text"].splitlines():
        line = line.strip().lstrip("•-* ")
        m = re.match(
            r"LOKASYON:\s*(.+?)\s*\|\s*(?:SAAT:\s*(.+?)\s*\|\s*)?OLAY:\s*(.+)", line
        )
        if not m:
            continue
        # map this line to the grounding chunks that support it
        idxs = set()
        for seg_text, chunk_idxs in res["supports"]:
            if seg_text[:80] in line or line[:80] in seg_text:
                idxs.update(chunk_idxs)
        line_sources = [res["sources"][i] for i in sorted(idxs) if i < len(res["sources"])]
        if not line_sources:
            continue  # unsupported claim — never report it
        reported_time = (m.group(2) or "belirsiz").strip()
        date_label = ("son 24 saat içinde, saat belirsiz"
                      if reported_time.lower() in ("belirsiz", "unknown", "")
                      else f"son 24 saat içinde, bildirilen saat: {reported_time}")
        clusters.append({
            "location": m.group(1)[:80],
            "event_type": "web_discovery",
            "date": date_label,
            "verification": None,  # caller re-labels from source domains
            "severity": 0,
            "snippet": m.group(3)[:600],
            "sources": line_sources[:4],
        })
        if len(clusters) >= max_incidents:
            break
    return clusters


def apply_web_enrichment(clusters: List[Dict[str, Any]], country_name: str,
                         max_clusters: int,
                         cooldown_s: float = WEB_ENRICH_COOLDOWN_S) -> Dict[str, Any]:
    """
    Full grounding pass, ordered by VALUE so a mid-run quota death costs the
    least: discovery and the strategic sweep (2 calls, whole-country coverage)
    run first; per-cluster enrichment (up to max_clusters calls, incremental
    detail) runs last. Returns {"strategic": {...}|None, "discovered":
    [cluster, ...]} — both empty when GEMINI_API_KEY is not configured.

    cooldown_s=7.0 caps the call rate at ~8.5/min even if grounded calls return
    instantly — under gemini-2.5-flash-lite's free-tier 10 RPM ceiling (the
    grounding-capable model since Gemini 3 Search grounding went to 0).
    MAX_GROUNDED_CALLS_PER_RUN is the per-run backstop on top of that.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.info("SITREP web enrichment skipped: GEMINI_API_KEY not set")
        return {"strategic": None, "discovered": []}
    known = [f'{c.get("location")}: {(c.get("snippet") or "")[:100]}' for c in clusters]
    discovered = discover_incidents(country_name, api_key, known)
    time.sleep(cooldown_s)
    sweep = strategic_sweep(country_name, api_key)
    time.sleep(cooldown_s)
    for cluster in clusters[:max_clusters]:
        if _quota_exhausted:
            break
        enrich_cluster(cluster, country_name, api_key)
        time.sleep(cooldown_s)
    if _quota_exhausted:
        logger.warning("SITREP web enrichment for %s ran with exhausted Gemini quota — "
                       "results may be partial", country_name)
    if discovered:
        logger.info("SITREP web discovery added %d incidents for %s", len(discovered), country_name)
    return {"strategic": sweep, "discovered": discovered}
