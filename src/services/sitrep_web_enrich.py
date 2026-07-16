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
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from src.core.sitrep_verify import registrable_domain

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("SITREP_GEMINI_MODEL", "gemini-3.1-flash-lite")
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

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
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
    }
    try:
        resp = httpx.post(
            _GEMINI_URL.format(model=GEMINI_MODEL),
            params={"key": api_key},
            json=body,
            timeout=30.0,
        )
        resp.raise_for_status()
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
        logger.warning("Gemini grounded call failed: %s", str(e)[:200])
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
        "- casualty and damage figures, evacuations, infrastructure/utility outages\n"
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
        "- aviation: airspace closures, flight suspensions/reroutes, EASA/FAA/ICAO notices\n"
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
                         max_clusters: int, cooldown_s: float = 2.0) -> Dict[str, Any]:
    """
    Full grounding pass: enrich the top clusters, discover incidents the ingest
    missed, and run the strategic sweep. Returns
    {"strategic": {...}|None, "discovered": [cluster, ...]} — both empty when
    GEMINI_API_KEY is not configured.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.info("SITREP web enrichment skipped: GEMINI_API_KEY not set")
        return {"strategic": None, "discovered": []}
    for cluster in clusters[:max_clusters]:
        enrich_cluster(cluster, country_name, api_key)
        time.sleep(cooldown_s)
    known = [f'{c.get("location")}: {(c.get("snippet") or "")[:100]}' for c in clusters]
    discovered = discover_incidents(country_name, api_key, known)
    time.sleep(cooldown_s)
    sweep = strategic_sweep(country_name, api_key)
    time.sleep(cooldown_s)
    if discovered:
        logger.info("SITREP web discovery added %d incidents for %s", len(discovered), country_name)
    return {"strategic": sweep, "discovered": discovered}
