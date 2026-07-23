"""Find a Gemini model that can still do Search grounding on the free tier.

Google retired gemini-2.5-flash and gemini-2.5-flash-lite on the
generativelanguage endpoint on 2026-07-09 — earlier than the announced
2026-10-16 date. Both SITREP keys are pinned to gemini-2.5-flash-lite in
daily-sitrep.yml, so every grounded call in run #13 returned 404 and web
enrichment produced nothing. The failure is silent: the pipeline is fail-soft
and still reports success.

The free-tier console shows Search-grounding quota in per-family buckets:

    Gemini 2      0 / 1.5K
    Gemini 2.5    0 / 1.5K     (the retired models drew from here)
    Gemini 3      0 / 0        (measured 2026-07-18: grounding 429s instantly)
    Default       0 / 1.5K     <-- unattributed bucket, 1500/day

No bucket is listed for the 3.1 / 3.5 / 3.6 families, which suggests they fall
under "Default" and therefore have grounding quota after all. If any of them
grounds, the whole problem is a one-line env change: gemini-3.1-flash-lite and
gemini-3.5-flash-lite carry 500 RPD against the 25 calls a run needs, versus the
20 RPD that made 2.5-flash-lite too tight even while it worked.

This script sends one real grounded request per candidate and reports, for each:
whether the model exists, whether the Search tool ran, and whether source URLs
came back. The last point is not optional — verification labels are derived from
source domains, so a model that answers without groundingMetadata is unusable
regardless of how good the prose looks.

Usage:
    GEMINI_API_KEY=... python -m scripts.check_gemini_grounding
    GEMINI_API_KEY=... python -m scripts.check_gemini_grounding --model gemini-3.6-flash

Costs nothing on the free tier, but each successful call spends one unit of the
model's RPD and one of its grounding bucket.
"""

import argparse
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# The key travels in the query string, so httpx exception messages (which quote
# the failing URL) carry it. This output is meant to be pasted into an issue or
# a chat, and the project has already leaked credentials once through a public
# CI artifact — mirror the redaction sitrep_web_enrich uses.
_KEY_IN_URL_RE = re.compile(r"([?&]key=)[\w.\-]+")


def _redact(text: str) -> str:
    return _KEY_IN_URL_RE.sub(r"\1***", text or "")

# Ordered by usefulness to us: the 500-RPD lites first — they would give the
# most headroom — then the 20-RPD flashes, then the known-dead 2.5 as a control.
# A 404 on gemini-2.5-flash-lite confirms the script is reaching the same
# endpoint the pipeline uses, so a 404 elsewhere means "retired", not "typo".
_CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash",
    "gemini-2.5-flash-lite",
]

# Deliberately a question that CANNOT be answered from weights: it asks for the
# last 24 hours. A model answering without grounding will either hedge or
# hallucinate, and either way returns no groundingMetadata — which is exactly
# the signal this script tests for.
_PROMPT = (
    "Search the web: in the LAST 24 HOURS, which named airlines suspended, "
    "cancelled or rerouted flights to or over Iran or the Gulf states, and "
    "which airports or airspace were closed? Answer in Turkish, one item per "
    "line starting with '- ', each naming the carrier and the route. Use only "
    "what you find in search results. If nothing, reply exactly: EK_BILGI_YOK"
)


def _api_key() -> str:
    for var in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    print("ERROR: set GEMINI_API_KEY before running.")
    sys.exit(2)


def _call(model: str, api_key: str) -> Tuple[Optional[Dict[str, Any]], int, str, float]:
    """Returns (payload, status_code, error_text, elapsed)."""
    body = {
        "contents": [{"parts": [{"text": _PROMPT}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }
    started = time.time()
    try:
        resp = httpx.post(_URL.format(model=model), params={"key": api_key},
                          json=body, timeout=60.0)
    except Exception as e:  # network-level failure, not a model verdict
        return None, 0, _redact(str(e))[:200], time.time() - started
    elapsed = time.time() - started
    if resp.status_code != 200:
        return None, resp.status_code, _redact(resp.text)[:400], elapsed
    return resp.json(), 200, "", elapsed


def _sources(payload: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    for cand in payload.get("candidates") or []:
        meta = cand.get("groundingMetadata") or {}
        for chunk in meta.get("groundingChunks") or []:
            web = chunk.get("web") or {}
            title = web.get("title") or web.get("uri")
            if title:
                urls.append(title)
    return urls


def _text(payload: Dict[str, Any]) -> str:
    out = []
    for cand in payload.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            if part.get("text"):
                out.append(part["text"])
    return "".join(out).strip()


def _queries(payload: Dict[str, Any]) -> List[str]:
    for cand in payload.get("candidates") or []:
        meta = cand.get("groundingMetadata") or {}
        if meta.get("webSearchQueries"):
            return meta["webSearchQueries"]
    return []


def _probe(model: str, api_key: str) -> Dict[str, Any]:
    print(f"\n{'=' * 70}\n{model}\n{'=' * 70}")
    payload, status, err, elapsed = _call(model, api_key)

    if payload is None:
        verdict = {
            404: "RETIRED / unknown model",
            429: "quota exhausted or ZERO grounding quota",
            400: "bad request — model may not accept the google_search tool",
        }.get(status, f"HTTP {status}")
        print(f"  FAIL ({elapsed:.1f}s): {verdict}")
        print(f"  {err}")
        return {"model": model, "ok": False, "grounded": False, "verdict": verdict}

    text = _text(payload)
    srcs = _sources(payload)
    queries = _queries(payload)

    print(f"  latency        : {elapsed:.1f}s")
    print(f"  search queries : {len(queries)} {queries[:4]}")
    print(f"  sources        : {len(srcs)}")
    for s in srcs[:8]:
        print(f"                   - {s}")
    print(f"\n  --- content ({len(text)} chars) ---")
    print("\n".join(f"  {line}" for line in text[:1200].splitlines()))

    grounded = bool(srcs) and bool(queries)
    if not grounded:
        print("\n  WARNING: answered WITHOUT grounding metadata — unusable, the "
              "verification labels need source domains.")
    return {"model": model, "ok": True, "grounded": grounded,
            "sources": len(srcs), "text": text,
            "verdict": "grounded" if grounded else "no grounding metadata"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", dest="models",
                        help="probe only this model (repeatable)")
    args = parser.parse_args()

    api_key = _api_key()
    models = args.models or _CANDIDATES

    results = []
    for i, model in enumerate(models):
        results.append(_probe(model, api_key))
        if i < len(models) - 1:
            time.sleep(5)  # stay under the 15 RPM free-tier ceiling

    print(f"\n{'=' * 70}\nVERDICT\n{'=' * 70}")
    winners = [r for r in results if r["grounded"]]
    for r in results:
        mark = "GROUNDED" if r["grounded"] else ("alive" if r["ok"] else "FAIL")
        print(f"  [{mark:>8}]  {r['model']:<26} {r['verdict']}")

    if winners:
        best = winners[0]
        print(f"\nUse {best['model']}. Set it in .github/workflows/daily-sitrep.yml:")
        print(f"  SITREP_GEMINI_MODEL:   {best['model']}")
        print(f"  SITREP_GEMINI_MODEL_2: {best['model']}")
        print("\nThen raise the budget — the 20-RPD ceiling that forced")
        print("max_grounded_calls_per_run=36 and a 7s cooldown was 2.5-flash-lite's;")
        print("a 500-RPD model needs neither.")
    else:
        print("\nNo Gemini model grounds on this key. Grounding must move to a")
        print("search API (Perplexity /search or Jina s.jina.ai) with synthesis")
        print("done by the existing router.")
    return 0 if winners else 1


if __name__ == "__main__":
    sys.exit(main())
