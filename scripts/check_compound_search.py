"""Probe groq/compound as a replacement for Gemini Search grounding.

Gemini's grounding is capped at 20 requests/day per project (the 2.5-flash-lite
model RPD — the only grounding-capable family), which is below what one SITREP
run needs. groq/compound does server-side web search in a single call with the
same request shape, at 250 RPD on the free tier.

Before committing to the swap, four things must hold. This script checks each
and prints a verdict:

  1. The call succeeds and returns NON-EMPTY prose. Compound runs on GPT-OSS
     120B, and this project has been bitten before by gpt-oss variants returning
     empty/garbage output unless reasoning is constrained (see the adjudicator
     fix: reasoning_effort=low + max_tokens >= 512). A silent empty response is
     the failure mode to catch here, not an exception.
  2. Source URLs come back. The SITREP verification labels are derived from
     source DOMAINS, and discovered incidents with no supporting source are
     dropped — without per-call source metadata the whole verification design
     collapses. Gemini supplies this via groundingChunks; compound is claimed to
     supply it via executed_tools[].search_results and/or citations.
  3. The model actually searched (rather than answering from weights) — an
     unused search tool would give us confident, stale, uncited text.
  4. Output honours the Turkish + line-format instructions the real prompts use.

Usage:
    export GROQ_API_KEY_A=...            # or GROQ_API_KEY
    python -m scripts.check_compound_search
    python -m scripts.check_compound_search --model groq/compound-mini

Exits non-zero if any hard check fails.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import httpx

_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Shaped like the real strategic_sweep prompt: Turkish out, bullet lines, hard
# "only what you found" instruction. Iran is used because it is a live,
# high-volume story — a quiet country would make "no results" ambiguous.
_SWEEP_PROMPT = (
    "Search for security developments about Iran in the LAST 24 HOURS only. "
    "Cover if reported:\n"
    "- aviation impact: WHICH named airlines suspended, cancelled, rerouted or "
    "resumed flights to/over the country; which airports were closed or attacked; "
    "which airspace was closed and to whom\n"
    "- travel advisories, embassy closures, sanctions, official military statements\n"
    "Summarize IN TURKISH, one development per line starting with '- ', each with "
    "concrete detail (who, what, figures). Only facts found in search results — no "
    "speculation. If nothing significant, reply exactly: EK_BILGI_YOK"
)

# Shaped like discover_incidents: strict machine-parseable line format. Tests
# whether compound respects an output contract while also searching.
_DISCOVERY_PROMPT = (
    "Search the web for security incidents in Iran in the LAST 24 HOURS: "
    "airstrikes, missile and drone attacks, explosions, armed clashes, attacks on "
    "airports, ports and energy infrastructure.\n\n"
    "Output up to 5 incidents, one per line, EXACTLY this format (no other text, "
    "no markdown):\n"
    "LOKASYON: <city or area> | SAAT: <local time ONLY if a source states it, "
    "otherwise 'belirsiz'> | OLAY: <2-4 sentence factual summary IN TURKISH>\n"
    "Only incidents found in actual search results. If none, reply exactly: "
    "EK_BILGI_YOK"
)


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY_A") or os.environ.get("GROQ_API_KEY")
    if not key:
        print("ERROR: set GROQ_API_KEY_A (or GROQ_API_KEY) before running.")
        sys.exit(2)
    return key


def _call(prompt: str, model: str, api_key: str,
          max_tokens: int = 1024) -> Tuple[Dict[str, Any], float]:
    """One compound call. max_tokens deliberately >= 512 (see docstring note 1)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    started = time.time()
    resp = httpx.post(
        _API_URL,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json=body,
        timeout=120.0,
    )
    elapsed = time.time() - started
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code}: {resp.text[:600]}")
        resp.raise_for_status()
    return resp.json(), elapsed


def _extract_sources(message: Dict[str, Any]) -> List[str]:
    """Collect result URLs from wherever compound exposes them.

    The field layout is not pinned down in the docs, so look in every documented
    place rather than assuming one: executed_tools[].search_results (dict with
    'results', or a bare list) and a top-level citations array.
    """
    urls: List[str] = []

    for tool in message.get("executed_tools") or []:
        results = tool.get("search_results")
        if isinstance(results, dict):
            results = results.get("results") or []
        for item in results or []:
            if isinstance(item, dict) and item.get("url"):
                urls.append(item["url"])
            elif isinstance(item, str) and item.startswith("http"):
                urls.append(item)

    for cite in message.get("citations") or []:
        if isinstance(cite, str) and cite.startswith("http"):
            urls.append(cite)
        elif isinstance(cite, dict) and cite.get("url"):
            urls.append(cite["url"])

    seen, unique = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _domains(urls: List[str]) -> List[str]:
    from src.core.sitrep_verify import registrable_domain
    seen, out = set(), []
    for u in urls:
        d = registrable_domain(u)
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _looks_turkish(text: str) -> bool:
    """Cheap heuristic: Turkish-specific characters or common function words."""
    if any(ch in text for ch in "çğışöüÇĞİŞÖÜ"):
        return True
    lowered = f" {text.lower()} "
    return any(w in lowered for w in (" ve ", " için ", " ile ", " bir ", " oldu"))


def _run_case(name: str, prompt: str, model: str, api_key: str,
              expect_format: str = "") -> Dict[str, Any]:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    data, elapsed = _call(prompt, model, api_key)

    message = (data.get("choices") or [{}])[0].get("message") or {}
    text = (message.get("content") or "").strip()
    tools = message.get("executed_tools") or []
    urls = _extract_sources(message)
    usage = data.get("usage") or {}

    print(f"latency        : {elapsed:.1f}s")
    print(f"tokens         : prompt={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')} "
          f"total={usage.get('total_tokens')}")
    print(f"executed_tools : {len(tools)} "
          f"({', '.join(str(t.get('type') or t.get('name') or '?') for t in tools) or 'none'})")
    print(f"source urls    : {len(urls)}")
    for d in _domains(urls)[:12]:
        print(f"                 - {d}")
    print(f"\n--- content ({len(text)} chars) ---\n{text[:1500]}")
    if len(text) > 1500:
        print(f"... [{len(text) - 1500} more chars]")

    empty_marker = text.startswith("EK_BILGI_YOK")
    checks = {
        "non-empty output (reasoning gate)": bool(text),
        "search tool actually ran": bool(tools),
        "source urls returned": bool(urls),
        "output in Turkish": _looks_turkish(text) or empty_marker,
    }
    if expect_format:
        checks[f"honours output contract ({expect_format})"] = (
            expect_format in text or empty_marker
        )
    if empty_marker:
        print("\nNOTE: model reported EK_BILGI_YOK (nothing found). Format and "
              "language checks pass trivially — rerun on a country with live "
              "coverage before trusting them.")
    return {"checks": checks, "text": text, "urls": urls, "raw_message": message}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="groq/compound",
                        help="groq/compound (multi tool call) or groq/compound-mini")
    parser.add_argument("--dump", action="store_true",
                        help="print the raw message object of the first case")
    args = parser.parse_args()

    api_key = _api_key()
    print(f"model: {args.model}")

    results = [
        _run_case("CASE 1 — strategic sweep shape (aviation focus)",
                  _SWEEP_PROMPT, args.model, api_key),
        _run_case("CASE 2 — discovery shape (strict line contract)",
                  _DISCOVERY_PROMPT, args.model, api_key,
                  expect_format="LOKASYON:"),
    ]

    if args.dump:
        print(f"\n{'=' * 70}\nRAW MESSAGE (case 1)\n{'=' * 70}")
        print(json.dumps(results[0]["raw_message"], ensure_ascii=False, indent=2)[:4000])

    print(f"\n{'=' * 70}\nVERDICT\n{'=' * 70}")
    failed = 0
    for i, res in enumerate(results, 1):
        for label, ok in res["checks"].items():
            print(f"  case {i}  [{'PASS' if ok else 'FAIL'}]  {label}")
            failed += not ok

    print()
    if failed:
        print(f"{failed} check(s) FAILED — do not swap Gemini out yet.")
        print("If 'source urls returned' is the failure, re-run with --dump and "
              "check where compound actually puts them; the verification labels "
              "depend on that field existing.")
    else:
        print("All checks passed. compound is a viable grounding replacement:")
        print("  - 250 RPD/key free tier vs Gemini's 20 → ~20x headroom")
        print("  - same single-call shape, so _call_gemini can be swapped 1:1")
        print("\nStill unverified by this script:")
        print("  - whether the search tool is billed separately (free tier has no")
        print("    billing attached, but the docs don't state it — watch the console)")
        print("  - recency control: compound has no time_range parameter, so the")
        print("    24h window stays a prompt-level request, same as Gemini today")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
