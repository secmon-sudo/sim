"""Bake-off harness: does a candidate model actually do Pass C's job?

Free model slots die often — three did on 2026-09-02 alone (Cerebras' free tier,
OpenRouter's whole free gpt-oss family, mistral-large leaving its subscription
tier). Picking the replacement from catalog copy is guesswork: a card that says
"1M context, reasoning" says nothing about whether the model returns the batch
schema Pass C parses. This runs the REAL classification prompt through a
candidate and prints what came back, so the choice is made on output.

Runs where the keys live (GitHub Actions, see model-probe.yml). Prints a verdict
per model and dumps raw content, since a malformed reply is the interesting case.

  python -m scripts.probe_models --models groq:qwen/qwen3.8-27b
  python -m scripts.probe_models --list gemini,groq   # discover exact model ids
"""

import argparse
import dataclasses
import difflib
import json
import os
import re
import sys
import time
import urllib.request

from src.core import llm_client
from src.core.llm_client import call_llm
from src.core.llm_router import LLMAccount, LLMRouter
from src.core.model_profiles import get_profile
from src.core.token_bucket import TokenBucket
from src.services.sitrep_generator import NARRATIVE_MAX_TOKENS
from src.services.sitrep_generator import _SYSTEM_PROMPT as SITREP_SYSTEM_PROMPT
from src.pipeline.pass_c_classify import (
    BATCH_SYSTEM_SUFFIX,
    CLASSIFICATION_SYSTEM_PROMPT,
    LLMParseError,
    _batch_prompt,
    _parse_batch_response,
)

# Fixed sample so two models are graded on identical input, and so a re-run months
# later is comparable. Deliberately spans the cases the gates argue about: an
# unambiguous attack, an aviation-security incident, a commentary piece that must
# NOT score as an incident, and an aftermath/anniversary piece.
SAMPLE_EVENTS = [
    {
        "source_title": "Drone attack halts all flights at Kabul airport, three killed",
        "source_domain": "reuters.com",
        "canonical_text": (
            "Afghan officials said a drone struck the perimeter of Hamid Karzai "
            "International Airport early on Tuesday, killing three ground staff and "
            "forcing the suspension of all departures. The Taliban-run aviation "
            "authority said airspace would remain closed until further notice."
        ),
    },
    {
        "source_title": "Passenger arrested after breaching security screening at Heathrow Terminal 5",
        "source_domain": "bbc.co.uk",
        "canonical_text": (
            "A man was detained after running through a security screening lane at "
            "Heathrow Terminal 5 on Monday evening. Police said the terminal was "
            "partially evacuated for about 40 minutes and several flights were delayed."
        ),
    },
    {
        "source_title": "Why the Sahel's security crisis will define the next decade",
        "source_domain": "foreignpolicy.com",
        "canonical_text": (
            "Analysts argue that the withdrawal of Western forces from the Sahel has "
            "reshaped the region's balance of power. This essay considers what the "
            "shift means for counterterrorism policy in the years ahead."
        ),
    },
    {
        "source_title": "Families mark one year since the Nice bus bombing that killed 12",
        "source_domain": "lemonde.fr",
        "canonical_text": (
            "Relatives gathered on Sunday for a memorial service marking the first "
            "anniversary of the bombing. A judicial inquiry into the attack remains open."
        ),
    },
]


# Where each provider publishes the ids it will actually accept. The rate-limit
# dashboards and deprecation emails name models by DISPLAY name ("Gemma 4 31B",
# "Qwen3.8-27B") and the API wants a slug; deriving one from the other is a guess,
# and a guessed slug is how a slot ends up 404-ing in production (gemma-4-26b-it,
# 2026-09-02).
_CATALOGUES = {
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        "GEMINI_API_KEY",
    ),
    "groq": ("https://api.groq.com/openai/v1/models", "GROQ_API_KEY_A"),
    # LLM7 is the only catalogue here that answers the capability checklist itself:
    # each entry carries json_mode/reasoning/context_window/tier and an hourly
    # availability percentage. Those are CLAIMS, which is why probing still happens —
    # but they say which models are worth spending a probe on.
    "llm7": ("https://api.llm7.io/v1/models", "LLM7_KEY"),
    # Pollinations publishes 370 entries; the free-tier ones are community-contributed
    # (an individual's upstream key registered into the router) and marked alpha.
    "pollinations": ("https://gen.pollinations.ai/models", "POLLINATIONS_API_KEY"),
}


def list_models(provider: str) -> int:
    """Print the model ids the key for `provider` can actually see."""
    url, key_env = _CATALOGUES[provider]
    key = os.environ.get(key_env, "")
    if not key:
        print(f"=== {provider}: {key_env} not set, skipping catalogue ===")
        return 0
    # Gemini takes the key in the query string, Groq as a bearer header.
    if provider == "gemini":
        req = urllib.request.Request(f"{url}&key={key}")
    else:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            data = json.loads(r.read())
    except Exception as exc:
        print(f"=== {provider}: catalogue fetch failed: {type(exc).__name__}: {exc} ===")
        return 0

    # Pollinations returns a bare JSON array; Gemini/Groq/LLM7 wrap it in an object.
    rows = data if isinstance(data, list) else (data.get("models") or data.get("data") or [])
    print(f"\n=== {len(rows)} models visible to {key_env} ({provider}) ===")
    for m in sorted(rows, key=lambda x: str(x.get("name") or x.get("id"))):
        name = str(m.get("name") or m.get("id")).replace("models/", "")
        methods = ",".join(m.get("supportedGenerationMethods", []))
        extra = f" methods={methods}" if methods else ""
        limit = m.get("inputTokenLimit") or m.get("context_window")
        # LLM7 nests it: {"tokens": 1048576, "chars": null}. Printing the dict buries
        # the one number the listing exists to show.
        if isinstance(limit, dict):
            limit = limit.get("tokens")
        owned = m.get("owned_by")
        if owned:
            extra += f" owned_by={owned}"
        # The capability flags LLM7 publishes map 1:1 onto checklist items 1 and 2, so
        # show them: they are what makes a model worth a probe slot.
        # usage_based_only is the field that decides whether a key can call the model
        # AT ALL: LLM7 answers 402 (not 401) for a valid key against a usage-based
        # model, so the first probe here spent all four of its slots on models the
        # allowance cannot reach (2026-09-02). Printing it makes that visible before
        # a probe is queued rather than after.
        for flag in ("json_mode", "reasoning", "tier", "usage_based_only"):
            if flag in m:
                extra += f" {flag}={m[flag]}"
        avail = m.get("availability_last_hour_percent")
        if avail is not None:
            extra += f" avail={avail}%"
        print(f"  {name:<44} ctx={limit}{extra}")
    return 0


def _one_slot_router(provider: str, model: str, key_env: str) -> LLMRouter:
    """A router holding exactly the model under test — no failover.

    The point is to see THIS model's answer; a cascade would quietly hand the
    prompt to a healthy slot and report a success that says nothing.
    """
    return LLMRouter([
        LLMAccount(
            provider=provider, account_id="A", model=model,
            api_key=os.environ.get(key_env, ""),
            rpm=30, rpd=10_000,
            bucket=TokenBucket(rate_per_minute=30, daily_limit=10_000, burst=4),
        )
    ])


def _grade(items: dict) -> tuple[bool, str]:
    """Did Pass C's own parser get a usable item for every report?

    `_parse_batch_response` returns dict[report_number, item] — NOT a list. An
    earlier version of this iterated it, which walks the KEYS, and duly reported
    "#1 not an object" for four flawless classifications. Grade the values, and
    check the report numbers are the ones we asked about.

    Grading the parser's output rather than the raw text is deliberate: the parser
    salvages partly-corrupt batches, so a stricter hand-rolled check would fail
    replies Pass C would happily accept.
    """
    if not items:
        return False, "parser recovered nothing"
    expected = set(range(1, len(SAMPLE_EVENTS) + 1))
    got = {int(k) for k in items}
    if got != expected:
        missing = sorted(expected - got)
        return False, f"{len(items)}/{len(SAMPLE_EVENTS)} reports (missing {missing})"
    problems = []
    for num in sorted(got):
        item = items[num] if num in items else items[str(num)]
        if not isinstance(item, dict):
            problems.append(f"#{num} is {type(item).__name__}")
            continue
        for field in ("event_type", "relevance_score"):
            if field not in item:
                problems.append(f"#{num} missing {field}")
    if problems:
        return False, "; ".join(problems[:6])
    return True, "schema OK"


# Fixed SITREP payload in run_sitrep_llm's exact shape, for --prose. The quality
# router's whole job is Turkish narrative and this harness could only measure Pass C
# classification, so a slot could pass here and still write unreadable prose in
# production — which is precisely the open question about a slot that only runs when
# the rung above it is already dead. Three clusters on purpose: an attack with
# casualties, an aviation-impact item (the section the prompt says must never be
# summarized away), and a strategic item, plus a country-scope airspace block, which
# is the case the prompt has the most rules about ("do not say which FIR").
SAMPLE_SITREP_PAYLOAD = {
    "country": "Afganistan (AF)",
    "window": "2026-09-01 06:00 — 2026-09-02 06:00 UTC",
    "events": [
        {
            "location": "Kabil",
            "event_type": "drone_attack_critical_infra",
            "date": "1 Eylül 2026",
            "verification": "Onaylandı (Çoklu kaynak)",
            "severity": 95,
            "snippet": "A drone struck the perimeter of Hamid Karzai International "
                       "Airport early on Tuesday, killing three ground staff. The "
                       "aviation authority suspended all departures.",
            "sources": [
                {"name": "reuters.com", "url": "https://reuters.com/a",
                 "title": "Drone attack halts flights at Kabul airport"},
                {"name": "apnews.com", "url": "https://apnews.com/b",
                 "title": "Three killed in Kabul airport drone strike"},
            ],
            "country_iso": "AF",
        },
        {
            "location": "Kabil",
            "event_type": "airspace_closure",
            "date": "1 Eylül 2026",
            "verification": "Onaylandı (Tek kaynak)",
            "severity": 70,
            "snippet": "Turkish Airlines suspended its Istanbul-Kabul route until "
                       "further notice; two other carriers rerouted around Afghan "
                       "airspace.",
            "sources": [{"name": "aviationweek.com", "url": "https://aviationweek.com/c",
                         "title": "Carriers suspend Kabul routes"}],
            "country_iso": "AF",
        },
    ],
    "spillover": [],
    "strategic": [
        {
            "location": "Ülke Geneli",
            "event_type": "travel_advisory",
            "date": "2 Eylül 2026",
            "verification": "Onaylandı (Resmî kaynak)",
            "severity": 55,
            "snippet": "Several foreign ministries raised their Afghanistan travel "
                       "advisories to the highest level, citing airport security.",
            "sources": [{"name": "gov.uk", "url": "https://gov.uk/d",
                         "title": "Afghanistan travel advice updated"}],
            "country_iso": "AF",
        },
    ],
    "airspace": {
        "kapsam": "country",
        "ulkenin_firlari": [{"kod": "OAKX", "ad": "Kabul FIR", "czib_active": True}],
        "ulkenin_baslica_havalimanlari": [
            {"kod": "OAKB", "ad": "Hamid Karzai Intl"},
            {"kod": "OAKN", "ad": "Kandahar Intl"},
        ],
    },
}

# Turkish function words that no English or machine-translated-from-scratch reply
# avoids for long. Cheap language check: the failure this catches is a model that
# silently answers in English, which has happened to non-Turkish-tuned slots.
_TURKISH_MARKERS = ("ve ", "bir ", "için", "olarak", "ile ", "bu ", "daha")


def _grade_prose(text: str, result: dict) -> tuple[bool, str]:
    """Objective, checkable properties of a narrative — NOT its literary quality.

    Prose quality is a human judgment and this returns none: the text is printed in
    full for that. What it does check is the handful of things the SITREP pipeline
    will actually break on, each one a bug that has shipped before: an empty or
    truncated narrative ([[sitrep truncation]], 25% of two weeks ended mid-sentence),
    a reply in the wrong language, a missing mandatory heading, and the markdown link
    syntax the prompt forbids because the source-credit extractor cannot parse it.
    """
    problems = []
    stripped = text.strip()
    if not stripped:
        return False, "empty completion"
    words = len(stripped.split())
    hits = sum(1 for m in _TURKISH_MARKERS if m in stripped.lower())
    if hits < 3:
        problems.append(f"probably not Turkish ({hits}/{len(_TURKISH_MARKERS)} markers)")
    if "YÖNETİCİ ÖZETİ" not in stripped.upper():
        problems.append("no 'YÖNETİCİ ÖZETİ' heading")
    if re.search(r"\[[^\]]+\]\(", stripped):
        problems.append("markdown links (prompt forbids)")
    if result.get("finish_reason") == "length":
        problems.append("truncated (finish_reason=length)")
    # Corrupted aviation identifiers. The prompt's hardest rule is that FIR and
    # airport codes may ONLY come from the airspace block, since that block is
    # computed from the event's coordinates and anything else is the model's own
    # geography. Caught on the very first --prose run: the payload says OAKX and
    # laguna wrote OAKWX.
    #
    # Deliberately NOT "any uppercase token missing from the payload": the report is
    # Turkish and the prompt DEMANDS all-caps section headings, so that rule flags
    # HAVA and every other Turkish word in a heading. What is checkable without a
    # code list is a NEAR MISS — a token close to a real code but not equal to it,
    # which is what a fabricated identifier actually looks like. A wholly invented
    # code that resembles nothing in the payload is left to the human read below.
    known_codes = {c for c in re.findall(r"\b[A-Z]{4}\b", json.dumps(
        SAMPLE_SITREP_PAYLOAD.get("airspace", {}), ensure_ascii=False))}
    mangled = set()
    for token in set(re.findall(r"\b[A-Z]{3,6}\b", stripped)):
        if token in known_codes:
            continue
        for code in known_codes:
            if difflib.SequenceMatcher(None, token, code).ratio() >= 0.75:
                mangled.add(f"{token}(≠{code})")
                break
    if mangled:
        problems.append(f"mangled codes: {', '.join(sorted(mangled))}")
    # Source URLs the model altered. validate_sitrep() checks every cited URL against
    # the payload's allowed_urls and rewrites what it cannot match, so a model that
    # silently drops a path turns a real citation into an unresolvable one. Caught on
    # minimax-m2.7, which wrote apnews.com for apnews.com/b and aviationweek.com for
    # aviationweek.com/c — the domain survives, the article does not.
    allowed = set(re.findall(r"https?://[^\s\"'),]+",
                             json.dumps(SAMPLE_SITREP_PAYLOAD, ensure_ascii=False)))
    cited = set(re.findall(r"https?://[^\s\"'),]+", stripped))
    altered = cited - allowed
    if altered:
        problems.append(f"altered URLs: {', '.join(sorted(altered)[:4])}")
    note = f"{words} words" + ("; " + "; ".join(problems) if problems else "")
    return not problems, note


def probe(provider: str, model: str, key_env: str, timeout: float | None = None,
          extras: dict | None = None, prose: bool = False) -> None:
    print(f"\n{'=' * 72}\n{provider}/{model}")
    if not os.environ.get(key_env):
        print(f"  SKIP: {key_env} not set")
        return

    router = _one_slot_router(provider, model, key_env)
    # A probe asks "can this model do the job at all", so it must not inherit the
    # production read timeout: gemma-4-31b-it blew the gemini profile's 30s and the
    # run learned nothing except that 30s was not enough. Widen it here, measure the
    # real latency, and let THAT decide the production timeout.
    patched = bool(timeout or extras)
    real_get_profile = llm_client.get_profile
    if patched:
        def _override(p, m):
            base = real_get_profile(p, m)
            return dataclasses.replace(
                base,
                request_timeout=timeout or base.request_timeout,
                wall_clock_timeout=(timeout + 30) if timeout else base.wall_clock_timeout,
                # Overriding payload_extras is the POINT of a probe: the right
                # reasoning knob for a new model is what we are trying to find out,
                # and encoding a guess in model_profiles before measuring it is how
                # a silently-thinking model reaches production.
                payload_extras={**base.payload_extras, **(extras or {})},
            )

        llm_client.get_profile = _override

    # Printed AFTER the override is installed, and read through llm_client, so the
    # line describes the request that is actually about to be sent. It used to be
    # printed from the unpatched profile at the top of this function, which meant a
    # run driven by --extras '{"reasoning_effort":"low"}' still reported extras={} —
    # the one number a reader needs to interpret the result, showing the value the
    # probe was overriding rather than the value it used.
    effective = llm_client.get_profile(provider, model)
    print(f"  profile: json_mode={effective.supports_json_mode} "
          f"extras={effective.payload_extras} timeout={effective.request_timeout}"
          + ("  (overridden)" if patched else ""))
    started = time.monotonic()
    try:
        if prose:
            # The REAL narrator prompt and the REAL budget: a 6K-token completion is
            # what exposes the timeout and truncation behavior that a 2K
            # classification batch never reaches.
            result = call_llm(
                router,
                prompt=(
                    "Aşağıdaki veriden Afganistan için 24 saatlik SITREP'i yaz. "
                    "RAPOR DİLİ: TÜRKÇE (veri İngilizce olsa bile).\n\n"
                    + json.dumps(SAMPLE_SITREP_PAYLOAD, ensure_ascii=False, indent=1)
                ),
                system_prompt=SITREP_SYSTEM_PROMPT,
                max_tokens=NARRATIVE_MAX_TOKENS,
                json_mode=False,
            )
        else:
            result = call_llm(
                router,
                prompt=_batch_prompt(SAMPLE_EVENTS),
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT + BATCH_SYSTEM_SUFFIX,
                max_tokens=2048,
                json_mode=True,
            )
    except Exception as exc:
        print(f"  FAILED after {time.monotonic() - started:.1f}s: "
              f"{type(exc).__name__}: {str(exc)[:400]}")
        return
    finally:
        if patched:
            llm_client.get_profile = real_get_profile

    content = result.get("content", "")
    if prose:
        ok, note = _grade_prose(content, result)
        print(f"  latency={result.get('latency_ms')}ms  verdict={'PASS' if ok else 'FAIL'} ({note})")
        # Printed in full and unclipped: the checks above cannot judge whether the
        # Turkish is any good, and that judgment is the entire reason for this mode.
        print("  --- narrative as written ---")
        print(content)
        return
    # _parse_batch_response RAISES on unparseable content rather than returning
    # empty. An unparseable reply is a probe's most informative outcome, so it must
    # be a graded verdict here, not an exception: letting it propagate killed the
    # whole run mid-sweep and the models queued behind it were never tested.
    try:
        items = _parse_batch_response(content, expected=len(SAMPLE_EVENTS))
        ok, note = _grade(items)
    except LLMParseError as exc:
        items, ok, note = [], False, f"unparseable: {str(exc)[:120]}"
    print(f"  latency={result.get('latency_ms')}ms  verdict={'PASS' if ok else 'FAIL'} ({note})")
    print("  --- what Pass C would extract ---")
    print(json.dumps(items, indent=2, ensure_ascii=False)[:5000])
    if not ok:
        print("  --- raw content ---")
        print(str(content)[:2500])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="",
                    help="comma-separated provider:model[:KEY_ENV] (KEY_ENV defaults per provider)")
    ap.add_argument("--list", dest="list_providers", default="",
                    help="comma-separated providers whose catalogue to print (gemini,groq)")
    ap.add_argument("--timeout", type=float, default=None,
                    help="read timeout for the probe, overriding the model profile "
                         "(default 120s, or 300s with --prose)")
    ap.add_argument("--extras", default="",
                    help='JSON merged into payload_extras, e.g. \'{"reasoning_effort":"none"}\'')
    ap.add_argument("--prose", action="store_true",
                    help="run the REAL Turkish SITREP narrator prompt instead of the "
                         "Pass C batch — for quality-router slots, whose job is prose")
    args = ap.parse_args()

    for provider in [p for p in args.list_providers.split(",") if p.strip()]:
        if provider.strip() in _CATALOGUES:
            list_models(provider.strip())
        else:
            print(f"unknown provider for catalogue: {provider}")
    if not args.models:
        return 0

    default_key = {
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY_A",
        "openrouter": "OPENROUTER_API_KEY_A",
        "mistral": "MISTRAL_API_KEY",
        "llm7": "LLM7_KEY",
        "pollinations": "POLLINATIONS_API_KEY",
    }
    extras = json.loads(args.extras) if args.extras.strip() else {}
    # A 6K-token narrative is a different order of work than a 2K classification
    # batch, and mistral's own production profile already allows 180s for it — a
    # probe that timed out at 120s would report "too slow" about the harness.
    timeout = args.timeout or (300.0 if args.prose else 120.0)
    for spec in [s for s in args.models.split(",") if s.strip()]:
        # Model ids may themselves contain colons (llm7's "deepseek-v4-flash:0731"),
        # so the optional KEY_ENV is recognized by SHAPE rather than by position:
        # env vars are upper snake case, model ids in these catalogues are not.
        # Splitting on the last colon unconditionally turned that id into model
        # "deepseek-v4-flash" with key_env "0731", i.e. a silent SKIP.
        provider, _, rest = spec.strip().partition(":")
        head, _, tail = rest.rpartition(":")
        if head and tail.replace("_", "").isalnum() and tail.isupper():
            model, key_env = head, tail
        else:
            model, key_env = rest, default_key.get(provider, "")
        try:
            probe(provider, model, key_env, timeout=timeout, extras=extras,
                  prose=args.prose)
        except Exception as exc:
            # One model's crash must not cancel the models queued behind it.
            print(f"  CRASHED: {type(exc).__name__}: {str(exc)[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
