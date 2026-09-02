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
import json
import os
import sys
import time
import urllib.request

from src.core import llm_client
from src.core.llm_client import call_llm
from src.core.llm_router import LLMAccount, LLMRouter
from src.core.model_profiles import get_profile
from src.core.token_bucket import TokenBucket
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

    rows = data.get("models") or data.get("data") or []
    print(f"\n=== {len(rows)} models visible to {key_env} ({provider}) ===")
    for m in sorted(rows, key=lambda x: str(x.get("name") or x.get("id"))):
        name = str(m.get("name") or m.get("id")).replace("models/", "")
        methods = ",".join(m.get("supportedGenerationMethods", []))
        extra = f" methods={methods}" if methods else ""
        limit = m.get("inputTokenLimit") or m.get("context_window")
        owned = m.get("owned_by")
        if owned:
            extra += f" owned_by={owned}"
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


def probe(provider: str, model: str, key_env: str, timeout: float | None = None,
          extras: dict | None = None) -> None:
    profile = get_profile(provider, model)
    print(f"\n{'=' * 72}\n{provider}/{model}")
    print(f"  profile: json_mode={profile.supports_json_mode} extras={profile.payload_extras} "
          f"timeout={profile.request_timeout}")
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
    started = time.monotonic()
    try:
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
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="read timeout for the probe, overriding the model profile")
    ap.add_argument("--extras", default="",
                    help='JSON merged into payload_extras, e.g. \'{"reasoning_effort":"none"}\'')
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
    }
    extras = json.loads(args.extras) if args.extras.strip() else {}
    for spec in [s for s in args.models.split(",") if s.strip()]:
        parts = spec.strip().split(":")
        provider, model = parts[0], parts[1]
        key_env = parts[2] if len(parts) > 2 else default_key.get(provider, "")
        try:
            probe(provider, model, key_env, timeout=args.timeout, extras=extras)
        except Exception as exc:
            # One model's crash must not cancel the models queued behind it.
            print(f"  CRASHED: {type(exc).__name__}: {str(exc)[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
