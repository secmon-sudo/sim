"""Bake-off harness: does a candidate model actually do Pass C's job?

Free model slots die often — three did on 2026-09-02 alone (Cerebras' free tier,
OpenRouter's whole free gpt-oss family, mistral-large leaving its subscription
tier). Picking the replacement from catalog copy is guesswork: a card that says
"1M context, reasoning" says nothing about whether the model returns the batch
schema Pass C parses. This runs the REAL classification prompt through a
candidate and prints what came back, so the choice is made on output.

Runs where the keys live (GitHub Actions, see model-probe.yml). Prints a verdict
per model and dumps raw content, since a malformed reply is the interesting case.

  python -m scripts.probe_models --models gemini:gemma-4-31b-it,gemini:gemma-4-26b-it
  python -m scripts.probe_models --list-gemini      # discover exact model ids
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


def list_gemini_models() -> int:
    """Ask the Gemini API for the exact model ids this key can see.

    Worth a separate mode: the rate-limit dashboard lists models by DISPLAY name
    ("Gemma 4 31B"), and the id the API wants is not derivable from it. Guessing
    the slug is how a slot ends up 404-ing in production.
    """
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("GEMINI_API_KEY not set")
        return 1
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}&pageSize=200"
    with urllib.request.urlopen(url, timeout=45) as r:
        data = json.loads(r.read())
    models = data.get("models", [])
    print(f"=== {len(models)} models visible to GEMINI_API_KEY ===")
    for m in sorted(models, key=lambda x: x.get("name", "")):
        name = m.get("name", "").replace("models/", "")
        methods = ",".join(m.get("supportedGenerationMethods", []))
        limit = m.get("inputTokenLimit")
        print(f"  {name:<44} in={limit} methods={methods}")
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


def _grade(items: list) -> tuple[bool, str]:
    """Did Pass C's own parser get usable items out of the reply?

    Grades what production extracts, not what the model literally sent: the
    parser has a salvage path for partly-corrupt batches, so hand-rolling a
    stricter check here would fail models that Pass C would actually accept —
    and pass ones it would not. `content` is the field pass_c reads; `response`
    is the raw provider envelope, which is what an earlier version of this
    graded, reporting FAIL for a flawless reply.
    """
    if not items:
        return False, "parser recovered nothing"
    if len(items) != len(SAMPLE_EVENTS):
        return False, f"{len(items)} items for {len(SAMPLE_EVENTS)} reports"
    missing = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            missing.append(f"#{i} not an object")
            continue
        for field in ("event_type", "relevance_score"):
            if field not in item:
                missing.append(f"#{i} missing {field}")
    if missing:
        return False, "; ".join(missing[:6])
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
    ap.add_argument("--list-gemini", action="store_true",
                    help="print exact model ids the Gemini key can see, then exit")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="read timeout for the probe, overriding the model profile")
    ap.add_argument("--extras", default="",
                    help='JSON merged into payload_extras, e.g. \'{"reasoning_effort":"none"}\'')
    args = ap.parse_args()

    if args.list_gemini:
        return list_gemini_models()

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
