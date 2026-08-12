"""
SIM — Declarative model capability profiles

Every provider/model quirk the pipeline has been burned by lives here as DATA,
so llm_client stays quirk-free and adding a model means answering a checklist,
not waiting for the next incident. One incident per rule below:

Checklist for adding a NEW model slot:
  1. json_mode — does the provider accept response_format={"type":"json_object"}?
     Answer per MODEL on OpenRouter, not per provider: free models 400'd on it
     across the board in 2026-07-08, but some now advertise it (see
     OPENROUTER_JSON_MODE_MODELS). Check the model's supported_parameters in
     https://openrouter.ai/api/v1/models before adding a slot to that list.
  2. reasoning — does the model reason by default, and which knob turns it off?
     max_tokens covers reasoning + answer COMBINED everywhere, so hidden thinking
     starves the actual reply. qwen accepts reasoning_effort="none"; gpt-oss only
     supports low/medium/high (Groq 400s on "none", 2026-07-10); Nemotron via
     OpenRouter ignores reasoning_effort and needs reasoning={"enabled": False} —
     and fails SILENTLY otherwise: HTTP 200 with garbage JSON (2026-07-10).
  3. max_request_tokens — the provider's per-request size ceiling. Groq free tier
     rejects requests above its 8K TPM window with HTTP 413 (2026-07-16); the
     client refuses oversized requests up front instead of burning a real call.
  4. request_timeout — how long a long completion actually takes on this provider.
     mistral-large needs >30s for a 4K-token SITREP; the old fixed 30s timeout
     made every call ReadTimeout and restart generation from scratch (2026-07-17).
     Raised again to 180s on 2026-08-10 when the SITREP budget went 4K → 6K tokens:
     a budget the timeout won't let the model spend is not a budget, it is a
     ReadTimeout with extra steps.
  5. wall_clock_timeout — how long ONE request may take end to end. Distinct from
     request_timeout, which httpx applies BETWEEN chunks: OpenRouter drips keepalive
     bytes while an upstream generates, so the 30s read timeout never fires no matter
     how slow the upstream is (2026-08-12: nemotron batches of 44-108s under a 30s
     read timeout, Pass C 322s → 1240s across three runs with no error logged).
     A read timeout bounds silence; only this bounds slowness.
"""

from dataclasses import dataclass, field

# Groq's free-tier TPM window doubles as a hard per-request ceiling (HTTP 413).
GROQ_MAX_REQUEST_TOKENS = 8000
# Cerebras free tier caps tokens at 30K/minute — a single request above that can
# never fit its window, so treat it as the per-request ceiling too.
CEREBRAS_MAX_REQUEST_TOKENS = 30000

# Wall-clock ceiling for a single request, enforced by llm_client on top of the httpx
# read timeout (see checklist item 5). A classification batch that outruns this is not
# a slow success worth waiting for — Pass C is sequential, so every extra second is a
# second of run duration, and the cascade's next slot answers the same prompt in ~10s.
# Sized off measured behavior: healthy nemotron batches ran ~30s, the degraded ones
# 44-108s (2026-08-12). Raise it if healthy slots start tripping; do not raise it to
# accommodate a slot that has simply gone slow — that is what the failover is for.
DEFAULT_WALL_CLOCK_SECONDS = 60.0
# mistral-large legitimately spends minutes on a 6K-token SITREP (request_timeout=180),
# and no other slot in the quality cascade writes prose as well, so its ceiling is set
# to outlast a full-length generation rather than to police it.
MISTRAL_WALL_CLOCK_SECONDS = 300.0

# One full classification batch's completion budget (450 * 4 + 512) — the size the
# ceilings above are calibrated on. See wall_clock_ceiling().
WALL_CLOCK_REFERENCE_TOKENS = 2312

# OpenRouter free slots that DO accept response_format. The blanket "OpenRouter free
# 400s on response_format" rule (2026-07-08) was true then but is no longer: both slots
# below advertise response_format + structured_outputs in the models API (re-checked
# 2026-08-06). It matters because unconstrained batch replies drift into malformed JSON
# mid-object — Nemotron corrupted ~7% of Pass C batches that way (2026-08-05/06), each
# one stranding a whole 6-event chunk. Anything NOT listed here keeps the old behavior.
# This list is a claim about the provider, so llm_client verifies it at runtime: a 400
# that disappears when response_format is dropped disables json mode for that slot
# instead of letting the slot die.
OPENROUTER_JSON_MODE_MODELS = frozenset({
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-20b:free",
})


@dataclass(frozen=True)
class ModelProfile:
    """Capabilities and limits of one (provider, model) slot."""

    # response_format={"type":"json_object"} is accepted (and worth sending).
    supports_json_mode: bool = False
    # Extra payload entries that minimize/disable hidden reasoning.
    payload_extras: dict = field(default_factory=dict)
    # Per-request token ceiling (estimated prompt + completion); None = no ceiling.
    max_request_tokens: int | None = None
    # HTTP read timeout for one request. Fast-inference providers finish a 4K-token
    # completion well under 30s; mistral-large does not (ReadTimeout storm,
    # 2026-07-17) — and each timeout retry restarts the generation from scratch.
    request_timeout: float = 30.0
    # Floor for the end-to-end ceiling enforced by llm_client (see wall_clock_ceiling).
    # Bounds SLOWNESS, which request_timeout cannot: a keepalive-dripping endpoint
    # resets the read clock forever. Exceeding it sidelines the slot and rotates.
    wall_clock_timeout: float = DEFAULT_WALL_CLOCK_SECONDS


def get_profile(provider: str, model: str) -> ModelProfile:
    """Resolve the capability profile for a (provider, model) pair."""
    if model.startswith("qwen"):
        extras = {"reasoning_effort": "none"}
    elif "gpt-oss" in model:
        extras = {"reasoning_effort": "low"}
    elif "nemotron" in model and provider == "openrouter":
        extras = {"reasoning": {"enabled": False}}
    else:
        extras = {}

    if provider == "groq":
        max_request = GROQ_MAX_REQUEST_TOKENS
    elif provider == "cerebras":
        max_request = CEREBRAS_MAX_REQUEST_TOKENS
    else:
        max_request = None

    # mistral-large is a plain (non-reasoning) model and Mistral's API accepts
    # response_format json_object. Cerebras serves gpt-oss with the same
    # reasoning_effort knob as Groq and supports json_object (verify on first
    # prod run per the checklist — a 400 would sideline the slot, not break it).
    # OpenRouter is per-model rather than per-provider: only the slots on the
    # verified list above take response_format.
    if provider == "openrouter":
        supports_json = model in OPENROUTER_JSON_MODE_MODELS
    else:
        supports_json = provider in ("groq", "gemini", "mistral", "cerebras")

    return ModelProfile(
        supports_json_mode=supports_json,
        payload_extras=extras,
        max_request_tokens=max_request,
        request_timeout=180.0 if provider == "mistral" else 30.0,
        wall_clock_timeout=(
            MISTRAL_WALL_CLOCK_SECONDS if provider == "mistral"
            else DEFAULT_WALL_CLOCK_SECONDS
        ),
    )


def wall_clock_ceiling(profile: ModelProfile, max_tokens: int) -> float:
    """End-to-end ceiling for one request that asked for `max_tokens` of completion.

    The profile's value is calibrated on a classification batch. A caller asking for a
    much larger completion — the 6K-token SITREP narrative — is not slow, it is doing
    more work, and policing it with the classification ceiling would sideline every slot
    capable of writing it: the quality cascade falls through to the main cascade, whose
    Groq slots the size guard already skips at that budget, leaving nothing. So the
    ceiling scales with the budget the caller asked for, and never drops below the
    slot's own floor.
    """
    scaled = DEFAULT_WALL_CLOCK_SECONDS * max(1.0, max_tokens / WALL_CLOCK_REFERENCE_TOKENS)
    return max(profile.wall_clock_timeout, scaled)
