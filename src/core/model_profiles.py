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
"""

from dataclasses import dataclass, field

# Groq's free-tier TPM window doubles as a hard per-request ceiling (HTTP 413).
GROQ_MAX_REQUEST_TOKENS = 8000
# Cerebras free tier caps tokens at 30K/minute — a single request above that can
# never fit its window, so treat it as the per-request ceiling too.
CEREBRAS_MAX_REQUEST_TOKENS = 30000

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
        request_timeout=120.0 if provider == "mistral" else 30.0,
    )
