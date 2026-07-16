"""
SIM — Declarative model capability profiles

Every provider/model quirk the pipeline has been burned by lives here as DATA,
so llm_client stays quirk-free and adding a model means answering a checklist,
not waiting for the next incident. One incident per rule below:

Checklist for adding a NEW model slot:
  1. json_mode — does the provider accept response_format={"type":"json_object"}?
     OpenRouter free models return HTTP 400 on it (2026-07-08).
  2. reasoning — does the model reason by default, and which knob turns it off?
     max_tokens covers reasoning + answer COMBINED everywhere, so hidden thinking
     starves the actual reply. qwen accepts reasoning_effort="none"; gpt-oss only
     supports low/medium/high (Groq 400s on "none", 2026-07-10); Nemotron via
     OpenRouter ignores reasoning_effort and needs reasoning={"enabled": False} —
     and fails SILENTLY otherwise: HTTP 200 with garbage JSON (2026-07-10).
  3. max_request_tokens — the provider's per-request size ceiling. Groq free tier
     rejects requests above its 8K TPM window with HTTP 413 (2026-07-16); the
     client refuses oversized requests up front instead of burning a real call.
"""

from dataclasses import dataclass, field

# Groq's free-tier TPM window doubles as a hard per-request ceiling (HTTP 413).
GROQ_MAX_REQUEST_TOKENS = 8000


@dataclass(frozen=True)
class ModelProfile:
    """Capabilities and limits of one (provider, model) slot."""

    # response_format={"type":"json_object"} is accepted (and worth sending).
    supports_json_mode: bool = False
    # Extra payload entries that minimize/disable hidden reasoning.
    payload_extras: dict = field(default_factory=dict)
    # Per-request token ceiling (estimated prompt + completion); None = no ceiling.
    max_request_tokens: int | None = None


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

    return ModelProfile(
        supports_json_mode=provider in ("groq", "gemini"),
        payload_extras=extras,
        max_request_tokens=GROQ_MAX_REQUEST_TOKENS if provider == "groq" else None,
    )
