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
    "minimax/minimax-m3:free",
})

# OpenRouter slots whose model REASONS BY DEFAULT, where the only knob that turns it
# off is OpenRouter's normalized `reasoning: {enabled: false}` — reasoning_effort is
# accepted and then ignored. This is the silent failure class: the request returns
# HTTP 200 while hidden thinking eats the max_tokens budget and the visible reply is
# empty or truncated garbage JSON (Nemotron, 2026-07-10). Membership is read off
# supported_parameters + the model card in https://openrouter.ai/api/v1/models: any
# model described as a reasoning model belongs here, and a model NOT listed is a
# claim that it answers directly.
OPENROUTER_REASONING_DISABLED_MODELS = frozenset({
    "nvidia/nemotron-3-super-120b-a12b:free",
    # MiniMax M3 reasons by default (advertises reasoning/reasoning_effort/
    # include_reasoning) and is tuned for "sustained, multi-step tasks", which is
    # exactly the shape that spends the budget on hidden thinking. Checked
    # 2026-09-02 when it took the secondary slots from GLM 5.2.
    "minimax/minimax-m3:free",
    # The paid floor (see llm_router._quality_slots). Gemini 3.1 Flash Lite
    # supports full thinking levels and reasons by default, and OpenRouter bills
    # reasoning tokens at the OUTPUT rate ($1.50/M). Two costs, and the second is
    # the one that matters: across all of OpenRouter's traffic to this model,
    # reasoning is 659M tokens against 2.97B completion — 22% — and every one of
    # those comes out of the 6,000-token ceiling the narrative is allowed. That is
    # the laguna problem measured on 2026-09-02, where hidden thinking took 42% of
    # max_tokens and the reader got half a report.
    #
    # Turning it off costs nothing measurable: probed both ways on the real SITREP
    # prompt, 3,299ms/278 words with reasoning and 3,288ms/241 words without, both
    # PASS. This is prose written from supplied facts, not a problem to think about.
    "google/gemini-3.1-flash-lite",
})

# LLM7 (aggregator, added 2026-09-02) publishes a per-model `json_mode` boolean in
# https://api.llm7.io/v1/models, so this is a DENYLIST rather than the allowlist used
# for OpenRouter: there the safe default was "no", here the catalogue answers the
# question for all 44 models and only these four say no. Read off the catalogue on
# 2026-09-02. Being wrong is cheap in this direction — llm_client sidelines json mode
# for a slot that 400s on response_format and retries without it — while being wrong
# the other way is not: an unconstrained batch reply drifts into malformed JSON
# mid-object (Nemotron corrupted ~7% of Pass C batches that way).
LLM7_NO_JSON_MODE_MODELS = frozenset({
    "gemini-3.7-flash",
    "glm-5.3-flash",
    "kimi-k3",
    "L3-8B-Lunaris-v1-Turbo",
})

# Checklist item 2 (which knob turns reasoning off) is DELIBERATELY UNANSWERED for
# LLM7: it is a proxy in front of many upstreams, and whether it forwards, normalizes
# or silently drops `reasoning_effort` is exactly the aggregator-parameter-fidelity
# question that got AnyAPI rejected — and the failure is silent (HTTP 200, hidden
# thinking eats max_tokens, garbage JSON comes back). Guessing a knob here would bake
# that guess into production. So llm7 slots send NO reasoning extras by default and
# scripts/probe_models.py --extras is the instrument that settles it per model; the
# answer belongs in this file afterwards, not before.
LLM7_REASONING_MODELS = frozenset({
    "Inkling", "Inkling-Small", "claude-fable-5", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-4-6", "claude-sonnet-5", "deepseek-v4-flash", "deepseek-v4-flash:0731",
    "gemini-3-flash", "gemini-3.5-flash-low", "glm-5.3", "glm-5.3-flash", "gpt-5.4",
    "gpt-5.4-mini", "gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-oss", "grok-4.6",
    "kimi-k3", "minimax-m2.7", "minimax-m3",
})

# LLM7 fronts upstream models over a proxy hop, so a completion that a first-party
# endpoint finishes in 30s can legitimately take longer here. Read timeout is widened
# accordingly; slowness stays bounded by wall_clock_timeout, which is the only thing
# that can bound it (checklist item 5).
# 180s for the same reason mistral gets it (checklist item 4): these slots write the
# 6K-token SITREP narrative, and a non-streaming completion arrives in one piece, so
# the read timeout has to cover the WHOLE generation, not a gap between chunks. The
# 60s that the classification probe justified would cap the narrative at a fraction
# of its budget — a budget the timeout won't let the model spend is not a budget.
# Slowness stays bounded by wall_clock_timeout, which scales with the asked-for
# max_tokens (wall_clock_ceiling): ~155s at a 6K budget, 60s at classification size.
LLM7_REQUEST_TIMEOUT = 180.0

# Pollinations' catalogue declares tool_calling/reasoning but says NOTHING about
# response_format, so unlike LLM7 there is no per-model answer to read off. Default to
# sending it anyway: llm_client already sidelines json mode for a slot that 400s and
# retries without it, so a wrong "yes" costs one request while a wrong "no" costs
# malformed batch JSON on every call (the Nemotron 7% corruption). The reasoning knob
# is unknown for the same reason as LLM7 — probe before declaring it.
#
# Every model reachable on the free tier there is community-contributed and flagged
# alpha: an individual's own upstream key registered into the router. They belong
# BEHIND first-party slots in a cascade, never in front of them.
POLLINATIONS_REQUEST_TIMEOUT = 180.0  # same narrative-length reasoning as LLM7 above


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
    if provider == "openrouter" and model in OPENROUTER_REASONING_DISABLED_MODELS:
        # Checked BEFORE the family rules: OpenRouter's normalized knob is the one
        # that actually lands, whatever the underlying family accepts natively.
        extras = {"reasoning": {"enabled": False}}
    elif provider in ("llm7", "pollinations", "cloudflare"):
        # Also before the family rules, and for the opposite reason: the family knobs
        # below are facts about first-party endpoints (Groq 400s on reasoning_effort
        # "none"), and nothing yet establishes that this proxy forwards them at all.
        # See LLM7_REASONING_MODELS — probe, then fill this in.
        extras = {}
    elif model.startswith("qwen"):
        extras = {"reasoning_effort": "none"}
    elif "gpt-oss" in model:
        extras = {"reasoning_effort": "low"}
    else:
        extras = {}

    if provider == "groq":
        max_request = GROQ_MAX_REQUEST_TOKENS
    else:
        max_request = None

    # mistral-large is a plain (non-reasoning) model and Mistral's API accepts
    # response_format json_object.
    # OpenRouter is per-model rather than per-provider: only the slots on the
    # verified list above take response_format.
    if provider == "openrouter":
        supports_json = model in OPENROUTER_JSON_MODE_MODELS
    elif provider == "llm7":
        # Per-model, from the provider's own catalogue — see LLM7_NO_JSON_MODE_MODELS
        # for why this one defaults to yes where OpenRouter defaults to no.
        supports_json = model not in LLM7_NO_JSON_MODE_MODELS
    elif provider == "pollinations":
        supports_json = True  # undeclared; verified at runtime, see the note above
    elif provider == "cloudflare":
        # Workers AI documents response_format on its OpenAI-compatible route, but
        # per model — and this slot exists for PROSE, where json_mode is wrong
        # anyway. Left off until a probe says otherwise; the 400-retry path in
        # llm_client is the safety net if a caller ever asks for it.
        supports_json = False
    else:
        supports_json = provider in ("groq", "gemini", "mistral")

    if provider == "mistral":
        request_timeout = 180.0
    elif provider == "llm7":
        request_timeout = LLM7_REQUEST_TIMEOUT
    elif provider == "pollinations":
        request_timeout = POLLINATIONS_REQUEST_TIMEOUT
    elif provider == "cloudflare":
        # Same allowance as Mistral, and for the same reason: this slot carries
        # 6K-token narratives, not classification batches, and a 30s read timeout
        # would abandon a generation that is progressing normally.
        request_timeout = 180.0
    else:
        request_timeout = 30.0

    return ModelProfile(
        supports_json_mode=supports_json,
        payload_extras=extras,
        max_request_tokens=max_request,
        request_timeout=request_timeout,
        wall_clock_timeout=(
            MISTRAL_WALL_CLOCK_SECONDS if provider in ("mistral", "cloudflare")
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
