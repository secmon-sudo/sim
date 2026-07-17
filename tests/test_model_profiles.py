"""Model capability profiles — the declarative quirk registry.

Each assertion pins a rule learned from a production incident (see the
checklist in src/core/model_profiles.py). If a profile changes, the matching
incident class reopens — these tests are the regression fence.
"""
from src.core.model_profiles import (
    CEREBRAS_MAX_REQUEST_TOKENS,
    GROQ_MAX_REQUEST_TOKENS,
    get_profile,
)


class TestJsonMode:
    def test_groq_and_gemini_support_json_mode(self):
        assert get_profile("groq", "openai/gpt-oss-120b").supports_json_mode
        assert get_profile("gemini", "gemini-3.1-flash-lite").supports_json_mode

    def test_quality_tier_providers_support_json_mode(self):
        assert get_profile("mistral", "mistral-large-2512").supports_json_mode
        assert get_profile("cerebras", "gpt-oss-120b").supports_json_mode

    def test_openrouter_free_models_do_not(self):
        # OpenRouter free models 400 on response_format (2026-07-08).
        assert not get_profile("openrouter", "openai/gpt-oss-20b:free").supports_json_mode
        assert not get_profile("openrouter", "nvidia/nemotron-3-super-120b-a12b:free").supports_json_mode


class TestReasoningGate:
    def test_qwen_disables_reasoning_entirely(self):
        assert get_profile("groq", "qwen/qwen3.6-27b").payload_extras == {"reasoning_effort": "none"}

    def test_gpt_oss_uses_lowest_valid_effort(self):
        # Groq 400s on reasoning_effort="none" for gpt-oss (2026-07-10).
        for provider, model in [("groq", "openai/gpt-oss-120b"),
                                ("groq", "openai/gpt-oss-20b"),
                                ("openrouter", "openai/gpt-oss-20b:free")]:
            assert get_profile(provider, model).payload_extras == {"reasoning_effort": "low"}

    def test_nemotron_on_openrouter_needs_full_toggle(self):
        # reasoning_effort does NOT tame Nemotron; it fails silently (2026-07-10).
        assert get_profile("openrouter", "nvidia/nemotron-3-super-120b-a12b:free") \
            .payload_extras == {"reasoning": {"enabled": False}}


class TestRequestSizeCeiling:
    def test_groq_has_8k_request_ceiling(self):
        # Groq rejects requests above its TPM window with HTTP 413 (2026-07-16).
        assert get_profile("groq", "openai/gpt-oss-20b").max_request_tokens == GROQ_MAX_REQUEST_TOKENS

    def test_openrouter_and_gemini_have_no_ceiling(self):
        assert get_profile("openrouter", "openai/gpt-oss-20b:free").max_request_tokens is None
        assert get_profile("gemini", "gemini-3.1-flash-lite").max_request_tokens is None

    def test_cerebras_ceiling_and_reasoning_gate(self):
        # Cerebras 30K tokens/min window doubles as the per-request ceiling; its
        # gpt-oss slot takes the same reasoning_effort knob as Groq's.
        profile = get_profile("cerebras", "gpt-oss-120b")
        assert profile.max_request_tokens == CEREBRAS_MAX_REQUEST_TOKENS
        assert profile.payload_extras == {"reasoning_effort": "low"}

    def test_mistral_large_is_plain_model(self):
        assert get_profile("mistral", "mistral-large-2512").payload_extras == {}
        assert get_profile("mistral", "mistral-large-2512").max_request_tokens is None


class TestRequestTimeout:
    def test_mistral_gets_long_completion_timeout(self):
        # mistral-large ReadTimeout storm on 4K-token SITREPs (2026-07-17).
        assert get_profile("mistral", "mistral-large-2512").request_timeout == 120.0

    def test_fast_providers_keep_default(self):
        assert get_profile("groq", "openai/gpt-oss-20b").request_timeout == 30.0
        assert get_profile("cerebras", "gpt-oss-120b").request_timeout == 30.0
