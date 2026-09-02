"""Model capability profiles — the declarative quirk registry.

Each assertion pins a rule learned from a production incident (see the
checklist in src/core/model_profiles.py). If a profile changes, the matching
incident class reopens — these tests are the regression fence.
"""
from src.core.model_profiles import (
    GROQ_MAX_REQUEST_TOKENS,
    get_profile,
)


class TestJsonMode:
    def test_groq_and_gemini_support_json_mode(self):
        assert get_profile("groq", "openai/gpt-oss-120b").supports_json_mode
        assert get_profile("gemini", "gemini-3.1-flash-lite").supports_json_mode

    def test_quality_tier_providers_support_json_mode(self):
        assert get_profile("mistral", "mistral-medium-latest").supports_json_mode

    def test_openrouter_verified_free_models_do(self):
        # Re-verified against the models API 2026-09-02: these two advertise
        # response_format + structured_outputs, and json mode is what stops batch
        # replies from drifting into malformed JSON mid-object.
        assert get_profile("openrouter", "minimax/minimax-m3:free").supports_json_mode
        assert get_profile("openrouter", "nvidia/nemotron-3-super-120b-a12b:free").supports_json_mode

    def test_openrouter_is_per_model_not_per_provider(self):
        # The 2026-07-08 blanket 400 still applies to anything unverified — an
        # unlisted OpenRouter slot must not inherit json mode from its neighbors.
        assert not get_profile("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free").supports_json_mode
        assert not get_profile("openrouter", "google/gemma-4-31b-it:free").supports_json_mode


class TestReasoningGate:
    def test_qwen_disables_reasoning_entirely(self):
        assert get_profile("groq", "qwen/qwen3.6-27b").payload_extras == {"reasoning_effort": "none"}

    def test_gpt_oss_uses_lowest_valid_effort(self):
        # Groq 400s on reasoning_effort="none" for gpt-oss (2026-07-10).
        for provider, model in [("groq", "openai/gpt-oss-120b"),
                                ("groq", "openai/gpt-oss-20b")]:
            assert get_profile(provider, model).payload_extras == {"reasoning_effort": "low"}

    def test_openrouter_reasoning_models_need_full_toggle(self):
        # reasoning_effort does NOT tame these; they fail silently — HTTP 200 with
        # hidden thinking eating the budget (Nemotron, 2026-07-10). MiniMax M3 joined
        # the list on 2026-09-02 as a self-described reasoning model.
        for model in ("nvidia/nemotron-3-super-120b-a12b:free", "minimax/minimax-m3:free"):
            assert get_profile("openrouter", model).payload_extras == \
                {"reasoning": {"enabled": False}}

    def test_retired_free_gpt_oss_slugs_are_not_configured(self):
        # OpenRouter dropped the whole free gpt-oss family from its catalog
        # (2026-09-02): both :free slugs 404'd on every call. Re-adding one is a
        # 404 storm, so the router must not name them.
        import src.core.llm_router as R
        source = open(R.__file__).read()
        assert "openai/gpt-oss-20b:free" not in source
        assert "openai/gpt-oss-120b:free" not in source


class TestRequestSizeCeiling:
    def test_groq_has_8k_request_ceiling(self):
        # Groq rejects requests above its TPM window with HTTP 413 (2026-07-16).
        assert get_profile("groq", "openai/gpt-oss-20b").max_request_tokens == GROQ_MAX_REQUEST_TOKENS

    def test_openrouter_and_gemini_have_no_ceiling(self):
        assert get_profile("openrouter", "minimax/minimax-m3:free").max_request_tokens is None
        assert get_profile("gemini", "gemini-3.1-flash-lite").max_request_tokens is None

    def test_mistral_large_is_plain_model(self):
        assert get_profile("mistral", "mistral-medium-latest").payload_extras == {}
        assert get_profile("mistral", "mistral-medium-latest").max_request_tokens is None


class TestRequestTimeout:
    def test_mistral_gets_long_completion_timeout(self):
        # mistral-large ReadTimeout storm on 4K-token SITREPs (2026-07-17). Raised
        # to 180s on 2026-08-10 with the narrative budget (4K → 6K tokens): the
        # timeout has to outlast the budget or the extra tokens are unspendable.
        from src.services.sitrep_generator import NARRATIVE_MAX_TOKENS
        assert get_profile("mistral", "mistral-medium-latest").request_timeout == 180.0
        assert NARRATIVE_MAX_TOKENS <= 6000, \
            "raising the narrative budget again needs the mistral timeout raised with it"

    def test_fast_providers_keep_default(self):
        assert get_profile("groq", "openai/gpt-oss-20b").request_timeout == 30.0
        assert get_profile("openrouter", "nvidia/nemotron-3-super-120b-a12b:free").request_timeout == 30.0
