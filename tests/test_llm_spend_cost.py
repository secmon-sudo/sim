"""Pricing and reconciliation for the spend report (6 Sep 2026).

Token counts answer "what did we spend it on"; they never answered "how much".
Until 2026-09-04 nothing here cost money, so the difference did not matter — and
once it did, the arithmetic was being done by hand, which is fine once and a
liability every day after.
"""

import pytest

from scripts.llm_spend import openrouter_actual, price_row, render_money


class TestPricing:
    PRICES = {
        "google/gemini-3.1-flash-lite": (0.25, 1.50),
        "anthropic/claude-haiku-4.5": (1.00, 5.00),
        "nvidia/nemotron-3-super-120b-a12b:free": (0.0, 0.0),
    }

    def test_it_matches_the_figure_reconciled_against_the_bill(self):
        """Five SITREP calls on 6 Sep: 50,096 prompt and 8,748 completion
        tokens. Computed by hand at $0.0256 that day and reconciled against
        OpenRouter's own total to 2.4%."""
        row = {"provider": "openrouter", "model": "google/gemini-3.1-flash-lite",
               "bucket": "x", "prompt_tokens": 50_096, "completion_tokens": 8_748}
        assert price_row(row, self.PRICES) == pytest.approx(0.0256, abs=0.0002)

    def test_a_free_slot_costs_nothing(self):
        row = {"provider": "openrouter",
               "model": "nvidia/nemotron-3-super-120b-a12b:free", "bucket": "x",
               "prompt_tokens": 500_000, "completion_tokens": 100_000}
        assert price_row(row, self.PRICES) == 0.0

    def test_an_unknown_model_is_free_not_an_error(self):
        """Only models we pay for appear in the catalogue, so an unknown key is
        a free slot. Raising here would make the whole report fail over a slot
        that cost nothing."""
        row = {"provider": "openrouter", "model": "some/slot-we-never-paid-for",
               "bucket": "x", "prompt_tokens": 9_000, "completion_tokens": 3_000}
        assert price_row(row, self.PRICES) == 0.0

    def test_purpose_rows_carry_no_model_and_price_at_zero(self):
        """A purpose spreads across slots at whatever mix the router chose, so
        pricing it per row would be a guess wearing a decimal point. The total
        is reported once at the bottom instead."""
        row = {"provider": None, "model": None, "bucket": "sitrep_country",
               "prompt_tokens": 50_000, "completion_tokens": 9_000}
        assert price_row(row, self.PRICES) == 0.0


class TestRendering:
    def test_no_prices_still_prints_something_useful(self, monkeypatch):
        """A spend report that cannot reach the internet must still print the
        token table it was always able to print."""
        import scripts.llm_spend as m

        monkeypatch.setattr(m, "fetch_prices", lambda: {})
        out = render_money([], 7, "model")
        assert "prices unavailable" in out
        assert "unaffected" in out

    def test_the_reconciliation_line_appears_when_the_key_answers(self, monkeypatch):
        import scripts.llm_spend as m

        monkeypatch.setattr(m, "fetch_prices",
                            lambda: {"a/b": (1.0, 2.0)})
        monkeypatch.setattr(m, "openrouter_actual",
                            lambda: {"usage": 0.1412, "limit": 10.0})
        out = m.render_money(
            [{"provider": "openrouter", "model": "a/b", "bucket": "openrouter a/b",
              "prompt_tokens": 1_000_000, "completion_tokens": 0}], 7, "model")
        assert "$0.1412" in out
        assert "credit left $9.86" in out


def test_no_key_means_no_reconciliation(monkeypatch):
    """The prices need no credential; only the bill does."""
    monkeypatch.delenv("OPENROUTER_API_KEY_A", raising=False)
    assert openrouter_actual() is None


def test_a_free_groq_slot_is_not_billed_at_openrouter_rates():
    """The bug this report found on its own first run. `qwen/qwen3.6-27b` runs
    here on Groq's free tier and is also sold by OpenRouter; pricing by model
    name alone reported $0.53 against a real bill of $0.14."""
    prices = {"qwen/qwen3.6-27b": (0.45, 3.20)}
    groq = {"provider": "groq", "model": "qwen/qwen3.6-27b", "bucket": "groq …",
            "prompt_tokens": 500_000, "completion_tokens": 60_000}
    assert price_row(groq, prices) == 0.0
    paid = dict(groq, provider="openrouter")
    assert price_row(paid, prices) > 0.0
