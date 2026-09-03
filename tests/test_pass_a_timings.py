"""Pass A phase timing, added 2026-08-24.

Pass A is 78% of run wall-clock (measured over 43 runs) and carried no timing at
all, so every account of WHY was inference from local reruns. Local measurement
put feeds at ~30s, article fetches at ~206s and content dedup at 130-260s against
a 952s total — the gap is what these counters exist to name, which is why
`unaccounted` is reported rather than quietly absorbed.

The translation counter is separate because translation is the only network call
Pass A makes per ITEM rather than per inserted event, and it runs BEFORE the noise
filter and content dedup discard that item — so its volume appears in no existing
counter.
"""

import time

from src.pipeline import ingest_sources as ing
from src.pipeline import pass_a_ingest as pa
from src.pipeline.ingest_sources import (
    reset_translation_counter,
    translate_to_english_if_needed,
    translation_call_count,
)


class TestPhaseTimer:
    def test_accumulates_across_calls(self):
        """Loop phases are summed over ~1000 iterations, not measured once."""
        acc = {}
        for _ in range(3):
            with pa._timed(acc, "phase"):
                time.sleep(0.01)
        assert acc["phase"] >= 0.03

    def test_records_even_when_the_body_raises(self):
        """A phase that throws still spent its time; losing it would flatter the total."""
        acc = {}
        try:
            with pa._timed(acc, "boom"):
                time.sleep(0.01)
                raise ValueError("x")
        except ValueError:
            pass
        assert acc["boom"] >= 0.01

    def test_separate_keys_do_not_collide(self):
        acc = {}
        with pa._timed(acc, "a"):
            time.sleep(0.01)
        with pa._timed(acc, "b"):
            time.sleep(0.01)
        assert set(acc) == {"a", "b"}


class TestTranslationCounter:
    def test_latin_text_is_not_a_network_call(self):
        reset_translation_counter()
        translate_to_english_if_needed("Russian strike on Kyiv kills 16")
        assert translation_call_count() == 0

    def test_empty_text_is_not_a_call(self):
        reset_translation_counter()
        translate_to_english_if_needed("")
        assert translation_call_count() == 0

    def test_reset_clears(self):
        reset_translation_counter()
        assert translation_call_count() == 0


class TestTranslationProviderChain:
    """Added 2 Sep 2026: Google answered the long-standing ``client=gtx`` route
    with a 429 abuse page from every IP. The failure path returns the ORIGINAL
    text, so eight production runs translated 0/12..0/28 items while every
    counter and the run's exit status stayed green. The chain exists so one
    route dying is survivable; the failure counter exists so it is visible.
    """

    def _chain(self, monkeypatch, *behaviours):
        """Replace the provider rungs with callables, keeping their names."""
        names = [name for name, _ in ing._TRANSLATE_PROVIDERS]
        assert len(behaviours) == len(names)
        monkeypatch.setattr(
            ing, "_TRANSLATE_PROVIDERS", tuple(zip(names, behaviours))
        )

    def test_first_working_rung_wins(self, monkeypatch):
        def boom(text, target):
            raise AssertionError("later rungs must not be reached")

        self._chain(monkeypatch, lambda t, tgt: "translated", boom, boom)
        reset_translation_counter()
        assert ing.google_translate("צה\"ל תוקף") == "translated"
        assert ing.translation_failure_count() == 0

    def test_falls_through_a_dead_rung(self, monkeypatch):
        def dead(text, target):
            raise RuntimeError("429")

        self._chain(monkeypatch, dead, lambda t, tgt: "from clients5", dead)
        reset_translation_counter()
        assert ing.google_translate("צה\"ל תוקף") == "from clients5"
        assert ing.translation_failure_count() == 0

    def test_empty_response_is_a_failed_rung_not_a_translation(self, monkeypatch):
        """A 200 carrying nothing must not overwrite the headline with ''."""
        self._chain(
            monkeypatch, lambda t, tgt: "", lambda t, tgt: "", lambda t, tgt: "last"
        )
        reset_translation_counter()
        assert ing.google_translate("צה\"ל תוקף") == "last"

    def test_total_outage_returns_the_original_and_is_counted(self, monkeypatch):
        def dead(text, target):
            raise RuntimeError("429")

        self._chain(monkeypatch, dead, dead, dead)
        reset_translation_counter()
        assert ing.google_translate("צה\"ל תוקף") == "צה\"ל תוקף"
        assert ing.translation_failure_count() == 1

    def test_reset_clears_failures(self, monkeypatch):
        def dead(text, target):
            raise RuntimeError("429")

        self._chain(monkeypatch, dead, dead, dead)
        reset_translation_counter()
        ing.google_translate("צה\"ל תוקף")
        assert ing.translation_failure_count() == 1
        reset_translation_counter()
        assert ing.translation_failure_count() == 0

    def test_blank_text_never_reaches_a_provider(self, monkeypatch):
        def boom(text, target):
            raise AssertionError("blank text must short-circuit")

        self._chain(monkeypatch, boom, boom, boom)
        reset_translation_counter()
        assert ing.google_translate("   ") == "   "
        assert ing.translation_failure_count() == 0
