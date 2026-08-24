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
