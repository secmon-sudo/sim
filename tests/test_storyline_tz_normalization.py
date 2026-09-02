"""Storyline linking across the naive/aware datetime boundary.

`events.occurred_at_est` is a bare `TIMESTAMP`, so everything read from the
database is naive — but `resolve_occurred_at_fallback` returns a tz-AWARE value
(it clamps against `now`). From 2026-08-31 the two met in Pass D and neither
outcome was visible: `should_link_storyline` swallowed the TypeError as "not a
match", so events dated by the fallback could never link and opened a fresh
storyline every run, while the identical-hint diagnostic did the same subtraction
unguarded and killed the whole event — 171 stranded in 'classified', +13 a run.

These tests fence both halves: the comparison must be CORRECT across the
boundary, not merely non-fatal.
"""

from datetime import datetime, timedelta, timezone

from src.core.storyline import as_naive_utc, should_link_storyline


def _event(dt, hint="kabul airport drone attack", iso="AF"):
    return {"storyline_hint": hint, "country_iso": iso, "occurred_at_est": dt}


class TestAsNaiveUtc:
    def test_none_passes_through(self):
        assert as_naive_utc(None) is None

    def test_naive_is_returned_unchanged(self):
        dt = datetime(2026, 9, 2, 6, 3, 19)
        assert as_naive_utc(dt) is dt

    def test_aware_utc_loses_only_the_tzinfo(self):
        aware = datetime(2026, 9, 2, 6, 3, 19, tzinfo=timezone.utc)
        assert as_naive_utc(aware) == datetime(2026, 9, 2, 6, 3, 19)
        assert as_naive_utc(aware).tzinfo is None

    def test_offset_datetime_is_converted_not_truncated(self):
        # +03:00 09:03 is 06:03 UTC. Dropping the tzinfo without converting would
        # shift the incident three hours and could push it out of the time window.
        aware = datetime(2026, 9, 2, 9, 3, 19, tzinfo=timezone(timedelta(hours=3)))
        assert as_naive_utc(aware) == datetime(2026, 9, 2, 6, 3, 19)

    def test_output_is_always_comparable_with_naive(self):
        # storyline_clusterer and flash_detector sort with `datetime.min` as the
        # default key; an aware element makes the whole list un-sortable.
        assert as_naive_utc(datetime(2026, 9, 2, tzinfo=timezone.utc)) > datetime.min


class TestMixedAwarenessLinking:
    def test_aware_event_links_to_naive_pool_row(self):
        # THE regression: identical hint, same country, minutes apart. Before the
        # fix the subtraction raised and the gate reported "no match".
        naive = datetime(2026, 9, 2, 6, 0, 0)
        aware = datetime(2026, 9, 2, 6, 30, 0, tzinfo=timezone.utc)
        assert should_link_storyline(_event(aware), _event(naive)) is True

    def test_links_in_both_directions(self):
        naive = datetime(2026, 9, 2, 6, 0, 0)
        aware = datetime(2026, 9, 2, 6, 30, 0, tzinfo=timezone.utc)
        assert should_link_storyline(_event(naive), _event(aware)) is True

    def test_time_window_still_refuses_across_the_boundary(self):
        # Normalizing must not turn the gate into a rubber stamp: a genuinely
        # distant pair stays refused even when the two sides disagree on tzinfo.
        naive = datetime(2026, 7, 1, 6, 0, 0)
        aware = datetime(2026, 9, 2, 6, 0, 0, tzinfo=timezone.utc)
        assert should_link_storyline(_event(aware), _event(naive)) is False

    def test_offset_pair_near_window_edge_uses_utc_instant(self):
        # 14-day window. The aware side is 13d23h after the naive one in real time
        # but reads as 14d02h if the offset is dropped instead of converted.
        naive = datetime(2026, 8, 19, 6, 0, 0)
        aware = datetime(2026, 9, 2, 8, 0, 0, tzinfo=timezone(timedelta(hours=3)))
        assert should_link_storyline(_event(aware), _event(naive)) is True

    def test_country_gate_still_applies_across_the_boundary(self):
        naive = datetime(2026, 9, 2, 6, 0, 0)
        aware = datetime(2026, 9, 2, 6, 30, 0, tzinfo=timezone.utc)
        assert should_link_storyline(
            _event(aware, iso="AF"), _event(naive, iso="PK")
        ) is False


class TestPassDProducesNaiveTimes:
    def test_fallback_output_is_normalized_before_use(self):
        # resolve_occurred_at_fallback deliberately returns an AWARE value; Pass D
        # must not put that shape into the event dict the linker consumes.
        from src.pipeline.pass_d_score import resolve_occurred_at_fallback

        published = datetime(2026, 9, 2, 6, 0, 0)
        now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
        raw = resolve_occurred_at_fallback(published, None, now=now)
        assert raw.tzinfo is not None, "fallback contract changed; this test is stale"
        assert as_naive_utc(raw).tzinfo is None

    def test_identical_hint_diagnostic_cannot_raise(self):
        # The crash site: the diagnostic subtracts the two datetimes directly.
        # With both sides normalized the subtraction is defined.
        a = as_naive_utc(datetime(2026, 9, 2, 6, 0, 0, tzinfo=timezone.utc))
        b = datetime(2026, 9, 2, 5, 0, 0)
        assert abs((a - b).total_seconds()) == 3600
