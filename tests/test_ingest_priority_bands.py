"""
The per-run insert budget must cut the least valuable items, not the deepest ones.

_interleave_by_domain round-robins across source domains for diversity. That
round-robin is depth-first: round 0 takes one item from EVERY domain before any
domain gets a second. SIM draws on more contributing domains than
max_events_per_run (100), so round 0 alone exhausted the budget and depth 1 was
unreachable — priority_score had no influence whatsoever on what survived.

Measured on the runs to 2026-08-10, the three that hit the cap:

    run 1440   inserted 100   median priority 1   dropped an item scoring 7
    run 1436   inserted 100   median priority 1   dropped an item scoring 5
    run 1431   inserted 100   median priority 1   dropped an item scoring 9

which is the exact inversion the priority mechanism was added to prevent
(2026-07-17). Banding restores it: high-priority items round-robin first, the rest
round-robin after, so diversity still orders each band and importance decides which
band the budget is spent on.
"""

from datetime import datetime, timezone

import pytest

from src.pipeline.pass_a_ingest import PRIORITY_BAND_MIN, _interleave_by_domain

T0 = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)

# Real headline shapes, chosen so priority_score puts them either side of the band.
HIGH = "Massive missile strike kills 21 in Kyiv as air defences fail"
LOW = "Minister discusses budget policy at annual economic forum"


def _item(domain, title, n=0):
    return {"domain": domain, "link": f"https://{domain}/{n}", "title": title,
            "description": "", "pub_dt": T0}


def _priorities(ordered):
    return [i["_priority"] for i in ordered]


class TestFixtureSanity:
    def test_the_two_headline_shapes_land_either_side_of_the_band(self):
        [hi] = _interleave_by_domain([_item("a.com", HIGH)])
        [lo] = _interleave_by_domain([_item("b.com", LOW)])
        assert hi["_priority"] >= PRIORITY_BAND_MIN > lo["_priority"]


class TestBudgetCutsTheLeastValuable:
    """Within a domain the bucket is already sorted by priority, so a domain's best
    item is always its round-0 item and always survives. The inversion lives at
    depth >= 1: a wire that files TWO major stories in one window got one slot, and
    the second was cut so that a hundred other domains' routine leads could be kept."""

    def test_a_domains_second_major_story_beats_other_domains_routine_leads(self):
        items = [_item("wire.com", HIGH, 0), _item("wire.com", HIGH, 1)]
        items += [_item(f"d{i}.com", LOW, 0) for i in range(150)]

        ordered = _interleave_by_domain(items)
        budget = ordered[:100]

        assert sum(1 for i in budget if i["title"] == HIGH) == 2, \
            "both major stories must fit before routine leads consume the budget"

    def test_dropped_priority_never_exceeds_inserted_median(self):
        """Restates the telemetry invariant that failed in production: median
        priority 1 inserted while items scoring 5, 7 and 9 were dropped."""
        import statistics

        items = []
        for i in range(10):                       # ten wires, two big stories each
            items += [_item(f"wire{i}.com", HIGH, n) for n in range(2)]
        items += [_item(f"d{i}.com", LOW, 0) for i in range(120)]

        ordered = _interleave_by_domain(items)
        inserted, dropped = ordered[:100], ordered[100:]
        assert dropped, "test needs the budget to actually bind"
        assert max(_priorities(dropped)) <= statistics.median(_priorities(inserted))


class TestDiversityIsPreserved:
    def test_one_domain_cannot_monopolise_the_high_band(self):
        # A single outlet publishing ten big stories must not take ten slots before
        # other outlets' big stories get one.
        items = [_item("flood.com", HIGH, n) for n in range(10)]
        items += [_item(f"other{i}.com", HIGH, 0) for i in range(5)]

        ordered = _interleave_by_domain(items)
        first_round = ordered[:6]
        assert len({i["domain"] for i in first_round}) == 6

    def test_low_band_still_round_robins(self):
        items = [_item("a.com", LOW, n) for n in range(3)]
        items += [_item("b.com", LOW, n) for n in range(3)]
        ordered = _interleave_by_domain(items)
        assert ordered[0]["domain"] != ordered[1]["domain"]

    def test_every_item_is_kept_exactly_once(self):
        items = [_item(f"d{i}.com", HIGH if i % 3 == 0 else LOW, 0) for i in range(30)]
        items += [_item(f"d{i}.com", LOW, 1) for i in range(10)]
        ordered = _interleave_by_domain(items)
        assert len(ordered) == len(items)
        assert {id(i) for i in ordered} == {id(i) for i in items}


class TestEdges:
    def test_empty_input(self):
        assert _interleave_by_domain([]) == []

    @pytest.mark.parametrize("title", [HIGH, LOW])
    def test_single_item(self, title):
        assert len(_interleave_by_domain([_item("a.com", title)])) == 1

    def test_all_one_band_behaves_like_the_old_round_robin(self):
        items = [_item(f"d{i}.com", LOW, 0) for i in range(5)]
        ordered = _interleave_by_domain(items)
        assert len({i["domain"] for i in ordered}) == 5
