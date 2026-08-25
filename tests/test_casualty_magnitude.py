"""Casualty figures read off a headline, added 2026-08-25.

_casualty_magnitude ranks SITREP clusters, and because severity saturates at 100
across every mass-casualty cluster (see sim-sitrep-cluster-representative) the death
count is effectively what decides the lead story and which clusters survive
cap_for_prompt. So a figure invented from a headline that reported none is not a
cosmetic error — it reorders the report.

Both halves of this were measured over 1373 real headlines (10 days of SITREP
clusters plus an 800-event ingest corpus): 1357 unchanged, 15 figures recovered,
one lost, and that one was the phantom.
"""

from src.services.sitrep_generator import _ANY_CASUALTY_RE, _DEATH_RE, _largest


def deaths(title: str) -> int:
    return _largest(_DEATH_RE, title)


def casualties(title: str) -> int:
    return _largest(_ANY_CASUALTY_RE, title)


class TestVerbsThatDoNotCarryTheOutcome:
    def test_claims_about_intercepted_drones_is_not_a_death_toll(self):
        """The headline that named this bug: it ranked first in the Russia SITREP."""
        title = "Russia claims 269 Ukrainian drones intercepted within 12 hours"
        assert deaths(title) == 0
        assert casualties(title) == 0

    def test_claims_about_arrests_is_not_a_death_toll(self):
        assert deaths("Iranian Police Commander Radan Claims 6,500 Arrests") == 0

    def test_leaves_something_uncounted_is_not_a_death_toll(self):
        """'Leaves 14 Children Injured' used to add 14 to the dead."""
        assert deaths("Shopping Center Attack Leaves 14 Children Injured") == 0
        assert casualties("Shopping Center Attack Leaves 14 Children Injured") == 14

    def test_kills_still_carries_its_own_number(self):
        """kills/killed can only mean one thing, so the verb-first form stays."""
        assert deaths("Kyiv strike kills 15") == 15
        assert deaths("Russian strike killed at least 17") == 17

    def test_injures_still_carries_its_own_number(self):
        assert casualties("Blast injures 18 in Karachi market") == 18
        assert deaths("Blast injures 18 in Karachi market") == 0


class TestOutcomeStatedAfterTheFigure:
    def test_leaves_n_dead_is_still_read(self):
        """Dropping the verb costs nothing because this form already covers it."""
        assert deaths("Gang attack in Haiti leaves at least 47 dead") == 47
        assert deaths("Russian Strike Leaves 2 Dead, 141 Homes Damaged") == 2

    def test_claims_n_soldiers_killed_is_still_read(self):
        assert deaths("BLA Claims 13 Soldiers Killed Across Balochistan") == 13

    def test_auxiliary_verb_between_figure_and_outcome(self):
        """'42 people WERE killed' — the fixed noun list had 'people' but not the
        auxiliary, so the whole class was invisible."""
        assert deaths("42 people were killed in gang attack on city in Haiti") == 42
        assert casualties("Palestinian medics say at least 2 people were injured") == 2

    def test_demonym_between_figure_and_outcome(self):
        assert deaths("Israel air strikes leave 10 Palestinians dead in Gaza") == 10
        assert deaths("9 more Gazans killed by Israeli fire in last 24 hours") == 9

    def test_qualifier_between_figure_and_outcome(self):
        assert deaths("16 confirmed dead more than 40 injured") == 16
        assert casualties("16 confirmed dead more than 40 injured") == 40
        assert deaths("3 reported killed in Israeli airstrikes") == 3

    def test_the_window_does_not_jump_a_clause(self):
        """Two words is a ceiling, not an invitation: a figure about property must
        not be picked up by a death word further along the sentence."""
        assert deaths("40 trucks modified for UAV launches destroyed, 2 dead") == 2
        assert deaths("Russia downed 351 Ukrainian drones overnight") == 0


class TestLargest:
    def test_takes_the_fullest_toll_in_the_headline(self):
        title = "Russia's twin drone strike on Kryvyi Rih mall kills 16, injures 130"
        assert deaths(title) == 16
        assert casualties(title) == 130

    def test_no_figure_is_zero(self):
        assert deaths("Russian drones strike shopping centre in Kryvyi Rih") == 0
        assert casualties("") == 0
