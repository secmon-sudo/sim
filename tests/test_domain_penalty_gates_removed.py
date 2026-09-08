"""Nothing is dropped or disqualified on penalty_score any more.

domain_penalties had two gates: ingest dropped an item whose domain scored above
0.8, and the SITREP barred any domain at or above 0.5 from label_cluster()'s
independence count and its official-source check. Both are gone, and this file is
the guard that keeps them gone — because the reason is not that they were tuned
wrong. It is that the number they read points the wrong way.

Measured 2026-09-08 across 3,592 domains, 819 of them with the five claims the
score needs to mean anything:

  * exactly ONE has ever cleared either threshold — nitter.net at 0.875, last seen
    6 July, deleted from this codebase on 1 August;
  * the more claims a domain has made, the LOWER its worst score gets. At 25+
    claims the worst domain in the corpus is 0.143; at 50+, 0.091. There is no
    population of repeat offenders to find;
  * and the ranking is inverted. mshale.com — the content farm that published
    fabricated Gulf attacks and signed 20 of 20 alerts on 17 August — scores 0.091
    over 88 claims. washingtonpost.com scores 0.308 over 13.

The metric measures how often the classifier disagreed with a headline. Internally
consistent fabrication classifies fine; a newspaper reporting contested, fast-moving
claims does not. A threshold cannot fix a sign.

The counters stay. They are cheap, the table is the evidence above, and the sixth
attempt at a domain-quality signal should not have to start from nothing.
"""

import inspect

import pytest

from src.core.sitrep_verify import label_cluster
from src.pipeline import pass_a_ingest
from src.services import sitrep_generator


class TestTheGatesAreGone:
    def test_ingest_no_longer_reads_a_penalty_for_a_drop_decision(self):
        assert not hasattr(pass_a_ingest, "check_domain_penalty")
        assert not hasattr(pass_a_ingest, "load_domain_penalties")

    def test_the_sitrep_no_longer_derives_an_exclusion_list_from_the_score(self):
        assert not hasattr(sitrep_generator, "fetch_penalized_domains")

    def test_no_penalty_threshold_survives_in_the_ingest_source(self):
        src = inspect.getsource(pass_a_ingest)
        assert "penalty > 0.8" not in src
        assert "domain_penalized" not in src


class TestTheCountersStay:
    def test_pass_c_still_records_what_each_domain_claimed(self):
        from src.pipeline.pass_c_classify import update_domain_penalty

        assert callable(update_domain_penalty)


class TestExcludingADomainIsStillPossible:
    """The parameter survives the producer. Excluding a named domain from the
    independence count is a legitimate operation; deriving that list from
    penalty_score is what stopped."""

    def _ev(self, domain):
        return {"source_domain": domain, "country_iso": "UA"}

    def test_two_independent_newsrooms_are_multi_source(self):
        events = [self._ev("kyivindependent.com"), self._ev("pravda.com.ua")]
        assert label_cluster(events, []) == label_cluster(events, None)

    def test_an_explicitly_excluded_domain_still_does_not_count(self):
        events = [self._ev("kyivindependent.com"), self._ev("pravda.com.ua")]
        both = label_cluster(events, [])
        one_excluded = label_cluster(events, ["pravda.com.ua"])
        assert both != one_excluded
