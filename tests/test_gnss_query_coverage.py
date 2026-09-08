"""The GNSS query asked for two things and got neither.

gnss_interference has been an event type since the taxonomy was written, the
prescreen has read "gnss|gps jamming|gps spoofing|jamming|spoofing" as an
aviation-security class term since 27 August, and there is a dedicated news query
for it. Over 30 days the corpus held ELEVEN events matching gnss/gps/jamming/
spoofing in the headline, four of which were archived unread.

The gates were not the problem. Seven realistic GNSS headlines were run through the
ingest keyword filter and the prescreen on 2026-09-08 and six passed both, scoring
25-45 against a floor of 15. The query was:

    "GPS jamming" OR "GNSS interference" OR "GPS spoofing" aviation OR flights when:2d

Fetched live, variant by variant, counting results that are actually about GNSS:

    current (as above)                     3 results,  1 on topic
    the same with parentheses              3 results,  1 on topic
    without `aviation OR flights`          6 results,  4 on topic
    without it, when:7d                   20 results, 14 on topic
    with four extra phrasings, when:7d    23 results, 14 on topic

So the OR/AND precedence was not the fault — parenthesising changes nothing — and
the extra phrasings buy nothing. The `aviation OR flights` conjunct was the whole
problem: a wire story about GPS jamming over the Baltic does not have to say
"aviation" to be one, and requiring it cost three quarters of the class.

Dropping it does not widen what gets INGESTED, because a news_queries result is
still subject to the acceptance filter. Of the 17 headlines the widened query
returned over 7 days, 7 survive ingest and the prescreen and 10 are dropped —
vendor press releases (TrustPoint/NovAtel, Höegh Autoliners, Inmarsat), a port
throughput report, an army components lab. The gates do that filtering already.

when:2d is left alone. Every other query in the file uses it, the pipeline runs
about every three hours, and widening the window would re-fetch the same items.
"""

import json
import urllib.parse
from pathlib import Path

import pytest

SETTINGS = json.loads(Path("config/settings.json").read_text())
QUERIES = [urllib.parse.unquote(u) for u in SETTINGS["sources"]["news_queries"]]
GNSS = [q for q in QUERIES if "GNSS interference" in q or "GPS jamming" in q]


class TestTheGnssQueryIsNotNarrowedByAnAviationWord:
    def test_there_is_still_exactly_one_gnss_query(self):
        assert len(GNSS) == 1

    def test_it_no_longer_requires_the_word_aviation_or_flights(self):
        """The conjunct that cost three quarters of the class. A wire story about
        GPS jamming over the Baltic does not have to say "aviation" to be one."""
        q = GNSS[0]
        assert "aviation" not in q
        assert "flights" not in q

    def test_it_still_asks_for_the_three_phrases(self):
        q = GNSS[0]
        for phrase in ('"GPS jamming"', '"GNSS interference"', '"GPS spoofing"'):
            assert phrase in q

    def test_it_keeps_the_house_recency_window(self):
        # Every other query uses when:2d and the pipeline runs about every 3 hours;
        # a wider window re-fetches the same items rather than finding new ones.
        assert "when:2d" in GNSS[0]


class TestTheGatesWereNeverTheProblem:
    @pytest.mark.parametrize("title", [
        "GPS jamming disrupts flights over the Baltic Sea, airlines warn",
        "GNSS interference reported near Kaliningrad affecting civil aviation",
        "GPS spoofing incidents surge near Iranian airspace, IATA says",
        "Finnair suspends Tartu flights over GPS interference",
        "Aircraft navigation systems disrupted by jamming near Black Sea",
        "Russia accused of GPS jamming affecting Baltic air traffic",
    ])
    def test_a_real_gnss_headline_clears_ingest_and_the_prescreen(self, title):
        from src.pipeline.pass_a_ingest import _matches_security_keywords
        from src.pipeline.pass_c_classify import deterministic_relevance

        assert _matches_security_keywords(title, "")
        assert deterministic_relevance(title, "")["score"] >= 15

    @pytest.mark.parametrize("title", [
        "TrustPoint, NovAtel demonstrate C-band PNT through GNSS interference",
        "Höegh Autoliners equips 38 ship fleet as GPS interference threatens navigation",
        "Port of Savannah handles 5.67 million teu in FY2026",
    ])
    def test_the_vendor_copy_the_wider_query_admits_is_dropped_downstream(self, title):
        """Widening the query does not widen ingestion: these are what the extra
        results look like, and the acceptance filter already refuses them."""
        from src.pipeline.pass_a_ingest import _matches_security_keywords

        assert not _matches_security_keywords(title, "")
