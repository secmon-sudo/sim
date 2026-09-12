"""CRITICAL's not-news exemption, conditioned on there actually being news.

The exemption's own argument is that "a roundup is sometimes the only carrier of a
genuinely major development, and missing that costs more than the noise it lets
through". That argument is about a development we have not otherwise heard, and it
stops being true the moment a card for the same storyline has already gone out: a
follow-up on a story the reader was carded on two hours ago is the same
development, filed again.

Measured 2026-09-12 over seven days: 53 CRITICAL cards carried followup or roundup,
and 49 of them belonged to a storyline that had ALREADY paged. Four were the first
card of their story — the case the exemption exists for, and the case these tests
pin as untouched.

The occasion was one Russian strike wave on Kyiv that sent 19 cards in a day. Two
of the 19 were this: a "death toll has risen" follow-up and a digest that opened
"Russia Attacked Kyiv Again, Canada Is Funding Interceptors, and Oil Approached…".
The classifier had labelled both correctly; the exemption sent them anyway.
"""

from src.core.alerts import (
    CRITICAL_NOTNEWS_KINDS,
    critical_notnews_is_redundant,
    storyline_already_paged,
)


class _Conn:
    """Answers the 'has this storyline paged' query with a fixed verdict."""

    def __init__(self, paged=True, raises=False):
        self.paged = paged
        self.raises = raises
        self.params = None

    def execute(self, _sql, params=None):
        if self.raises:
            raise RuntimeError("pooler went away")
        self.params = params
        conn = self

        class _R:
            def fetchone(self_inner):
                return (1,) if conn.paged else None
        return _R()


def _event(kind="followup", storyline="s-1", event_id="e-2"):
    return {"report_kind": kind, "storyline_id": storyline, "id": event_id}


class TestRedundantCriticalFollowups:
    def test_a_followup_on_a_carded_storyline_is_redundant(self):
        assert critical_notnews_is_redundant(_Conn(paged=True), _event(), "CRITICAL")

    def test_a_roundup_on_a_carded_storyline_is_redundant(self):
        assert critical_notnews_is_redundant(_Conn(paged=True),
                                             _event(kind="roundup"), "CRITICAL")

    def test_the_first_card_of_a_story_still_pages(self):
        """The case the exemption was written for: four of 53 over seven days."""
        assert not critical_notnews_is_redundant(_Conn(paged=False), _event(), "CRITICAL")

    def test_a_new_incident_is_never_touched(self):
        assert not critical_notnews_is_redundant(_Conn(paged=True),
                                                 _event(kind="new_incident"), "CRITICAL")

    def test_lower_tiers_are_not_this_gate_s_business(self):
        """ALERT and WATCH already veto these kinds outright — see
        REPORT_KIND_NOT_NEWS. Reaching here at all would mean the ladder changed."""
        for tier in ("ALERT", "WATCH", None):
            assert not critical_notnews_is_redundant(_Conn(paged=True), _event(), tier)

    def test_commentary_is_not_in_this_set(self):
        """It loses the exemption outright, unconditionally, and has since 8 Sep."""
        assert "commentary" not in CRITICAL_NOTNEWS_KINDS

    def test_the_kind_can_arrive_nested_under_llm_parsed(self):
        """Pass D builds one dict shape, Pass E another; the gate reads both."""
        event = {"llm_parsed": {"report_kind": "followup"},
                 "storyline_id": "s-1", "id": "e-2"}
        assert critical_notnews_is_redundant(_Conn(paged=True), event, "CRITICAL")


class TestStorylinePagedProbe:
    def test_an_event_does_not_count_as_its_own_predecessor(self):
        """Pass E rescoring dispatches the same event a second time. Counting its
        own earlier row would make every rescore look redundant."""
        conn = _Conn(paged=True)
        storyline_already_paged(conn, "s-1", "e-2")
        assert conn.params[0] == "s-1" and conn.params[1] == "e-2"

    def test_no_storyline_means_no_evidence_of_a_previous_card(self):
        conn = _Conn(paged=True)
        assert not storyline_already_paged(conn, None, "e-2")
        assert conn.params is None

    def test_a_failed_query_sends_the_card(self):
        """Fails open. This predicate exists to withhold something redundant, and
        withholding on a broken query would silence a real page."""
        assert not storyline_already_paged(_Conn(raises=True), "s-1", "e-2")
        assert not critical_notnews_is_redundant(_Conn(raises=True), _event(), "CRITICAL")
