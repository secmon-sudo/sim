"""A social platform is a carrier; the publisher is who posted.

Google News indexes publishers' own social posts and its <source url=> then names
the PLATFORM. Measured over the 30 days to 2026-08-19, 127 events were filed under
"facebook.com" — genuine DW News, New York Times, Washington Post and ABS-CBN
stories, all wearing one domain.

source_domain is an identity, so that collapse is not cosmetic: an NYT post and a
DW post looked like one outlet republishing itself and never corroborated each
other, while dw.com plus facebook.com/deutschewellenews counted as two independent
outlets and inflated a single source into "multiple sources".

These tests pin both directions of the repair, and the rule that an unrecognised
page is left alone rather than given an invented identity.
"""

import pytest

from src.pipeline.ingest_filters import (
    is_social_platform,
    social_publisher_domain,
)


class TestPlatformDetection:
    @pytest.mark.parametrize("domain", [
        "facebook.com", "www.facebook.com", "FACEBOOK.COM", "x.com", "twitter.com",
        "instagram.com", "t.me", "threads.net",
    ])
    def test_carriers_are_recognised(self, domain):
        assert is_social_platform(domain) is True

    @pytest.mark.parametrize("domain", [
        "dw.com", "nytimes.com", "reuters.com", "bbc.co.uk", "", None,
    ])
    def test_publishers_are_not_platforms(self, domain):
        assert is_social_platform(domain) is False


class TestPublisherRecovery:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.facebook.com/nytimes/posts/heavy-rain-in-japan", "nytimes.com"),
        ("https://www.facebook.com/deutschewellenews/videos/ukraine-drone", "dw.com"),
        ("https://www.facebook.com/washingtonpost/posts/a-74-magnitude", "washingtonpost.com"),
        ("https://www.facebook.com/abscbnNEWS/posts/north-korea-fired", "abs-cbn.com"),
        ("https://www.facebook.com/ManchesterEveningNews/photos/engineer",
         "manchestereveningnews.co.uk"),
    ])
    def test_known_pages_resolve_to_the_publisher(self, url, expected):
        assert social_publisher_domain(url) == expected

    def test_page_slug_is_case_insensitive(self):
        assert social_publisher_domain("https://facebook.com/NYTimes/posts/x") == "nytimes.com"

    def test_an_unmapped_page_is_left_alone(self):
        # Commentators and aggregators are deliberately absent from the map: giving
        # them a publisher identity would let the corroboration count trust them.
        assert social_publisher_domain(
            "https://www.facebook.com/officialbenshapiro/posts/x") is None
        assert social_publisher_domain(
            "https://www.facebook.com/lonewolfnewsandmedia/posts/x") is None

    def test_a_url_with_no_page_slug_is_none(self):
        assert social_publisher_domain("https://www.facebook.com/profile.php?id=123") is None
        assert social_publisher_domain("https://www.facebook.com/") is None

    def test_a_non_social_url_is_none(self):
        assert social_publisher_domain("https://www.nytimes.com/2026/08/19/world.html") is None
        assert social_publisher_domain(None) is None

    def test_a_recovered_domain_is_a_real_registrable_domain(self):
        """It feeds registrable_domain(); a value that collapses would defeat the fix."""
        from src.core.sitrep_verify import registrable_domain
        for url, expected in [
            ("https://facebook.com/nytimes/posts/x", "nytimes.com"),
            ("https://facebook.com/abscbnNEWS/posts/x", "abs-cbn.com"),
            ("https://facebook.com/sunstardavaonews/posts/x", "sunstar.com.ph"),
            ("https://facebook.com/theliverpoolecho/posts/x", "liverpoolecho.co.uk"),
        ]:
            got = social_publisher_domain(url)
            assert got == expected
            assert registrable_domain(got) == expected

    def test_two_outlets_on_the_same_platform_no_longer_collapse(self):
        """The corroboration bug in one assertion."""
        nyt = social_publisher_domain("https://facebook.com/nytimes/posts/x")
        dw = social_publisher_domain("https://facebook.com/deutschewellenews/videos/y")
        assert nyt != dw

    def test_an_outlet_matches_its_own_website(self):
        """The inflation bug: the post and the site must be ONE identity."""
        from src.core.sitrep_verify import registrable_domain
        post = social_publisher_domain("https://facebook.com/deutschewellenews/videos/y")
        site = registrable_domain("https://www.dw.com/en/story/a-123")
        assert registrable_domain(post) == site
