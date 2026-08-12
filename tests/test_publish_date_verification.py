"""
Tests for publication-date verification (Pass A).

Regression cover for the 2026-08-05 reprint incident: Google News served three
Yeni Safak archive stories (2016-10-25, 2016-10-27, 2021) stamped with the
current crawl date. The feed's pubDate is the field max_article_age_days reads,
so all three cleared the age gate and fired ALERT-tier Telegram cards.
"""

from datetime import datetime, timezone

from src.pipeline.ingest_sources import extract_published_date


class TestExtractPublishedDate:
    def test_jsonld_date_published(self):
        # The exact shape Yeni Safak serves on the 2016 Kenya story.
        html = '<script type="application/ld+json">{"@type":"NewsArticle",' \
               '"datePublished":"2016-10-25T10:53:20+03:00",' \
               '"dateModified":"2016-10-25T10:53:20+03:00"}</script>'
        assert extract_published_date(html) == datetime(2016, 10, 25, 7, 53, 20,
                                                        tzinfo=timezone.utc)

    def test_opengraph_meta(self):
        html = '<meta property="article:published_time" content="2026-08-04T22:10:14+03:00"/>'
        assert extract_published_date(html) == datetime(2026, 8, 4, 19, 10, 14,
                                                        tzinfo=timezone.utc)

    def test_jsonld_wins_over_time_tag(self):
        # <time> tags also appear on "related articles" teasers, so the
        # publisher-declared field must take precedence.
        html = ('<script type="application/ld+json">{"datePublished":"2016-10-25"}</script>'
                '<time datetime="2026-08-04T18:50:24Z">Aug 4</time>')
        assert extract_published_date(html).year == 2016

    def test_time_tag_fallback(self):
        html = '<article><time datetime="2026-08-03T08:17:46Z">Aug 3</time></article>'
        assert extract_published_date(html) == datetime(2026, 8, 3, 8, 17, 46,
                                                        tzinfo=timezone.utc)

    def test_naive_date_is_treated_as_utc(self):
        html = '<meta name="pubdate" content="2026-08-04"/>'
        dt = extract_published_date(html)
        assert dt.tzinfo is not None and dt.date().isoformat() == "2026-08-04"

    def test_no_date_returns_none(self):
        assert extract_published_date("<html><body>no dates here</body></html>") is None

    def test_empty_html_returns_none(self):
        assert extract_published_date("") is None

    def test_publisher_namespaced_published_time(self):
        # Hankyoreh declares the date only as `h:published_time`. Missing it made a
        # severity-100 Zaporizhzhia strike look like an unverifiable aggregator stamp
        # and the date gate withheld its page (2026-08-12).
        html = '<meta name="h:published_time" content="2026-08-12T17:26:00+09:00"/>'
        assert extract_published_date(html) == datetime(2026, 8, 12, 8, 26,
                                                        tzinfo=timezone.utc)

    def test_namespaced_suffix_does_not_match_other_fields(self):
        # The suffix rule must not turn every namespaced meta tag into a date source.
        html = ('<meta name="h:modify_date" content="2026-08-12T17:26:00+09:00"/>'
                '<meta name="h:section" content="International"/>')
        assert extract_published_date(html) is None

    def test_unparseable_date_does_not_raise(self):
        html = '<meta property="article:published_time" content="not a date"/>'
        assert extract_published_date(html) is None


class TestReprintGate:
    """The age comparison Pass A applies to the page-declared date."""

    @staticmethod
    def _is_reprint(page_iso: str, now: datetime, max_age_days: int = 2) -> bool:
        page_dt = extract_published_date(
            f'<meta property="article:published_time" content="{page_iso}"/>'
        )
        return (now - page_dt).total_seconds() / 86400 > max_age_days

    def test_2016_archive_is_a_reprint(self):
        now = datetime(2026, 8, 5, tzinfo=timezone.utc)
        assert self._is_reprint("2016-10-25T10:53:20+03:00", now) is True

    def test_todays_article_passes(self):
        now = datetime(2026, 8, 5, tzinfo=timezone.utc)
        assert self._is_reprint("2026-08-04T22:10:14+03:00", now) is False

    def test_boundary_just_inside_window(self):
        now = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
        assert self._is_reprint("2026-08-03T18:00:00+00:00", now) is False


class TestVerificationDownAlarm:
    """The layer breaking must page, not degrade quietly."""

    def test_zero_verified_on_real_sample_is_a_degradation(self):
        from src.pipeline.orchestrator import _collect_degradations
        problems = _collect_degradations(
            {"pass_a": {"full_text_attempted": 47, "publish_dates_verified": 0}}
        )
        assert any("verification is DOWN" in p for p in problems)

    def test_healthy_run_is_silent(self):
        from src.pipeline.orchestrator import _collect_degradations
        assert _collect_degradations(
            {"pass_a": {"full_text_attempted": 47, "publish_dates_verified": 41}}
        ) == []

    def test_tiny_sample_does_not_page(self):
        # A quiet run that inserted three events proves nothing about the layer.
        from src.pipeline.orchestrator import _collect_degradations
        assert _collect_degradations(
            {"pass_a": {"full_text_attempted": 3, "publish_dates_verified": 0}}
        ) == []


class TestExtractDateFromUrl:
    """Free fallback for publishers that 403 the fetch (France24 among them)."""

    def test_france24_compact_path_date(self):
        from src.pipeline.ingest_sources import extract_date_from_url
        dt = extract_date_from_url(
            "https://www.france24.com/en/middle-east/20260805-us-says-iran-hormuz-deal")
        assert dt.date().isoformat() == "2026-08-05"

    def test_slashed_path_date(self):
        from src.pipeline.ingest_sources import extract_date_from_url
        dt = extract_date_from_url("https://example.com/news/2016/10/25/kenya-hotel-bomb")
        assert dt.date().isoformat() == "2016-10-25"

    def test_undated_path_returns_none(self):
        from src.pipeline.ingest_sources import extract_date_from_url
        assert extract_date_from_url(
            "https://en.yenisafak.com/world/kenya-12-killed-in-al-shabaab-hotel-bomb-2553444"
        ) is None

    def test_article_id_is_not_mistaken_for_a_date(self):
        from src.pipeline.ingest_sources import extract_date_from_url
        assert extract_date_from_url(
            "https://www.standardmedia.co.ke/opinion/article/2001554553/why-citizens") is None

    def test_impossible_date_rejected(self):
        from src.pipeline.ingest_sources import extract_date_from_url
        assert extract_date_from_url("https://example.com/2026/13/45/story") is None

    def test_month_precision_path_resolves_to_end_of_month(self):
        # WordPress's /YYYY/MM/slug permalink. Ignoring it let a 2022 bombing through
        # as a 2026-08-05 ALERT at severity 95 (thenewsmill, measured 2026-08-12):
        # Google News stamps its crawl date and the page declared none.
        from src.pipeline.ingest_sources import extract_date_from_url
        dt = extract_date_from_url(
            "https://thenewsmill.com/2022/09/pakistan-one-killed-after-blast-in-south-waziristan/")
        assert dt.date().isoformat() == "2022-09-30"

    def test_day_precision_wins_over_month(self):
        from src.pipeline.ingest_sources import extract_date_from_url
        dt = extract_date_from_url("https://example.com/2026/08/05/story/")
        assert dt.date().isoformat() == "2026-08-05"

    def test_impossible_month_rejected(self):
        from src.pipeline.ingest_sources import extract_date_from_url
        assert extract_date_from_url("https://example.com/2026/13/story") is None

    def test_bare_year_segment_is_not_a_date(self):
        from src.pipeline.ingest_sources import extract_date_from_url
        assert extract_date_from_url("https://example.com/section/2026/tag/") is None


class TestDayPrecisionIsConservative:
    """A date without a time must never make an article look older than it is."""

    def test_url_date_resolves_to_end_of_day(self):
        from src.pipeline.ingest_sources import extract_date_from_url
        dt = extract_date_from_url("https://www.france24.com/en/20260803-us-israel-pause")
        assert (dt.hour, dt.minute) == (23, 59)

    def test_date_only_meta_resolves_to_end_of_day(self):
        dt = extract_published_date('<meta name="pubdate" content="2026-08-03"/>')
        assert (dt.hour, dt.minute) == (23, 59)

    def test_timestamped_date_keeps_its_time(self):
        dt = extract_published_date(
            '<meta property="article:published_time" content="2026-08-03T08:17:46Z"/>')
        assert (dt.hour, dt.minute) == (8, 17)

    def test_same_day_article_survives_a_two_day_window(self):
        # The France24 regression: published 2026-08-03, read 2026-08-05 02:00.
        from src.pipeline.ingest_sources import extract_date_from_url
        now = datetime(2026, 8, 5, 2, 0, tzinfo=timezone.utc)
        dt = extract_date_from_url("https://www.france24.com/en/20260803-story")
        assert (now - dt).total_seconds() / 86400 <= 2


class TestNoHeuristicGuessing:
    """Run #1349 regression: a page with no declared date must yield UNKNOWN.

    htmldate inferred dates from copyright lines ("2026-01-01") on metadata-less
    pages, which discarded three same-day reports of the 2026-08-05 Kyiv strike
    as reprints. Unknown must never justify a drop.
    """

    def test_page_without_metadata_yields_none(self):
        html = ("<html><head><title>At least 15 killed in and around Kyiv</title></head>"
                "<body><p>Story text.</p><footer>© 2026 bluewin.ch</footer></body></html>")
        assert extract_published_date(html) is None

    def test_copyright_year_is_not_a_date(self):
        assert extract_published_date("<footer>Copyright 2026 Example News</footer>") is None

    def test_visible_dateline_text_is_not_parsed(self):
        # Prose datelines are exactly what the heuristic layer used to mine.
        html = "<article><p>KYIV, August 5, 2026 — Russian missiles struck.</p></article>"
        assert extract_published_date(html) is None

    def test_declared_date_still_wins(self):
        html = ('<meta property="article:published_time" content="2026-08-05T06:31:58+00:00"/>'
                "<footer>© 2026</footer>")
        assert extract_published_date(html).date().isoformat() == "2026-08-05"


class TestOccurredAtFallbackNeverLandsInTheFuture:
    """The end-of-day sentinel is a freshness comparison, not an incident time.

    Regression cover for the 2026-08-06 dashboard reading: same-day articles whose
    date came only from the URL path carried occurred_at_est = 23:59:59, showed a
    time that had not happened yet, and pinned their storyline to the top of the
    board until midnight.
    """

    def test_same_day_end_of_day_sentinel_is_clamped_to_now(self):
        from src.pipeline.pass_d_score import resolve_occurred_at_fallback
        now = datetime(2026, 8, 6, 10, 46, tzinfo=timezone.utc)
        sentinel = datetime(2026, 8, 6, 23, 59, 59, tzinfo=timezone.utc)
        assert resolve_occurred_at_fallback(sentinel, None, now=now) == now

    def test_past_publication_time_is_left_alone(self):
        from src.pipeline.pass_d_score import resolve_occurred_at_fallback
        now = datetime(2026, 8, 6, 10, 46, tzinfo=timezone.utc)
        published = datetime(2026, 8, 5, 6, 31, 58, tzinfo=timezone.utc)
        assert resolve_occurred_at_fallback(published, None, now=now) == published

    def test_yesterdays_sentinel_survives_because_it_is_already_past(self):
        from src.pipeline.pass_d_score import resolve_occurred_at_fallback
        now = datetime(2026, 8, 6, 10, 46, tzinfo=timezone.utc)
        sentinel = datetime(2026, 8, 5, 23, 59, 59, tzinfo=timezone.utc)
        assert resolve_occurred_at_fallback(sentinel, None, now=now) == sentinel

    def test_naive_published_at_is_read_as_utc(self):
        from src.pipeline.pass_d_score import resolve_occurred_at_fallback
        now = datetime(2026, 8, 6, 10, 46, tzinfo=timezone.utc)
        naive = datetime(2026, 8, 6, 23, 59, 59)
        assert resolve_occurred_at_fallback(naive, None, now=now) == now

    def test_missing_publication_date_falls_through_to_ingest_time(self):
        from src.pipeline.pass_d_score import resolve_occurred_at_fallback
        ingested = datetime(2026, 8, 6, 9, 0, tzinfo=timezone.utc)
        assert resolve_occurred_at_fallback(None, ingested) is ingested
