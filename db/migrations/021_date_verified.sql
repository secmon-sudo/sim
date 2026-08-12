-- 021: Did the PUBLISHER declare this article's date, or did we inherit an
-- aggregator's crawl stamp?
--
-- Google News stamps a re-crawled archive page with the date IT saw the page, so an
-- item whose publisher page declares no date and carries none in its path enters the
-- pipeline dated "now". Measured 2026-08-12: a 2026-03-25 Nepalnews explainer about
-- the September 2025 uprising was stored as published 2026-08-12 01:33 and reached the
-- daily report as a same-day event. Heuristic date extraction is not a fix — htmldate
-- and trafilatura both read that page as 2026-08-12, picking up sidebar links.
--
-- So the date is kept, and its provenance is recorded alongside it. FALSE means "this
-- timestamp is the aggregator's, not the publisher's": such an event may still be
-- ingested, classified, clustered and reported, but it may not claim to be fresh.
--
-- DEFAULT FALSE governs anything inserted without an opinion: Pass A always states the
-- flag explicitly, so a row that arrives silent is a row written by code that does not
-- know about provenance, and that row should not get to assert a verified date.
ALTER TABLE events
    ADD COLUMN IF NOT EXISTS date_verified BOOLEAN NOT NULL DEFAULT FALSE;

-- Rows that predate the column carry no evidence either way, and "no evidence" must not
-- silence an alert — the same fail-open direction the gate uses for a missing key. The
-- cutoff is a fixed literal, not NOW(), because this file re-runs on every pipeline run:
-- a relative bound would keep sweeping newly-ingested rows to TRUE forever and the flag
-- would never mean anything. Events ingested between this timestamp and the deploy read
-- as unverified for one cycle, which costs at most a few freshness-only pages.
UPDATE events SET date_verified = TRUE
 WHERE ingested_at < TIMESTAMP '2026-08-12 08:00:00' AND date_verified = FALSE;

COMMENT ON COLUMN events.date_verified IS
    'TRUE when published_at came from the publisher (page metadata, URL path, or the '
    'publisher''s own feed); FALSE when it is an aggregator crawl stamp we could not '
    'confirm. Unverified dates cannot satisfy the alert gates freshness requirement.';
