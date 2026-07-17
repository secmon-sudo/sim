-- 017: Capture corroboration evidence from ingest-time duplicates.
--
-- Pass A drops cross-source content duplicates to save the insert budget, but
-- the dropped article IS the corroboration signal the SITREP verification
-- labels depend on ("Onaylandı (Çoklu kaynak)" needs >= 2 independent domains).
-- Instead of discarding it, the duplicate's source is appended to the surviving
-- event's corroborating_sources: [{"domain": ..., "url": ..., "title": ...}].
ALTER TABLE events ADD COLUMN IF NOT EXISTS corroborating_sources JSONB NOT NULL DEFAULT '[]'::jsonb;
