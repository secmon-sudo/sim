-- Iran theatre bulletin (3 Sep 2026). A separate table rather than a row in
-- sitreps: every SITREP consumer keys off country_iso, and a bulletin has no
-- single country — filing it under a synthetic ISO would put it inside
-- select_sitrep_countries, the digest and the per-country queries, where it does
-- not belong. Keeping it apart also means the war ending is a DROP, not a
-- migration through live SITREP data.
CREATE TABLE IF NOT EXISTS iran_bulletins (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    window_start   timestamp NOT NULL,
    window_end     timestamp NOT NULL,
    report_text    text,
    -- Per-section counts and the extracted actor/standing per event, so a later
    -- review can ask what the model attributed WITHOUT re-running extraction.
    sections_json  jsonb,
    event_count    integer DEFAULT 0,
    status         varchar(20) NOT NULL,
    llm_provider   varchar(40),
    llm_model      varchar(120),
    r2_url         text,
    error_message  text,
    created_at     timestamp NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_iran_bulletins_created
    ON iran_bulletins (created_at DESC);
