-- ============================================================
-- SIM — CZIB (Conflict Zone Information Bulletin) Schema
-- EASA CZIB data synced from https://www.easa.europa.eu/en/domains/air-operations/czibs/export-json
-- ============================================================

CREATE TABLE IF NOT EXISTS czib_zones (
    id              SERIAL PRIMARY KEY,
    czib_id         VARCHAR(20)  NOT NULL UNIQUE,
    name            VARCHAR(200) NOT NULL,
    status          VARCHAR(20)  NOT NULL CHECK (status IN ('Active', 'Suspended', 'Withdrawn')),
    countries       TEXT[]       DEFAULT '{}',
    country_names   TEXT         DEFAULT '',
    coordinates     VARCHAR(100) DEFAULT '',
    issued_date     TIMESTAMP,
    valid_until     VARCHAR(50)  DEFAULT '',
    valid_descr     TEXT         DEFAULT '',
    updated_at      TIMESTAMP    DEFAULT NOW(),
    synced_at       TIMESTAMP    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_czib_status     ON czib_zones(status);
CREATE INDEX IF NOT EXISTS idx_czib_countries  ON czib_zones USING GIN(countries);
CREATE INDEX IF NOT EXISTS idx_czib_synced     ON czib_zones(synced_at DESC);

-- View: Active CZIB zones only
CREATE OR REPLACE VIEW v_czib_active AS
SELECT * FROM czib_zones WHERE status = 'Active';
