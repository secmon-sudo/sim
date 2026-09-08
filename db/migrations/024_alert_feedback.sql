-- 024: Did the card that we sent actually help anybody?
--
-- Everything this pipeline calibrates against is a PROXY: what the classifier said,
-- how many independent sources corroborated, what report_kind came back, how many
-- false positives a replay found. None of them answers "was this card worth sending".
-- Measured 2026-09-08: ~119 Telegram cards a day (96 CRITICAL + 409 ALERT + 327 WATCH
-- over 7 days), `report_feedback` 0 rows, `report_validations` 9. So every threshold
-- argument — is 119 the right number, or 30, or 300 — has been a guess.
--
-- Two buttons under every card write here. That is the whole mechanism.
--
-- No foreign key to events, ON PURPOSE. Pass F deletes: reconciled events leave in the
-- archive batch and archived noise is purged at 30 days, so a FK with ON DELETE CASCADE
-- would destroy exactly the longitudinal record this table exists to accumulate — the
-- signal is worth least on the day it is written and most a quarter later. Everything a
-- later analysis needs is therefore SNAPSHOT here at press time rather than joined back.
CREATE TABLE IF NOT EXISTS alert_feedback (
    -- Telegram's own update id, as the primary key: the drain re-reads a window it may
    -- have already written (a crash between the write and the cursor advance, or two
    -- drains racing), and idempotency has to come from the update itself, not from our
    -- bookkeeping about it.
    update_id      BIGINT PRIMARY KEY,
    event_id       UUID NOT NULL,
    verdict        VARCHAR(16) NOT NULL CHECK (verdict IN ('useful', 'noise')),

    -- The tier the CARD carried, read out of the button's own callback_data — not
    -- events.alert_tier, which is not a record that a card was sent (see ced1565) and
    -- which Pass E may have overwritten since.
    card_tier      VARCHAR(10),

    -- Snapshot of the event as it stood when the button was pressed. NULL means the
    -- event was already purged — the verdict still counts, it just cannot be sliced.
    event_type     VARCHAR(60),
    country_iso    CHAR(2),
    severity_score INT,
    source_domain  VARCHAR(255),
    source_title   TEXT,

    tg_user_id     BIGINT,
    tg_username    VARCHAR(64),
    tg_message_id  BIGINT,
    pressed_at     TIMESTAMPTZ,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per (event, person): pressing the other button changes your mind rather than
-- casting a second vote, and the drain upserts on this.
CREATE UNIQUE INDEX IF NOT EXISTS uq_alert_feedback_event_user
    ON alert_feedback (event_id, tg_user_id);

CREATE INDEX IF NOT EXISTS idx_alert_feedback_created
    ON alert_feedback (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alert_feedback_tier_verdict
    ON alert_feedback (card_tier, verdict);

-- Where the drain got to in Telegram's update stream. A single row by construction.
--
-- The alternative — deriving the offset as max(update_id)+1 over alert_feedback — looks
-- tidy and quietly breaks: any update that is NOT one of our callbacks (somebody typing
-- in the group) never lands in that table, so it would be re-delivered on every drain
-- until Telegram expires it 24h later, and a busy chat would keep the window pinned.
CREATE TABLE IF NOT EXISTS telegram_update_cursor (
    id             SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    last_update_id BIGINT NOT NULL DEFAULT 0,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO telegram_update_cursor (id, last_update_id)
VALUES (1, 0)
ON CONFLICT (id) DO NOTHING;

COMMENT ON TABLE alert_feedback IS
    'One analyst button press per row. The only non-proxy signal SIM has about whether '
    'an alert card was worth sending; deliberately outlives the event it describes.';
