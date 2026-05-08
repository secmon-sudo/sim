-- SIM — Add 'noise' event type to catalog
-- Required for LLM classification to mark irrelevant content

INSERT INTO event_type_catalog (code, label_en, parent_code, severity_base, active, created_at) VALUES
    ('noise', 'Noise / Irrelevant Content', NULL, 0, TRUE, NOW())
ON CONFLICT (code) DO NOTHING;
