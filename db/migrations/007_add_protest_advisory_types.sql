-- SIM (Security Incident Monitor) — Protest & Travel Advisory Event Types
-- Adds protest/demonstration subtypes under civil_unrest,
-- and travel advisory types as new root categories.

INSERT INTO event_type_catalog (code, label_en, parent_code, severity_base, active, created_at) VALUES
    -- Protest & Civil Unrest subtypes (parent: civil_unrest)
    ('protest',              'Protest / Demonstration',           'civil_unrest', 65, TRUE, NOW()),
    ('mass_demonstration',   'Mass Demonstration',                'civil_unrest', 72, TRUE, NOW()),
    ('riot',                 'Riot / Violent Unrest',             'civil_unrest', 78, TRUE, NOW()),
    ('general_strike',       'General Strike',                    'civil_unrest', 70, TRUE, NOW()),
    ('coup_attempt',         'Coup Attempt',                      'civil_unrest', 90, TRUE, NOW()),

    -- Travel Advisory types (new root category)
    ('travel_advisory',      'Country Travel Advisory',           NULL, 60, TRUE, NOW()),
    ('travel_ban',           'Travel Ban / Do Not Travel',        'travel_advisory', 75, TRUE, NOW()),
    ('embassy_closure',      'Embassy / Consulate Closure',       'travel_advisory', 70, TRUE, NOW())
ON CONFLICT (code) DO NOTHING;
