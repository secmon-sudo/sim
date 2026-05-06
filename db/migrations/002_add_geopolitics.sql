-- SIM (Security Incident Monitor) — Database Schema
-- Add geopolitical event types

INSERT INTO event_type_catalog (code, label_en, parent_code, severity_base, active, created_at) VALUES
    ('geopolitical_conflict',  'Geopolitical Conflict',   NULL,                85, TRUE, NOW()),
    ('military_action',        'Military Action',         'geopolitical_conflict', 95, TRUE, NOW()),
    ('missile_strike',         'Missile Strike',          'geopolitical_conflict', 100, TRUE, NOW()),
    ('political_event',        'Political Event',         NULL,                60, TRUE, NOW()),
    ('civil_unrest',           'Civil Unrest',            NULL,                70, TRUE, NOW()),
    ('terrorism',              'Terrorism',               'security_incident', 95, TRUE, NOW())
ON CONFLICT (code) DO NOTHING;
