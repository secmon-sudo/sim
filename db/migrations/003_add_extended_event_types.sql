-- SIM (Security Incident Monitor) — Extended Event Types
-- Adds new threat categories: drone attacks, mass casualty, African terror, war escalation

INSERT INTO event_type_catalog (code, label_en, parent_code, severity_base, active, created_at) VALUES
    -- Drone attacks on critical infrastructure
    ('drone_attack_critical_infra', 'Drone Attack on Critical Infrastructure', 'security_incident', 90, TRUE, NOW()),
    ('drone_airport_attack',        'Drone Attack on Airport',               'drone_attack_critical_infra', 95, TRUE, NOW()),
    ('drone_military_base_attack',  'Drone Attack on Military Base',         'drone_attack_critical_infra', 92, TRUE, NOW()),
    ('drone_energy_attack',         'Drone Attack on Energy Facility',       'drone_attack_critical_infra', 88, TRUE, NOW()),
    ('drone_port_attack',           'Drone Attack on Port',                  'drone_attack_critical_infra', 88, TRUE, NOW()),

    -- Mass casualty events
    ('mass_casualty_event',         'Mass Casualty Event',                   'security_incident', 95, TRUE, NOW()),
    ('mass_shooting',               'Mass Shooting',                         'mass_casualty_event', 95, TRUE, NOW()),
    ('mass_stabbing',               'Mass Stabbing',                         'mass_casualty_event', 90, TRUE, NOW()),
    ('suicide_bombing',             'Suicide Bombing',                       'mass_casualty_event', 100, TRUE, NOW()),
    ('vehicle_ramming',             'Vehicle Ramming Attack',                'mass_casualty_event', 88, TRUE, NOW()),

    -- War escalation & civilian casualties
    ('war_escalation',              'War Escalation',                        'geopolitical_conflict', 90, TRUE, NOW()),
    ('ceasefire_violation',         'Ceasefire Violation',                   'geopolitical_conflict', 85, TRUE, NOW()),
    ('civilian_casualties',         'Civilian Casualties in Conflict',       'geopolitical_conflict', 92, TRUE, NOW()),
    ('humanitarian_crisis',         'Humanitarian Crisis',                   'geopolitical_conflict', 80, TRUE, NOW()),

    -- African terrorism & insurgency
    ('african_terrorism',           'African Terrorism',                     'terrorism', 95, TRUE, NOW()),
    ('insurgency_attack',           'Insurgency Attack',                     'african_terrorism', 90, TRUE, NOW()),
    ('extremist_violence',          'Extremist Violence',                    'african_terrorism', 88, TRUE, NOW()),
    ('jihadist_attack',             'Jihadist Attack',                       'african_terrorism', 92, TRUE, NOW()),

    -- Aviation personnel attacks
    ('aviation_personnel_attack',   'Aviation Personnel Attack',             'security_incident', 85, TRUE, NOW()),
    ('pilot_attacked',              'Pilot Attacked',                        'aviation_personnel_attack', 88, TRUE, NOW()),
    ('cabin_crew_attacked',         'Cabin Crew Attacked',                   'aviation_personnel_attack', 85, TRUE, NOW()),
    ('ground_staff_attacked',       'Ground Staff Attacked',                 'aviation_personnel_attack', 82, TRUE, NOW()),
    ('air_traffic_controller_threat', 'Air Traffic Controller Threat',       'aviation_personnel_attack', 80, TRUE, NOW()),

    -- Resort / tourism attacks
    ('resort_attack',               'Resort Attack',                         'security_incident', 88, TRUE, NOW()),
    ('beach_attack',                'Beach Attack',                          'resort_attack', 85, TRUE, NOW()),
    ('tourist_bus_attack',          'Tourist Bus Attack',                    'resort_attack', 90, TRUE, NOW()),
    ('cruise_ship_attack',          'Cruise Ship Attack',                    'resort_attack', 90, TRUE, NOW())
ON CONFLICT (code) DO NOTHING;
