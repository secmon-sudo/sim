-- SIM — A category for GNSS interference, the aviation threat the pipeline was
-- already hunting but had nowhere to put
--
-- config/keywords.json has searched for "GPS jamming", "GPS spoofing", "navigation
-- jamming" and "signal jamming aircraft" since the aviation keyword set was built,
-- and pass_c's prompt names GPS jamming as an example of aviation relevance. But
-- event_type had no code for it, so every one of those hits had to land in
-- security_incident or other_aviation_related — indistinguishable, in the SITREP and
-- in every severity replay, from an unrelated airport disturbance.
--
-- It matters operationally in a way the generic codes cannot express: spoofed GNSS
-- corrupts inertial reference and terrain-awareness systems in flight, so the hazard
-- follows the aircraft along a route rather than sitting at a place. Eastern
-- Mediterranean, Black Sea, Baltic and Gulf corridors have been the persistent
-- sources.
--
-- severity_base 45: above security_incident's generic tail because it is a confirmed
-- aviation-system compromise, below the 60+ band reserved for events with casualties
-- or an actual closure. Forward-only; existing rows keep their type.
INSERT INTO event_type_catalog (code, label_en, parent_code, severity_base, active, created_at) VALUES
    ('gnss_interference', 'GNSS Jamming / Spoofing', NULL, 45, TRUE, NOW())
ON CONFLICT (code) DO NOTHING;
