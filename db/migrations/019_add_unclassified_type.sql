-- SIM — Separate the classifier's fallback sink from the genuine aviation category
--
-- 'other_aviation_related' had been doing double duty: (1) a real LLM category for
-- aviation events that fit no specific subtype, AND (2) the FK-safe sink for every
-- parse failure, 'noise' verdict, missing type, and sub-relevance (<50) tail. Role
-- (2) leaked into SITREP daily records, mislabeling geopolitical/diplomatic tail
-- items (e.g. "Rubio warns Iran on nuclear talks", "Trump's nuclear-deal demand
-- puts Saudis in a bind") as "Other Aviation Related".
--
-- This adds a neutral fallback code. Pass C now routes non-aviation fallbacks here,
-- leaving 'other_aviation_related' to mean only what the LLM genuinely tagged as
-- aviation. severity_base matches other_aviation_related (20) so scoring is
-- unchanged — only the label differs. Forward-only: existing rows keep their type.
INSERT INTO event_type_catalog (code, label_en, parent_code, severity_base, active, created_at) VALUES
    ('unclassified', 'Unclassified / Low Relevance', NULL, 20, TRUE, NOW())
ON CONFLICT (code) DO NOTHING;
