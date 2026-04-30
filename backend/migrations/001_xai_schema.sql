-- PostgreSQL schema for concept index, audit, overrides (optional — SQLite supported via XAI_AUDIT_SQLITE)

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS concept_index (
    id              SERIAL PRIMARY KEY,
    chunk_id        VARCHAR(255),
    entity          VARCHAR(100),
    claim_type      VARCHAR(100),
    value           VARCHAR(255),
    section_id      VARCHAR(255),
    edition_date    DATE,
    report_type     VARCHAR(20)
);
CREATE INDEX IF NOT EXISTS idx_concept_entity ON concept_index(entity, claim_type);
CREATE INDEX IF NOT EXISTS idx_concept_edition ON concept_index(edition_date);

CREATE TABLE IF NOT EXISTS edition_registry (
    id              SERIAL PRIMARY KEY,
    report_type     VARCHAR(20),
    edition_date    DATE UNIQUE,
    ingestion_ts    TIMESTAMP DEFAULT NOW(),
    changed_sections TEXT[],
    supersedes      DATE[]
);

CREATE TABLE IF NOT EXISTS audit_log (
    id              SERIAL PRIMARY KEY,
    session_id      UUID DEFAULT gen_random_uuid(),
    created_at      TIMESTAMP DEFAULT NOW(),
    query           TEXT,
    answer          TEXT,
    trust_gate      VARCHAR(30),
    confidence      FLOAT,
    hallucination   BOOLEAN,
    claims_json     JSONB,
    sources_json    JSONB,
    nli_json        JSONB,
    conflicts_json  JSONB,
    ragas_json      JSONB,
    artifact_json   JSONB
);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_trust   ON audit_log(trust_gate);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at DESC);

CREATE TABLE IF NOT EXISTS attribution_log (
    id              SERIAL PRIMARY KEY,
    session_id      UUID,
    claim_text      TEXT,
    source_doc_id   VARCHAR(255),
    source_section  VARCHAR(255),
    similarity      FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS override_store (
    override_id     UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    query           TEXT,
    query_embedding JSONB,
    original_verdict VARCHAR(30),
    human_verdict   VARCHAR(30),
    reasoning       TEXT,
    relevant_section VARCHAR(255),
    edition_date    DATE,
    timestamp       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS drift_log (
    id              SERIAL PRIMARY KEY,
    session_id      UUID,
    query           TEXT,
    old_verdict     VARCHAR(30),
    new_verdict     VARCHAR(30),
    changed_section VARCHAR(255),
    detected_at     TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paraphrase_log (
    id              SERIAL PRIMARY KEY,
    session_id      UUID,
    mean_similarity FLOAT,
    min_similarity  FLOAT,
    citation_overlap FLOAT,
    verdict_agreement FLOAT,
    is_stable       BOOLEAN,
    created_at      TIMESTAMP DEFAULT NOW()
);
