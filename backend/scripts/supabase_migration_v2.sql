-- PS-14 Supabase Migration v2 — Schema-Isolated
-- Run this in Supabase SQL Editor (Dashboard -> SQL Editor -> New Query)
-- Each service gets its own schema to preserve the SQLite-per-file isolation.

-- === Create Schemas ===
CREATE SCHEMA IF NOT EXISTS identity;
CREATE SCHEMA IF NOT EXISTS privacy;
CREATE SCHEMA IF NOT EXISTS risk;
CREATE SCHEMA IF NOT EXISTS audit;

-- === Identity Service (identity.*) ===
CREATE TABLE IF NOT EXISTS identity.users (
    user_id VARCHAR(36) NOT NULL,
    full_name TEXT NOT NULL,
    phone_encrypted BYTEA NOT NULL,
    email_encrypted BYTEA NOT NULL,
    email_hash VARCHAR(64) NOT NULL,
    kyc_doc_ref TEXT,
    account_number VARCHAR(20),
    account_type VARCHAR(20),
    address_encrypted BYTEA,
    role VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (user_id)
);

CREATE TABLE IF NOT EXISTS identity.auth_credentials (
    credential_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    credential_type VARCHAR(32) NOT NULL,
    secret_hash TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (credential_id),
    FOREIGN KEY(user_id) REFERENCES identity.users (user_id)
);

CREATE TABLE IF NOT EXISTS identity.pseudonym_mapping (
    fraud_id VARCHAR(16) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    generated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    rotation_group INTEGER NOT NULL,
    access_log_required BOOLEAN NOT NULL,
    PRIMARY KEY (fraud_id),
    FOREIGN KEY(user_id) REFERENCES identity.users (user_id)
);

CREATE TABLE IF NOT EXISTS identity.pseudonym_access_log (
    id SERIAL NOT NULL,
    fraud_id VARCHAR(16) NOT NULL,
    actor VARCHAR(64) NOT NULL,
    action VARCHAR(64) NOT NULL,
    reason TEXT NOT NULL,
    accessed_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (id)
);

-- === Privacy Service (privacy.*) ===
CREATE TABLE IF NOT EXISTS privacy.fraud_profiles (
    fraud_id VARCHAR(16) NOT NULL,
    avg_txn_amount_90d FLOAT NOT NULL,
    txn_freq_7d INTEGER NOT NULL,
    typical_txn_hours TEXT NOT NULL,
    known_device_count INTEGER NOT NULL,
    usual_locations TEXT NOT NULL,
    usual_recipients TEXT NOT NULL,
    behavioral_baseline_vector TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    last_updated TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (fraud_id)
);

CREATE TABLE IF NOT EXISTS privacy.device_fingerprints (
    device_hash VARCHAR(32) NOT NULL,
    fraud_id VARCHAR(64) NOT NULL,
    first_seen TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    last_seen TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (device_hash, fraud_id),
    FOREIGN KEY(fraud_id) REFERENCES privacy.fraud_profiles (fraud_id)
);

CREATE TABLE IF NOT EXISTS privacy.transaction_features (
    event_id VARCHAR(36) NOT NULL,
    fraud_id VARCHAR(16) NOT NULL,
    amount_ratio FLOAT NOT NULL,
    txn_amount_bucket VARCHAR(32) NOT NULL,
    txn_freq_last_24h INTEGER NOT NULL,
    txn_time_unusual BOOLEAN NOT NULL,
    new_device_flag BOOLEAN NOT NULL,
    unusual_location_flag BOOLEAN NOT NULL,
    unusual_recipient_flag BOOLEAN NOT NULL,
    failed_auth_count_24h INTEGER NOT NULL,
    days_since_last_similar_txn FLOAT NOT NULL,
    gradual_escalation_score FLOAT NOT NULL,
    known_device_count INTEGER NOT NULL,
    account_tenure_days FLOAT NOT NULL,
    hour_of_day INTEGER NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    shared_device_accounts INTEGER NOT NULL,
    shared_recipient_accounts INTEGER NOT NULL,
    mule_ring_score FLOAT NOT NULL,
    baseline_committed BOOLEAN DEFAULT false NOT NULL,
    device_hash VARCHAR(32),
    location_id VARCHAR(64),
    recipient_id VARCHAR(64),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (event_id),
    FOREIGN KEY(fraud_id) REFERENCES privacy.fraud_profiles (fraud_id)
);

-- === Risk + Verification Service (risk.*) ===
CREATE TABLE IF NOT EXISTS risk.risk_scores (
    score_id VARCHAR(36) NOT NULL,
    fraud_id VARCHAR(16) NOT NULL,
    event_id VARCHAR(64) NOT NULL,
    risk_score INTEGER NOT NULL,
    risk_band VARCHAR(16) NOT NULL,
    reason_codes TEXT NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    ml_score FLOAT NOT NULL,
    rule_score FLOAT NOT NULL,
    degraded BOOLEAN NOT NULL,
    scored_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (score_id),
    CONSTRAINT ck_risk_score_range CHECK (risk_score BETWEEN 0 AND 100)
);

CREATE TABLE IF NOT EXISTS risk.verification_outcomes (
    verification_id VARCHAR(36) NOT NULL,
    score_id VARCHAR(36) NOT NULL,
    fraud_id VARCHAR(16) NOT NULL,
    event_id VARCHAR(64) NOT NULL,
    outcome VARCHAR(16) NOT NULL,
    case_id VARCHAR(32) NOT NULL,
    resolved_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (verification_id),
    FOREIGN KEY(score_id) REFERENCES risk.risk_scores (score_id)
);

CREATE TABLE IF NOT EXISTS risk.investigator_cases (
    case_id VARCHAR(36) NOT NULL,
    fraud_id VARCHAR(16) NOT NULL,
    event_id VARCHAR(64) NOT NULL,
    score_id VARCHAR(36) NOT NULL,
    status VARCHAR(24) NOT NULL,
    investigator_id VARCHAR(16),
    confidence VARCHAR(8) NOT NULL,
    priority FLOAT NOT NULL,
    risk_score INTEGER NOT NULL,
    reason_codes TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    closed_at TIMESTAMP WITHOUT TIME ZONE,
    PRIMARY KEY (case_id),
    FOREIGN KEY(score_id) REFERENCES risk.risk_scores (score_id)
);

-- === Audit Service (audit.*) ===
CREATE TABLE IF NOT EXISTS audit.audit_events (
    seq SERIAL NOT NULL,
    event_id VARCHAR(36) NOT NULL,
    fraud_id VARCHAR(16) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    prev_hash VARCHAR(64) NOT NULL,
    entry_hash VARCHAR(64) NOT NULL,
    payload_summary TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (seq),
    UNIQUE (event_id)
);

CREATE TABLE IF NOT EXISTS audit.audit_access_log (
    id SERIAL NOT NULL,
    actor VARCHAR(64) NOT NULL,
    action VARCHAR(64) NOT NULL,
    query_summary TEXT NOT NULL,
    accessed_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (id)
);

-- === Performance Indexes (per schema) ===
-- Privacy
CREATE INDEX IF NOT EXISTS ix_tf_fraud_ts ON privacy.transaction_features (fraud_id, created_at);
CREATE INDEX IF NOT EXISTS ix_tf_device_ts ON privacy.transaction_features (device_hash, created_at);
CREATE INDEX IF NOT EXISTS ix_tf_recipient ON privacy.transaction_features (recipient_id);
CREATE INDEX IF NOT EXISTS ix_tf_baseline ON privacy.transaction_features (fraud_id, baseline_committed) WHERE baseline_committed = true;

-- Risk
CREATE INDEX IF NOT EXISTS ix_rs_event ON risk.risk_scores (event_id);
CREATE INDEX IF NOT EXISTS ix_rs_fraud ON risk.risk_scores (fraud_id);

-- Audit
CREATE INDEX IF NOT EXISTS ix_ae_type ON audit.audit_events (event_type);
CREATE INDEX IF NOT EXISTS ix_ae_fraud ON audit.audit_events (fraud_id);

-- Identity
CREATE INDEX IF NOT EXISTS ix_user_email ON identity.users (email_hash);
CREATE INDEX IF NOT EXISTS ix_pseudo_user ON identity.pseudonym_mapping (user_id);

-- === Append-only triggers for audit_events ===
-- Use DO blocks to safely drop and recreate triggers
DO $$
BEGIN
    -- Drop triggers on public.audit_events (from v1)
    IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_audit_no_update' AND tgrelid = 'public.audit_events'::regclass) THEN
        DROP TRIGGER trg_audit_no_update ON public.audit_events;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_audit_no_delete' AND tgrelid = 'public.audit_events'::regclass) THEN
        DROP TRIGGER trg_audit_no_delete ON public.audit_events;
    END IF;
    -- Drop triggers on audit.audit_events (from v1 if schema existed)
    IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_audit_no_update' AND tgrelid = 'audit.audit_events'::regclass) THEN
        DROP TRIGGER trg_audit_no_update ON audit.audit_events;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_audit_no_delete' AND tgrelid = 'audit.audit_events'::regclass) THEN
        DROP TRIGGER trg_audit_no_delete ON audit.audit_events;
    END IF;
END $$;

-- Create functions in audit schema
CREATE OR REPLACE FUNCTION audit.prevent_audit_update()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit_events is append-only; UPDATE not allowed';
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION audit.prevent_audit_delete()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit_events is append-only; DELETE not allowed';
END;
$$ LANGUAGE plpgsql;

-- Create triggers on audit.audit_events
CREATE TRIGGER trg_audit_no_update
    BEFORE UPDATE ON audit.audit_events
    FOR EACH ROW
    EXECUTE FUNCTION audit.prevent_audit_update();

CREATE TRIGGER trg_audit_no_delete
    BEFORE DELETE ON audit.audit_events
    FOR EACH ROW
    EXECUTE FUNCTION audit.prevent_audit_delete();

-- === Grant usage to the default postgres role (Supabase default) ===
GRANT USAGE ON SCHEMA identity TO postgres;
GRANT USAGE ON SCHEMA privacy TO postgres;
GRANT USAGE ON SCHEMA risk TO postgres;
GRANT USAGE ON SCHEMA audit TO postgres;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA identity TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA privacy TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA risk TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO postgres;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA identity TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA privacy TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA risk TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO postgres;

-- Allow future tables to inherit grants
ALTER DEFAULT PRIVILEGES IN SCHEMA identity GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA privacy GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA risk GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA identity GRANT ALL ON SEQUENCES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA privacy GRANT ALL ON SEQUENCES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA risk GRANT ALL ON SEQUENCES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT ALL ON SEQUENCES TO postgres;
