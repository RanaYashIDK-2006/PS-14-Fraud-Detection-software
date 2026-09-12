-- PS-14 PostgreSQL Initialization Script
-- Runs automatically when the PostgreSQL container starts for the first time

-- Create schemas for each service
CREATE SCHEMA IF NOT EXISTS identity;
CREATE SCHEMA IF NOT EXISTS privacy;
CREATE SCHEMA IF NOT EXISTS risk;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS verification;

-- Grant usage to the application user
GRANT USAGE ON SCHEMA identity TO CURRENT_USER;
GRANT USAGE ON SCHEMA privacy TO CURRENT_USER;
GRANT USAGE ON SCHEMA risk TO CURRENT_USER;
GRANT USAGE ON SCHEMA audit TO CURRENT_USER;
GRANT USAGE ON SCHEMA verification TO CURRENT_USER;

-- Grant all privileges on schemas (for table creation)
GRANT ALL ON SCHEMA identity TO CURRENT_USER;
GRANT ALL ON SCHEMA privacy TO CURRENT_USER;
GRANT ALL ON SCHEMA risk TO CURRENT_USER;
GRANT ALL ON SCHEMA audit TO CURRENT_USER;
GRANT ALL ON SCHEMA verification TO CURRENT_USER;

-- Set default search path for the database
ALTER DATABASE ps14 SET search_path TO public;

-- Enable UUID extension (used by some models)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'PS-14 PostgreSQL initialization complete';
    RAISE NOTICE 'Schemas created: identity, privacy, risk, audit, verification';
END $$;
