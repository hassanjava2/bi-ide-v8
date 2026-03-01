-- BI-IDE v8 PostgreSQL Schema
-- Standardized database schema for all persistent state

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- WAVE A: Authentication & Users
-- ============================================

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- Token blacklist for logout
CREATE TABLE IF NOT EXISTS token_blacklist (
    id SERIAL PRIMARY KEY,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_token_blacklist_hash ON token_blacklist(token_hash);

-- ============================================
-- WAVE B: Council System
-- ============================================

CREATE TABLE IF NOT EXISTS council_members (
    id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL,
    expertise TEXT[] DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    current_focus TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_council_members_role ON council_members(role);
CREATE INDEX idx_council_members_active ON council_members(is_active);

CREATE TABLE IF NOT EXISTS council_decisions (
    id SERIAL PRIMARY KEY,
    decision_id VARCHAR(50) UNIQUE NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    votes JSONB DEFAULT '{}',
    confidence FLOAT DEFAULT 0.0,
    consensus_score FLOAT DEFAULT 0.0,
    evidence JSONB DEFAULT '[]',
    proposed_by INTEGER REFERENCES users(id),
    decided_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_council_decisions_status ON council_decisions(status);
CREATE INDEX idx_council_decisions_created ON council_decisions(created_at);

CREATE TABLE IF NOT EXISTS council_votes (
    id SERIAL PRIMARY KEY,
    decision_id VARCHAR(50) REFERENCES council_decisions(decision_id) ON DELETE CASCADE,
    member_id VARCHAR(10) REFERENCES council_members(id),
    vote VARCHAR(20) NOT NULL, -- approve, reject, abstain
    comment TEXT,
    voted_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(decision_id, member_id)
);

CREATE INDEX idx_council_votes_decision ON council_votes(decision_id);

-- ============================================
-- WAVE C: Training System
-- ============================================

CREATE TABLE IF NOT EXISTS training_jobs (
    id VARCHAR(50) PRIMARY KEY,
    job_name VARCHAR(200) NOT NULL,
    model_name VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, paused, completed, failed, cancelled
    config JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    error_message TEXT,
    assigned_worker VARCHAR(50),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_created ON training_jobs(created_at);

CREATE TABLE IF NOT EXISTS workers (
    id VARCHAR(50) PRIMARY KEY,
    hostname VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'offline', -- online, offline, throttled, busy
    labels TEXT[] DEFAULT '{}',
    hardware JSONB DEFAULT '{}',
    resources JSONB DEFAULT '{}',
    current_job_id VARCHAR(50) REFERENCES training_jobs(id),
    last_heartbeat TIMESTAMPTZ,
    registered_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_workers_status ON workers(status);
CREATE INDEX idx_workers_labels ON workers USING GIN(labels);

CREATE TABLE IF NOT EXISTS models (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) DEFAULT '1.0.0',
    status VARCHAR(20) DEFAULT 'training', -- training, ready, deployed, archived
    architecture VARCHAR(50),
    parameters_count BIGINT,
    dataset_size INTEGER,
    trained_epochs INTEGER DEFAULT 0,
    job_id VARCHAR(50) REFERENCES training_jobs(id),
    metrics JSONB DEFAULT '{}',
    storage_path TEXT,
    is_deployed BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_deployed ON models(is_deployed);

-- ============================================
-- Monitoring & Metrics
-- ============================================

CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    value FLOAT NOT NULL,
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp);

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- info, warning, critical
    message TEXT NOT NULL,
    source VARCHAR(50),
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_resolved ON alerts(is_resolved);

-- ============================================
-- Brain System (Phase 5)
-- ============================================

CREATE TABLE IF NOT EXISTS brain_schedules (
    id SERIAL PRIMARY KEY,
    schedule_name VARCHAR(100) NOT NULL,
    layer_name VARCHAR(50),
    priority INTEGER DEFAULT 5,
    config JSONB DEFAULT '{}',
    cron_expression VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    last_run TIMESTAMPTZ,
    next_run TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS brain_evaluations (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) REFERENCES models(id),
    job_id VARCHAR(50) REFERENCES training_jobs(id),
    evaluation_type VARCHAR(50), -- pre_deploy, periodic, benchmark
    metrics JSONB DEFAULT '{}',
    passed_threshold BOOLEAN DEFAULT FALSE,
    improvement_delta FLOAT,
    evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- Knowledge Base
-- ============================================

CREATE TABLE IF NOT EXISTS knowledge_entries (
    id VARCHAR(50) PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536), -- Requires pgvector extension
    source VARCHAR(100),
    confidence FLOAT DEFAULT 0.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_knowledge_category ON knowledge_entries(category);

-- ============================================
-- Data Pipeline
-- ============================================

CREATE TABLE IF NOT EXISTS data_cleaning_runs (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    records_in INTEGER NOT NULL,
    records_out INTEGER NOT NULL,
    duplicates_removed INTEGER DEFAULT 0,
    noise_removed INTEGER DEFAULT 0,
    validation_errors JSONB DEFAULT '[]',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- ============================================
-- Triggers for updated_at
-- ============================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_council_members_updated_at BEFORE UPDATE ON council_members
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_jobs_updated_at BEFORE UPDATE ON training_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- Seed Data
-- ============================================

-- Insert default council members (fixing duplicate S002)
INSERT INTO council_members (id, name, role, expertise) VALUES
    ('S001', 'حكيم الهوية', 'identity', ARRAY['identity', 'culture', 'values']),
    ('S002', 'حكيم الاستراتيجيا', 'strategy', ARRAY['strategy', 'planning', 'vision']),
    ('S003', 'حكيم الأخلاق', 'ethics', ARRAY['ethics', 'morality', 'fairness']),
    ('S004', 'حكيم التوازن', 'balance', ARRAY['balance', 'harmony', 'equilibrium']),
    ('S005', 'حكيم المعرفة', 'knowledge', ARRAY['knowledge', 'wisdom', 'learning']),
    ('S006', 'حكيم العلاقات', 'relations', ARRAY['relations', 'diplomacy', 'communication']),
    ('S007', 'حكيم الابتكار', 'innovation', ARRAY['innovation', 'creativity', 'invention']),
    ('S008', 'حكيم الحماية', 'protection', ARRAY['protection', 'security', 'defense'])
ON CONFLICT (id) DO NOTHING;

-- Insert operations council
INSERT INTO council_members (id, name, role, expertise) VALUES
    ('O001', 'حكيم النظام', 'system', ARRAY['system', 'architecture', 'integration']),
    ('O002', 'حكيم التنفيذ', 'execution', ARRAY['execution', 'operations', 'deployment']),
    ('O003', 'حكيم الربط', 'bridge', ARRAY['bridge', 'connection', 'interfaces']),
    ('O004', 'حكيم التقارير', 'reports', ARRAY['reports', 'monitoring', 'analytics']),
    ('O005', 'حكيم التنسيق', 'coordination', ARRAY['coordination', 'management', 'organization']),
    ('O006', 'حكيم المتابعة', 'monitoring', ARRAY['monitoring', 'tracking', 'observation']),
    ('O007', 'حكيم التدقيق', 'verification', ARRAY['verification', 'validation', 'quality']),
    ('O008', 'حكيم الطوارئ', 'emergency', ARRAY['emergency', 'crisis', 'recovery'])
ON CONFLICT (id) DO NOTHING;
