-- =====================================================
-- BI IDE v8 - Database Initialization
-- تهيئة قاعدة بيانات BI IDE
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- =====================================================
-- Knowledge Entries - سجل المعرفة
-- =====================================================
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100) NOT NULL DEFAULT 'general',
    source VARCHAR(200),
    confidence FLOAT DEFAULT 0.5,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    embedding_vector BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_entries(category);
CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_title_trgm ON knowledge_entries USING gin(title gin_trgm_ops);

-- =====================================================
-- Learning Experiences - تجارب التعلم
-- =====================================================
CREATE TABLE IF NOT EXISTS learning_experiences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experience_type VARCHAR(100) NOT NULL,
    input_data JSONB NOT NULL DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    model_version VARCHAR(50),
    success BOOLEAN DEFAULT true,
    score FLOAT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_learning_type ON learning_experiences(experience_type);
CREATE INDEX IF NOT EXISTS idx_learning_timestamp ON learning_experiences(timestamp);
CREATE INDEX IF NOT EXISTS idx_learning_success ON learning_experiences(success);

-- =====================================================
-- Council Discussions - مناقشات المجلس
-- =====================================================
CREATE TABLE IF NOT EXISTS council_discussions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    topic VARCHAR(1000) NOT NULL,
    council_member VARCHAR(200) NOT NULL,
    response TEXT NOT NULL,
    source VARCHAR(50) DEFAULT 'local',
    response_source VARCHAR(100),
    confidence FLOAT DEFAULT 0.5,
    evidence JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_council_timestamp ON council_discussions(timestamp);
CREATE INDEX IF NOT EXISTS idx_council_member ON council_discussions(council_member);

-- =====================================================
-- System Metrics - مقاييس النظام
-- =====================================================
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(200) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_type VARCHAR(50) DEFAULT 'gauge',
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp);

-- Partition by time for large-scale metric data (optional)
-- CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON system_metrics(metric_name, timestamp);

-- =====================================================
-- ERP: Invoices - الفواتير
-- =====================================================
CREATE TABLE IF NOT EXISTS invoices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    invoice_number VARCHAR(50) UNIQUE NOT NULL,
    customer_id VARCHAR(100) NOT NULL,
    customer_name VARCHAR(300) NOT NULL,
    amount NUMERIC(15, 2) NOT NULL DEFAULT 0,
    tax NUMERIC(15, 2) NOT NULL DEFAULT 0,
    total NUMERIC(15, 2) NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    items JSONB DEFAULT '[]',
    notes TEXT,
    due_date DATE,
    paid_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(status);
CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices(customer_id);

-- =====================================================
-- ERP: Inventory - المخزون
-- =====================================================
CREATE TABLE IF NOT EXISTS inventory_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sku VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    category VARCHAR(200),
    quantity INTEGER NOT NULL DEFAULT 0,
    reorder_point INTEGER DEFAULT 10,
    unit_price NUMERIC(15, 2) DEFAULT 0,
    cost_price NUMERIC(15, 2) DEFAULT 0,
    supplier VARCHAR(300),
    location VARCHAR(200),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inventory_sku ON inventory_items(sku);
CREATE INDEX IF NOT EXISTS idx_inventory_category ON inventory_items(category);

-- =====================================================
-- ERP: Employees - الموظفين
-- =====================================================
CREATE TABLE IF NOT EXISTS employees (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    employee_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(300) NOT NULL,
    email VARCHAR(300) UNIQUE,
    phone VARCHAR(50),
    department VARCHAR(200),
    position VARCHAR(200),
    salary NUMERIC(15, 2) DEFAULT 0,
    hire_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_employees_dept ON employees(department);
CREATE INDEX IF NOT EXISTS idx_employees_status ON employees(status);

-- =====================================================
-- System Configuration - إعدادات النظام
-- =====================================================
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(200) PRIMARY KEY,
    value TEXT,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- Initial Data
-- =====================================================
INSERT INTO system_config (key, value, description) VALUES
    ('version', '8.0.0', 'Current system version'),
    ('initialized_at', NOW()::TEXT, 'Database initialization timestamp')
ON CONFLICT (key) DO NOTHING;

-- Done
-- تم تهيئة قاعدة البيانات بنجاح ✅

-- =====================================================
-- Training Tables - جداول التدريب
-- =====================================================

-- Training Runs - جولات التدريب
CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_preset VARCHAR(20) NOT NULL,
    model_params BIGINT,
    epochs_total INT,
    epochs_done INT DEFAULT 0,
    batch_size INT,
    learning_rate FLOAT,
    device VARCHAR(20),
    worker_id VARCHAR(100),
    status VARCHAR(20) DEFAULT 'queued',
    loss_final FLOAT,
    accuracy_final FLOAT,
    throughput_sps FLOAT,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    config_json JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model Checkpoints - نقاط التحقق من النموذج
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_run_id UUID REFERENCES training_runs(id),
    epoch INT,
    loss FLOAT,
    accuracy FLOAT,
    file_path TEXT,
    file_size_mb FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- Council Tables - جداول المجلس
-- =====================================================

-- Council Decisions - قرارات المجلس
CREATE TABLE IF NOT EXISTS council_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    decision VARCHAR(20),
    confidence FLOAT,
    votes_json JSONB,
    reasoning TEXT,
    shadow_analysis TEXT,
    light_suggestion TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Council Votes - أصوات المجلس
CREATE TABLE IF NOT EXISTS council_votes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID REFERENCES council_decisions(id),
    member_id VARCHAR(100) NOT NULL,
    vote VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- Monitoring Tables - جداول المراقبة
-- =====================================================

-- Worker Metrics - مقاييس العمال
CREATE TABLE IF NOT EXISTS worker_metrics (
    id SERIAL PRIMARY KEY,
    worker_id VARCHAR(100) NOT NULL,
    cpu_percent FLOAT,
    gpu_percent FLOAT,
    gpu_temp_c FLOAT,
    ram_percent FLOAT,
    gpu_vram_used FLOAT,
    gpu_vram_total FLOAT,
    is_training BOOLEAN DEFAULT FALSE,
    measured_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_worker_metrics_time ON worker_metrics(worker_id, measured_at DESC);

-- Training Metrics - مقاييس التدريب
CREATE TABLE IF NOT EXISTS training_metrics (
    id SERIAL PRIMARY KEY,
    training_run_id UUID REFERENCES training_runs(id),
    epoch INT NOT NULL,
    step INT,
    loss FLOAT,
    accuracy FLOAT,
    learning_rate FLOAT,
    throughput_sps FLOAT,
    gpu_utilization FLOAT,
    gpu_memory_used FLOAT,
    recorded_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_training_metrics_run ON training_metrics(training_run_id, recorded_at DESC);

-- =====================================================
-- Learning Log - سجل التعلم
-- =====================================================
CREATE TABLE IF NOT EXISTS learning_log (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50),
    content_type VARCHAR(50),
    content TEXT,
    learned_topics TEXT[],
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- Alerts - التنبيهات
-- =====================================================
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    severity VARCHAR(20) NOT NULL,
    source VARCHAR(100),
    message TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(resolved, created_at DESC);

-- Done - تم ✅
