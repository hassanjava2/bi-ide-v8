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
