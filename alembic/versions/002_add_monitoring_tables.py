"""Add monitoring tables

Add training_metrics, worker_metrics, and alerts tables for BI-IDE v8

Revision ID: 002
Revises: 001
Create Date: 2026-02-28 23:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Training Metrics Table
    op.create_table(
        'training_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('training_run_id', postgresql.UUID(), nullable=False),
        sa.Column('epoch', sa.Integer(), nullable=False),
        sa.Column('step', sa.Integer(), nullable=True),
        sa.Column('loss', sa.Float(), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('learning_rate', sa.Float(), nullable=True),
        sa.Column('throughput_sps', sa.Float(), nullable=True),
        sa.Column('gpu_utilization', sa.Float(), nullable=True),
        sa.Column('gpu_memory_used', sa.BigInteger(), nullable=True),
        sa.Column('gpu_memory_total', sa.BigInteger(), nullable=True),
        sa.Column('cpu_percent', sa.Float(), nullable=True),
        sa.Column('ram_used', sa.BigInteger(), nullable=True),
        sa.Column('recorded_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('metadata_json', postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['training_run_id'], ['training_runs.id'], ondelete='CASCADE'),
        comment='Detailed training metrics'
    )
    
    # Training metrics indexes
    op.create_index('idx_training_metrics_run_epoch', 'training_metrics', ['training_run_id', 'epoch'])
    op.create_index('idx_training_metrics_recorded', 'training_metrics', [sa.text('recorded_at DESC')])
    
    # Worker Metrics Table
    op.create_table(
        'worker_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('worker_id', sa.String(length=100), nullable=False),
        sa.Column('worker_type', sa.String(length=50), server_default='general', nullable=True),
        sa.Column('hostname', sa.String(length=255), nullable=True),
        sa.Column('cpu_percent', sa.Float(), nullable=True),
        sa.Column('cpu_count', sa.Integer(), nullable=True),
        sa.Column('gpu_percent', sa.Float(), nullable=True),
        sa.Column('gpu_temp_c', sa.Float(), nullable=True),
        sa.Column('gpu_power_w', sa.Float(), nullable=True),
        sa.Column('ram_percent', sa.Float(), nullable=True),
        sa.Column('ram_used_mb', sa.BigInteger(), nullable=True),
        sa.Column('ram_total_mb', sa.BigInteger(), nullable=True),
        sa.Column('gpu_vram_used', sa.BigInteger(), nullable=True),
        sa.Column('gpu_vram_total', sa.BigInteger(), nullable=True),
        sa.Column('disk_used_gb', sa.Float(), nullable=True),
        sa.Column('disk_total_gb', sa.Float(), nullable=True),
        sa.Column('network_rx_mb', sa.Float(), nullable=True),
        sa.Column('network_tx_mb', sa.Float(), nullable=True),
        sa.Column('is_training', sa.Boolean(), server_default='FALSE', nullable=True),
        sa.Column('active_tasks', sa.Integer(), server_default='0', nullable=True),
        sa.Column('queue_depth', sa.Integer(), server_default='0', nullable=True),
        sa.Column('measured_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('metadata_json', postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        comment='System worker metrics'
    )
    
    # Worker metrics indexes
    op.create_index('idx_worker_metrics_worker_time', 'worker_metrics', ['worker_id', sa.text('measured_at DESC')])
    op.create_index('idx_worker_metrics_training', 'worker_metrics', ['is_training', sa.text('measured_at DESC')])
    op.create_index('idx_worker_metrics_hostname', 'worker_metrics', ['hostname'])
    
    # Alerts Table
    op.create_table(
        'alerts',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('category', sa.String(length=50), server_default='general', nullable=True),
        sa.Column('source', sa.String(length=100), nullable=True),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('details_json', postgresql.JSONB(), nullable=True),
        sa.Column('resolved', sa.Boolean(), server_default='FALSE', nullable=True),
        sa.Column('resolved_by', sa.String(length=100), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('resolved_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('auto_resolve', sa.Boolean(), server_default='FALSE', nullable=True),
        sa.Column('notification_sent', sa.Boolean(), server_default='FALSE', nullable=True),
        sa.Column('notification_channels', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=True, onupdate=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        comment='System alerts'
    )
    
    # Alerts indexes
    op.create_index('idx_alerts_active', 'alerts', ['resolved', sa.text('created_at DESC')])
    op.create_index('idx_alerts_severity', 'alerts', ['severity', 'resolved', sa.text('created_at DESC')])
    op.create_index('idx_alerts_source', 'alerts', ['source', sa.text('created_at DESC')])
    op.create_index('idx_alerts_category', 'alerts', ['category', sa.text('created_at DESC')])
    
    # Alert Rules Table
    op.create_table(
        'alert_rules',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False, unique=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('enabled', sa.Boolean(), server_default='TRUE', nullable=True),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('condition_operator', sa.String(length=10), nullable=False),
        sa.Column('threshold_value', sa.Float(), nullable=False),
        sa.Column('duration_seconds', sa.Integer(), server_default='60', nullable=True),
        sa.Column('cooldown_seconds', sa.Integer(), server_default='300', nullable=True),
        sa.Column('notification_channels', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('auto_resolve', sa.Boolean(), server_default='TRUE', nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=True, onupdate=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        comment='Alert generation rules'
    )
    
    # Alert rules indexes
    op.create_index('idx_alert_rules_enabled', 'alert_rules', ['enabled', 'metric_type'])
    
    # Insert default rules
    op.execute("""
        INSERT INTO alert_rules 
            (name, description, severity, metric_type, condition_operator, threshold_value, notification_channels)
        VALUES 
            ('high_cpu', 'High CPU usage', 'warning', 'cpu', 'gt', 80.0, ARRAY['log', 'email']),
            ('critical_cpu', 'Critical CPU usage', 'critical', 'cpu', 'gt', 95.0, ARRAY['log', 'email', 'slack']),
            ('high_memory', 'High memory usage', 'warning', 'memory', 'gt', 85.0, ARRAY['log', 'email']),
            ('critical_memory', 'Critical memory usage', 'critical', 'memory', 'gt', 95.0, ARRAY['log', 'email', 'slack']),
            ('high_gpu_temp', 'High GPU temperature', 'warning', 'gpu_temp', 'gt', 80.0, ARRAY['log', 'email']),
            ('critical_gpu_temp', 'Critical GPU temperature', 'critical', 'gpu_temp', 'gt', 90.0, ARRAY['log', 'email', 'slack']),
            ('disk_full', 'Disk nearly full', 'warning', 'disk', 'gt', 85.0, ARRAY['log', 'email']),
            ('disk_critical', 'Disk full', 'critical', 'disk', 'gt', 95.0, ARRAY['log', 'email', 'slack'])
        ON CONFLICT (name) DO NOTHING
    """)


def downgrade() -> None:
    """Rollback migration"""
    op.drop_index('idx_alert_rules_enabled', table_name='alert_rules')
    op.drop_table('alert_rules')
    
    op.drop_index('idx_alerts_category', table_name='alerts')
    op.drop_index('idx_alerts_source', table_name='alerts')
    op.drop_index('idx_alerts_severity', table_name='alerts')
    op.drop_index('idx_alerts_active', table_name='alerts')
    op.drop_table('alerts')
    
    op.drop_index('idx_worker_metrics_hostname', table_name='worker_metrics')
    op.drop_index('idx_worker_metrics_training', table_name='worker_metrics')
    op.drop_index('idx_worker_metrics_worker_time', table_name='worker_metrics')
    op.drop_table('worker_metrics')
    
    op.drop_index('idx_training_metrics_recorded', table_name='training_metrics')
    op.drop_index('idx_training_metrics_run_epoch', table_name='training_metrics')
    op.drop_table('training_metrics')
