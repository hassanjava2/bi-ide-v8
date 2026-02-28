"""
001_initial_tables

Training, Council, Monitoring tables for BI-IDE
جداول التدريب والمجلس والمراقبة لـ BI-IDE

Revision ID: 001
Revises: 
Create Date: 2026-02-28 23:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # =====================================================
    # Training Tables - جداول التدريب
    # =====================================================
    
    # training_runs - جولات التدريب
    op.create_table(
        'training_runs',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('model_preset', sa.String(length=20), nullable=False),
        sa.Column('model_params', sa.BigInteger(), nullable=True),
        sa.Column('epochs_total', sa.Integer(), nullable=True),
        sa.Column('epochs_done', sa.Integer(), server_default='0', nullable=True),
        sa.Column('batch_size', sa.Integer(), nullable=True),
        sa.Column('learning_rate', sa.Float(), nullable=True),
        sa.Column('device', sa.String(length=20), nullable=True),
        sa.Column('worker_id', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=20), server_default='queued', nullable=True),
        sa.Column('loss_final', sa.Float(), nullable=True),
        sa.Column('accuracy_final', sa.Float(), nullable=True),
        sa.Column('throughput_sps', sa.Float(), nullable=True),
        sa.Column('started_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('finished_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('config_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # model_checkpoints - نقاط التحقق من النموذج
    op.create_table(
        'model_checkpoints',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('training_run_id', postgresql.UUID(), nullable=True),
        sa.Column('epoch', sa.Integer(), nullable=True),
        sa.Column('loss', sa.Float(), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('file_path', sa.Text(), nullable=True),
        sa.Column('file_size_mb', sa.Float(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=True),
        sa.ForeignKeyConstraint(['training_run_id'], ['training_runs.id']),
        sa.PrimaryKeyConstraint('id')
    )
    
    # =====================================================
    # Council Tables - جداول المجلس
    # =====================================================
    
    # council_decisions - قرارات المجلس
    op.create_table(
        'council_decisions',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('decision', sa.String(length=20), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('votes_json', postgresql.JSONB(), nullable=True),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('shadow_analysis', sa.Text(), nullable=True),
        sa.Column('light_suggestion', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # council_votes - أصوات المجلس
    op.create_table(
        'council_votes',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('decision_id', postgresql.UUID(), nullable=True),
        sa.Column('member_id', sa.String(length=100), nullable=False),
        sa.Column('vote', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=True),
        sa.ForeignKeyConstraint(['decision_id'], ['council_decisions.id']),
        sa.PrimaryKeyConstraint('id')
    )
    
    # =====================================================
    # Monitoring Tables - جداول المراقبة
    # =====================================================
    
    # worker_metrics - مقاييس العمال
    op.create_table(
        'worker_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('worker_id', sa.String(length=100), nullable=False),
        sa.Column('cpu_percent', sa.Float(), nullable=True),
        sa.Column('gpu_percent', sa.Float(), nullable=True),
        sa.Column('gpu_temp_c', sa.Float(), nullable=True),
        sa.Column('ram_percent', sa.Float(), nullable=True),
        sa.Column('gpu_vram_used', sa.Float(), nullable=True),
        sa.Column('gpu_vram_total', sa.Float(), nullable=True),
        sa.Column('is_training', sa.Boolean(), server_default='FALSE', nullable=True),
        sa.Column('measured_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_worker_metrics_time', 'worker_metrics', ['worker_id', sa.text('measured_at DESC')])
    
    # training_metrics - مقاييس التدريب
    op.create_table(
        'training_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('training_run_id', postgresql.UUID(), nullable=True),
        sa.Column('epoch', sa.Integer(), nullable=False),
        sa.Column('step', sa.Integer(), nullable=True),
        sa.Column('loss', sa.Float(), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('learning_rate', sa.Float(), nullable=True),
        sa.Column('throughput_sps', sa.Float(), nullable=True),
        sa.Column('gpu_utilization', sa.Float(), nullable=True),
        sa.Column('gpu_memory_used', sa.Float(), nullable=True),
        sa.Column('recorded_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=True),
        sa.ForeignKeyConstraint(['training_run_id'], ['training_runs.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_training_metrics_run', 'training_metrics', ['training_run_id', sa.text('recorded_at DESC')])
    
    # =====================================================
    # Learning Log - سجل التعلم
    # =====================================================
    op.create_table(
        'learning_log',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('source', sa.String(length=50), nullable=True),
        sa.Column('content_type', sa.String(length=50), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('learned_topics', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # =====================================================
    # Alerts - التنبيهات
    # =====================================================
    op.create_table(
        'alerts',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('source', sa.String(length=100), nullable=True),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('resolved', sa.Boolean(), server_default='FALSE', nullable=True),
        sa.Column('resolved_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_alerts_active', 'alerts', ['resolved', sa.text('created_at DESC')])


def downgrade() -> None:
    # Drop in reverse order to handle foreign key constraints
    # حذف بالترتيب العكسي للتعامل مع قيود المفاتيح الخارجية
    
    op.drop_index('idx_alerts_active', table_name='alerts')
    op.drop_table('alerts')
    
    op.drop_table('learning_log')
    
    op.drop_index('idx_training_metrics_run', table_name='training_metrics')
    op.drop_table('training_metrics')
    
    op.drop_index('idx_worker_metrics_time', table_name='worker_metrics')
    op.drop_table('worker_metrics')
    
    op.drop_table('council_votes')
    op.drop_table('council_decisions')
    
    op.drop_table('model_checkpoints')
    op.drop_table('training_runs')
