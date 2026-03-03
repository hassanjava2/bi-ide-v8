"""
003_add_notifications_backups_tables

Add notifications and backups tables for persistent storage
جداول الإشعارات والنسخ الاحتياطي للتخزين الدائم

Revision ID: 003
Revises: 002
Create Date: 2026-03-03 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # =====================================================
    # Notifications Tables - جداول الإشعارات
    # =====================================================
    
    op.create_table(
        'notifications',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=False, index=True),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('channel', sa.String(length=20), server_default='in_app', nullable=False),
        sa.Column('priority', sa.String(length=20), server_default='medium', nullable=False),
        sa.Column('is_read', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('metadata_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('read_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('idx_notifications_user_unread', 'notifications', ['user_id', 'is_read'])
    op.create_index('idx_notifications_created', 'notifications', ['created_at'])
    
    # =====================================================
    # Backup Tables - جداول النسخ الاحتياطي
    # =====================================================
    
    op.create_table(
        'backups',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=False, index=True),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('backup_type', sa.String(length=20), server_default='full', nullable=False),
        sa.Column('status', sa.String(length=20), server_default='pending', nullable=False),
        sa.Column('size_bytes', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('storage_path', sa.Text(), nullable=True),
        sa.Column('manifest_json', postgresql.JSONB(), nullable=True),
        sa.Column('metadata_json', postgresql.JSONB(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('completed_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table(
        'backup_schedules',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=False, index=True),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('backup_type', sa.String(length=20), server_default='full', nullable=False),
        sa.Column('cron_expression', sa.String(length=100), nullable=False),
        sa.Column('retention_days', sa.Integer(), server_default='30', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('metadata_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('last_run', sa.TIMESTAMP(), nullable=True),
        sa.Column('next_run', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('idx_backups_user_status', 'backups', ['user_id', 'status'])
    op.create_index('idx_backups_created', 'backups', ['created_at'])
    op.create_index('idx_backup_schedules_user', 'backup_schedules', ['user_id', 'is_active'])
    
    # =====================================================
    # Training Jobs Table - جدول مهام التدريب
    # =====================================================
    
    op.create_table(
        'training_jobs',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('job_id', sa.String(length=100), unique=True, nullable=False, index=True),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=20), server_default='pending', nullable=False),
        sa.Column('config_json', postgresql.JSONB(), nullable=True),
        sa.Column('metrics_json', postgresql.JSONB(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('started_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('completed_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table(
        'trained_models',
        sa.Column('id', postgresql.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('model_id', sa.String(length=100), unique=True, nullable=False, index=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('version', sa.String(length=20), nullable=False),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('is_deployed', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('metadata_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('idx_training_jobs_status', 'training_jobs', ['status'])
    op.create_index('idx_trained_models_deployed', 'trained_models', ['is_deployed'])


def downgrade() -> None:
    op.drop_table('trained_models')
    op.drop_table('training_jobs')
    op.drop_table('backup_schedules')
    op.drop_table('backups')
    op.drop_table('notifications')
