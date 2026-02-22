"""Initial migration - create all tables

Revision ID: 001_initial
Revises:
Create Date: 2026-02-22 11:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables."""

    # ── Knowledge Entries ──
    op.create_table(
        'knowledge_entries',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('category', sa.String(), nullable=True, index=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('embedding', sa.JSON(), nullable=True),
        sa.Column('source', sa.String(), nullable=True),
        sa.Column('confidence', sa.Float(), default=0.0),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
    )

    # ── Learning Experiences ──
    op.create_table(
        'learning_experiences',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('experience_type', sa.String(), nullable=True, index=True),
        sa.Column('context', sa.JSON(), nullable=True),
        sa.Column('action', sa.Text(), nullable=True),
        sa.Column('outcome', sa.Text(), nullable=True),
        sa.Column('reward', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
    )

    # ── Council Discussions ──
    op.create_table(
        'council_discussions',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('topic', sa.Text(), nullable=True),
        sa.Column('wise_men_input', sa.JSON(), nullable=True),
        sa.Column('consensus_score', sa.Float(), nullable=True),
        sa.Column('final_decision', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
    )

    # ── System Metrics ──
    op.create_table(
        'system_metrics',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('metric_name', sa.String(), nullable=True, index=True),
        sa.Column('value', sa.Float(), nullable=True),
        sa.Column('labels', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
    )

    # ── Invoices ──
    op.create_table(
        'invoices',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('invoice_number', sa.String(50), unique=True, nullable=False),
        sa.Column('customer_id', sa.String(100), nullable=False, index=True),
        sa.Column('customer_name', sa.String(300), nullable=False),
        sa.Column('amount', sa.Numeric(15, 2), nullable=False, server_default='0'),
        sa.Column('tax', sa.Numeric(15, 2), nullable=False, server_default='0'),
        sa.Column('total', sa.Numeric(15, 2), nullable=False, server_default='0'),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending', index=True),
        sa.Column('items', sa.JSON(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('due_date', sa.Date(), nullable=True),
        sa.Column('paid_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )

    # ── Inventory Items ──
    op.create_table(
        'inventory_items',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('sku', sa.String(100), unique=True, nullable=False, index=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(200), nullable=True, index=True),
        sa.Column('quantity', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('reorder_point', sa.Integer(), server_default='10'),
        sa.Column('unit_price', sa.Numeric(15, 2), server_default='0'),
        sa.Column('cost_price', sa.Numeric(15, 2), server_default='0'),
        sa.Column('supplier', sa.String(300), nullable=True),
        sa.Column('location', sa.String(200), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )

    # ── Employees ──
    op.create_table(
        'employees',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('employee_id', sa.String(50), unique=True, nullable=False, index=True),
        sa.Column('name', sa.String(300), nullable=False),
        sa.Column('email', sa.String(300), unique=True, nullable=True),
        sa.Column('phone', sa.String(50), nullable=True),
        sa.Column('department', sa.String(200), nullable=True, index=True),
        sa.Column('position', sa.String(200), nullable=True),
        sa.Column('salary', sa.Numeric(15, 2), server_default='0'),
        sa.Column('hire_date', sa.Date(), nullable=True),
        sa.Column('status', sa.String(20), server_default='active', index=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )

    # ── Transactions ──
    op.create_table(
        'transactions',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('date', sa.DateTime(), nullable=True),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('category', sa.String(100), nullable=True, index=True),
        sa.Column('amount', sa.Numeric(15, 2), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('reference', sa.String(200), nullable=True),
        sa.Column('invoice_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    # ── Users (RBAC) ──
    op.create_table(
        'users',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('username', sa.String(100), unique=True, nullable=False, index=True),
        sa.Column('hashed_password', sa.String(500), nullable=False),
        sa.Column('full_name', sa.String(300), nullable=True),
        sa.Column('email', sa.String(300), unique=True, nullable=True),
        sa.Column('role', sa.String(50), nullable=False, server_default='viewer', index=True),
        sa.Column('is_active', sa.Boolean(), server_default='1'),
        sa.Column('permissions', sa.JSON(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )

    # ── System Config ──
    op.create_table(
        'system_config',
        sa.Column('key', sa.String(200), primary_key=True),
        sa.Column('value', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('system_config')
    op.drop_table('users')
    op.drop_table('transactions')
    op.drop_table('employees')
    op.drop_table('inventory_items')
    op.drop_table('invoices')
    op.drop_table('system_metrics')
    op.drop_table('council_discussions')
    op.drop_table('learning_experiences')
    op.drop_table('knowledge_entries')
