"""
ERP Database Models - نماذج قاعدة البيانات لنظام ERP
SQLAlchemy models backed by PostgreSQL/SQLite
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, JSON, Boolean,
    Numeric, Date, Enum as SQLEnum, ForeignKey,
)
from sqlalchemy.orm import relationship

from core.database import Base


def gen_uuid():
    return str(uuid.uuid4())


# ─────────────────── Invoices ───────────────────


class InvoiceDB(Base):
    """فاتورة — مخزنة في قاعدة البيانات"""
    __tablename__ = "invoices"

    id = Column(String, primary_key=True, default=gen_uuid)
    invoice_number = Column(String(50), unique=True, nullable=False)
    customer_id = Column(String(100), nullable=False, index=True)
    customer_name = Column(String(300), nullable=False)
    amount = Column(Numeric(15, 2), nullable=False, default=0)
    tax = Column(Numeric(15, 2), nullable=False, default=0)
    total = Column(Numeric(15, 2), nullable=False, default=0)
    status = Column(String(20), nullable=False, default="pending", index=True)
    items = Column(JSON, default=list)
    notes = Column(Text, default="")
    due_date = Column(Date, default=lambda: (datetime.now() + timedelta(days=30)).date())
    paid_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────── Inventory ───────────────────


class InventoryItemDB(Base):
    """عنصر مخزون — مخزن في قاعدة البيانات"""
    __tablename__ = "inventory_items"

    id = Column(String, primary_key=True, default=gen_uuid)
    sku = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(500), nullable=False)
    description = Column(Text, default="")
    category = Column(String(200), index=True)
    quantity = Column(Integer, nullable=False, default=0)
    reorder_point = Column(Integer, default=10)
    unit_price = Column(Numeric(15, 2), default=0)
    cost_price = Column(Numeric(15, 2), default=0)
    supplier = Column(String(300), default="")
    location = Column(String(200), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────── Employees ───────────────────


class EmployeeDB(Base):
    """موظف — مخزن في قاعدة البيانات"""
    __tablename__ = "employees"

    id = Column(String, primary_key=True, default=gen_uuid)
    employee_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(300), nullable=False)
    email = Column(String(300), unique=True, nullable=True)
    phone = Column(String(50), default="")
    department = Column(String(200), index=True)
    position = Column(String(200))
    salary = Column(Numeric(15, 2), default=0)
    hire_date = Column(Date, nullable=True)
    status = Column(String(20), default="active", index=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────── Transactions ───────────────────


class TransactionDB(Base):
    """معاملة مالية"""
    __tablename__ = "transactions"

    id = Column(String, primary_key=True, default=gen_uuid)
    date = Column(DateTime, default=datetime.utcnow)
    type = Column(String(50), nullable=False)  # income, expense, transfer
    category = Column(String(100), index=True)
    amount = Column(Numeric(15, 2), nullable=False)
    description = Column(Text, default="")
    reference = Column(String(200), default="")
    invoice_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────── Users (for RBAC) ───────────────────


class UserDB(Base):
    """مستخدم النظام — لنظام RBAC"""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_uuid)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(500), nullable=False)
    full_name = Column(String(300), default="")
    email = Column(String(300), unique=True, nullable=True)
    role = Column(String(50), nullable=False, default="viewer", index=True)
    is_active = Column(Boolean, default=True)
    permissions = Column(JSON, default=list)  # Additional fine-grained permissions
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────── System Config ───────────────────


class SystemConfigDB(Base):
    """إعدادات النظام"""
    __tablename__ = "system_config"

    key = Column(String(200), primary_key=True)
    value = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
