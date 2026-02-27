"""
ERP Database Models - نماذج قاعدة البيانات لنظام ERP
SQLAlchemy models backed by PostgreSQL/SQLite
حسابات + مخزون + موارد بشرية
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict
import enum

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean,
    Numeric, Date, Enum as SQLEnum, ForeignKey,
)
from sqlalchemy.orm import relationship

from core.database import Base


class ERPBase(Base):
    """Base for ERP DB models.

    Some legacy modules define duplicate ORM models (same table names) under the
    same metadata. Setting extend_existing avoids import-time crashes.
    """

    __abstract__ = True
    __table_args__ = {"extend_existing": True}


def gen_uuid():
    """Generate UUID string"""
    return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════

class AccountTypeDB(enum.Enum):
    """أنواع الحسابات المحاسبية"""
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    REVENUE = "revenue"
    EXPENSE = "expense"


class InvoiceStatusDB(enum.Enum):
    """حالات الفاتورة"""
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class EmployeeStatusDB(enum.Enum):
    """حالات الموظف"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ON_LEAVE = "on_leave"
    TERMINATED = "terminated"


# ═══════════════════════════════════════════════════════════════
# ACCOUNTING MODELS - نماذج المحاسبة
# ═══════════════════════════════════════════════════════════════

class AccountDB(ERPBase):
    """حساب محاسبي — مخزن في قاعدة البيانات"""
    __tablename__ = "erp_accounts"

    id = Column(String, primary_key=True, default=gen_uuid)
    code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(300), nullable=False)
    name_ar = Column(String(300), nullable=True)
    type = Column(SQLEnum(AccountTypeDB), nullable=False, index=True)
    balance = Column(Numeric(15, 2), nullable=False, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    debit_transactions = relationship("TransactionDB", foreign_keys="TransactionDB.debit_account_id", back_populates="debit_account")
    credit_transactions = relationship("TransactionDB", foreign_keys="TransactionDB.credit_account_id", back_populates="credit_account")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "name_ar": self.name_ar,
            "type": self.type.value if self.type else None,
            "balance": float(self.balance) if self.balance else 0,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TransactionDB(ERPBase):
    """معاملة مالية — مخزنة في قاعدة البيانات"""
    __tablename__ = "erp_transactions"

    id = Column(String, primary_key=True, default=gen_uuid)
    date = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    reference = Column(String(200), default="")
    description = Column(Text, default="")
    
    # Accounting entries
    debit_account_id = Column(String, ForeignKey("erp_accounts.id"), nullable=False)
    credit_account_id = Column(String, ForeignKey("erp_accounts.id"), nullable=False)
    amount = Column(Numeric(15, 2), nullable=False)
    currency = Column(String(3), default="SAR")
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    debit_account = relationship("AccountDB", foreign_keys=[debit_account_id], back_populates="debit_transactions")
    credit_account = relationship("AccountDB", foreign_keys=[credit_account_id], back_populates="credit_transactions")


class InvoiceDB(ERPBase):
    """فاتورة — مخزنة في قاعدة البيانات"""
    __tablename__ = "erp_invoices"

    id = Column(String, primary_key=True, default=gen_uuid)
    invoice_number = Column(String(50), unique=True, nullable=False, index=True)
    customer_id = Column(String(100), ForeignKey("erp_customers.id"), nullable=False, index=True)
    customer_name = Column(String(300), nullable=False)
    customer_email = Column(String(300), nullable=True)
    
    # Amounts
    amount = Column(Numeric(15, 2), nullable=False, default=0)
    tax_amount = Column(Numeric(15, 2), nullable=False, default=0)
    total = Column(Numeric(15, 2), nullable=False, default=0)
    
    # Status
    status = Column(SQLEnum(InvoiceStatusDB), nullable=False, default=InvoiceStatusDB.DRAFT, index=True)
    
    # Items & Notes
    items = Column(Text, default="")  # JSON string or comma-separated
    notes = Column(Text, default="")
    
    # Dates
    due_date = Column(Date, nullable=True)
    paid_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    invoice_items = relationship("InvoiceItemDB", back_populates="invoice")
    customer = relationship("CustomerDB", back_populates="invoices")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "invoice_number": self.invoice_number,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "customer_email": self.customer_email,
            "amount": float(self.amount) if self.amount else 0,
            "tax_amount": float(self.tax_amount) if self.tax_amount else 0,
            "total": float(self.total) if self.total else 0,
            "status": self.status.value if self.status else "draft",
            "items": self.items,
            "notes": self.notes,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class InvoiceItemDB(ERPBase):
    """بند فاتورة — Line items for invoices"""
    __tablename__ = "erp_invoice_items"
    
    id = Column(String, primary_key=True, default=gen_uuid)
    invoice_id = Column(String, ForeignKey("erp_invoices.id"), nullable=False, index=True)
    product_id = Column(String, ForeignKey("erp_products.id"), nullable=True, index=True)
    description = Column(String(500), nullable=False)
    quantity = Column(Integer, nullable=False, default=1)
    unit_price = Column(Numeric(15, 2), nullable=False, default=0)
    total = Column(Numeric(15, 2), nullable=False, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    invoice = relationship("InvoiceDB", back_populates="invoice_items")
    product = relationship("ProductDB")


# ═══════════════════════════════════════════════════════════════
# INVENTORY MODELS - نماذج المخزون
# ═══════════════════════════════════════════════════════════════

class ProductDB(ERPBase):
    """منتج — مخزن في قاعدة البيانات"""
    __tablename__ = "erp_products"

    id = Column(String, primary_key=True, default=gen_uuid)
    sku = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(500), nullable=False)
    description = Column(Text, default="")
    
    # Stock
    quantity = Column(Integer, nullable=False, default=0)
    reorder_point = Column(Integer, default=10)
    
    # Pricing
    unit_price = Column(Numeric(15, 2), default=0)
    cost_price = Column(Numeric(15, 2), default=0)
    
    # Location & Category
    category = Column(String(200), index=True)
    location = Column(String(200), default="")
    supplier = Column(String(300), default="")
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    stock_movements = relationship("StockMovementDB", back_populates="product")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "sku": self.sku,
            "name": self.name,
            "description": self.description,
            "quantity": self.quantity,
            "reorder_point": self.reorder_point,
            "unit_price": float(self.unit_price) if self.unit_price else 0,
            "cost_price": float(self.cost_price) if self.cost_price else 0,
            "category": self.category,
            "location": self.location,
            "supplier": self.supplier,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class StockMovementDB(ERPBase):
    """حركة مخزون — مخزنة في قاعدة البيانات"""
    __tablename__ = "erp_stock_movements"

    id = Column(String, primary_key=True, default=gen_uuid)
    product_id = Column(String, ForeignKey("erp_products.id"), nullable=False, index=True)
    movement_type = Column(String(20), nullable=False)  # in, out, adjustment
    quantity = Column(Integer, nullable=False)
    reason = Column(String(500), default="")
    reference = Column(String(200), default="")  # invoice, PO, etc.
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    product = relationship("ProductDB", back_populates="stock_movements")


# ═══════════════════════════════════════════════════════════════
# HR MODELS - نماذج الموارد البشرية
# ═══════════════════════════════════════════════════════════════

class EmployeeDB(ERPBase):
    """موظف — مخزن في قاعدة البيانات"""
    __tablename__ = "erp_employees"

    id = Column(String, primary_key=True, default=gen_uuid)
    employee_number = Column(String(50), unique=True, nullable=False, index=True)
    
    # Personal Info
    first_name = Column(String(150), nullable=False)
    last_name = Column(String(150), nullable=False)
    email = Column(String(300), unique=True, nullable=True)
    phone = Column(String(50), default="")
    
    # Job Info
    department = Column(String(200), index=True)
    position = Column(String(200))
    salary = Column(Numeric(15, 2), default=0)
    
    # Status
    status = Column(SQLEnum(EmployeeStatusDB), default=EmployeeStatusDB.ACTIVE, index=True)
    hire_date = Column(Date, nullable=True)
    termination_date = Column(Date, nullable=True)
    
    # Metadata
    metadata_json = Column(Text, default="")  # JSON string for extra data
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    payroll_records = relationship("PayrollRecordDB", back_populates="employee")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "employee_number": self.employee_number,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "department": self.department,
            "position": self.position,
            "salary": float(self.salary) if self.salary else 0,
            "status": self.status.value if self.status else None,
            "hire_date": self.hire_date.isoformat() if self.hire_date else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PayrollRecordDB(ERPBase):
    """سجل راتب — مخزن في قاعدة البيانات"""
    __tablename__ = "erp_payroll_records"

    id = Column(String, primary_key=True, default=gen_uuid)
    employee_id = Column(String, ForeignKey("erp_employees.id"), nullable=False, index=True)
    
    # Period
    month = Column(Integer, nullable=False)
    year = Column(Integer, nullable=False)
    
    # Amounts
    base_salary = Column(Numeric(15, 2), nullable=False)
    allowances = Column(Numeric(15, 2), default=0)
    deductions = Column(Numeric(15, 2), default=0)
    net_salary = Column(Numeric(15, 2), nullable=False)
    overtime = Column(Numeric(15, 2), default=0)
    
    # Status
    status = Column(String(20), default="pending")  # pending, paid
    paid_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    employee = relationship("EmployeeDB", back_populates="payroll_records")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "month": self.month,
            "year": self.year,
            "base_salary": float(self.base_salary) if self.base_salary else 0,
            "allowances": float(self.allowances) if self.allowances else 0,
            "deductions": float(self.deductions) if self.deductions else 0,
            "overtime": float(self.overtime) if self.overtime else 0,
            "net_salary": float(self.net_salary) if self.net_salary else 0,
            "status": self.status,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ═══════════════════════════════════════════════════════════════
# CRM MODELS - نماذج إدارة العلاقات مع العملاء
# ═══════════════════════════════════════════════════════════════

class CustomerDB(ERPBase):
    """عميل — مخزن في قاعدة البيانات"""
    __tablename__ = "erp_customers"

    id = Column(String, primary_key=True, default=gen_uuid)
    customer_code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(300), nullable=False)
    email = Column(String(300), nullable=True)
    phone = Column(String(50), default="")
    address = Column(Text, default="")
    
    # Classification
    customer_type = Column(String(50), default="regular")  # regular, vip, wholesale
    credit_limit = Column(Numeric(15, 2), default=0)
    balance = Column(Numeric(15, 2), default=0)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    invoices = relationship("InvoiceDB", back_populates="customer")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "customer_code": self.customer_code,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "customer_type": self.customer_type,
            "credit_limit": float(self.credit_limit) if self.credit_limit else 0,
            "balance": float(self.balance) if self.balance else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SupplierDB(ERPBase):
    """مورد — مخزن في قاعدة البيانات"""
    __tablename__ = "erp_suppliers"

    id = Column(String, primary_key=True, default=gen_uuid)
    supplier_code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(300), nullable=False)
    email = Column(String(300), nullable=True)
    phone = Column(String(50), default="")
    address = Column(Text, default="")
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════
# ALL MODELS EXPORT
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "AccountTypeDB",
    "InvoiceStatusDB",
    "EmployeeStatusDB",
    # Accounting
    "AccountDB",
    "TransactionDB",
    "InvoiceDB",
    "InvoiceItemDB",
    # Inventory
    "ProductDB",
    "StockMovementDB",
    # HR
    "EmployeeDB",
    "PayrollRecordDB",
    # CRM
    "CustomerDB",
    "SupplierDB",
]
