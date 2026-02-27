"""
CRM Module - إدارة علاقات العملاء
Customer relationship management
"""
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any
from enum import Enum

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Base
from sqlalchemy import Column, String, Numeric, DateTime, Text, ForeignKey, Enum as SQLEnum, Date, JSON, Integer, func
from sqlalchemy.orm import relationship


class CustomerType(str, Enum):
    INDIVIDUAL = "individual"
    COMPANY = "company"
    GOVERNMENT = "government"
    NON_PROFIT = "non_profit"


class CustomerStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PROSPECT = "prospect"
    BLACKLISTED = "blacklisted"


class Customer(Base):
    """Customer / العميل"""
    __tablename__ = "erp_customers"
    __table_args__ = {"extend_existing": True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_code = Column(String(50), unique=True, nullable=False, index=True)
    customer_type = Column(SQLEnum(CustomerType), default=CustomerType.COMPANY)
    
    # Company info
    company_name = Column(String(300), nullable=True)
    company_name_ar = Column(String(300), nullable=True)
    tax_number = Column(String(50), nullable=True)
    commercial_registration = Column(String(100), nullable=True)
    
    # Contact info
    contact_name = Column(String(300), nullable=False)
    contact_name_ar = Column(String(300), nullable=True)
    email = Column(String(300), nullable=True)
    phone = Column(String(50), default="")
    mobile = Column(String(50), nullable=True)
    website = Column(String(300), nullable=True)
    
    # Address
    address = Column(Text, nullable=True)
    city = Column(String(100), nullable=True)
    region = Column(String(100), nullable=True)
    country = Column(String(100), default="Saudi Arabia")
    postal_code = Column(String(20), nullable=True)
    
    # Business info
    industry = Column(String(200), nullable=True)
    status = Column(SQLEnum(CustomerStatus), default=CustomerStatus.PROSPECT)
    credit_limit = Column(Numeric(15, 2), default=Decimal("0.00"))
    payment_terms_days = Column(Integer, default=30)
    
    # Marketing
    source = Column(String(200), nullable=True)  # How they found us
    assigned_to = Column(String, nullable=True)  # Sales rep user ID
    tags = Column(JSON, default=list)
    
    # Metadata
    notes = Column(Text, nullable=True)
    custom_fields = Column(JSON, default=dict)
    is_active = Column(String, default="true")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    invoices = relationship("Invoice", back_populates="customer")
    contacts = relationship("CustomerContact", back_populates="customer", cascade="all, delete-orphan")
    activities = relationship("CustomerActivity", back_populates="customer", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_code": self.customer_code,
            "customer_type": self.customer_type.value if self.customer_type else None,
            "company_name": self.company_name,
            "company_name_ar": self.company_name_ar,
            "tax_number": self.tax_number,
            "commercial_registration": self.commercial_registration,
            "contact_name": self.contact_name,
            "contact_name_ar": self.contact_name_ar,
            "email": self.email,
            "phone": self.phone,
            "mobile": self.mobile,
            "website": self.website,
            "address": self.address,
            "city": self.city,
            "region": self.region,
            "country": self.country,
            "postal_code": self.postal_code,
            "industry": self.industry,
            "status": self.status.value if self.status else None,
            "credit_limit": float(self.credit_limit) if self.credit_limit else 0.0,
            "payment_terms_days": self.payment_terms_days or 30,
            "source": self.source,
            "assigned_to": self.assigned_to,
            "tags": self.tags or [],
            "notes": self.notes,
            "custom_fields": self.custom_fields or {},
            "is_active": self.is_active == "true",
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class CustomerContact(Base):
    """Additional contact person for a customer"""
    __tablename__ = "erp_customer_contacts"
    __table_args__ = {"extend_existing": True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String, ForeignKey("erp_customers.id"), nullable=False)
    name = Column(String(300), nullable=False)
    title = Column(String(200), nullable=True)
    email = Column(String(300), nullable=True)
    phone = Column(String(50), nullable=True)
    mobile = Column(String(50), nullable=True)
    is_primary = Column(String, default="false")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    customer = relationship("Customer", back_populates="contacts")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "name": self.name,
            "title": self.title,
            "email": self.email,
            "phone": self.phone,
            "mobile": self.mobile,
            "is_primary": self.is_primary == "true",
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class ActivityType(str, Enum):
    CALL = "call"
    EMAIL = "email"
    MEETING = "meeting"
    NOTE = "note"
    TASK = "task"
    DEAL_WON = "deal_won"
    DEAL_LOST = "deal_lost"


class CustomerActivity(Base):
    """Activity log for customer interactions"""
    __tablename__ = "erp_customer_activities"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String, ForeignKey("erp_customers.id"), nullable=False, index=True)
    activity_type = Column(SQLEnum(ActivityType), nullable=False)
    subject = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    due_date = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    completed_by = Column(String, nullable=True)
    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    customer = relationship("Customer", back_populates="activities")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "activity_type": self.activity_type.value if self.activity_type else None,
            "subject": self.subject,
            "description": self.description,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "completed_by": self.completed_by,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


async def create_customer(
    session: AsyncSession,
    customer_code: str,
    contact_name: str,
    customer_type: str = "company",
    company_name: Optional[str] = None,
    company_name_ar: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    mobile: Optional[str] = None,
    address: Optional[str] = None,
    city: Optional[str] = None,
    tax_number: Optional[str] = None,
    industry: Optional[str] = None,
    credit_limit: float = 0.0,
    payment_terms_days: int = 30,
    source: Optional[str] = None,
    assigned_to: Optional[str] = None,
    notes: Optional[str] = None
) -> Customer:
    """Create a new customer / إنشاء عميل جديد"""
    customer = Customer(
        customer_code=customer_code,
        customer_type=CustomerType(customer_type),
        company_name=company_name,
        company_name_ar=company_name_ar,
        contact_name=contact_name,
        email=email,
        phone=phone or "",
        mobile=mobile,
        address=address,
        city=city,
        tax_number=tax_number,
        industry=industry,
        credit_limit=Decimal(str(credit_limit)),
        payment_terms_days=payment_terms_days,
        source=source,
        assigned_to=assigned_to,
        notes=notes,
        status=CustomerStatus.PROSPECT if source else CustomerStatus.ACTIVE
    )
    session.add(customer)
    await session.flush()
    return customer


async def get_customer(session: AsyncSession, customer_id: str) -> Optional[Customer]:
    """Get customer by ID"""
    result = await session.execute(
        select(Customer).where(Customer.id == customer_id)
    )
    return result.scalar_one_or_none()


async def get_customer_by_code(session: AsyncSession, customer_code: str) -> Optional[Customer]:
    """Get customer by code"""
    result = await session.execute(
        select(Customer).where(Customer.customer_code == customer_code)
    )
    return result.scalar_one_or_none()


async def update_customer(
    session: AsyncSession,
    customer_id: str,
    **kwargs
) -> Optional[Customer]:
    """Update customer fields"""
    customer = await get_customer(session, customer_id)
    if not customer:
        return None
    
    # Convert Enum fields
    if 'customer_type' in kwargs and isinstance(kwargs['customer_type'], str):
        kwargs['customer_type'] = CustomerType(kwargs['customer_type'])
    if 'status' in kwargs and isinstance(kwargs['status'], str):
        kwargs['status'] = CustomerStatus(kwargs['status'])
    
    # Convert Decimal fields
    if 'credit_limit' in kwargs:
        kwargs['credit_limit'] = Decimal(str(kwargs['credit_limit']))
    
    for key, value in kwargs.items():
        if hasattr(customer, key):
            setattr(customer, key, value)
    
    customer.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return customer


async def add_customer_contact(
    session: AsyncSession,
    customer_id: str,
    name: str,
    title: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    mobile: Optional[str] = None,
    is_primary: bool = False,
    notes: Optional[str] = None
) -> Optional[CustomerContact]:
    """Add a contact person to a customer"""
    customer = await get_customer(session, customer_id)
    if not customer:
        return None
    
    # If this is primary, unset others
    if is_primary:
        await session.execute(
            select(CustomerContact).where(
                and_(
                    CustomerContact.customer_id == customer_id,
                    CustomerContact.is_primary == "true"
                )
            )
        )
    
    contact = CustomerContact(
        customer_id=customer_id,
        name=name,
        title=title,
        email=email,
        phone=phone,
        mobile=mobile,
        is_primary="true" if is_primary else "false",
        notes=notes
    )
    session.add(contact)
    await session.flush()
    return contact


async def log_customer_activity(
    session: AsyncSession,
    customer_id: str,
    activity_type: str,
    subject: str,
    description: Optional[str] = None,
    due_date: Optional[datetime] = None,
    created_by: Optional[str] = None
) -> Optional[CustomerActivity]:
    """Log an activity for a customer"""
    customer = await get_customer(session, customer_id)
    if not customer:
        return None
    
    activity = CustomerActivity(
        customer_id=customer_id,
        activity_type=ActivityType(activity_type),
        subject=subject,
        description=description,
        due_date=due_date,
        created_by=created_by
    )
    session.add(activity)
    await session.flush()
    return activity


async def complete_activity(
    session: AsyncSession,
    activity_id: str,
    completed_by: str
) -> Optional[CustomerActivity]:
    """Mark an activity as completed"""
    result = await session.execute(
        select(CustomerActivity).where(CustomerActivity.id == activity_id)
    )
    activity = result.scalar_one_or_none()
    
    if not activity:
        return None
    
    activity.completed_at = datetime.now(timezone.utc)
    activity.completed_by = completed_by
    await session.flush()
    return activity


async def calculate_customer_ltv(
    session: AsyncSession,
    customer_id: str
) -> Dict[str, Any]:
    """
    Calculate Customer Lifetime Value / حساب قيمة العمى مدى الحياة
    
    Returns total revenue, average order value, and estimated LTV.
    """
    # Get all paid invoices
    from erp.invoices import Invoice, InvoiceStatus
    
    result = await session.execute(
        select(
            func.count(Invoice.id).label("invoice_count"),
            func.sum(Invoice.total_amount).label("total_revenue"),
            func.avg(Invoice.total_amount).label("avg_order"),
            func.min(Invoice.created_at).label("first_invoice"),
            func.max(Invoice.created_at).label("last_invoice")
        )
        .where(
            and_(
                Invoice.customer_id == customer_id,
                Invoice.status == InvoiceStatus.PAID
            )
        )
    )
    row = result.one()
    
    total_revenue = float(row.total_revenue or 0)
    invoice_count = row.invoice_count or 0
    avg_order = float(row.avg_order or 0)
    
    # Calculate customer age in months
    if row.first_invoice and row.last_invoice:
        days_as_customer = (row.last_invoice - row.first_invoice).days
        months_as_customer = max(1, days_as_customer / 30)
        monthly_value = total_revenue / months_as_customer
        
        # Estimate 3-year LTV (conservative)
        estimated_ltv = monthly_value * 36
    else:
        months_as_customer = 0
        monthly_value = 0
        estimated_ltv = total_revenue * 3  # Assume 3x for new customers
    
    return {
        "customer_id": customer_id,
        "total_revenue": total_revenue,
        "invoice_count": invoice_count,
        "average_order_value": round(avg_order, 2),
        "months_as_customer": round(months_as_customer, 1),
        "monthly_value": round(monthly_value, 2),
        "estimated_3yr_ltv": round(estimated_ltv, 2)
    }


async def get_customer_outstanding_balance(
    session: AsyncSession,
    customer_id: str
) -> Dict[str, Any]:
    """Get customer's outstanding invoice balance"""
    from erp.invoices import Invoice, InvoiceStatus
    
    result = await session.execute(
        select(
            func.count(Invoice.id).label("invoice_count"),
            func.sum(Invoice.total_amount).label("total_invoiced"),
            func.sum(Invoice.paid_amount).label("total_paid")
        )
        .where(
            and_(
                Invoice.customer_id == customer_id,
                Invoice.status.notin_([InvoiceStatus.CANCELLED, InvoiceStatus.DRAFT])
            )
        )
    )
    row = result.one()
    
    total_invoiced = float(row.total_invoiced or 0)
    total_paid = float(row.total_paid or 0)
    
    return {
        "customer_id": customer_id,
        "invoice_count": row.invoice_count or 0,
        "total_invoiced": total_invoiced,
        "total_paid": total_paid,
        "outstanding_balance": round(total_invoiced - total_paid, 2)
    }


async def get_customers_by_status(
    session: AsyncSession,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get customers grouped or filtered by status"""
    query = select(Customer).where(Customer.is_active == "true")
    
    if status:
        query = query.where(Customer.status == CustomerStatus(status))
    
    result = await session.execute(query.order_by(Customer.company_name))
    customers = result.scalars().all()
    
    return [c.to_dict() for c in customers]


async def search_customers(
    session: AsyncSession,
    search_term: str,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Search customers by name, email, or code"""
    search_pattern = f"%{search_term}%"
    
    result = await session.execute(
        select(Customer).where(
            and_(
                Customer.is_active == "true",
                func.or_(
                    Customer.company_name.ilike(search_pattern),
                    Customer.contact_name.ilike(search_pattern),
                    Customer.customer_code.ilike(search_pattern),
                    Customer.email.ilike(search_pattern)
                )
            )
        ).limit(limit)
    )
    customers = result.scalars().all()
    
    return [c.to_dict() for c in customers]
