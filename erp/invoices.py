"""
Invoices Module - الفواتير
Billing and invoicing management
"""
import uuid
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any
from enum import Enum

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Base
from sqlalchemy import Column, String, Numeric, DateTime, Text, ForeignKey, Enum as SQLEnum, Date, Integer
from sqlalchemy.orm import relationship


class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    VIEWED = "viewed"
    PARTIAL = "partial"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class Invoice(Base):
    """Invoice / الفاتورة"""
    __tablename__ = "erp_invoices"
    __table_args__ = {"extend_existing": True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_number = Column(String(50), unique=True, nullable=False, index=True)
    customer_id = Column(String, ForeignKey("erp_customers.id"), nullable=False, index=True)
    issue_date = Column(Date, default=date.today)
    due_date = Column(Date, nullable=False)
    subtotal = Column(Numeric(15, 2), default=Decimal("0.00"))
    tax_rate = Column(Numeric(5, 2), default=Decimal("15.00"))  # Default 15% VAT
    tax_amount = Column(Numeric(15, 2), default=Decimal("0.00"))
    discount_amount = Column(Numeric(15, 2), default=Decimal("0.00"))
    total_amount = Column(Numeric(15, 2), default=Decimal("0.00"))
    paid_amount = Column(Numeric(15, 2), default=Decimal("0.00"))
    status = Column(SQLEnum(InvoiceStatus), default=InvoiceStatus.DRAFT)
    notes = Column(Text, nullable=True)
    terms = Column(Text, nullable=True)
    reference = Column(String(200), nullable=True)  # Customer PO reference
    created_by = Column(String, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    paid_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    customer = relationship("Customer", back_populates="invoices")
    items = relationship("InvoiceItem", back_populates="invoice", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "invoice_number": self.invoice_number,
            "customer_id": self.customer_id,
            "issue_date": self.issue_date.isoformat() if self.issue_date else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "subtotal": float(self.subtotal) if self.subtotal else 0.0,
            "tax_rate": float(self.tax_rate) if self.tax_rate else 15.0,
            "tax_amount": float(self.tax_amount) if self.tax_amount else 0.0,
            "discount_amount": float(self.discount_amount) if self.discount_amount else 0.0,
            "total_amount": float(self.total_amount) if self.total_amount else 0.0,
            "paid_amount": float(self.paid_amount) if self.paid_amount else 0.0,
            "balance_due": float(self.total_amount - self.paid_amount) if self.total_amount and self.paid_amount else float(self.total_amount or 0),
            "status": self.status.value if self.status else None,
            "notes": self.notes,
            "terms": self.terms,
            "reference": self.reference,
            "created_by": self.created_by,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class InvoiceItem(Base):
    """Invoice line item / بند الفاتورة"""
    __tablename__ = "erp_invoice_items"
    __table_args__ = {"extend_existing": True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_id = Column(String, ForeignKey("erp_invoices.id"), nullable=False, index=True)
    product_id = Column(String, ForeignKey("erp_products.id"), nullable=True)
    description = Column(String(500), nullable=False)
    quantity = Column(Numeric(10, 2), default=Decimal("1.00"))
    unit_price = Column(Numeric(15, 2), default=Decimal("0.00"))
    discount_percent = Column(Numeric(5, 2), default=Decimal("0.00"))
    tax_percent = Column(Numeric(5, 2), default=Decimal("15.00"))
    line_total = Column(Numeric(15, 2), default=Decimal("0.00"))
    sort_order = Column(Integer, default=0)
    
    # Relationships
    invoice = relationship("Invoice", back_populates="items")
    product = relationship("Product")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "invoice_id": self.invoice_id,
            "product_id": self.product_id,
            "description": self.description,
            "quantity": float(self.quantity) if self.quantity else 1.0,
            "unit_price": float(self.unit_price) if self.unit_price else 0.0,
            "discount_percent": float(self.discount_percent) if self.discount_percent else 0.0,
            "tax_percent": float(self.tax_percent) if self.tax_percent else 15.0,
            "line_total": float(self.line_total) if self.line_total else 0.0,
            "sort_order": self.sort_order or 0
        }


async def create_invoice(
    session: AsyncSession,
    invoice_number: str,
    customer_id: str,
    due_date: date,
    items: List[Dict[str, Any]],
    tax_rate: float = 15.0,
    discount_amount: float = 0.0,
    notes: Optional[str] = None,
    terms: Optional[str] = None,
    reference: Optional[str] = None,
    created_by: Optional[str] = None
) -> Invoice:
    """
    Create a new invoice with items / إنشاء فاتورة جديدة
    
    Args:
        items: List of dicts with keys: description, quantity, unit_price, 
               optionally product_id, discount_percent, tax_percent
    """
    # Calculate totals
    subtotal = Decimal("0.00")
    invoice_items = []
    
    for idx, item_data in enumerate(items):
        quantity = Decimal(str(item_data.get("quantity", 1)))
        unit_price = Decimal(str(item_data.get("unit_price", 0)))
        discount_pct = Decimal(str(item_data.get("discount_percent", 0)))
        item_tax_rate = Decimal(str(item_data.get("tax_percent", tax_rate)))
        
        # Calculate line total
        line_subtotal = quantity * unit_price
        line_discount = line_subtotal * (discount_pct / 100)
        line_taxable = line_subtotal - line_discount
        line_tax = line_taxable * (item_tax_rate / 100)
        line_total = line_taxable + line_tax
        
        invoice_item = InvoiceItem(
            product_id=item_data.get("product_id"),
            description=item_data["description"],
            quantity=quantity,
            unit_price=unit_price,
            discount_percent=discount_pct,
            tax_percent=item_tax_rate,
            line_total=line_total,
            sort_order=idx
        )
        invoice_items.append(invoice_item)
        subtotal += line_taxable
    
    # Calculate invoice totals
    decimal_discount = Decimal(str(discount_amount))
    taxable_after_discount = subtotal - decimal_discount
    decimal_tax_rate = Decimal(str(tax_rate))
    tax_amount = taxable_after_discount * (decimal_tax_rate / 100)
    total_amount = taxable_after_discount + tax_amount
    
    # Create invoice
    invoice = Invoice(
        invoice_number=invoice_number,
        customer_id=customer_id,
        due_date=due_date,
        subtotal=subtotal,
        tax_rate=decimal_tax_rate,
        tax_amount=tax_amount,
        discount_amount=decimal_discount,
        total_amount=total_amount,
        paid_amount=Decimal("0.00"),
        status=InvoiceStatus.DRAFT,
        notes=notes,
        terms=terms,
        reference=reference,
        created_by=created_by
    )
    session.add(invoice)
    await session.flush()  # Get invoice ID
    
    # Add items with invoice ID
    for item in invoice_items:
        item.invoice_id = invoice.id
        session.add(item)
    
    await session.flush()
    return invoice


async def get_invoice(session: AsyncSession, invoice_id: str) -> Optional[Invoice]:
    """Get invoice by ID with items"""
    result = await session.execute(
        select(Invoice).where(Invoice.id == invoice_id)
    )
    return result.scalar_one_or_none()


async def get_invoice_by_number(session: AsyncSession, invoice_number: str) -> Optional[Invoice]:
    """Get invoice by invoice number"""
    result = await session.execute(
        select(Invoice).where(Invoice.invoice_number == invoice_number)
    )
    return result.scalar_one_or_none()


async def get_invoice_with_items(session: AsyncSession, invoice_id: str) -> Optional[Dict[str, Any]]:
    """Get invoice with all items loaded"""
    invoice = await get_invoice(session, invoice_id)
    if not invoice:
        return None
    
    # Load items
    result = await session.execute(
        select(InvoiceItem)
        .where(InvoiceItem.invoice_id == invoice_id)
        .order_by(InvoiceItem.sort_order)
    )
    items = result.scalars().all()
    
    invoice_dict = invoice.to_dict()
    invoice_dict["items"] = [item.to_dict() for item in items]
    return invoice_dict


async def add_invoice_item(
    session: AsyncSession,
    invoice_id: str,
    description: str,
    quantity: float,
    unit_price: float,
    product_id: Optional[str] = None,
    discount_percent: float = 0.0,
    tax_percent: float = 15.0
) -> Optional[InvoiceItem]:
    """Add an item to an existing invoice"""
    invoice = await get_invoice(session, invoice_id)
    if not invoice or invoice.status != InvoiceStatus.DRAFT:
        return None
    
    quantity_dec = Decimal(str(quantity))
    unit_price_dec = Decimal(str(unit_price))
    discount_dec = Decimal(str(discount_percent))
    tax_dec = Decimal(str(tax_percent))
    
    # Calculate line total
    line_subtotal = quantity_dec * unit_price_dec
    line_discount = line_subtotal * (discount_dec / 100)
    line_taxable = line_subtotal - line_discount
    line_tax = line_taxable * (tax_dec / 100)
    line_total = line_taxable + line_tax
    
    # Get next sort order
    result = await session.execute(
        select(func.max(InvoiceItem.sort_order)).where(InvoiceItem.invoice_id == invoice_id)
    )
    max_order = result.scalar() or 0
    
    item = InvoiceItem(
        invoice_id=invoice_id,
        product_id=product_id,
        description=description,
        quantity=quantity_dec,
        unit_price=unit_price_dec,
        discount_percent=discount_dec,
        tax_percent=tax_dec,
        line_total=line_total,
        sort_order=max_order + 1
    )
    session.add(item)
    
    # Recalculate invoice totals
    await _recalculate_invoice_totals(session, invoice)
    
    await session.flush()
    return item


async def _recalculate_invoice_totals(session: AsyncSession, invoice: Invoice):
    """Recalculate invoice totals after item changes"""
    result = await session.execute(
        select(InvoiceItem).where(InvoiceItem.invoice_id == invoice.id)
    )
    items = result.scalars().all()
    
    subtotal = Decimal("0.00")
    for item in items:
        line_subtotal = item.quantity * item.unit_price
        line_discount = line_subtotal * (item.discount_percent / 100)
        subtotal += (line_subtotal - line_discount)
    
    taxable = subtotal - invoice.discount_amount
    invoice.tax_amount = taxable * (invoice.tax_rate / 100)
    invoice.subtotal = subtotal
    invoice.total_amount = taxable + invoice.tax_amount
    invoice.updated_at = datetime.now(timezone.utc)


async def mark_invoice_sent(session: AsyncSession, invoice_id: str) -> Optional[Invoice]:
    """Mark invoice as sent / تحديد الفاتورة كمرسلة"""
    invoice = await get_invoice(session, invoice_id)
    if not invoice or invoice.status != InvoiceStatus.DRAFT:
        return None
    
    invoice.status = InvoiceStatus.SENT
    invoice.sent_at = datetime.now(timezone.utc)
    invoice.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return invoice


async def record_invoice_payment(
    session: AsyncSession,
    invoice_id: str,
    amount: float,
    payment_method: str = "transfer",
    reference: Optional[str] = None,
    paid_by: Optional[str] = None
) -> Optional[Invoice]:
    """
    Record a payment on an invoice / تسجيل دفع الفاتورة
    
    Updates paid amount and status based on payment.
    """
    invoice = await get_invoice(session, invoice_id)
    if not invoice:
        return None
    
    if invoice.status in [InvoiceStatus.PAID, InvoiceStatus.CANCELLED]:
        return None
    
    decimal_amount = Decimal(str(amount))
    invoice.paid_amount = (invoice.paid_amount or Decimal("0.00")) + decimal_amount
    
    # Update status based on payment
    if invoice.paid_amount >= invoice.total_amount:
        invoice.status = InvoiceStatus.PAID
        invoice.paid_at = datetime.now(timezone.utc)
    elif invoice.paid_amount > 0:
        invoice.status = InvoiceStatus.PARTIAL
    
    invoice.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return invoice


async def mark_invoice_paid(
    session: AsyncSession,
    invoice_id: str,
    paid_by: Optional[str] = None
) -> Optional[Invoice]:
    """Mark invoice as fully paid / تحديد الفاتورة كمدفوعة"""
    invoice = await get_invoice(session, invoice_id)
    if not invoice or invoice.status == InvoiceStatus.CANCELLED:
        return None
    
    invoice.paid_amount = invoice.total_amount
    invoice.status = InvoiceStatus.PAID
    invoice.paid_at = datetime.now(timezone.utc)
    invoice.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return invoice


async def cancel_invoice(session: AsyncSession, invoice_id: str) -> Optional[Invoice]:
    """Cancel an invoice / إلغاء الفاتورة"""
    invoice = await get_invoice(session, invoice_id)
    if not invoice or invoice.status == InvoiceStatus.PAID:
        return None
    
    invoice.status = InvoiceStatus.CANCELLED
    invoice.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return invoice


async def get_overdue_invoices(
    session: AsyncSession,
    as_of_date: Optional[date] = None
) -> List[Dict[str, Any]]:
    """Get all overdue invoices / الفواتير المتأخرة"""
    as_of = as_of_date or date.today()
    
    result = await session.execute(
        select(Invoice, func.coalesce(func.sum(InvoiceItem.line_total), 0))
        .outerjoin(InvoiceItem)
        .where(
            and_(
                Invoice.due_date < as_of,
                Invoice.status.notin_([InvoiceStatus.PAID, InvoiceStatus.CANCELLED])
            )
        )
        .group_by(Invoice.id)
        .order_by(Invoice.due_date)
    )
    
    overdue = []
    for invoice, _ in result.all():
        days_overdue = (as_of - invoice.due_date).days
        balance = invoice.total_amount - (invoice.paid_amount or Decimal("0.00"))
        
        overdue.append({
            "invoice_id": invoice.id,
            "invoice_number": invoice.invoice_number,
            "customer_id": invoice.customer_id,
            "due_date": invoice.due_date.isoformat(),
            "days_overdue": days_overdue,
            "total_amount": float(invoice.total_amount),
            "balance_due": float(balance),
            "status": invoice.status.value
        })
    
    return overdue


async def get_invoice_summary_by_status(
    session: AsyncSession,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> List[Dict[str, Any]]:
    """Get invoice totals grouped by status"""
    query = select(
        Invoice.status,
        func.count(Invoice.id).label("count"),
        func.sum(Invoice.total_amount).label("total"),
        func.sum(Invoice.paid_amount).label("paid")
    )
    
    if start_date:
        query = query.where(Invoice.issue_date >= start_date)
    if end_date:
        query = query.where(Invoice.issue_date <= end_date)
    
    result = await session.execute(query.group_by(Invoice.status))
    
    return [
        {
            "status": row.status.value,
            "count": row.count,
            "total_amount": float(row.total or 0),
            "paid_amount": float(row.paid or 0),
            "balance": float((row.total or 0) - (row.paid or 0))
        }
        for row in result.all()
    ]
