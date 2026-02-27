"""
Accounts Payable & Receivable - الذمم المدينة والدائنة

وحدة إدارة المدفوعات والمقبوضات مع:
- Aging Reports (تقرير تقادم الديون)
- Payment Tracking (تتبع المدفوعات)
- Vendor/Customer management
"""

import uuid
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class PayableStatus(Enum):
    """حالات المدفوعات"""
    DRAFT = "draft"           # مسودة
    PENDING = "pending"       # معلقة
    APPROVED = "approved"     # معتمدة
    PARTIAL = "partial"       # مدفوعة جزئياً
    PAID = "paid"             # مدفوعة بالكامل
    OVERDUE = "overdue"       # متأخرة
    CANCELLED = "cancelled"   # ملغاة


class ReceivableStatus(Enum):
    """حالات المقبوضات"""
    INVOICED = "invoiced"     # مفوترة
    SENT = "sent"             # مرسلة
    PARTIAL = "partial"       # مستلمة جزئياً
    PAID = "paid"             # مستلمة بالكامل
    OVERDUE = "overdue"       # متأخرة
    WRITTEN_OFF = "written_off"  # معدومة


@dataclass
class Payable:
    """فاتورة مستحقة الدفع (مدفوعات)"""
    id: str
    vendor_id: str            # معرف المورد
    vendor_name: str          # اسم المورد
    invoice_number: str       # رقم الفاتورة
    amount: Decimal           # المبلغ
    currency: str = "SAR"
    description: str = ""
    status: PayableStatus = PayableStatus.PENDING
    issue_date: date = field(default_factory=date.today)
    due_date: date = field(default_factory=lambda: date.today() + timedelta(days=30))
    paid_amount: Decimal = field(default_factory=lambda: Decimal('0'))
    paid_date: Optional[date] = None
    payment_terms: int = 30   # Net days
    purchase_order_id: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def balance(self) -> Decimal:
        """المبلغ المتبقي"""
        return self.amount - self.paid_amount
    
    @property
    def is_overdue(self) -> bool:
        """هل الفاتورة متأخرة؟"""
        return date.today() > self.due_date and self.status != PayableStatus.PAID
    
    @property
    def days_overdue(self) -> int:
        """عدد أيام التأخير"""
        if not self.is_overdue:
            return 0
        return (date.today() - self.due_date).days
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "vendor_id": self.vendor_id,
            "vendor_name": self.vendor_name,
            "invoice_number": self.invoice_number,
            "amount": float(self.amount),
            "currency": self.currency,
            "description": self.description,
            "status": self.status.value,
            "issue_date": self.issue_date.isoformat(),
            "due_date": self.due_date.isoformat(),
            "paid_amount": float(self.paid_amount),
            "balance": float(self.balance),
            "paid_date": self.paid_date.isoformat() if self.paid_date else None,
            "payment_terms": self.payment_terms,
            "is_overdue": self.is_overdue,
            "days_overdue": self.days_overdue,
            "purchase_order_id": self.purchase_order_id,
            "notes": self.notes,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Receivable:
    """فاتورة مستحقة القبض (مقبوضات)"""
    id: str
    customer_id: str          # معرف العميل
    customer_name: str        # اسم العميل
    invoice_number: str       # رقم الفاتورة
    amount: Decimal           # المبلغ
    currency: str = "SAR"
    description: str = ""
    status: ReceivableStatus = ReceivableStatus.INVOICED
    issue_date: date = field(default_factory=date.today)
    due_date: date = field(default_factory=lambda: date.today() + timedelta(days=30))
    received_amount: Decimal = field(default_factory=lambda: Decimal('0'))
    received_date: Optional[date] = None
    payment_terms: int = 30
    sales_order_id: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def balance(self) -> Decimal:
        """المبلغ المتبقي"""
        return self.amount - self.received_amount
    
    @property
    def is_overdue(self) -> bool:
        """هل الفاتورة متأخرة؟"""
        return date.today() > self.due_date and self.status != ReceivableStatus.PAID
    
    @property
    def days_overdue(self) -> int:
        """عدد أيام التأخير"""
        if not self.is_overdue:
            return 0
        return (date.today() - self.due_date).days
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "invoice_number": self.invoice_number,
            "amount": float(self.amount),
            "currency": self.currency,
            "description": self.description,
            "status": self.status.value,
            "issue_date": self.issue_date.isoformat(),
            "due_date": self.due_date.isoformat(),
            "received_amount": float(self.received_amount),
            "balance": float(self.balance),
            "received_date": self.received_date.isoformat() if self.received_date else None,
            "payment_terms": self.payment_terms,
            "is_overdue": self.is_overdue,
            "days_overdue": self.days_overdue,
            "sales_order_id": self.sales_order_id,
            "notes": self.notes,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Payment:
    """سجل دفعة"""
    id: str
    payable_id: Optional[str] = None
    receivable_id: Optional[str] = None
    amount: Decimal = field(default_factory=lambda: Decimal('0'))
    payment_date: date = field(default_factory=date.today)
    payment_method: str = "bank_transfer"  # cash, check, bank_transfer, card
    reference_number: str = ""
    notes: str = ""
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "payable_id": self.payable_id,
            "receivable_id": self.receivable_id,
            "amount": float(self.amount),
            "payment_date": self.payment_date.isoformat(),
            "payment_method": self.payment_method,
            "reference_number": self.reference_number,
            "notes": self.notes,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat()
        }


class AgingReport:
    """
    تقرير تقادم الديون (Aging Report)
    تصنيف الديون حسب الفترات الزمنية
    """
    
    BUCKETS = [
        (0, 30, "current"),
        (31, 60, "1-30_days"),
        (61, 90, "31-60_days"),
        (91, 120, "61-90_days"),
        (121, float('inf'), "over_90_days")
    ]
    
    def __init__(self):
        self.buckets: Dict[str, Decimal] = {
            "current": Decimal('0'),
            "1-30_days": Decimal('0'),
            "31-60_days": Decimal('0'),
            "61-90_days": Decimal('0'),
            "over_90_days": Decimal('0')
        }
        self.details: List[Dict] = []
    
    def add_amount(self, days_overdue: int, amount: Decimal, 
                   reference: str = "", details: Dict = None):
        """إضافة مبلغ للتصنيف المناسب"""
        for min_days, max_days, bucket in self.BUCKETS:
            if min_days <= days_overdue <= max_days:
                self.buckets[bucket] += amount
                self.details.append({
                    "bucket": bucket,
                    "days_overdue": days_overdue,
                    "amount": float(amount),
                    "reference": reference,
                    **(details or {})
                })
                break
    
    def to_dict(self) -> Dict[str, Any]:
        total = sum(self.buckets.values())
        return {
            "buckets": {
                k: float(v) for k, v in self.buckets.items()
            },
            "total": float(total),
            "percentages": {
                k: float(v / total * 100) if total > 0 else 0
                for k, v in self.buckets.items()
            },
            "details": self.details
        }


class AccountsPayable:
    """
    إدارة المدفوعات (Accounts Payable)
    """
    
    def __init__(self):
        self.payables: Dict[str, Payable] = {}
        self.payments: Dict[str, Payment] = {}
        self.vendors: Dict[str, Dict] = {}
    
    def add_vendor(self, vendor_id: str, name: str, 
                   contact_info: Dict = None, payment_terms: int = 30):
        """إضافة مورد جديد"""
        self.vendors[vendor_id] = {
            "id": vendor_id,
            "name": name,
            "contact_info": contact_info or {},
            "payment_terms": payment_terms,
            "total_payables": Decimal('0'),
            "total_paid": Decimal('0'),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    
    def create_payable(self, vendor_id: str, vendor_name: str,
                      invoice_number: str, amount: Decimal,
                      description: str = "", due_date: Optional[date] = None,
                      purchase_order_id: Optional[str] = None) -> Payable:
        """إنشاء فاتورة مستحقة الدفع"""
        if due_date is None:
            terms = self.vendors.get(vendor_id, {}).get("payment_terms", 30)
            due_date = date.today() + timedelta(days=terms)
        
        payable = Payable(
            id=str(uuid.uuid4()),
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            invoice_number=invoice_number,
            amount=Decimal(str(amount)),
            description=description,
            due_date=due_date,
            purchase_order_id=purchase_order_id
        )
        
        self.payables[payable.id] = payable
        
        # Update vendor stats
        if vendor_id in self.vendors:
            self.vendors[vendor_id]["total_payables"] += Decimal(str(amount))
        
        return payable
    
    def record_payment(self, payable_id: str, amount: Decimal,
                      payment_method: str = "bank_transfer",
                      reference_number: str = "",
                      notes: str = "") -> Payment:
        """تسجيل دفعة"""
        payable = self.payables.get(payable_id)
        if not payable:
            raise ValueError(f"Payable {payable_id} not found")
        
        payment = Payment(
            id=str(uuid.uuid4()),
            payable_id=payable_id,
            amount=Decimal(str(amount)),
            payment_method=payment_method,
            reference_number=reference_number,
            notes=notes
        )
        
        self.payments[payment.id] = payment
        
        # Update payable
        payable.paid_amount += Decimal(str(amount))
        payable.paid_date = date.today()
        
        if payable.paid_amount >= payable.amount:
            payable.status = PayableStatus.PAID
        elif payable.paid_amount > 0:
            payable.status = PayableStatus.PARTIAL
        
        # Update vendor stats
        if payable.vendor_id in self.vendors:
            self.vendors[payable.vendor_id]["total_paid"] += Decimal(str(amount))
        
        return payment
    
    def get_overdue_payables(self) -> List[Payable]:
        """الحصول على الفواتير المتأخرة"""
        return [p for p in self.payables.values() if p.is_overdue]
    
    def get_aging_report(self) -> AgingReport:
        """تقرير تقادم الديون للمدفوعات"""
        report = AgingReport()
        
        for payable in self.payables.values():
            if payable.status != PayableStatus.PAID and payable.balance > 0:
                report.add_amount(
                    payable.days_overdue,
                    payable.balance,
                    payable.invoice_number,
                    {
                        "vendor_name": payable.vendor_name,
                        "due_date": payable.due_date.isoformat(),
                        "payable_id": payable.id
                    }
                )
        
        return report
    
    def get_vendor_summary(self, vendor_id: str) -> Dict[str, Any]:
        """ملخص المورد"""
        vendor = self.vendors.get(vendor_id, {})
        vendor_payables = [p for p in self.payables.values() if p.vendor_id == vendor_id]
        
        return {
            "vendor": vendor,
            "total_payables": float(vendor.get("total_payables", 0)),
            "total_paid": float(vendor.get("total_paid", 0)),
            "outstanding": float(vendor.get("total_payables", 0) - vendor.get("total_paid", 0)),
            "invoices_count": len(vendor_payables),
            "overdue_count": len([p for p in vendor_payables if p.is_overdue])
        }
    
    def get_payables_summary(self) -> Dict[str, Any]:
        """ملخص المدفوعات"""
        total = sum(p.amount for p in self.payables.values())
        paid = sum(p.paid_amount for p in self.payables.values())
        outstanding = total - paid
        overdue = sum(p.balance for p in self.payables.values() if p.is_overdue)
        
        return {
            "total_payables": float(total),
            "total_paid": float(paid),
            "total_outstanding": float(outstanding),
            "total_overdue": float(overdue),
            "payables_count": len(self.payables),
            "overdue_count": len([p for p in self.payables.values() if p.is_overdue]),
            "vendors_count": len(self.vendors)
        }


class AccountsReceivable:
    """
    إدارة المقبوضات (Accounts Receivable)
    """
    
    def __init__(self):
        self.receivables: Dict[str, Receivable] = {}
        self.receipts: Dict[str, Payment] = {}
        self.customers: Dict[str, Dict] = {}
    
    def add_customer(self, customer_id: str, name: str,
                     contact_info: Dict = None, credit_limit: Decimal = None,
                     payment_terms: int = 30):
        """إضافة عميل جديد"""
        self.customers[customer_id] = {
            "id": customer_id,
            "name": name,
            "contact_info": contact_info or {},
            "credit_limit": float(credit_limit) if credit_limit else None,
            "payment_terms": payment_terms,
            "total_receivables": Decimal('0'),
            "total_received": Decimal('0'),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    
    def create_receivable(self, customer_id: str, customer_name: str,
                         invoice_number: str, amount: Decimal,
                         description: str = "", due_date: Optional[date] = None,
                         sales_order_id: Optional[str] = None) -> Receivable:
        """إنشاء فاتورة مستحقة القبض"""
        if due_date is None:
            terms = self.customers.get(customer_id, {}).get("payment_terms", 30)
            due_date = date.today() + timedelta(days=terms)
        
        receivable = Receivable(
            id=str(uuid.uuid4()),
            customer_id=customer_id,
            customer_name=customer_name,
            invoice_number=invoice_number,
            amount=Decimal(str(amount)),
            description=description,
            due_date=due_date,
            sales_order_id=sales_order_id
        )
        
        self.receivables[receivable.id] = receivable
        
        # Update customer stats
        if customer_id in self.customers:
            self.customers[customer_id]["total_receivables"] += Decimal(str(amount))
        
        return receivable
    
    def record_receipt(self, receivable_id: str, amount: Decimal,
                      payment_method: str = "bank_transfer",
                      reference_number: str = "",
                      notes: str = "") -> Payment:
        """تسجيل مقبوض"""
        receivable = self.receivables.get(receivable_id)
        if not receivable:
            raise ValueError(f"Receivable {receivable_id} not found")
        
        receipt = Payment(
            id=str(uuid.uuid4()),
            receivable_id=receivable_id,
            amount=Decimal(str(amount)),
            payment_method=payment_method,
            reference_number=reference_number,
            notes=notes
        )
        
        self.receipts[receipt.id] = receipt
        
        # Update receivable
        receivable.received_amount += Decimal(str(amount))
        receivable.received_date = date.today()
        
        if receivable.received_amount >= receivable.amount:
            receivable.status = ReceivableStatus.PAID
        elif receivable.received_amount > 0:
            receivable.status = ReceivableStatus.PARTIAL
        
        # Update customer stats
        if receivable.customer_id in self.customers:
            self.customers[receivable.customer_id]["total_received"] += Decimal(str(amount))
        
        return receipt
    
    def write_off(self, receivable_id: str, reason: str = ""):
        """إعدام دين"""
        receivable = self.receivables.get(receivable_id)
        if receivable:
            receivable.status = ReceivableStatus.WRITTEN_OFF
            receivable.notes += f" | Written off: {reason}"
    
    def get_overdue_receivables(self) -> List[Receivable]:
        """الحصول على الفواتير المتأخرة"""
        return [r for r in self.receivables.values() if r.is_overdue]
    
    def get_aging_report(self) -> AgingReport:
        """تقرير تقادم الديون للمقبوضات"""
        report = AgingReport()
        
        for receivable in self.receivables.values():
            if receivable.status not in [ReceivableStatus.PAID, ReceivableStatus.WRITTEN_OFF]:
                if receivable.balance > 0:
                    report.add_amount(
                        receivable.days_overdue,
                        receivable.balance,
                        receivable.invoice_number,
                        {
                            "customer_name": receivable.customer_name,
                            "due_date": receivable.due_date.isoformat(),
                            "receivable_id": receivable.id
                        }
                    )
        
        return report
    
    def get_customer_summary(self, customer_id: str) -> Dict[str, Any]:
        """ملخص العميل"""
        customer = self.customers.get(customer_id, {})
        customer_receivables = [r for r in self.receivables.values() if r.customer_id == customer_id]
        
        total_receivables = sum(r.amount for r in customer_receivables)
        total_received = sum(r.received_amount for r in customer_receivables)
        
        return {
            "customer": customer,
            "total_receivables": float(total_receivables),
            "total_received": float(total_received),
            "outstanding": float(total_receivables - total_received),
            "invoices_count": len(customer_receivables),
            "overdue_count": len([r for r in customer_receivables if r.is_overdue]),
            "credit_utilization": (
                float((total_receivables - total_received) / Decimal(str(customer.get("credit_limit", 1))))
                if customer.get("credit_limit") else None
            )
        }
    
    def get_receivables_summary(self) -> Dict[str, Any]:
        """ملخص المقبوضات"""
        total = sum(r.amount for r in self.receivables.values())
        received = sum(r.received_amount for r in self.receivables.values())
        outstanding = total - received
        overdue = sum(r.balance for r in self.receivables.values() if r.is_overdue)
        
        return {
            "total_receivables": float(total),
            "total_received": float(received),
            "total_outstanding": float(outstanding),
            "total_overdue": float(overdue),
            "receivables_count": len(self.receivables),
            "overdue_count": len([r for r in self.receivables.values() if r.is_overdue]),
            "customers_count": len(self.customers),
            "collection_rate": float(received / total * 100) if total > 0 else 0
        }
