"""
Supplier Management - إدارة الموردين

إدارة الموردين مع:
- معلومات الاتصال وشروط الدفع
- تقييم الأداء
- تاريخ الطلبات
"""

import uuid
from datetime import datetime, date, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class SupplierRating(Enum):
    """تقييم المورد"""
    EXCELLENT = 5      # ممتاز
    GOOD = 4           # جيد
    AVERAGE = 3        # متوسط
    POOR = 2           # ضعيف
    UNACCEPTABLE = 1   # غير مقبول


@dataclass
class SupplierContact:
    """جهة اتصال المورد"""
    name: str
    email: str = ""
    phone: str = ""
    position: str = ""
    is_primary: bool = False


@dataclass
class Supplier:
    """مورد"""
    id: str
    supplier_code: str            # كود المورد
    name: str                     # اسم المورد
    legal_name: str = ""          # الاسم القانوني
    
    # Contact Information
    contacts: List[SupplierContact] = field(default_factory=list)
    address: str = ""
    city: str = ""
    country: str = "Saudi Arabia"
    postal_code: str = ""
    email: str = ""
    phone: str = ""
    website: str = ""
    
    # Business Information
    tax_number: str = ""          # الرقم الضريبي
    commercial_registration: str = ""  # السجل التجاري
    bank_account: str = ""
    bank_name: str = ""
    
    # Terms
    payment_terms: int = 30       # شروط الدفع (أيام)
    credit_limit: Decimal = field(default_factory=lambda: Decimal('0'))
    currency: str = "SAR"
    minimum_order_amount: Decimal = field(default_factory=lambda: Decimal('0'))
    lead_time_days: int = 7       # مدة التوريد
    
    # Performance
    rating: SupplierRating = SupplierRating.AVERAGE
    on_time_delivery_rate: float = 0.0  # نسبة التسليم في الموعد
    quality_score: float = 0.0    # درجة الجودة
    
    # Status
    is_active: bool = True
    is_approved: bool = False     # معتمد للتعامل
    
    # Notes
    notes: str = ""
    categories: List[str] = field(default_factory=list)  # فئات المنتجات الموردة
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_primary_contact(self) -> Optional[SupplierContact]:
        """الحصول على جهة الاتصال الرئيسية"""
        for contact in self.contacts:
            if contact.is_primary:
                return contact
        return self.contacts[0] if self.contacts else None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "supplier_code": self.supplier_code,
            "name": self.name,
            "legal_name": self.legal_name,
            "contacts": [
                {
                    "name": c.name,
                    "email": c.email,
                    "phone": c.phone,
                    "position": c.position,
                    "is_primary": c.is_primary
                } for c in self.contacts
            ],
            "address": self.address,
            "city": self.city,
            "country": self.country,
            "email": self.email,
            "phone": self.phone,
            "website": self.website,
            "tax_number": self.tax_number,
            "commercial_registration": self.commercial_registration,
            "payment_terms": self.payment_terms,
            "credit_limit": float(self.credit_limit),
            "currency": self.currency,
            "minimum_order_amount": float(self.minimum_order_amount),
            "lead_time_days": self.lead_time_days,
            "rating": self.rating.value,
            "on_time_delivery_rate": self.on_time_delivery_rate,
            "quality_score": self.quality_score,
            "is_active": self.is_active,
            "is_approved": self.is_approved,
            "categories": self.categories,
            "notes": self.notes,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class SupplierPerformance:
    """أداء المورد"""
    supplier_id: str
    period_start: date
    period_end: date
    
    # Metrics
    total_orders: int = 0
    total_value: Decimal = field(default_factory=lambda: Decimal('0'))
    on_time_deliveries: int = 0
    late_deliveries: int = 0
    rejected_items: int = 0
    accepted_items: int = 0
    
    # Calculated
    @property
    def on_time_rate(self) -> float:
        """نسبة التسليم في الموعد"""
        total = self.on_time_deliveries + self.late_deliveries
        if total == 0:
            return 0
        return (self.on_time_deliveries / total) * 100
    
    @property
    def quality_rate(self) -> float:
        """نسبة الجودة"""
        total = self.accepted_items + self.rejected_items
        if total == 0:
            return 0
        return (self.accepted_items / total) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "supplier_id": self.supplier_id,
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat()
            },
            "total_orders": self.total_orders,
            "total_value": float(self.total_value),
            "on_time_deliveries": self.on_time_deliveries,
            "late_deliveries": self.late_deliveries,
            "on_time_rate": self.on_time_rate,
            "quality_rate": self.quality_rate,
            "rejected_items": self.rejected_items,
            "accepted_items": self.accepted_items
        }


@dataclass
class SupplierOrderHistory:
    """سجل طلبات المورد"""
    order_id: str
    po_number: str
    order_date: date
    expected_delivery: date
    actual_delivery: Optional[date]
    total_amount: Decimal
    status: str
    items: List[Dict] = field(default_factory=list)
    
    @property
    def was_on_time(self) -> bool:
        """هل تم التسليم في الموعد؟"""
        if not self.actual_delivery:
            return False
        return self.actual_delivery <= self.expected_delivery
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "po_number": self.po_number,
            "order_date": self.order_date.isoformat(),
            "expected_delivery": self.expected_delivery.isoformat(),
            "actual_delivery": self.actual_delivery.isoformat() if self.actual_delivery else None,
            "was_on_time": self.was_on_time,
            "total_amount": float(self.total_amount),
            "status": self.status,
            "items": self.items
        }


class SupplierManager:
    """
    مدير الموردين
    """
    
    def __init__(self):
        self.suppliers: Dict[str, Supplier] = {}
        self.performance_records: List[SupplierPerformance] = []
        self.order_history: Dict[str, List[SupplierOrderHistory]] = {}  # supplier_id -> orders
        self._supplier_counter = 0
    
    def create_supplier(self, name: str, supplier_code: str = None,
                       email: str = "", phone: str = "", address: str = "",
                       categories: List[str] = None,
                       payment_terms: int = 30,
                       **kwargs) -> Supplier:
        """إنشاء مورد جديد"""
        if not supplier_code:
            self._supplier_counter += 1
            supplier_code = f"SUP-{self._supplier_counter:04d}"
        
        supplier = Supplier(
            id=str(uuid.uuid4()),
            supplier_code=supplier_code,
            name=name,
            email=email,
            phone=phone,
            address=address,
            categories=categories or [],
            payment_terms=payment_terms,
            **kwargs
        )
        
        self.suppliers[supplier.id] = supplier
        return supplier
    
    def get_supplier(self, supplier_id: str) -> Optional[Supplier]:
        """الحصول على مورد"""
        return self.suppliers.get(supplier_id)
    
    def get_supplier_by_code(self, code: str) -> Optional[Supplier]:
        """الحصول على مورد بالكود"""
        for supplier in self.suppliers.values():
            if supplier.supplier_code == code:
                return supplier
        return None
    
    def update_supplier(self, supplier_id: str, **kwargs) -> Supplier:
        """تحديث بيانات مورد"""
        supplier = self.suppliers.get(supplier_id)
        if not supplier:
            raise ValueError(f"Supplier {supplier_id} not found")
        
        for key, value in kwargs.items():
            if hasattr(supplier, key):
                setattr(supplier, key, value)
        
        supplier.updated_at = datetime.now(timezone.utc)
        return supplier
    
    def add_contact(self, supplier_id: str, name: str, email: str = "",
                   phone: str = "", position: str = "",
                   is_primary: bool = False) -> SupplierContact:
        """إضافة جهة اتصال للمورد"""
        supplier = self.suppliers.get(supplier_id)
        if not supplier:
            raise ValueError(f"Supplier {supplier_id} not found")
        
        contact = SupplierContact(
            name=name,
            email=email,
            phone=phone,
            position=position,
            is_primary=is_primary
        )
        
        # If this is primary, unset others
        if is_primary:
            for c in supplier.contacts:
                c.is_primary = False
        
        supplier.contacts.append(contact)
        supplier.updated_at = datetime.now(timezone.utc)
        return contact
    
    def rate_supplier(self, supplier_id: str, rating: SupplierRating,
                     on_time_rate: float = None, quality_score: float = None):
        """تقييم مورد"""
        supplier = self.suppliers.get(supplier_id)
        if not supplier:
            raise ValueError(f"Supplier {supplier_id} not found")
        
        supplier.rating = rating
        if on_time_rate is not None:
            supplier.on_time_delivery_rate = on_time_rate
        if quality_score is not None:
            supplier.quality_score = quality_score
        
        supplier.updated_at = datetime.now(timezone.utc)
    
    def record_order(self, supplier_id: str, order_id: str, po_number: str,
                    order_date: date, expected_delivery: date,
                    total_amount: Decimal, status: str,
                    items: List[Dict] = None,
                    actual_delivery: date = None):
        """تسجيل طلب للمورد"""
        history = SupplierOrderHistory(
            order_id=order_id,
            po_number=po_number,
            order_date=order_date,
            expected_delivery=expected_delivery,
            actual_delivery=actual_delivery,
            total_amount=total_amount,
            status=status,
            items=items or []
        )
        
        if supplier_id not in self.order_history:
            self.order_history[supplier_id] = []
        
        self.order_history[supplier_id].append(history)
        
        # Update supplier stats
        supplier = self.suppliers.get(supplier_id)
        if supplier:
            # Recalculate on-time delivery rate
            orders = self.order_history[supplier_id]
            completed = [o for o in orders if o.actual_delivery]
            if completed:
                on_time = sum(1 for o in completed if o.was_on_time)
                supplier.on_time_delivery_rate = (on_time / len(completed)) * 100
    
    def get_supplier_performance(self, supplier_id: str,
                                period_start: Optional[date] = None,
                                period_end: Optional[date] = None) -> SupplierPerformance:
        """الحصول على أداء مورد"""
        if period_end is None:
            period_end = date.today()
        if period_start is None:
            period_start = period_end.replace(day=1)
        
        history = self.order_history.get(supplier_id, [])
        
        # Filter by period
        period_history = [
            h for h in history
            if period_start <= h.order_date <= period_end
        ]
        
        performance = SupplierPerformance(
            supplier_id=supplier_id,
            period_start=period_start,
            period_end=period_end,
            total_orders=len(period_history),
            total_value=sum(h.total_amount for h in period_history),
            on_time_deliveries=sum(1 for h in period_history if h.was_on_time),
            late_deliveries=sum(1 for h in period_history if h.actual_delivery and not h.was_on_time)
        )
        
        return performance
    
    def get_order_history(self, supplier_id: str,
                         limit: int = None) -> List[SupplierOrderHistory]:
        """الحصول على سجل طلبات المورد"""
        history = self.order_history.get(supplier_id, [])
        sorted_history = sorted(history, key=lambda x: x.order_date, reverse=True)
        
        if limit:
            return sorted_history[:limit]
        return sorted_history
    
    def get_suppliers_by_category(self, category: str) -> List[Supplier]:
        """الحصول على الموردين حسب الفئة"""
        return [
            s for s in self.suppliers.values()
            if category in s.categories
        ]
    
    def get_top_suppliers(self, by: str = "value", limit: int = 10) -> List[Dict]:
        """الحصول على أفضل الموردين"""
        supplier_stats = []
        
        for supplier in self.suppliers.values():
            history = self.order_history.get(supplier.id, [])
            total_value = sum(h.total_amount for h in history)
            total_orders = len(history)
            
            supplier_stats.append({
                "supplier": supplier.to_dict(),
                "total_orders": total_orders,
                "total_value": float(total_value),
                "average_order_value": float(total_value / total_orders) if total_orders > 0 else 0,
                "on_time_rate": supplier.on_time_delivery_rate,
                "quality_score": supplier.quality_score,
                "rating": supplier.rating.value
            })
        
        # Sort
        if by == "value":
            supplier_stats.sort(key=lambda x: x["total_value"], reverse=True)
        elif by == "orders":
            supplier_stats.sort(key=lambda x: x["total_orders"], reverse=True)
        elif by == "rating":
            supplier_stats.sort(key=lambda x: x["rating"], reverse=True)
        
        return supplier_stats[:limit]
    
    def get_suppliers_summary(self) -> Dict[str, Any]:
        """ملخص الموردين"""
        total_suppliers = len(self.suppliers)
        active_suppliers = len([s for s in self.suppliers.values() if s.is_active])
        approved_suppliers = len([s for s in self.suppliers.values() if s.is_approved])
        
        # Rating distribution
        ratings = {}
        for rating in SupplierRating:
            count = len([s for s in self.suppliers.values() if s.rating == rating])
            ratings[rating.name] = count
        
        return {
            "total_suppliers": total_suppliers,
            "active_suppliers": active_suppliers,
            "approved_suppliers": approved_suppliers,
            "rating_distribution": ratings,
            "categories": list(set(
                cat for s in self.suppliers.values() for cat in s.categories
            ))
        }
    
    def deactivate_supplier(self, supplier_id: str):
        """إلغاء تفعيل مورد"""
        supplier = self.suppliers.get(supplier_id)
        if supplier:
            supplier.is_active = False
            supplier.updated_at = datetime.now(timezone.utc)
    
    def approve_supplier(self, supplier_id: str):
        """اعتماد مورد للتعامل"""
        supplier = self.suppliers.get(supplier_id)
        if supplier:
            supplier.is_approved = True
            supplier.updated_at = datetime.now(timezone.utc)
