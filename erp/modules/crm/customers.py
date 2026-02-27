"""
Customer Management - إدارة العملاء

إدارة العملاء مع:
- معلومات الاتصال
- التصنيف (lead, prospect, customer)
- قيمة العميل مدى الحياة (LTV)
- بوابة العملاء (portal integration)
"""

import uuid
from datetime import datetime, date, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class CustomerClassification(Enum):
    """تصنيف العميل"""
    LEAD = "lead"                     # عميل محتمل
    PROSPECT = "prospect"             # prospect
    CUSTOMER = "customer"             # عميل نشط
    VIP = "vip"                       # عميل VIP
    INACTIVE = "inactive"             # غير نشط


class CustomerType(Enum):
    """نوع العميل"""
    INDIVIDUAL = "individual"         # فرد
    COMPANY = "company"               # شركة
    GOVERNMENT = "government"         # حكومي
    NON_PROFIT = "non_profit"         # غير ربحي


@dataclass
class CustomerContact:
    """جهة اتصال العميل"""
    id: str
    name: str
    email: str = ""
    phone: str = ""
    position: str = ""                # المنصب
    is_primary: bool = False          # جهة الاتصال الرئيسية
    is_decision_maker: bool = False   # صانع القرار
    notes: str = ""


@dataclass
class CustomerAddress:
    """عنوان العميل"""
    id: str
    address_type: str = "billing"     # billing, shipping, office
    street: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = "Saudi Arabia"
    is_default: bool = False


@dataclass
class Customer:
    """عميل"""
    id: str
    customer_code: str                # كود العميل
    
    # Basic Info
    name: str                         # الاسم/اسم الشركة
    type: CustomerType = CustomerType.INDIVIDUAL
    classification: CustomerClassification = CustomerClassification.LEAD
    
    # Contact Info
    email: str = ""
    phone: str = ""
    mobile: str = ""
    website: str = ""
    
    # Additional contacts (for companies)
    contacts: List[CustomerContact] = field(default_factory=list)
    addresses: List[CustomerAddress] = field(default_factory=list)
    
    # Business Info (for B2B)
    company_name: str = ""
    industry: str = ""                # مجال العمل
    company_size: str = ""            # حجم الشركة
    tax_number: str = ""              # الرقم الضريبي
    commercial_registration: str = ""  # السجل التجاري
    
    # Source
    source: str = ""                  # مصدر العميل: referral, website, social_media, etc.
    source_detail: str = ""           # تفاصيل المصدر
    assigned_to: Optional[str] = None # معرف المندوب المسؤول
    
    # Lifetime Value
    first_purchase_date: Optional[date] = None
    last_purchase_date: Optional[date] = None
    total_orders: int = 0
    total_revenue: Decimal = field(default_factory=lambda: Decimal('0'))
    total_paid: Decimal = field(default_factory=lambda: Decimal('0'))
    outstanding_balance: Decimal = field(default_factory=lambda: Decimal('0'))
    
    # Scoring
    lead_score: int = 0               # درجة العميل المحتمل (0-100)
    
    # Portal
    portal_enabled: bool = False
    portal_username: str = ""
    portal_last_login: Optional[datetime] = None
    
    # Notes & Tags
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Status
    is_active: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def lifetime_value(self) -> Decimal:
        """قيمة العميل مدى الحياة (LTV)"""
        return self.total_revenue
    
    @property
    def average_order_value(self) -> Decimal:
        """متوسط قيمة الطلب"""
        if self.total_orders == 0:
            return Decimal('0')
        return self.total_revenue / Decimal(str(self.total_orders))
    
    @property
    def is_overdue(self) -> bool:
        """هل للعميل مستحقات متأخرة؟"""
        return self.outstanding_balance > 0
    
    def add_contact(self, name: str, email: str = "", phone: str = "",
                   position: str = "", is_primary: bool = False,
                   is_decision_maker: bool = False) -> CustomerContact:
        """إضافة جهة اتصال"""
        contact = CustomerContact(
            id=str(uuid.uuid4()),
            name=name,
            email=email,
            phone=phone,
            position=position,
            is_primary=is_primary,
            is_decision_maker=is_decision_maker
        )
        
        if is_primary:
            for c in self.contacts:
                c.is_primary = False
        
        self.contacts.append(contact)
        self.updated_at = datetime.now(timezone.utc)
        return contact
    
    def add_address(self, street: str, city: str, state: str = "",
                   postal_code: str = "", country: str = "Saudi Arabia",
                   address_type: str = "billing", is_default: bool = False) -> CustomerAddress:
        """إضافة عنوان"""
        address = CustomerAddress(
            id=str(uuid.uuid4()),
            address_type=address_type,
            street=street,
            city=city,
            state=state,
            postal_code=postal_code,
            country=country,
            is_default=is_default
        )
        
        if is_default:
            for a in self.addresses:
                if a.address_type == address_type:
                    a.is_default = False
        
        self.addresses.append(address)
        self.updated_at = datetime.now(timezone.utc)
        return address
    
    def record_purchase(self, amount: Decimal):
        """تسجيل عملية شراء"""
        today = date.today()
        
        if self.first_purchase_date is None:
            self.first_purchase_date = today
        
        self.last_purchase_date = today
        self.total_orders += 1
        self.total_revenue += Decimal(str(amount))
        
        # Update classification if needed
        if self.classification == CustomerClassification.LEAD:
            self.classification = CustomerClassification.PROSPECT
        elif self.classification == CustomerClassification.PROSPECT and self.total_orders >= 2:
            self.classification = CustomerClassification.CUSTOMER
        
        self.updated_at = datetime.now(timezone.utc)
    
    def record_payment(self, amount: Decimal):
        """تسجيل دفعة"""
        self.total_paid += Decimal(str(amount))
        self.outstanding_balance = self.total_revenue - self.total_paid
        self.updated_at = datetime.now(timezone.utc)
    
    def enable_portal(self, username: str = None):
        """تفعيل بوابة العميل"""
        self.portal_enabled = True
        self.portal_username = username or self.email
    
    def record_portal_login(self):
        """تسجيل دخول العميل للبوابة"""
        self.portal_last_login = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_code": self.customer_code,
            "name": self.name,
            "type": self.type.value,
            "classification": self.classification.value,
            "email": self.email,
            "phone": self.phone,
            "mobile": self.mobile,
            "website": self.website,
            "contacts": [
                {
                    "id": c.id,
                    "name": c.name,
                    "email": c.email,
                    "phone": c.phone,
                    "position": c.position,
                    "is_primary": c.is_primary,
                    "is_decision_maker": c.is_decision_maker
                } for c in self.contacts
            ],
            "addresses": [
                {
                    "id": a.id,
                    "type": a.address_type,
                    "street": a.street,
                    "city": a.city,
                    "country": a.country,
                    "is_default": a.is_default
                } for a in self.addresses
            ],
            "company_name": self.company_name,
            "industry": self.industry,
            "source": self.source,
            "assigned_to": self.assigned_to,
            "lifetime_value": float(self.lifetime_value),
            "average_order_value": float(self.average_order_value),
            "total_orders": self.total_orders,
            "total_revenue": float(self.total_revenue),
            "outstanding_balance": float(self.outstanding_balance),
            "lead_score": self.lead_score,
            "portal_enabled": self.portal_enabled,
            "portal_username": self.portal_username,
            "portal_last_login": self.portal_last_login.isoformat() if self.portal_last_login else None,
            "tags": self.tags,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat()
        }


class CustomerManager:
    """
    مدير العملاء
    """
    
    def __init__(self):
        self.customers: Dict[str, Customer] = {}
        self.interactions: List[Dict] = []  # سجل التفاعلات
        self._customer_counter = 0
    
    def create_customer(self, name: str, email: str = "", phone: str = "",
                       type: CustomerType = CustomerType.INDIVIDUAL,
                       classification: CustomerClassification = CustomerClassification.LEAD,
                       **kwargs) -> Customer:
        """إنشاء عميل جديد"""
        self._customer_counter += 1
        customer_code = f"CUST-{datetime.now().strftime('%Y')}-{self._customer_counter:05d}"
        
        customer = Customer(
            id=str(uuid.uuid4()),
            customer_code=customer_code,
            name=name,
            email=email,
            phone=phone,
            type=type,
            classification=classification,
            **kwargs
        )
        
        self.customers[customer.id] = customer
        return customer
    
    def get_customer(self, customer_id: str) -> Optional[Customer]:
        """الحصول على عميل"""
        return self.customers.get(customer_id)
    
    def get_customer_by_code(self, code: str) -> Optional[Customer]:
        """الحصول على عميل بالكود"""
        for customer in self.customers.values():
            if customer.customer_code == code:
                return customer
        return None
    
    def update_customer(self, customer_id: str, **kwargs) -> Customer:
        """تحديث بيانات عميل"""
        customer = self.customers.get(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")
        
        for key, value in kwargs.items():
            if hasattr(customer, key):
                setattr(customer, key, value)
        
        customer.updated_at = datetime.now(timezone.utc)
        return customer
    
    def classify_customer(self, customer_id: str, 
                         classification: CustomerClassification) -> Customer:
        """تصنيف عميل"""
        customer = self.customers.get(customer_id)
        if customer:
            customer.classification = classification
            customer.updated_at = datetime.now(timezone.utc)
        return customer
    
    def assign_to_salesperson(self, customer_id: str, salesperson_id: str) -> Customer:
        """تعيين عميل لمندوب مبيعات"""
        customer = self.customers.get(customer_id)
        if customer:
            customer.assigned_to = salesperson_id
            customer.updated_at = datetime.now(timezone.utc)
        return customer
    
    def record_interaction(self, customer_id: str, interaction_type: str,
                          notes: str, created_by: str = None):
        """تسجيل تفاعل مع العميل"""
        interaction = {
            "id": str(uuid.uuid4()),
            "customer_id": customer_id,
            "type": interaction_type,  # call, email, meeting, note
            "notes": notes,
            "created_by": created_by,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        self.interactions.append(interaction)
        return interaction
    
    def get_customer_interactions(self, customer_id: str) -> List[Dict]:
        """الحصول على تفاعلات العميل"""
        return [
            i for i in self.interactions
            if i["customer_id"] == customer_id
        ]
    
    def search_customers(self, query: str) -> List[Customer]:
        """البحث في العملاء"""
        query = query.lower()
        results = []
        
        for customer in self.customers.values():
            if (query in customer.name.lower() or
                query in customer.email.lower() or
                query in customer.phone.lower() or
                query in customer.customer_code.lower() or
                query in [t.lower() for t in customer.tags]):
                results.append(customer)
        
        return results
    
    def get_customers_by_classification(self, 
                                       classification: CustomerClassification) -> List[Customer]:
        """الحصول على العملاء حسب التصنيف"""
        return [
            c for c in self.customers.values()
            if c.classification == classification
        ]
    
    def get_customers_by_salesperson(self, salesperson_id: str) -> List[Customer]:
        """الحصول على عملاء مندوب مبيعات"""
        return [
            c for c in self.customers.values()
            if c.assigned_to == salesperson_id
        ]
    
    def get_top_customers(self, limit: int = 10) -> List[Customer]:
        """الحصول على أفضل العملاء حسب القيمة"""
        sorted_customers = sorted(
            self.customers.values(),
            key=lambda x: x.lifetime_value,
            reverse=True
        )
        return sorted_customers[:limit]
    
    def get_customer_summary(self) -> Dict[str, Any]:
        """ملخص العملاء"""
        total_customers = len(self.customers)
        active_customers = len([c for c in self.customers.values() if c.is_active])
        
        by_classification = {
            c.value: len(self.get_customers_by_classification(c))
            for c in CustomerClassification
        }
        
        by_type = {
            t.value: len([c for c in self.customers.values() if c.type == t])
            for t in CustomerType
        }
        
        total_revenue = sum(c.total_revenue for c in self.customers.values())
        total_outstanding = sum(c.outstanding_balance for c in self.customers.values())
        
        return {
            "total_customers": total_customers,
            "active_customers": active_customers,
            "by_classification": by_classification,
            "by_type": by_type,
            "total_lifetime_value": float(total_revenue),
            "total_outstanding": float(total_outstanding),
            "average_order_value": float(
                sum(c.average_order_value for c in self.customers.values()) / total_customers
            ) if total_customers > 0 else 0
        }
    
    def deactivate_customer(self, customer_id: str):
        """إلغاء تفعيل عميل"""
        customer = self.customers.get(customer_id)
        if customer:
            customer.is_active = False
            customer.classification = CustomerClassification.INACTIVE
            customer.updated_at = datetime.now(timezone.utc)
    
    def delete_customer(self, customer_id: str) -> bool:
        """حذف عميل"""
        if customer_id in self.customers:
            del self.customers[customer_id]
            return True
        return False
