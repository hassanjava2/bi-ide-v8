"""
Purchase Orders - أوامر الشراء

إدارة دورة حياة أمر الشراء:
Draft → Sent → Confirmed → Partially Received → Received → Paid
"""

import uuid
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class POStatus(Enum):
    """حالات أمر الشراء"""
    DRAFT = "draft"                     # مسودة
    SENT = "sent"                       # مرسل للمورد
    CONFIRMED = "confirmed"             # مؤكد من المورد
    PARTIALLY_RECEIVED = "partial"     # مستلم جزئياً
    RECEIVED = "received"               # مستلم بالكامل
    CANCELLED = "cancelled"             # ملغي
    PAID = "paid"                       # مدفوع


@dataclass
class POLineItem:
    """بند في أمر الشراء"""
    id: str
    line_number: int
    item_sku: str                     # رمز المنتج
    item_name: str                    # اسم المنتج
    description: str = ""
    quantity: int = 0
    unit_price: Decimal = field(default_factory=lambda: Decimal('0'))
    discount_percent: Decimal = field(default_factory=lambda: Decimal('0'))
    tax_percent: Decimal = field(default_factory=lambda: Decimal('15'))  # VAT 15%
    
    # Received tracking
    quantity_received: int = 0
    
    @property
    def subtotal(self) -> Decimal:
        """المجموع قبل الخصم والضريبة"""
        return Decimal(str(self.quantity)) * self.unit_price
    
    @property
    def discount_amount(self) -> Decimal:
        """مبلغ الخصم"""
        return self.subtotal * (self.discount_percent / Decimal('100'))
    
    @property
    def net_amount(self) -> Decimal:
        """المبلغ بعد الخصم"""
        return self.subtotal - self.discount_amount
    
    @property
    def tax_amount(self) -> Decimal:
        """مبلغ الضريبة"""
        return self.net_amount * (self.tax_percent / Decimal('100'))
    
    @property
    def total(self) -> Decimal:
        """الإجمالي"""
        return self.net_amount + self.tax_amount
    
    @property
    def remaining_quantity(self) -> int:
        """الكمية المتبقية للاستلام"""
        return self.quantity - self.quantity_received
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "line_number": self.line_number,
            "item_sku": self.item_sku,
            "item_name": self.item_name,
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": float(self.unit_price),
            "discount_percent": float(self.discount_percent),
            "tax_percent": float(self.tax_percent),
            "subtotal": float(self.subtotal),
            "discount_amount": float(self.discount_amount),
            "tax_amount": float(self.tax_amount),
            "total": float(self.total),
            "quantity_received": self.quantity_received,
            "remaining_quantity": self.remaining_quantity
        }


@dataclass
class PurchaseOrder:
    """أمر شراء"""
    id: str
    po_number: str                    # رقم أمر الشراء
    supplier_id: str                  # معرف المورد
    supplier_name: str                # اسم المورد
    
    # Dates
    order_date: date = field(default_factory=date.today)
    expected_delivery: Optional[date] = None
    actual_delivery: Optional[date] = None
    
    # Status
    status: POStatus = POStatus.DRAFT
    
    # Line items
    line_items: List[POLineItem] = field(default_factory=list)
    
    # Totals
    currency: str = "SAR"
    
    # References
    reference_number: str = ""        # رقم مرجعي داخلي
    supplier_reference: str = ""      # رقم مرجع المورد
    quotation_id: Optional[str] = None  # رقم عرض السعر
    
    # Shipping
    shipping_address: str = ""
    shipping_method: str = ""
    shipping_cost: Decimal = field(default_factory=lambda: Decimal('0'))
    
    # Terms
    payment_terms: int = 30           # أيام الاستحقاق
    notes: str = ""
    terms_conditions: str = ""
    
    # Tracking
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def subtotal(self) -> Decimal:
        """المجموع الفرعي"""
        return sum(item.subtotal for item in self.line_items)
    
    @property
    def total_discount(self) -> Decimal:
        """إجمالي الخصم"""
        return sum(item.discount_amount for item in self.line_items)
    
    @property
    def total_tax(self) -> Decimal:
        """إجمالي الضريبة"""
        return sum(item.tax_amount for item in self.line_items)
    
    @property
    def total(self) -> Decimal:
        """الإجمالي الكلي"""
        items_total = sum(item.total for item in self.line_items)
        return items_total + self.shipping_cost
    
    @property
    def is_fully_received(self) -> bool:
        """هل تم الاستلام بالكامل؟"""
        if not self.line_items:
            return False
        return all(item.remaining_quantity == 0 for item in self.line_items)
    
    @property
    def received_percentage(self) -> float:
        """نسبة الاستلام"""
        if not self.line_items:
            return 0
        total_qty = sum(item.quantity for item in self.line_items)
        if total_qty == 0:
            return 0
        received_qty = sum(item.quantity_received for item in self.line_items)
        return (received_qty / total_qty) * 100
    
    def add_line_item(self, item_sku: str, item_name: str,
                     quantity: int, unit_price: Decimal,
                     description: str = "", 
                     discount_percent: Decimal = Decimal('0'),
                     tax_percent: Decimal = Decimal('15')) -> POLineItem:
        """إضافة بند لأمر الشراء"""
        line_item = POLineItem(
            id=str(uuid.uuid4()),
            line_number=len(self.line_items) + 1,
            item_sku=item_sku,
            item_name=item_name,
            description=description,
            quantity=quantity,
            unit_price=Decimal(str(unit_price)),
            discount_percent=Decimal(str(discount_percent)),
            tax_percent=Decimal(str(tax_percent))
        )
        
        self.line_items.append(line_item)
        self.updated_at = datetime.now(timezone.utc)
        return line_item
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "po_number": self.po_number,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "order_date": self.order_date.isoformat(),
            "expected_delivery": self.expected_delivery.isoformat() if self.expected_delivery else None,
            "actual_delivery": self.actual_delivery.isoformat() if self.actual_delivery else None,
            "status": self.status.value,
            "line_items": [item.to_dict() for item in self.line_items],
            "subtotal": float(self.subtotal),
            "total_discount": float(self.total_discount),
            "total_tax": float(self.total_tax),
            "shipping_cost": float(self.shipping_cost),
            "total": float(self.total),
            "currency": self.currency,
            "reference_number": self.reference_number,
            "supplier_reference": self.supplier_reference,
            "shipping_address": self.shipping_address,
            "payment_terms": self.payment_terms,
            "notes": self.notes,
            "is_fully_received": self.is_fully_received,
            "received_percentage": self.received_percentage,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class GoodsReceipt:
    """سند استلام بضاعة"""
    id: str
    po_id: str                        # معرف أمر الشراء
    receipt_number: str               # رقم سند الاستلام
    received_date: date = field(default_factory=date.today)
    items_received: List[Dict] = field(default_factory=list)
    notes: str = ""
    received_by: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "po_id": self.po_id,
            "receipt_number": self.receipt_number,
            "received_date": self.received_date.isoformat(),
            "items_received": self.items_received,
            "notes": self.notes,
            "received_by": self.received_by
        }


class PurchaseOrderManager:
    """
    مدير أوامر الشراء
    """
    
    def __init__(self):
        self.purchase_orders: Dict[str, PurchaseOrder] = {}
        self.goods_receipts: Dict[str, GoodsReceipt] = {}
        self._po_counter = 0
    
    def create_po(self, supplier_id: str, supplier_name: str,
                  expected_delivery: Optional[date] = None,
                  payment_terms: int = 30,
                  created_by: Optional[str] = None,
                  **kwargs) -> PurchaseOrder:
        """إنشاء أمر شراء جديد"""
        self._po_counter += 1
        po_number = f"PO-{datetime.now().strftime('%Y%m')}-{self._po_counter:04d}"
        
        if expected_delivery is None:
            expected_delivery = date.today() + timedelta(days=14)
        
        po = PurchaseOrder(
            id=str(uuid.uuid4()),
            po_number=po_number,
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            expected_delivery=expected_delivery,
            payment_terms=payment_terms,
            created_by=created_by,
            **kwargs
        )
        
        self.purchase_orders[po.id] = po
        return po
    
    def get_po(self, po_id: str) -> Optional[PurchaseOrder]:
        """الحصول على أمر شراء"""
        return self.purchase_orders.get(po_id)
    
    def get_po_by_number(self, po_number: str) -> Optional[PurchaseOrder]:
        """الحصول على أمر شراء برقمه"""
        for po in self.purchase_orders.values():
            if po.po_number == po_number:
                return po
        return None
    
    def update_status(self, po_id: str, new_status: POStatus) -> PurchaseOrder:
        """تحديث حالة أمر الشراء"""
        po = self.purchase_orders.get(po_id)
        if not po:
            raise ValueError(f"Purchase order {po_id} not found")
        
        # Validate status transition
        valid_transitions = {
            POStatus.DRAFT: [POStatus.SENT, POStatus.CANCELLED],
            POStatus.SENT: [POStatus.CONFIRMED, POStatus.CANCELLED],
            POStatus.CONFIRMED: [POStatus.PARTIALLY_RECEIVED, POStatus.RECEIVED, POStatus.CANCELLED],
            POStatus.PARTIALLY_RECEIVED: [POStatus.RECEIVED, POStatus.CANCELLED],
            POStatus.RECEIVED: [POStatus.PAID, POStatus.CANCELLED],
            POStatus.PAID: [POStatus.CANCELLED],
            POStatus.CANCELLED: []
        }
        
        if new_status not in valid_transitions.get(po.status, []):
            raise ValueError(f"Invalid status transition from {po.status.value} to {new_status.value}")
        
        po.status = new_status
        po.updated_at = datetime.now(timezone.utc)
        
        if new_status == POStatus.RECEIVED:
            po.actual_delivery = date.today()
        
        return po
    
    def receive_goods(self, po_id: str, items_received: List[Dict],
                     received_by: Optional[str] = None,
                     notes: str = "") -> GoodsReceipt:
        """
        استلام بضاعة
        
        Args:
            po_id: معرف أمر الشراء
            items_received: قائمة بالبنود المستلمة [{"line_id": str, "quantity": int}]
            received_by: معرف المستلم
            notes: ملاحظات
        """
        po = self.purchase_orders.get(po_id)
        if not po:
            raise ValueError(f"Purchase order {po_id} not found")
        
        # Create goods receipt
        receipt = GoodsReceipt(
            id=str(uuid.uuid4()),
            po_id=po_id,
            receipt_number=f"GR-{po.po_number}-{len([r for r in self.goods_receipts.values() if r.po_id == po_id]) + 1}",
            items_received=items_received,
            received_by=received_by,
            notes=notes
        )
        
        self.goods_receipts[receipt.id] = receipt
        
        # Update line items
        for received in items_received:
            line_id = received.get("line_id")
            quantity = received.get("quantity", 0)
            
            for line_item in po.line_items:
                if line_item.id == line_id:
                    line_item.quantity_received += quantity
                    break
        
        # Update PO status
        if po.is_fully_received:
            po.status = POStatus.RECEIVED
        else:
            po.status = POStatus.PARTIALLY_RECEIVED
        
        po.actual_delivery = date.today()
        po.updated_at = datetime.now(timezone.utc)
        
        return receipt
    
    def get_pos_by_supplier(self, supplier_id: str) -> List[PurchaseOrder]:
        """الحصول على أوامر شراء مورد معين"""
        return [po for po in self.purchase_orders.values() if po.supplier_id == supplier_id]
    
    def get_pos_by_status(self, status: POStatus) -> List[PurchaseOrder]:
        """الحصول على أوامر شراء بحالة معينة"""
        return [po for po in self.purchase_orders.values() if po.status == status]
    
    def get_pending_receipts(self) -> List[PurchaseOrder]:
        """الحصول على أوامر الشراء المعلقة للاستلام"""
        return [
            po for po in self.purchase_orders.values()
            if po.status in [POStatus.CONFIRMED, POStatus.PARTIALLY_RECEIVED]
        ]
    
    def get_overdue_pos(self) -> List[PurchaseOrder]:
        """الحصول على أوامر الشراء المتأخرة"""
        today = date.today()
        return [
            po for po in self.purchase_orders.values()
            if po.status in [POStatus.SENT, POStatus.CONFIRMED, POStatus.PARTIALLY_RECEIVED]
            and po.expected_delivery and po.expected_delivery < today
        ]
    
    def get_po_summary(self) -> Dict[str, Any]:
        """ملخص أوامر الشراء"""
        total_pos = len(self.purchase_orders)
        by_status = {}
        
        for status in POStatus:
            count = len([po for po in self.purchase_orders.values() if po.status == status])
            by_status[status.value] = count
        
        total_value = sum(po.total for po in self.purchase_orders.values())
        
        return {
            "total_pos": total_pos,
            "by_status": by_status,
            "total_value": float(total_value),
            "overdue_count": len(self.get_overdue_pos()),
            "pending_receipt_count": len(self.get_pending_receipts())
        }
    
    def cancel_po(self, po_id: str, reason: str = "") -> PurchaseOrder:
        """إلغاء أمر شراء"""
        po = self.update_status(po_id, POStatus.CANCELLED)
        po.notes += f" | Cancelled: {reason}"
        return po
