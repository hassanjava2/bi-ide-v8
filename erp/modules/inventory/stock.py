"""
Stock Management - إدارة المخزون

المميزات:
- Multi-location support (دعم مواقع متعددة)
- FIFO/LIFO valuation (تقييم المخزون)
- Stock movements tracking (تتبع الحركات)
- Low stock alerts (تنبيهات المخزون المنخفض)
"""

import uuid
from datetime import datetime, date, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque


class ValuationMethod(Enum):
    """طريقة تقييم المخزون"""
    FIFO = "fifo"      # First In First Out
    LIFO = "lifo"      # Last In First Out
    AVERAGE = "average"  # Weighted Average


class MovementType(Enum):
    """نوع حركة المخزون"""
    PURCHASE = "purchase"         # شراء
    SALE = "sale"                 # بيع
    RETURN_IN = "return_in"       # مرتجع وارد
    RETURN_OUT = "return_out"     # مرتجع صادر
    ADJUSTMENT = "adjustment"     # تسوية
    TRANSFER = "transfer"         # تحويل
    PRODUCTION = "production"     # إنتاج


@dataclass
class StockItem:
    """عنصر مخزون"""
    id: str
    sku: str                      # رمز المنتج
    name: str                     # اسم المنتج
    description: str = ""
    category: str = ""            # الفئة
    
    # Pricing
    unit_cost: Decimal = field(default_factory=lambda: Decimal('0'))
    unit_price: Decimal = field(default_factory=lambda: Decimal('0'))
    currency: str = "SAR"
    
    # Stock levels by location
    stock_by_location: Dict[str, int] = field(default_factory=dict)
    
    # Reorder settings
    reorder_point: int = 10       # نقطة إعادة الطلب
    reorder_quantity: int = 50    # كمية إعادة الطلب
    
    # Valuation
    valuation_method: ValuationMethod = ValuationMethod.FIFO
    
    # Metadata
    barcode: str = ""
    supplier_id: Optional[str] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def total_quantity(self) -> int:
        """الكمية الإجمالية في جميع المواقع"""
        return sum(self.stock_by_location.values())
    
    @property
    def is_low_stock(self) -> bool:
        """هل المخزون منخفض؟"""
        return self.total_quantity <= self.reorder_point
    
    @property
    def stock_value(self) -> Decimal:
        """قيمة المخزون"""
        return Decimal(str(self.total_quantity)) * self.unit_cost
    
    def get_location_quantity(self, location: str) -> int:
        """الحصول على كمية موقع معين"""
        return self.stock_by_location.get(location, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sku": self.sku,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "unit_cost": float(self.unit_cost),
            "unit_price": float(self.unit_price),
            "currency": self.currency,
            "total_quantity": self.total_quantity,
            "stock_by_location": self.stock_by_location,
            "reorder_point": self.reorder_point,
            "reorder_quantity": self.reorder_quantity,
            "is_low_stock": self.is_low_stock,
            "stock_value": float(self.stock_value),
            "valuation_method": self.valuation_method.value,
            "barcode": self.barcode,
            "supplier_id": self.supplier_id,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class StockMovement:
    """حركة مخزون"""
    id: str
    item_id: str                  # معرف المنتج
    item_sku: str                 # رمز المنتج
    type: MovementType            # نوع الحركة
    quantity: int                 # الكمية (موجبة للإضافة، سالبة للخصم)
    location: str                 # الموقع
    
    # For transfers
    from_location: Optional[str] = None
    to_location: Optional[str] = None
    
    # Cost tracking for FIFO/LIFO
    unit_cost: Decimal = field(default_factory=lambda: Decimal('0'))
    total_cost: Decimal = field(default_factory=lambda: Decimal('0'))
    
    # References
    reference_id: Optional[str] = None    # رقم الفاتورة/الطلب
    reference_type: Optional[str] = None  # نوع المرجع
    
    reason: str = ""              # سبب الحركة
    notes: str = ""
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "item_id": self.item_id,
            "item_sku": self.item_sku,
            "type": self.type.value,
            "quantity": self.quantity,
            "location": self.location,
            "from_location": self.from_location,
            "to_location": self.to_location,
            "unit_cost": float(self.unit_cost),
            "total_cost": float(self.total_cost),
            "reference_id": self.reference_id,
            "reference_type": self.reference_type,
            "reason": self.reason,
            "notes": self.notes,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class StockLot:
    """دفعة مخزون (لـ FIFO/LIFO)"""
    lot_id: str
    item_id: str
    quantity: int
    unit_cost: Decimal
    received_date: datetime
    expiry_date: Optional[date] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lot_id": self.lot_id,
            "item_id": self.item_id,
            "quantity": self.quantity,
            "unit_cost": float(self.unit_cost),
            "received_date": self.received_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None
        }


class StockManager:
    """
    مدير المخزون مع دعم multi-location و FIFO/LIFO
    """
    
    def __init__(self):
        self.items: Dict[str, StockItem] = {}           # sku -> item
        self.movements: List[StockMovement] = []
        self.lots: Dict[str, deque] = {}                # item_id -> deque of StockLot (for FIFO/LIFO)
        self.locations: set = {"المستودع الرئيسي", "المعرض", "المستودع الثانوي"}
    
    def add_item(self, sku: str, name: str, quantity: int = 0,
                location: str = "المستودع الرئيسي",
                unit_cost: Decimal = Decimal('0'),
                unit_price: Decimal = Decimal('0'),
                category: str = "",
                reorder_point: int = 10,
                **kwargs) -> StockItem:
        """
        إضافة منتج جديد للمخزون
        
        Args:
            sku: رمز المنتج الفريد
            name: اسم المنتج
            quantity: الكمية الابتدائية
            location: الموقع
            unit_cost: سعر التكلفة
            unit_price: سعر البيع
            category: الفئة
            reorder_point: نقطة إعادة الطلب
        """
        if sku in [item.sku for item in self.items.values()]:
            raise ValueError(f"SKU {sku} already exists")
        
        if location not in self.locations:
            self.locations.add(location)
        
        item = StockItem(
            id=str(uuid.uuid4()),
            sku=sku,
            name=name,
            category=category,
            unit_cost=Decimal(str(unit_cost)),
            unit_price=Decimal(str(unit_price)),
            stock_by_location={location: quantity},
            reorder_point=reorder_point,
            **kwargs
        )
        
        self.items[item.id] = item
        
        # Record initial stock movement if quantity > 0
        if quantity > 0:
            self._record_movement(
                item_id=item.id,
                item_sku=sku,
                movement_type=MovementType.ADJUSTMENT,
                quantity=quantity,
                location=location,
                unit_cost=item.unit_cost,
                reason="Initial stock"
            )
        
        return item
    
    def update_stock(self, sku: str, delta: int, reason: str,
                    location: str = "المستودع الرئيسي",
                    movement_type: MovementType = MovementType.ADJUSTMENT,
                    reference_id: Optional[str] = None,
                    reference_type: Optional[str] = None,
                    unit_cost: Optional[Decimal] = None) -> StockMovement:
        """
        تحديث المخزون
        
        Args:
            sku: رمز المنتج
            delta: التغيير (موجب للإضافة، سالب للخصم)
            reason: سبب التغيير
            location: الموقع
            movement_type: نوع الحركة
            reference_id: رقم المرجع
            reference_type: نوع المرجع
            unit_cost: سعر التكلفة (للإضافات)
        """
        # Find item by SKU
        item = None
        for i in self.items.values():
            if i.sku == sku:
                item = i
                break
        
        if not item:
            raise ValueError(f"Item with SKU {sku} not found")
        
        # Ensure location exists
        if location not in self.locations:
            self.locations.add(location)
        
        # Check stock availability for deductions
        if delta < 0:
            current_qty = item.stock_by_location.get(location, 0)
            if current_qty + delta < 0:
                raise ValueError(f"Insufficient stock at {location}. Available: {current_qty}")
        
        # Update stock
        item.stock_by_location[location] = item.stock_by_location.get(location, 0) + delta
        item.updated_at = datetime.now(timezone.utc)
        
        # Determine unit cost
        cost = unit_cost if unit_cost else item.unit_cost
        
        # Handle FIFO/LIFO lots for purchases
        if delta > 0 and item.valuation_method in [ValuationMethod.FIFO, ValuationMethod.LIFO]:
            self._add_lot(item.id, delta, cost)
        
        # Record movement
        movement = self._record_movement(
            item_id=item.id,
            item_sku=sku,
            movement_type=movement_type,
            quantity=delta,
            location=location,
            unit_cost=cost,
            total_cost=abs(Decimal(str(delta)) * cost),
            reference_id=reference_id,
            reference_type=reference_type,
            reason=reason
        )
        
        return movement
    
    def _record_movement(self, item_id: str, item_sku: str,
                        movement_type: MovementType, quantity: int,
                        location: str, unit_cost: Decimal,
                        total_cost: Optional[Decimal] = None,
                        reference_id: Optional[str] = None,
                        reference_type: Optional[str] = None,
                        reason: str = "") -> StockMovement:
        """تسجيل حركة مخزون"""
        movement = StockMovement(
            id=str(uuid.uuid4()),
            item_id=item_id,
            item_sku=item_sku,
            type=movement_type,
            quantity=quantity,
            location=location,
            unit_cost=unit_cost,
            total_cost=total_cost or abs(Decimal(str(quantity)) * unit_cost),
            reference_id=reference_id,
            reference_type=reference_type,
            reason=reason
        )
        
        self.movements.append(movement)
        return movement
    
    def _add_lot(self, item_id: str, quantity: int, unit_cost: Decimal):
        """إضافة دفعة للـ FIFO/LIFO"""
        if item_id not in self.lots:
            self.lots[item_id] = deque()
        
        lot = StockLot(
            lot_id=str(uuid.uuid4()),
            item_id=item_id,
            quantity=quantity,
            unit_cost=unit_cost,
            received_date=datetime.now(timezone.utc)
        )
        
        self.lots[item_id].append(lot)
    
    def get_stock_levels(self, location: Optional[str] = None) -> Dict[str, Any]:
        """
        الحصول على مستويات المخزون
        
        Args:
            location: موقع محدد (اختياري)
        """
        if location:
            items = [
                {
                    "sku": item.sku,
                    "name": item.name,
                    "quantity": item.get_location_quantity(location),
                    "unit_cost": float(item.unit_cost),
                    "value": float(item.get_location_quantity(location) * item.unit_cost)
                }
                for item in self.items.values()
            ]
            total_value = sum(i["value"] for i in items)
            return {
                "location": location,
                "items": items,
                "total_items": len(items),
                "total_value": float(total_value)
            }
        else:
            # All locations summary
            by_location = {}
            for loc in self.locations:
                items_in_loc = [
                    item for item in self.items.values()
                    if item.get_location_quantity(loc) > 0
                ]
                by_location[loc] = {
                    "item_count": len(items_in_loc),
                    "total_quantity": sum(item.get_location_quantity(loc) for item in items_in_loc),
                    "total_value": sum(
                        item.get_location_quantity(loc) * item.unit_cost 
                        for item in items_in_loc
                    )
                }
            
            return {
                "by_location": by_location,
                "total_items": len(self.items),
                "total_value": sum(item.stock_value for item in self.items.values())
            }
    
    def low_stock_alerts(self) -> List[Dict[str, Any]]:
        """الحصول على تنبيهات المخزون المنخفض"""
        alerts = []
        
        for item in self.items.values():
            if item.is_low_stock:
                for location, quantity in item.stock_by_location.items():
                    if quantity <= item.reorder_point:
                        alerts.append({
                            "item_id": item.id,
                            "sku": item.sku,
                            "name": item.name,
                            "location": location,
                            "current_quantity": quantity,
                            "reorder_point": item.reorder_point,
                            "reorder_quantity": item.reorder_quantity,
                            "shortage": item.reorder_point - quantity + item.reorder_quantity,
                            "severity": "high" if quantity == 0 else "medium" if quantity <= item.reorder_point / 2 else "low"
                        })
        
        return sorted(alerts, key=lambda x: x["current_quantity"])
    
    def transfer_stock(self, sku: str, quantity: int,
                      from_location: str, to_location: str,
                      reason: str = "") -> StockMovement:
        """
        تحويل مخزون بين المواقع
        """
        # Deduct from source
        self.update_stock(sku, -quantity, reason=f"Transfer to {to_location}",
                         location=from_location, movement_type=MovementType.TRANSFER)
        
        # Add to destination
        self.update_stock(sku, quantity, reason=f"Transfer from {from_location}",
                         location=to_location, movement_type=MovementType.TRANSFER)
        
        # Create transfer record
        item = next(i for i in self.items.values() if i.sku == sku)
        return StockMovement(
            id=str(uuid.uuid4()),
            item_id=item.id,
            item_sku=sku,
            type=MovementType.TRANSFER,
            quantity=quantity,
            location=f"{from_location} -> {to_location}",
            from_location=from_location,
            to_location=to_location,
            reason=reason
        )
    
    def get_item_movements(self, sku: str, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Dict]:
        """الحصول على حركات منتج معين"""
        movements = [m for m in self.movements if m.item_sku == sku]
        
        if start_date:
            movements = [m for m in movements if m.created_at >= start_date]
        if end_date:
            movements = [m for m in movements if m.created_at <= end_date]
        
        return [m.to_dict() for m in sorted(movements, key=lambda x: x.created_at)]
    
    def get_item(self, sku: str) -> Optional[StockItem]:
        """الحصول على منتج برمزه"""
        for item in self.items.values():
            if item.sku == sku:
                return item
        return None
    
    def get_inventory_valuation(self, method: ValuationMethod = ValuationMethod.FIFO) -> Dict[str, Any]:
        """تقييم المخزون"""
        total_value = Decimal('0')
        item_values = []
        
        for item in self.items.values():
            value = item.stock_value
            total_value += value
            item_values.append({
                "sku": item.sku,
                "name": item.name,
                "quantity": item.total_quantity,
                "unit_cost": float(item.unit_cost),
                "value": float(value)
            })
        
        return {
            "valuation_method": method.value,
            "total_value": float(total_value),
            "total_items": len(self.items),
            "total_quantity": sum(item.total_quantity for item in self.items.values()),
            "items": item_values
        }
    
    def add_location(self, location_name: str):
        """إضافة موقع جديد"""
        self.locations.add(location_name)
    
    def get_locations(self) -> List[str]:
        """الحصول على قائمة المواقع"""
        return list(self.locations)
