"""
Inventory Module - إدارة المخزون
Stock management with products and stock movements
"""
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any
from enum import Enum

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Base
from sqlalchemy import Column, String, Numeric, DateTime, Text, ForeignKey, Enum as SQLEnum, Integer
from sqlalchemy.orm import relationship


class MovementType(str, Enum):
    IN = "in"           # Stock in / إدخال
    OUT = "out"         # Stock out / إخراج
    ADJUSTMENT = "adjustment"  # Adjustment / تعديل
    RETURN = "return"   # Return / مرتجع


class Product(Base):
    """Product / المنتج"""
    __tablename__ = "erp_products"
    __table_args__ = {"extend_existing": True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sku = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(500), nullable=False)
    name_ar = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    category = Column(String(200), index=True)
    unit_price = Column(Numeric(15, 2), default=Decimal("0.00"))
    cost_price = Column(Numeric(15, 2), default=Decimal("0.00"))
    quantity_in_stock = Column(Integer, default=0)
    reorder_point = Column(Integer, default=10)
    reorder_quantity = Column(Integer, default=50)
    supplier_id = Column(String, ForeignKey("erp_customers.id"), nullable=True)
    location = Column(String(200), default="")
    barcode = Column(String(100), unique=True, nullable=True)
    is_active = Column(String, default="true")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    stock_movements = relationship("StockMovement", back_populates="product")
    supplier = relationship("Customer", foreign_keys=[supplier_id])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sku": self.sku,
            "name": self.name,
            "name_ar": self.name_ar,
            "description": self.description,
            "category": self.category,
            "unit_price": float(self.unit_price) if self.unit_price else 0.0,
            "cost_price": float(self.cost_price) if self.cost_price else 0.0,
            "quantity_in_stock": self.quantity_in_stock or 0,
            "reorder_point": self.reorder_point or 10,
            "reorder_quantity": self.reorder_quantity or 50,
            "supplier_id": self.supplier_id,
            "location": self.location,
            "barcode": self.barcode,
            "is_active": self.is_active == "true",
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class StockMovement(Base):
    """Stock movement / حركة المخزون"""
    __tablename__ = "erp_stock_movements"
    __table_args__ = {"extend_existing": True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id = Column(String, ForeignKey("erp_products.id"), nullable=False, index=True)
    movement_type = Column(SQLEnum(MovementType), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_cost = Column(Numeric(15, 2), default=Decimal("0.00"))
    total_cost = Column(Numeric(15, 2), default=Decimal("0.00"))
    reference = Column(String(200), nullable=True)  # Invoice number, PO, etc.
    notes = Column(Text, nullable=True)
    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    product = relationship("Product", back_populates="stock_movements")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "product_id": self.product_id,
            "movement_type": self.movement_type.value if self.movement_type else None,
            "quantity": self.quantity,
            "unit_cost": float(self.unit_cost) if self.unit_cost else 0.0,
            "total_cost": float(self.total_cost) if self.total_cost else 0.0,
            "reference": self.reference,
            "notes": self.notes,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


async def create_product(
    session: AsyncSession,
    sku: str,
    name: str,
    category: Optional[str] = None,
    unit_price: float = 0.0,
    cost_price: float = 0.0,
    initial_quantity: int = 0,
    reorder_point: int = 10,
    name_ar: Optional[str] = None,
    description: Optional[str] = None,
    supplier_id: Optional[str] = None,
    location: Optional[str] = None,
    barcode: Optional[str] = None
) -> Product:
    """Create a new product / إنشاء منتج جديد"""
    product = Product(
        sku=sku,
        name=name,
        name_ar=name_ar,
        description=description,
        category=category,
        unit_price=Decimal(str(unit_price)),
        cost_price=Decimal(str(cost_price)),
        quantity_in_stock=initial_quantity,
        reorder_point=reorder_point,
        supplier_id=supplier_id,
        location=location or "",
        barcode=barcode
    )
    session.add(product)
    await session.flush()
    
    # Create initial stock movement if quantity > 0
    if initial_quantity > 0:
        movement = StockMovement(
            product_id=product.id,
            movement_type=MovementType.IN,
            quantity=initial_quantity,
            unit_cost=Decimal(str(cost_price)),
            total_cost=Decimal(str(cost_price)) * initial_quantity,
            notes="Initial stock"
        )
        session.add(movement)
        await session.flush()
    
    return product


async def get_product(session: AsyncSession, product_id: str) -> Optional[Product]:
    """Get product by ID"""
    result = await session.execute(
        select(Product).where(Product.id == product_id)
    )
    return result.scalar_one_or_none()


async def get_product_by_sku(session: AsyncSession, sku: str) -> Optional[Product]:
    """Get product by SKU"""
    result = await session.execute(
        select(Product).where(Product.sku == sku)
    )
    return result.scalar_one_or_none()


async def update_product(
    session: AsyncSession,
    product_id: str,
    **kwargs
) -> Optional[Product]:
    """Update product fields"""
    product = await get_product(session, product_id)
    if not product:
        return None
    
    # Convert Decimal fields
    decimal_fields = ['unit_price', 'cost_price']
    for field in decimal_fields:
        if field in kwargs:
            kwargs[field] = Decimal(str(kwargs[field]))
    
    for key, value in kwargs.items():
        if hasattr(product, key):
            setattr(product, key, value)
    
    product.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return product


async def adjust_stock(
    session: AsyncSession,
    product_id: str,
    quantity_change: int,
    movement_type: str,
    unit_cost: float = 0.0,
    reference: Optional[str] = None,
    notes: Optional[str] = None,
    created_by: Optional[str] = None
) -> StockMovement:
    """
    Adjust stock quantity / تعديل المخزون
    
    Args:
        quantity_change: Positive for IN, negative for OUT
        movement_type: 'in', 'out', 'adjustment', 'return'
    """
    product = await get_product(session, product_id)
    if not product:
        raise ValueError(f"Product {product_id} not found")
    
    movement_enum = MovementType(movement_type)
    
    # Calculate total cost
    decimal_unit_cost = Decimal(str(unit_cost))
    total_cost = decimal_unit_cost * abs(quantity_change)
    
    # Create movement record
    movement = StockMovement(
        product_id=product_id,
        movement_type=movement_enum,
        quantity=quantity_change,
        unit_cost=decimal_unit_cost,
        total_cost=total_cost,
        reference=reference,
        notes=notes,
        created_by=created_by
    )
    session.add(movement)
    
    # Update product quantity
    product.quantity_in_stock = (product.quantity_in_stock or 0) + quantity_change
    product.updated_at = datetime.now(timezone.utc)
    
    await session.flush()
    return movement


async def get_low_stock_items(
    session: AsyncSession,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get products below reorder point / منتجات تحت حد الطلب"""
    result = await session.execute(
        select(Product).where(
            and_(
                Product.is_active == "true",
                Product.quantity_in_stock <= Product.reorder_point
            )
        ).limit(limit)
    )
    products = result.scalars().all()
    
    return [
        {
            "product_id": p.id,
            "sku": p.sku,
            "name": p.name,
            "current_stock": p.quantity_in_stock or 0,
            "reorder_point": p.reorder_point or 10,
            "suggested_order": (p.reorder_quantity or 50) - (p.quantity_in_stock or 0)
        }
        for p in products
    ]


async def get_stock_valuation(session: AsyncSession) -> Dict[str, Any]:
    """Get total stock valuation / تقييم المخزون"""
    result = await session.execute(
        select(
            func.sum(Product.quantity_in_stock * Product.cost_price).label("total_cost"),
            func.sum(Product.quantity_in_stock * Product.unit_price).label("total_value"),
            func.count(Product.id).label("product_count")
        ).where(Product.is_active == "true")
    )
    row = result.one()
    
    return {
        "total_cost": float(row.total_cost or 0),
        "total_value": float(row.total_value or 0),
        "potential_profit": float((row.total_value or 0) - (row.total_cost or 0)),
        "product_count": row.product_count or 0
    }


async def get_product_movements(
    session: AsyncSession,
    product_id: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get stock movements for a product"""
    result = await session.execute(
        select(StockMovement)
        .where(StockMovement.product_id == product_id)
        .order_by(StockMovement.created_at.desc())
        .limit(limit)
    )
    movements = result.scalars().all()
    
    return [m.to_dict() for m in movements]
