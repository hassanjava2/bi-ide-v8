"""
Unit Tests for Inventory Module
اختبارات وحدة المخزون
"""

import pytest
from decimal import Decimal
from datetime import datetime

from erp.modules.inventory.stock import (
    StockManager, StockItem, StockMovement, 
    MovementType, ValuationMethod
)
from erp.modules.inventory.purchase_orders import (
    PurchaseOrderManager, PurchaseOrder, POStatus
)
from erp.modules.inventory.suppliers import (
    SupplierManager, Supplier, SupplierRating
)


class TestStockManager:
    """اختبارات مدير المخزون"""
    
    @pytest.fixture
    def stock_manager(self):
        """إنشاء مدير مخزون للاختبارات"""
        return StockManager()
    
    def test_add_item(self, stock_manager):
        """اختبار إضافة منتج"""
        item = stock_manager.add_item(
            sku="TEST-001",
            name="Test Product",
            quantity=100,
            location="Warehouse A",
            unit_cost=Decimal("50.00"),
            unit_price=Decimal("75.00"),
            category="Electronics"
        )
        
        assert item.sku == "TEST-001"
        assert item.name == "Test Product"
        assert item.total_quantity == 100
        assert item.unit_cost == Decimal("50.00")
    
    def test_add_item_duplicate_sku(self, stock_manager):
        """اختبار إضافة منتج بـ SKU مكرر"""
        stock_manager.add_item(sku="TEST-001", name="Test Product", quantity=100)
        
        with pytest.raises(ValueError, match="already exists"):
            stock_manager.add_item(sku="TEST-001", name="Another Product", quantity=50)
    
    def test_update_stock(self, stock_manager):
        """اختبار تحديث المخزون"""
        stock_manager.add_item(sku="TEST-001", name="Test Product", quantity=100, location="WH-A")
        
        movement = stock_manager.update_stock(
            sku="TEST-001",
            delta=50,
            reason="Purchase",
            location="WH-A",
            movement_type=MovementType.PURCHASE
        )
        
        assert movement.quantity == 50
        assert movement.type == MovementType.PURCHASE
        
        item = stock_manager.get_item("TEST-001")
        assert item.total_quantity == 150
    
    def test_update_stock_insufficient(self, stock_manager):
        """اختبار خصم كمية أكبر من المتاح"""
        stock_manager.add_item(sku="TEST-001", name="Test Product", quantity=10, location="WH-A")
        
        with pytest.raises(ValueError, match="Insufficient stock"):
            stock_manager.update_stock("TEST-001", -20, "Sale", "WH-A", MovementType.SALE)
    
    def test_transfer_stock(self, stock_manager):
        """اختبار تحويل مخزون"""
        stock_manager.add_item(sku="TEST-001", name="Test Product", quantity=100, location="WH-A")
        
        transfer = stock_manager.transfer_stock(
            sku="TEST-001",
            quantity=30,
            from_location="WH-A",
            to_location="WH-B",
            reason="Restocking"
        )
        
        item = stock_manager.get_item("TEST-001")
        assert item.get_location_quantity("WH-A") == 70
        assert item.get_location_quantity("WH-B") == 30
    
    def test_low_stock_alerts(self, stock_manager):
        """اختبار تنبيهات المخزون المنخفض"""
        # Add item with low stock
        stock_manager.add_item(
            sku="LOW-001",
            name="Low Stock Item",
            quantity=5,
            location="WH-A",
            reorder_point=10
        )
        
        # Add item with sufficient stock
        stock_manager.add_item(
            sku="OK-001",
            name="OK Stock Item",
            quantity=100,
            location="WH-A",
            reorder_point=10
        )
        
        alerts = stock_manager.low_stock_alerts()
        
        assert len(alerts) == 1
        assert alerts[0]["sku"] == "LOW-001"
        assert alerts[0]["severity"] in ["high", "medium", "critical"]
    
    def test_get_stock_levels(self, stock_manager):
        """اختبار الحصول على مستويات المخزون"""
        stock_manager.add_item("ITEM-001", "Item 1", 100, "WH-A")
        stock_manager.add_item("ITEM-002", "Item 2", 50, "WH-A")
        stock_manager.add_item("ITEM-003", "Item 3", 75, "WH-B")
        
        levels = stock_manager.get_stock_levels()
        
        assert "by_location" in levels
        assert levels["total_items"] == 3
    
    def test_multi_location_support(self, stock_manager):
        """اختبار دعم مواقع متعددة"""
        item = stock_manager.add_item(
            sku="MULTI-001",
            name="Multi-location Item",
            quantity=100,
            location="Warehouse A"
        )
        
        # Add to another location
        stock_manager.update_stock("MULTI-001", 50, "Transfer", "Warehouse B")
        
        assert item.get_location_quantity("Warehouse A") == 100
        assert item.get_location_quantity("Warehouse B") == 50
        assert item.total_quantity == 150
    
    def test_fifo_lots(self, stock_manager):
        """اختبار دفعات FIFO"""
        item = stock_manager.add_item(
            sku="FIFO-001",
            name="FIFO Item",
            quantity=0,
            valuation_method=ValuationMethod.FIFO
        )
        
        # Add lots with different costs
        stock_manager.update_stock("FIFO-001", 100, "Purchase", unit_cost=Decimal("10"))
        stock_manager.update_stock("FIFO-001", 100, "Purchase", unit_cost=Decimal("12"))
        
        # Check lots were created (keyed by item ID, not SKU)
        item = stock_manager.get_item("FIFO-001")
        assert item.id in stock_manager.lots
        assert len(stock_manager.lots[item.id]) == 2


class TestPurchaseOrderManager:
    """اختبارات مدير أوامر الشراء"""
    
    @pytest.fixture
    def po_manager(self):
        """إنشاء مدير أوامر شراء للاختبارات"""
        return PurchaseOrderManager()
    
    def test_create_po(self, po_manager):
        """اختبار إنشاء أمر شراء"""
        po = po_manager.create_po(
            supplier_id="SUP001",
            supplier_name="Test Supplier"
        )
        
        assert po.supplier_id == "SUP001"
        assert po.status == POStatus.DRAFT
        assert po.po_number.startswith("PO-")
    
    def test_add_line_item(self, po_manager):
        """اختبار إضافة بند"""
        po = po_manager.create_po("SUP001", "Test Supplier")
        
        line_item = po.add_line_item(
            item_sku="ITEM-001",
            item_name="Test Item",
            quantity=10,
            unit_price=Decimal("100.00"),
            discount_percent=Decimal("5")
        )
        
        assert line_item.quantity == 10
        assert line_item.unit_price == Decimal("100.00")
        assert line_item.discount_percent == Decimal("5")
        assert len(po.line_items) == 1
    
    def test_po_totals(self, po_manager):
        """اختبار حساب إجماليات أمر الشراء"""
        po = po_manager.create_po("SUP001", "Test Supplier")
        
        po.add_line_item("ITEM-001", "Item 1", 10, Decimal("100"))
        po.add_line_item("ITEM-002", "Item 2", 5, Decimal("200"))
        
        # 10*100 + 5*200 = 2000
        assert po.subtotal == Decimal("2000")
        assert po.total > 0
    
    def test_update_status(self, po_manager):
        """اختبار تحديث الحالة"""
        po = po_manager.create_po("SUP001", "Test Supplier")
        
        # Valid transition: DRAFT -> SENT
        po_manager.update_status(po.id, POStatus.SENT)
        assert po.status == POStatus.SENT
        
        # Valid transition: SENT -> CONFIRMED
        po_manager.update_status(po.id, POStatus.CONFIRMED)
        assert po.status == POStatus.CONFIRMED
    
    def test_invalid_status_transition(self, po_manager):
        """اختبار انتقال حالة غير صالح"""
        po = po_manager.create_po("SUP001", "Test Supplier")
        
        # Cannot go from DRAFT to RECEIVED
        with pytest.raises(ValueError, match="Invalid status transition"):
            po_manager.update_status(po.id, POStatus.RECEIVED)
    
    def test_receive_goods(self, po_manager):
        """اختبار استلام بضاعة"""
        po = po_manager.create_po("SUP001", "Test Supplier")
        po.add_line_item("ITEM-001", "Item 1", 10, Decimal("100"))
        po.add_line_item("ITEM-002", "Item 2", 5, Decimal("200"))
        
        po_manager.update_status(po.id, POStatus.SENT)
        po_manager.update_status(po.id, POStatus.CONFIRMED)
        
        # Partial receipt
        receipt = po_manager.receive_goods(
            po_id=po.id,
            items_received=[
                {"line_id": po.line_items[0].id, "quantity": 10},
                {"line_id": po.line_items[1].id, "quantity": 2}
            ]
        )
        
        assert receipt is not None
        assert po.status == POStatus.PARTIALLY_RECEIVED
        assert po.line_items[0].quantity_received == 10
        assert po.line_items[1].quantity_received == 2
    
    def test_full_receipt(self, po_manager):
        """اختبار استلام كامل"""
        po = po_manager.create_po("SUP001", "Test Supplier")
        po.add_line_item("ITEM-001", "Item 1", 10, Decimal("100"))
        
        po_manager.update_status(po.id, POStatus.SENT)
        po_manager.update_status(po.id, POStatus.CONFIRMED)
        
        po_manager.receive_goods(
            po_id=po.id,
            items_received=[{"line_id": po.line_items[0].id, "quantity": 10}]
        )
        
        assert po.status == POStatus.RECEIVED
        assert po.is_fully_received is True
        assert po.received_percentage == 100.0
    
    def test_get_overdue_pos(self, po_manager):
        """اختبار الحصول على أوامر الشراء المتأخرة"""
        from datetime import date, timedelta
        
        # Create overdue PO
        overdue_po = po_manager.create_po(
            "SUP001", "Test Supplier",
            expected_delivery=date.today() - timedelta(days=10)
        )
        po_manager.update_status(overdue_po.id, POStatus.SENT)
        
        # Create on-time PO
        ontime_po = po_manager.create_po(
            "SUP002", "Another Supplier",
            expected_delivery=date.today() + timedelta(days=10)
        )
        po_manager.update_status(ontime_po.id, POStatus.SENT)
        
        overdue = po_manager.get_overdue_pos()
        
        assert len(overdue) == 1
        assert overdue[0].id == overdue_po.id


class TestSupplierManager:
    """اختبارات مدير الموردين"""
    
    @pytest.fixture
    def supplier_manager(self):
        """إنشاء مدير موردين للاختبارات"""
        return SupplierManager()
    
    def test_create_supplier(self, supplier_manager):
        """اختبار إنشاء مورد"""
        supplier = supplier_manager.create_supplier(
            name="Test Supplier Co.",
            supplier_code="SUP001",
            email="supplier@test.com",
            phone="+966501234567",
            payment_terms=30
        )
        
        assert supplier.name == "Test Supplier Co."
        assert supplier.supplier_code == "SUP001"
        assert supplier.payment_terms == 30
    
    def test_add_contact(self, supplier_manager):
        """اختبار إضافة جهة اتصال"""
        supplier = supplier_manager.create_supplier("Test Supplier")
        
        contact = supplier_manager.add_contact(
            supplier_id=supplier.id,
            name="John Doe",
            email="john@supplier.com",
            phone="+966501234567",
            position="Sales Manager",
            is_primary=True
        )
        
        assert contact.name == "John Doe"
        assert contact.is_primary is True
        assert len(supplier.contacts) == 1
    
    def test_rate_supplier(self, supplier_manager):
        """اختبار تقييم مورد"""
        supplier = supplier_manager.create_supplier("Test Supplier")
        
        supplier_manager.rate_supplier(
            supplier.id,
            SupplierRating.GOOD,
            on_time_rate=85.5,
            quality_score=90.0
        )
        
        assert supplier.rating == SupplierRating.GOOD
        assert supplier.on_time_delivery_rate == 85.5
        assert supplier.quality_score == 90.0
    
    def test_record_order(self, supplier_manager):
        """اختبار تسجيل طلب"""
        supplier = supplier_manager.create_supplier("Test Supplier")
        
        from datetime import date
        
        supplier_manager.record_order(
            supplier_id=supplier.id,
            order_id="PO-001",
            po_number="PO-2024-001",
            order_date=date.today(),
            expected_delivery=date.today(),
            total_amount=Decimal("50000"),
            status="completed",
            actual_delivery=date.today()
        )
        
        history = supplier_manager.get_order_history(supplier.id)
        assert len(history) == 1
        assert history[0].po_number == "PO-2024-001"
    
    def test_performance_tracking(self, supplier_manager):
        """اختبار تتبع الأداء"""
        from datetime import date
        
        supplier = supplier_manager.create_supplier("Test Supplier")
        
        # Record orders with different delivery times
        supplier_manager.record_order(
            supplier.id, "PO-001", "PO-001",
            date.today(), date.today(),
            Decimal("10000"), "completed",
            actual_delivery=date.today()  # On time
        )
        
        supplier_manager.record_order(
            supplier.id, "PO-002", "PO-002",
            date.today(), date.today(),
            Decimal("15000"), "completed",
            actual_delivery=date.today()  # On time
        )
        
        performance = supplier_manager.get_supplier_performance(supplier.id)
        
        assert performance.total_orders == 2
        assert performance.total_value == Decimal("25000")
        assert performance.on_time_rate == 100.0
    
    def test_get_top_suppliers(self, supplier_manager):
        """اختبار الحصول على أفضل الموردين"""
        from datetime import date
        
        # Create suppliers with different order volumes
        sup1 = supplier_manager.create_supplier("Supplier A")
        sup2 = supplier_manager.create_supplier("Supplier B")
        
        supplier_manager.record_order(
            sup1.id, "PO-001", "PO-001", date.today(), date.today(),
            Decimal("100000"), "completed", date.today()
        )
        
        supplier_manager.record_order(
            sup2.id, "PO-002", "PO-002", date.today(), date.today(),
            Decimal("50000"), "completed", date.today()
        )
        
        top = supplier_manager.get_top_suppliers(by="value", limit=2)
        
        assert len(top) == 2
        assert top[0]["supplier"]["id"] == sup1.id  # Higher value first
