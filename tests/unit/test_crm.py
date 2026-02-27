"""
Unit Tests for CRM Module
اختبارات وحدة إدارة العملاء
"""

import pytest
from decimal import Decimal
from datetime import date, datetime, timedelta, timezone

from erp.modules.crm.customers import (
    CustomerManager, Customer, CustomerClassification, CustomerType
)
from erp.modules.crm.sales_pipeline import (
    SalesPipeline, Deal, PipelineStage, DealPriority
)
from erp.modules.crm.support_tickets import (
    TicketManager, SupportTicket, TicketStatus, TicketPriority
)


class TestCustomerManager:
    """اختبارات مدير العملاء"""
    
    @pytest.fixture
    def crm(self):
        """إنشاء مدير عملاء للاختبارات"""
        return CustomerManager()
    
    def test_create_customer(self, crm):
        """اختبار إنشاء عميل"""
        customer = crm.create_customer(
            name="شركة التقنية المتقدمة",
            email="info@techcorp.com",
            phone="+966501234567",
            type=CustomerType.COMPANY,
            classification=CustomerClassification.LEAD
        )
        
        assert customer.name == "شركة التقنية المتقدمة"
        assert customer.email == "info@techcorp.com"
        assert customer.type == CustomerType.COMPANY
        assert customer.classification == CustomerClassification.LEAD
        assert customer.customer_code.startswith("CUST-")
    
    def test_add_contact(self, crm):
        """اختبار إضافة جهة اتصال"""
        customer = crm.create_customer("شركة التقنية", type=CustomerType.COMPANY)
        
        contact = customer.add_contact(
            name="أحمد محمد",
            email="ahmed@techcorp.com",
            phone="+966501234567",
            position="مدير المشتريات",
            is_primary=True,
            is_decision_maker=True
        )
        
        assert contact.name == "أحمد محمد"
        assert contact.is_primary is True
        assert contact.is_decision_maker is True
        assert len(customer.contacts) == 1
    
    def test_add_address(self, crm):
        """اختبار إضافة عنوان"""
        customer = crm.create_customer("شركة التقنية")
        
        address = customer.add_address(
            street="شارع الملك فهد",
            city="الرياض",
            postal_code="12345",
            address_type="billing",
            is_default=True
        )
        
        assert address.street == "شارع الملك فهد"
        assert address.city == "الرياض"
        assert address.is_default is True
    
    def test_record_purchase(self, crm):
        """اختبار تسجيل عملية شراء"""
        customer = crm.create_customer(
            "شركة التقنية",
            classification=CustomerClassification.LEAD
        )
        
        customer.record_purchase(Decimal("50000"))
        
        assert customer.total_orders == 1
        assert customer.total_revenue == Decimal("50000")
        assert customer.classification == CustomerClassification.PROSPECT
        assert customer.first_purchase_date == date.today()
    
    def test_lifetime_value(self, crm):
        """اختبار قيمة العميل مدى الحياة"""
        customer = crm.create_customer("شركة التقنية")
        
        customer.record_purchase(Decimal("50000"))
        customer.record_purchase(Decimal("75000"))
        customer.record_purchase(Decimal("25000"))
        
        assert customer.lifetime_value == Decimal("150000")
        assert customer.average_order_value == Decimal("50000")
    
    def test_classify_customer(self, crm):
        """اختبار تصنيف العميل"""
        customer = crm.create_customer("شركة التقنية")
        
        crm.classify_customer(customer.id, CustomerClassification.VIP)
        
        assert customer.classification == CustomerClassification.VIP
    
    def test_enable_portal(self, crm):
        """اختبار تفعيل بوابة العميل"""
        customer = crm.create_customer("شركة التقنية", email="info@techcorp.com")
        
        customer.enable_portal()
        
        assert customer.portal_enabled is True
        assert customer.portal_username == "info@techcorp.com"
    
    def test_search_customers(self, crm):
        """اختبار البحث في العملاء"""
        crm.create_customer("شركة التقنية", email="tech@corp.com")
        crm.create_customer("مؤسسة النور", email="noor@corp.com")
        crm.create_customer("شركة البناء", email="building@corp.com")
        
        results = crm.search_customers("شركة")
        
        assert len(results) == 2
    
    def test_get_top_customers(self, crm):
        """اختبار الحصول على أفضل العملاء"""
        cust1 = crm.create_customer("عميل 1")
        cust1.record_purchase(Decimal("100000"))
        
        cust2 = crm.create_customer("عميل 2")
        cust2.record_purchase(Decimal("50000"))
        
        cust3 = crm.create_customer("عميل 3")
        cust3.record_purchase(Decimal("150000"))
        
        top = crm.get_top_customers(limit=2)
        
        assert len(top) == 2
        assert top[0].id == cust3.id  # Highest value first
    
    def test_get_customer_summary(self, crm):
        """اختبار ملخص العملاء"""
        crm.create_customer("عميل 1", classification=CustomerClassification.CUSTOMER)
        crm.create_customer("عميل 2", classification=CustomerClassification.LEAD)
        crm.create_customer("عميل 3", classification=CustomerClassification.VIP)
        
        summary = crm.get_customer_summary()
        
        assert summary["total_customers"] == 3
        assert summary["by_classification"]["customer"] == 1
        assert summary["by_classification"]["lead"] == 1
        assert summary["by_classification"]["vip"] == 1


class TestSalesPipeline:
    """اختبارات خط أنابيب المبيعات"""
    
    @pytest.fixture
    def pipeline(self):
        """إنشاء خط أنابيب للاختبارات"""
        return SalesPipeline()
    
    def test_create_deal(self, pipeline):
        """اختبار إنشاء صفقة"""
        deal = pipeline.create_deal(
            deal_name="مشروع النظام المحاسبي",
            customer_id="CUST001",
            customer_name="شركة التقنية",
            value=Decimal("150000"),
            owner_id="SALES001"
        )
        
        assert deal.deal_name == "مشروع النظام المحاسبي"
        assert deal.value == Decimal("150000")
        assert deal.stage == PipelineStage.LEAD
        assert deal.probability == 10
    
    def test_add_product_to_deal(self, pipeline):
        """اختبار إضافة منتج للصفقة"""
        deal = pipeline.create_deal("مشروع", "CUST001", "شركة التقنية")
        
        deal.add_product(
            product_id="PROD001",
            product_name="نظام ERP",
            quantity=1,
            unit_price=Decimal("100000"),
            discount=Decimal("10000")
        )
        
        assert len(deal.items) == 1
        assert deal.value == Decimal("90000")  # 100000 - 10000
    
    def test_move_deal_stage(self, pipeline):
        """اختبار نقل الصفقة بين المراحل"""
        deal = pipeline.create_deal("مشروع", "CUST001", "شركة التقنية")
        
        pipeline.move_deal(deal.id, PipelineStage.QUALIFIED)
        assert deal.stage == PipelineStage.QUALIFIED
        assert deal.probability == 25
        
        pipeline.move_deal(deal.id, PipelineStage.PROPOSAL)
        assert deal.stage == PipelineStage.PROPOSAL
        assert deal.probability == 50
        
        pipeline.move_deal(deal.id, PipelineStage.NEGOTIATION)
        assert deal.stage == PipelineStage.NEGOTIATION
        assert deal.probability == 75
    
    def test_close_deal_won(self, pipeline):
        """اختبار إغلاق صفقة بنجاح"""
        deal = pipeline.create_deal("مشروع", "CUST001", "شركة التقنية")
        
        deal.close_won("SALES001")
        
        assert deal.stage == PipelineStage.CLOSED_WON
        assert deal.probability == 100
        assert deal.actual_close_date == date.today()
    
    def test_close_deal_lost(self, pipeline):
        """اختبار إغلاق صفقة بخسارة"""
        deal = pipeline.create_deal("مشروع", "CUST001", "شركة التقنية")
        
        deal.close_lost("Competitor had better price", "SALES001")
        
        assert deal.stage == PipelineStage.CLOSED_LOST
        assert deal.probability == 0
        assert deal.lost_reason == "Competitor had better price"
    
    def test_weighted_value(self, pipeline):
        """اختبار القيمة المرجحة"""
        deal = pipeline.create_deal("مشروع", "CUST001", "شركة التقنية", value=Decimal("100000"))
        deal.probability = 50
        
        assert deal.weighted_value == Decimal("50000")
    
    def test_deal_overdue(self, pipeline):
        """اختبار تحديد صفقة متأخرة"""
        deal = pipeline.create_deal("مشروع", "CUST001", "شركة التقنية")
        deal.expected_close_date = date.today() - timedelta(days=10)
        
        assert deal.is_overdue is True
    
    def test_pipeline_summary(self, pipeline):
        """اختبار ملخص خط الأنابيب"""
        # Create deals in different stages
        deal1 = pipeline.create_deal("صفقة 1", "CUST001", "عميل 1", Decimal("100000"))
        deal1.close_won()
        
        deal2 = pipeline.create_deal("صفقة 2", "CUST002", "عميل 2", Decimal("50000"))
        deal2.close_lost()
        
        deal3 = pipeline.create_deal("صفقة 3", "CUST003", "عميل 3", Decimal("75000"))
        pipeline.move_deal(deal3.id, PipelineStage.NEGOTIATION)
        
        summary = pipeline.get_pipeline_summary()
        
        assert summary["total_deals"] == 3
        assert summary["won_deals"] == 1
        assert summary["lost_deals"] == 1
        assert summary["win_rate"] == 50.0
    
    def test_forecast_report(self, pipeline):
        """اختبار تقرير التوقعات"""
        deal1 = pipeline.create_deal("صفقة 1", "CUST001", "عميل 1", Decimal("100000"))
        deal1.expected_close_date = date.today() + timedelta(days=30)
        pipeline.move_deal(deal1.id, PipelineStage.NEGOTIATION)  # 75% probability
        
        deal2 = pipeline.create_deal("صفقة 2", "CUST002", "عميل 2", Decimal("50000"))
        deal2.expected_close_date = date.today() + timedelta(days=60)
        pipeline.move_deal(deal2.id, PipelineStage.PROPOSAL)  # 50% probability
        
        forecast = pipeline.get_forecast_report(period_months=3)
        
        assert "monthly_forecast" in forecast
        assert forecast["total_forecast"] > 0


class TestTicketManager:
    """اختبارات مدير تذاكر الدعم"""
    
    @pytest.fixture
    def ticket_manager(self):
        """إنشاء مدير تذاكر للاختبارات"""
        return TicketManager()
    
    def test_create_ticket(self, ticket_manager):
        """اختبار إنشاء تذكرة"""
        ticket = ticket_manager.create_ticket(
            customer_id="CUST001",
            customer_name="شركة التقنية",
            subject="مشكلة في النظام",
            description="لا يمكن تسجيل الدخول",
            priority=TicketPriority.HIGH,
            category="technical"
        )
        
        assert ticket.subject == "مشكلة في النظام"
        assert ticket.priority == TicketPriority.HIGH
        assert ticket.status == TicketStatus.NEW
        assert ticket.ticket_number.startswith("TKT-")
        assert ticket.sla_response_deadline is not None
    
    def test_assign_ticket(self, ticket_manager):
        """اختبار تعيين تذكرة"""
        ticket = ticket_manager.create_ticket(
            "CUST001", "شركة التقنية", "مشكلة", "وصف"
        )
        
        assigned = ticket_manager.assign_ticket(
            ticket.id, "AGENT001", "محمد علي", "Support Team"
        )
        
        assert assigned.assigned_to == "AGENT001"
        assert assigned.assigned_team == "Support Team"
        assert assigned.status == TicketStatus.ASSIGNED
    
    def test_add_comment(self, ticket_manager):
        """اختبار إضافة تعليق"""
        ticket = ticket_manager.create_ticket(
            "CUST001", "شركة التقنية", "مشكلة", "وصف"
        )
        
        comment = ticket_manager.add_comment(
            ticket.id, "AGENT001", "محمد علي",
            "جاري العمل على المشكلة"
        )
        
        assert comment.content == "جاري العمل على المشكلة"
        assert ticket.first_response_at is not None
    
    def test_resolve_ticket(self, ticket_manager):
        """اختبار حل تذكرة"""
        ticket = ticket_manager.create_ticket(
            "CUST001", "شركة التقنية", "مشكلة", "وصف"
        )
        
        ticket_manager.update_ticket_status(
            ticket.id, TicketStatus.RESOLVED, "AGENT001"
        )
        
        assert ticket.status == TicketStatus.RESOLVED
        assert ticket.resolved_at is not None
    
    def test_escalate_ticket(self, ticket_manager):
        """اختبار تصعيد تذكرة"""
        ticket = ticket_manager.create_ticket(
            "CUST001", "شركة التقنية", "مشكلة", "وصف",
            priority=TicketPriority.MEDIUM
        )
        
        ticket_manager.escalate_ticket(ticket.id, "AGENT001", "تحتاج خبرة متقدمة")
        
        assert ticket.status == TicketStatus.ESCALATED
        assert ticket.priority == TicketPriority.HIGH  # Escalated by one level
    
    def test_sla_breach(self, ticket_manager):
        """اختبار تجاوز SLA"""
        from datetime import datetime, timedelta
        
        ticket = ticket_manager.create_ticket(
            "CUST001", "شركة التقنية", "مشكلة", "وصف",
            priority=TicketPriority.HIGH
        )
        
        # Simulate past response deadline
        ticket.sla_response_deadline = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Add comment after deadline
        ticket.add_comment("AGENT001", "Agent", "Response", is_internal=False)
        
        assert ticket.sla_breached is True
    
    def test_link_kb_article(self, ticket_manager):
        """اختبار ربط مقالة قاعدة معرفة"""
        ticket = ticket_manager.create_ticket(
            "CUST001", "شركة التقنية", "مشكلة", "وصف"
        )
        
        ticket.link_kb_article("KB-001")
        
        assert ticket.kb_article_id == "KB-001"
    
    def test_customer_satisfaction(self, ticket_manager):
        """اختبار تقييم رضا العميل"""
        ticket = ticket_manager.create_ticket(
            "CUST001", "شركة التقنية", "مشكلة", "وصف"
        )
        
        ticket.set_satisfaction_rating(5, "خدمة ممتازة")
        
        assert ticket.satisfaction_rating == 5
        assert ticket.satisfaction_comment == "خدمة ممتازة"
    
    def test_get_sla_at_risk(self, ticket_manager):
        """اختبار الحصول على التذاكر المهددة بتجاوز SLA"""
        from datetime import datetime, timedelta
        
        # Create ticket with approaching deadline
        ticket = ticket_manager.create_ticket(
            "CUST001", "شركة التقنية", "مشكلة", "وصف",
            priority=TicketPriority.HIGH
        )
        ticket.sla_resolution_deadline = datetime.now(timezone.utc) + timedelta(hours=2)
        
        at_risk = ticket_manager.get_sla_at_risk_tickets(hours=4)
        
        assert len(at_risk) >= 1
    
    def test_ticket_summary(self, ticket_manager):
        """اختبار ملخص التذاكر"""
        ticket_manager.create_ticket(
            "CUST001", "عميل 1", "مشكلة 1", "وصف", TicketPriority.HIGH
        )
        ticket_manager.create_ticket(
            "CUST002", "عميل 2", "مشكلة 2", "وصف", TicketPriority.MEDIUM
        )
        
        summary = ticket_manager.get_ticket_summary()
        
        assert summary["total_tickets"] == 2
        assert summary["by_priority"]["high"] == 1
        assert summary["by_priority"]["medium"] == 1
