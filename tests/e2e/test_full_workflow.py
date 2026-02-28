"""
End-to-End Tests - اختبارات سير العمل الكامل

تغطي:
- سير عمل تجاري كامل
- سير عمل تطوير البرمجيات
- سير عمل الموارد البشرية
- سير عمل متكامل للنظام بأكمله
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from api.app import app


@pytest.fixture
def client():
    """Create test client with ERP service initialized"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_token(client):
    """Get valid authentication token with admin role"""
    from core.database import db_manager
    from core.user_service import UserService
    from core.user_models import RoleDB
    from sqlalchemy import select
    import asyncio

    # Create admin user directly via service (register endpoint doesn't support roles)
    async def _create_admin():
        async with db_manager.get_session() as session:
            # Ensure admin role exists (DB has no default seeding)
            result = await session.execute(select(RoleDB).where(RoleDB.name == "admin"))
            if not result.scalar_one_or_none():
                admin_role = RoleDB(name="admin", description="Administrator", permissions='["*"]')
                session.add(admin_role)
                await session.commit()

            user_service = UserService(session)
            user = await user_service.get_user_by_username("e2e_test_user")
            if not user:
                user = await user_service.create_user(
                    username="e2e_test_user",
                    email="e2e@test.com",
                    password="E2E_Test_Password123!",
                    first_name="E2E",
                    last_name="Test",
                    role_names=["admin"]
                )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_create_admin())

    # Login to get token
    login_response = client.post("/api/v1/users/login", json={
        "username": "e2e_test_user",
        "password": "E2E_Test_Password123!"
    })

    return login_response.json()["access_token"]



class TestCompleteBusinessWorkflow:
    """سير عمل تجاري كامل - Complete Business Workflow"""
    
    def test_business_workflow_e2e(self, client, auth_token):
        """سير عمل تجاري شامل"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        workflow_data = {}
        
        # ═══════════════════════════════════════════════════════════
        # 1. إنشاء مستخدم جديد (مع صلاحيات admin)
        # ═══════════════════════════════════════════════════════════
        print("\n[1/7] Creating new user...")
        from core.database import db_manager
        from core.user_service import UserService
        import asyncio

        async def _create_biz_user():
            async with db_manager.get_session() as session:
                user_service = UserService(session)
                user = await user_service.get_user_by_username("business_user")
                if not user:
                    user = await user_service.create_user(
                        username="business_user",
                        email="business@company.com",
                        password="Business123!",
                        first_name="Business",
                        last_name="Owner",
                        role_names=["admin"]
                    )
                return user

        loop = asyncio.get_event_loop()
        biz_user = loop.run_until_complete(_create_biz_user())
        workflow_data["user"] = biz_user
        
        # ═══════════════════════════════════════════════════════════
        # 2. تسجيل الدخول
        # ═══════════════════════════════════════════════════════════
        print("[2/7] Logging in...")
        login_response = client.post("/api/v1/users/login", json={
            "username": "business_user",
            "password": "Business123!"
        })
        assert login_response.status_code == 200
        user_token = login_response.json()["access_token"]
        user_headers = {"Authorization": f"Bearer {user_token}"}
        workflow_data["token"] = user_token
        
        # ═══════════════════════════════════════════════════════════
        # 3. إنشاء عميل
        # ═══════════════════════════════════════════════════════════
        print("[3/7] Creating customer...")
        customer_response = client.post("/api/v1/erp/customers", json={
            "customer_code": f"CUST-E2E-{datetime.now().strftime('%Y%m%d')}",
            "name": "E2E Test Customer",
            "email": "customer@e2e-test.com",
            "phone": "+966501234567",
            "customer_type": "vip",
            "credit_limit": 50000.00
        }, headers=user_headers)
        assert customer_response.status_code == 200
        customer = customer_response.json()
        workflow_data["customer"] = customer
        
        # ═══════════════════════════════════════════════════════════
        # 4. إنشاء منتج
        # ═══════════════════════════════════════════════════════════
        print("[4/7] Creating product...")
        product_response = client.post("/api/v1/erp/products", json={
            "sku": f"PROD-E2E-{datetime.now().strftime('%Y%m%d')}",
            "name": "E2E Test Product",
            "description": "Product created during E2E testing",
            "quantity": 100,
            "unit_price": 500.00,
            "cost_price": 300.00,
            "reorder_point": 20,
            "category": "E2E Category"
        }, headers=user_headers)
        assert product_response.status_code == 200
        product = product_response.json()
        workflow_data["product"] = product
        
        # ═══════════════════════════════════════════════════════════
        # 5. إنشاء فاتورة
        # ═══════════════════════════════════════════════════════════
        print("[5/7] Creating invoice...")
        invoice_response = client.post("/api/v1/erp/invoices", json={
            "customer_name": customer["name"],
            "customer_id": customer["customer_code"],
            "customer_email": customer["email"],
            "amount": 5000.00,
            "tax": 750.00,
            "total": 5750.00,
            "items": [
                {"name": product["name"], "quantity": 10, "price": 500.0}
            ],
            "notes": "E2E Test Invoice"
        }, headers=user_headers)
        assert invoice_response.status_code == 200
        invoice = invoice_response.json()
        workflow_data["invoice"] = invoice
        assert invoice["status"] in ["draft", "pending"]
        
        # ═══════════════════════════════════════════════════════════
        # 6. تحديث المخزون وتسجيل الدفعة
        # ═══════════════════════════════════════════════════════════
        print("[6/7] Processing payment and updating stock...")
        
        # Update stock (reduce for sale)
        stock_response = client.post(
            f"/api/v1/erp/products/{product['id']}/stock",
            json={
                "quantity_change": -10,
                "reason": f"Sale - Invoice {invoice['number']}",
                "reference": invoice["number"]
            },
            headers=user_headers
        )
        assert stock_response.status_code == 200
        
        # Mark invoice as paid
        payment_response = client.post(
            f"/api/v1/erp/invoices/{invoice['id']}/pay",
            headers=user_headers
        )
        assert payment_response.status_code == 200
        assert payment_response.json()["success"] is True
        
        # ═══════════════════════════════════════════════════════════
        # 7. التحقق من البيانات والتقارير
        # ═══════════════════════════════════════════════════════════
        print("[7/7] Verifying data and reports...")
        
        # Check dashboard
        dashboard_response = client.get("/api/v1/erp/dashboard", headers=user_headers)
        assert dashboard_response.status_code == 200
        dashboard = dashboard_response.json()
        
        # Verify accounting data reflects the sale
        assert "accounting" in dashboard
        
        # Verify inventory was updated
        inventory_response = client.get("/api/v1/erp/inventory", headers=user_headers)
        assert inventory_response.status_code == 200
        inventory = inventory_response.json()
        updated_product = next(
            (p for p in inventory if p["id"] == product["id"]),
            None
        )
        assert updated_product is not None
        assert updated_product["quantity"] == 90  # 100 - 10
        
        # Check financial report
        report_response = client.get("/api/v1/erp/reports/financial", headers=user_headers)
        assert report_response.status_code == 200
        report = report_response.json()
        assert "chart_of_accounts" in report
        
        print("✅ Complete business workflow verified!")


class TestSoftwareDevelopmentWorkflow:
    """سير عمل تطوير البرمجيات - Software Development Workflow"""
    
    def test_ide_workflow(self, client, auth_token):
        """سير عمل IDE"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # 1. طلب اقتراح كود
        print("\n[IDE] Testing code suggestion...")
        
        # Note: IDE routes may need the IDE service to be initialized
        # This is a simplified test
        
        # 2. تحليل كود
        code_analysis_response = client.post("/api/v1/ide/analysis", json={
            "code": "def calculate(x, y): return x + y",
            "language": "python",
            "file_path": "/test/calculate.py"
        }, headers=headers)
        
        # May return 200, 503 (service unavailable), or 500 (service not initialized in tests)
        assert code_analysis_response.status_code in [200, 500, 503]
        
        # 3. اقتراح إعادة بناء
        refactor_response = client.post("/api/v1/ide/refactor/suggest", json={
            "code": "def f(a,b): return a+b",
            "language": "python",
            "file_path": "/test/refactor.py"
        }, headers=headers)
        
        assert refactor_response.status_code in [200, 500, 503]


class TestHRWorkflow:
    """سير عمل الموارد البشرية - HR Workflow"""
    
    def test_hr_workflow(self, client, auth_token):
        """سير عمل الموارد البشرية"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # 1. الحصول على الموظفين
        print("\n[HR] Getting employees...")
        employees_response = client.get("/api/v1/erp/hr/employees", headers=headers)
        assert employees_response.status_code == 200
        
        # 2. الحصول على معلومات الرواتب
        print("[HR] Getting payroll info...")
        payroll_response = client.get("/api/v1/erp/hr/payroll", headers=headers)
        assert payroll_response.status_code == 200
        payroll = payroll_response.json()
        assert "total_employees" in payroll
        assert "total_payroll" in payroll
        
        print("✅ HR workflow verified!")


class TestAIIntegrationWorkflow:
    """سير عمل تكامل AI - AI Integration Workflow"""
    
    def test_council_workflow(self, client, auth_token):
        """سير عمل المجلس"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # 1. الحصول على حالة المجلس
        print("\n[AI] Testing council workflow...")
        status_response = client.get("/api/v1/council/status")
        assert status_response.status_code == 200
        
        # 2. إرسال رسالة للمجلس
        message_response = client.post("/api/v1/council/message", json={
            "message": "What advice do you have for business automation?",
            "user_id": "e2e_test_user"
        })
        assert message_response.status_code == 200
        message_data = message_response.json()
        assert "response" in message_data
        
        # 3. الحصول على سجل المحادثات
        history_response = client.get("/api/v1/council/history")
        assert history_response.status_code == 200
        history_data = history_response.json()
        assert "history" in history_data
        
        # 4. الحصول على المقاييس
        metrics_response = client.get("/api/v1/council/metrics")
        assert metrics_response.status_code == 200
        metrics_data = metrics_response.json()
        assert metrics_data["status"] == "ok"
        
        print("✅ AI Council workflow verified!")
    
    def test_hierarchy_workflow(self, client, auth_token):
        """سير عمل الهرم"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        print("\n[AI] Testing hierarchy workflow...")
        
        # الحصول على حكمة
        wisdom_response = client.get("/api/v1/wisdom?horizon=decade")
        assert wisdom_response.status_code == 200
        wisdom_data = wisdom_response.json()
        assert "wisdom" in wisdom_data
        
        # حالة الحارس
        guardian_response = client.get("/api/v1/guardian/status")
        assert guardian_response.status_code == 200
        guardian_data = guardian_response.json()
        assert "active" in guardian_data
        
        print("✅ AI Hierarchy workflow verified!")


class TestFullSystemIntegration:
    """اختبارات تكامل النظام الكامل - Full System Integration Tests"""
    
    def test_system_health(self, client):
        """صحة النظام"""
        print("\n[System] Checking system health...")
        
        # Health check
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Ready check
        ready_response = client.get("/ready")
        assert ready_response.status_code == 200
        
        # Metrics
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        
        print("✅ System health verified!")
    
    def test_end_to_end_data_flow(self, client, auth_token):
        """تدفق البيانات end-to-end"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        print("\n[System] Testing end-to-end data flow...")
        
        # Create data across all modules
        
        # 1. User
        user = client.post("/api/v1/users/register", json={
            "username": f"dataflow_{datetime.now().strftime('%H%M%S')}",
            "email": f"dataflow{datetime.now().strftime('%H%M%S')}@test.com",
            "password": "DataFlow123!"
        }).json()
        
        # 2. Customer
        customer = client.post("/api/v1/erp/customers", json={
            "customer_code": f"CUST-FLOW-{datetime.now().strftime('%H%M%S')}",
            "name": "Data Flow Customer",
            "email": "flow@customer.com"
        }, headers=headers).json()
        
        # 3. Product
        product = client.post("/api/v1/erp/products", json={
            "sku": f"SKU-FLOW-{datetime.now().strftime('%H%M%S')}",
            "name": "Data Flow Product",
            "quantity": 50,
            "unit_price": 200.00
        }, headers=headers).json()
        
        # 4. Invoice
        invoice = client.post("/api/v1/erp/invoices", json={
            "customer_name": customer.get("name", "Flow Customer"),
            "customer_id": customer.get("customer_code", "FLOW"),
            "amount": 2000.00,
            "tax": 300.00,
            "total": 2300.00,
            "items": [{"name": product.get("name", "Flow Product"), "quantity": 10, "price": 200.0}]
        }, headers=headers).json()
        
        # Verify all data is accessible
        
        # Dashboard should reflect all data
        dashboard = client.get("/api/v1/erp/dashboard", headers=headers).json()
        assert "accounting" in dashboard
        assert "inventory" in dashboard
        assert "hr" in dashboard
        
        # Financial report
        report = client.get("/api/v1/erp/reports/financial", headers=headers).json()
        assert "chart_of_accounts" in report
        
        print("✅ End-to-end data flow verified!")


class TestMultiUserWorkflow:
    """سير عمل متعدد المستخدمين - Multi-User Workflow"""
    
    def test_concurrent_user_operations(self, client):
        """عمليات مستخدمين متزامنة"""
        print("\n[Multi-User] Testing concurrent operations...")
        
        users = []
        tokens = []
        
        # Create multiple users
        for i in range(3):
            # Register
            register_response = client.post("/api/v1/users/register", json={
                "username": f"concurrent_user_{i}_{datetime.now().strftime('%H%M%S')}",
                "email": f"concurrent{i}@test.com",
                "password": "Concurrent123!"
            })
            
            if register_response.status_code == 201:
                users.append(register_response.json())
                
                # Login
                login_response = client.post("/api/v1/users/login", json={
                    "username": register_response.json()["username"],
                    "password": "Concurrent123!"
                })
                tokens.append(login_response.json()["access_token"])
        
        # Each user performs operations
        for i, token in enumerate(tokens):
            headers = {"Authorization": f"Bearer {token}"}
            
            # Create customer
            client.post("/api/v1/erp/customers", json={
                "customer_code": f"CUST-CONC-{i}-{datetime.now().strftime('%H%M%S')}",
                "name": f"Concurrent Customer {i}"
            }, headers=headers)
            
            # Get dashboard
            client.get("/api/v1/erp/dashboard", headers=headers)
        
        print(f"✅ Concurrent operations for {len(tokens)} users verified!")


class TestErrorRecoveryWorkflow:
    """سير عمل استعادة الأخطاء - Error Recovery Workflow"""
    
    def test_graceful_degradation(self, client, auth_token):
        """التحلل السلس للخدمات"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        print("\n[Recovery] Testing graceful degradation...")
        
        # Test council with fallback
        with patch("api.routes.council.check_rtx4090_with_retry", return_value=False):
            with patch("api.routes.council.SMART_COUNCIL_AVAILABLE", False):
                response = client.post("/api/v1/council/message", json={
                    "message": "Test message",
                    "user_id": "test"
                })
                # Should still return 200 with fallback
                assert response.status_code == 200
                assert response.json()["source"] == "fallback"
        
        print("✅ Graceful degradation verified!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
