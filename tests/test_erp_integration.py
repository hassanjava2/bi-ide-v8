"""
ERP Integration Tests - اختبارات تكامل ERP
Tests all ERP modules via ERPDatabaseService
"""
import pytest
import asyncio
from datetime import datetime, timedelta


@pytest.fixture
async def erp_service():
    """Create and initialize ERPDatabaseService for testing."""
    from erp.erp_database_service import ERPDatabaseService
    service = ERPDatabaseService(hierarchy=None)
    await service.initialize()
    yield service


class TestAccounting:
    """Test Accounting Module / اختبارات المحاسبة"""

    @pytest.mark.asyncio
    async def test_create_account(self, erp_service):
        """Test creating an account / إنشاء حساب"""
        account = await erp_service.create_account({
            "code": "9000",
            "name": "Test Cash Account",
            "name_ar": "حساب نقد اختباري",
            "type": "asset",
        })

        assert account is not None
        # Returns ORM object (AccountDB)
        assert account.code == "9000"
        assert account.name == "Test Cash Account"

    @pytest.mark.asyncio
    async def test_chart_of_accounts(self, erp_service):
        """Test chart of accounts / شجرة الحسابات"""
        accounts = await erp_service.get_chart_of_accounts()

        assert isinstance(accounts, list)
        assert len(accounts) > 0

    @pytest.mark.asyncio
    async def test_trial_balance(self, erp_service):
        """Test trial balance report / ميزان المراجعة"""
        trial_balance = await erp_service.get_trial_balance()

        assert trial_balance is not None
        # Actual keys from get_trial_balance
        assert "assets" in trial_balance
        assert "liabilities" in trial_balance
        assert "accounts" in trial_balance


class TestInventory:
    """Test Inventory Module / اختبارات المخزون"""

    @pytest.mark.asyncio
    async def test_create_product(self, erp_service):
        """Test creating a product / إنشاء منتج"""
        product = await erp_service.create_product({
            "sku": "TEST-PROD-001",
            "name": "Test Product",
            "description": "A test product",
            "unit_price": 100.0,
            "quantity": 50,
        })

        assert product is not None
        # Returns ORM object (ProductDB)
        assert product.sku == "TEST-PROD-001"
        assert product.quantity == 50

    @pytest.mark.asyncio
    async def test_get_inventory(self, erp_service):
        """Test getting inventory / الحصول على المخزون"""
        inventory = await erp_service.get_inventory()

        assert isinstance(inventory, list)
        assert len(inventory) > 0

    @pytest.mark.asyncio
    async def test_low_stock_items(self, erp_service):
        """Test low stock detection / تنبيه المخزون المنخفض"""
        low_stock = await erp_service.get_low_stock_items(threshold=99999)

        assert isinstance(low_stock, list)


class TestHR:
    """Test HR Module / اختبارات الموارد البشرية"""

    @pytest.mark.asyncio
    async def test_create_employee(self, erp_service):
        """Test creating an employee / إنشاء موظف"""
        employee = await erp_service.create_employee({
            "employee_number": "EMP-TEST-001",
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.test@example.com",
            "department": "Engineering",
            "position": "Developer",
            "salary": 5000.0,
        })

        assert employee is not None
        # Returns ORM object (EmployeeDB)
        assert employee.first_name == "John"

    @pytest.mark.asyncio
    async def test_get_employees(self, erp_service):
        """Test getting employees / الحصول على الموظفين"""
        employees = await erp_service.get_employees()

        assert isinstance(employees, list)
        assert len(employees) > 0

    @pytest.mark.asyncio
    async def test_process_payroll(self, erp_service):
        """Test payroll processing / معالجة الرواتب"""
        # Get an existing employee
        employees = await erp_service.get_employees()
        if not employees:
            pytest.skip("No employees to test payroll with")

        employee = employees[0]
        emp_id = employee["id"] if isinstance(employee, dict) else employee.id
        payroll = await erp_service.process_payroll(
            employee_id=emp_id,
            month=2,
            year=2026,
        )

        assert payroll is not None
        # PayrollRecord.to_dict() returns net_salary, base_salary (not gross_salary)
        assert "net_salary" in payroll
        assert payroll["net_salary"] > 0


class TestInvoices:
    """Test Invoice Module / اختبارات الفواتير"""

    @pytest.mark.asyncio
    async def test_create_invoice(self, erp_service):
        """Test creating an invoice / إنشاء فاتورة"""
        customers = await erp_service.get_customers()
        customer = customers[0] if customers else None

        cust_name = customer["name"] if isinstance(customer, dict) else (customer.name if customer else "Test Corp")
        cust_id = customer["id"] if isinstance(customer, dict) else (customer.id if customer else None)

        invoice = await erp_service.create_invoice({
            "customer_name": cust_name,
            "customer_id": cust_id,
            "amount": 2000.0,
            "tax": 300.0,
            "total": 2300.0,
            "items": [
                {"name": "Service A", "quantity": 2, "unit_price": 500.0},
                {"name": "Service B", "quantity": 1, "unit_price": 1000.0},
            ],
        })

        assert invoice is not None

    @pytest.mark.asyncio
    async def test_get_invoices(self, erp_service):
        """Test getting invoices / الحصول على الفواتير"""
        invoices = await erp_service.get_invoices()

        assert isinstance(invoices, list)

    @pytest.mark.asyncio
    async def test_mark_invoice_paid(self, erp_service):
        """Test marking invoice as paid / تسديد فاتورة"""
        invoices = await erp_service.get_invoices()
        if not invoices:
            pytest.skip("No invoices to mark as paid")

        # Find an unpaid invoice
        unpaid = None
        for inv in invoices:
            status = inv.get("status") if isinstance(inv, dict) else getattr(inv, "status", None)
            if status != "paid":
                unpaid = inv
                break

        if not unpaid:
            pytest.skip("No unpaid invoices available")

        inv_id = unpaid["id"] if isinstance(unpaid, dict) else unpaid.id
        result = await erp_service.mark_invoice_paid(inv_id)
        assert result is not None


class TestCRM:
    """Test CRM Module / اختبارات إدارة العملاء"""

    @pytest.mark.asyncio
    async def test_create_customer(self, erp_service):
        """Test creating a customer / إنشاء عميل"""
        customer = await erp_service.create_customer({
            "customer_code": "CUST-TEST-001",
            "name": "ABC Test Company",
            "email": "contact@abctest.com",
            "phone": "+1234567890",
        })

        assert customer is not None
        # Returns ORM object or dict depending on implementation
        cust_name = customer["name"] if isinstance(customer, dict) else customer.name
        assert cust_name == "ABC Test Company"

    @pytest.mark.asyncio
    async def test_get_customers(self, erp_service):
        """Test getting customers / الحصول على العملاء"""
        customers = await erp_service.get_customers()

        assert isinstance(customers, list)
        assert len(customers) > 0


class TestERPDashboard:
    """Test ERP Dashboard / اختبارات لوحة التحكم"""

    @pytest.mark.asyncio
    async def test_dashboard_metrics(self, erp_service):
        """Test dashboard metrics / مقاييس لوحة التحكم"""
        dashboard = await erp_service.get_dashboard()

        assert dashboard is not None
        assert "accounting" in dashboard
        assert "inventory" in dashboard
        assert "hr" in dashboard

    @pytest.mark.asyncio
    async def test_financial_report(self, erp_service):
        """Test financial report / التقرير المالي"""
        report = await erp_service.get_financial_report()

        assert report is not None
        assert "chart_of_accounts" in report
