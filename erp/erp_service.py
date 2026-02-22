"""
ERP Service - نظام إدارة الموارد المؤسسية
محاسبة + مخزون + موارد بشرية + مبيعات
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio


class InvoiceStatus(Enum):
    PENDING = "معلقة"
    PAID = "مدفوعة"
    OVERDUE = "متأخرة"
    CANCELLED = "ملغاة"


class EmployeeStatus(Enum):
    ACTIVE = "نشط"
    INACTIVE = "غير نشط"
    ON_LEAVE = "في إجازة"
    TERMINATED = "مفصول"


@dataclass
class Invoice:
    """فاتورة"""
    id: str
    invoice_number: str
    customer_id: str
    customer_name: str
    amount: float
    tax: float
    total: float
    status: InvoiceStatus
    items: List[Dict]
    created_at: datetime
    due_date: datetime
    paid_at: Optional[datetime] = None
    notes: str = ""


@dataclass
class InventoryItem:
    """عنصر مخزون"""
    id: str
    sku: str
    name: str
    description: str
    quantity: int
    reorder_point: int
    unit_cost: float
    unit_price: float
    category: str
    supplier: str
    location: str
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class Employee:
    """موظف"""
    id: str
    employee_id: str
    name: str
    email: str
    phone: str
    department: str
    position: str
    salary: float
    join_date: datetime
    status: EmployeeStatus
    attendance: Dict = field(default_factory=dict)


@dataclass
class Transaction:
    """معاملة مالية"""
    id: str
    date: datetime
    type: str  # 'income', 'expense', 'transfer'
    category: str
    amount: float
    description: str
    reference: str


class AccountingManager:
    """
    مدير المحاسبة
    """
    
    def __init__(self):
        self.invoices: Dict[str, Invoice] = {}
        self.transactions: List[Transaction] = []
        self.chart_of_accounts = {
            "assets": {"cash": 0, "accounts_receivable": 0, "inventory": 0},
            "liabilities": {"accounts_payable": 0, "loans": 0},
            "equity": {"capital": 0, "retained_earnings": 0},
            "revenue": {"sales": 0, "services": 0},
            "expenses": {"salaries": 0, "rent": 0, "utilities": 0}
        }
        self._init_sample_data()
    
    def _init_sample_data(self):
        """بيانات نموذجية"""
        # فواتير نموذجية
        for i in range(5):
            invoice = Invoice(
                id=str(uuid.uuid4()),
                invoice_number=f"INV-2024-{1000+i}",
                customer_id=f"CUST-{i}",
                customer_name=["شركة التقنية", "مؤسسة النور", "مكتب المحاماة", "شركة البناء", "مؤسسة التعليم"][i],
                amount=5000 + (i * 1000),
                tax=750 + (i * 150),
                total=5750 + (i * 1150),
                status=[InvoiceStatus.PAID, InvoiceStatus.PENDING, InvoiceStatus.PAID, InvoiceStatus.OVERDUE, InvoiceStatus.PENDING][i],
                items=[
                    {"name": "خدمة استشارية", "quantity": 1, "price": 5000 + (i * 1000)}
                ],
                created_at=datetime.now() - timedelta(days=i*5),
                due_date=datetime.now() + timedelta(days=30-i*5)
            )
            self.invoices[invoice.id] = invoice
    
    def get_dashboard_stats(self) -> Dict:
        """إحصائيات لوحة التحكم"""
        total_sales = sum(inv.total for inv in self.invoices.values() if inv.status == InvoiceStatus.PAID)
        pending = sum(inv.total for inv in self.invoices.values() if inv.status == InvoiceStatus.PENDING)
        overdue = sum(inv.total for inv in self.invoices.values() if inv.status == InvoiceStatus.OVERDUE)
        
        return {
            "total_sales": total_sales,
            "pending_revenue": pending,
            "overdue_amount": overdue,
            "invoice_count": len(self.invoices),
            "paid_invoices": len([i for i in self.invoices.values() if i.status == InvoiceStatus.PAID]),
            "pending_invoices": len([i for i in self.invoices.values() if i.status == InvoiceStatus.PENDING]),
            "overdue_invoices": len([i for i in self.invoices.values() if i.status == InvoiceStatus.OVERDUE])
        }
    
    def get_invoices(self, status: Optional[str] = None) -> List[Invoice]:
        """الحصول على الفواتير"""
        invoices = list(self.invoices.values())
        if status:
            status_enum = InvoiceStatus(status)
            invoices = [i for i in invoices if i.status == status_enum]
        return sorted(invoices, key=lambda x: x.created_at, reverse=True)
    
    def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        """الحصول على فاتورة واحدة"""
        return self.invoices.get(invoice_id)
    
    def create_invoice(self, data: Dict) -> Invoice:
        """إنشاء فاتورة جديدة"""
        invoice = Invoice(
            id=str(uuid.uuid4()),
            invoice_number=f"INV-{datetime.now().year}-{1000+len(self.invoices)}",
            customer_id=data.get("customer_id", ""),
            customer_name=data.get("customer_name", ""),
            amount=data.get("amount", 0),
            tax=data.get("tax", 0),
            total=data.get("total", 0),
            status=InvoiceStatus.PENDING,
            items=data.get("items", []),
            created_at=datetime.now(),
            due_date=datetime.now() + timedelta(days=30),
            notes=data.get("notes", "")
        )
        self.invoices[invoice.id] = invoice
        return invoice
    
    def mark_paid(self, invoice_id: str) -> bool:
        """تحديد فاتورة كمدفوعة"""
        if invoice_id in self.invoices:
            self.invoices[invoice_id].status = InvoiceStatus.PAID
            self.invoices[invoice_id].paid_at = datetime.now()
            return True
        return False
    
    def get_financial_report(self, period: str = "month") -> Dict:
        """تقرير مالي"""
        return {
            "period": period,
            "total_revenue": sum(inv.total for inv in self.invoices.values() if inv.status == InvoiceStatus.PAID),
            "outstanding": sum(inv.total for inv in self.invoices.values() if inv.status != InvoiceStatus.PAID),
            "chart_of_accounts": self.chart_of_accounts,
            "trends": {
                "revenue_growth": 15.5,
                "expense_growth": 8.2,
                "profit_margin": 42.3
            }
        }


class InventoryManager:
    """
    مدير المخزون
    """
    
    def __init__(self):
        self.items: Dict[str, InventoryItem] = {}
        self._init_sample_data()
    
    def _init_sample_data(self):
        """بيانات نموذجية"""
        items = [
            ("LAPTOP-001", "لابتوب Dell XPS", 15, 5, 3500, 5000, "إلكترونيات"),
            ("MOUSE-001", "ماوس لاسلكي", 50, 10, 25, 45, "إكسسوارات"),
            ("KEYBOARD-001", "كيبورد ميكانيكي", 20, 5, 150, 250, "إكسسوارات"),
            ("MONITOR-001", "شاشة 27 بوصة", 8, 3, 1200, 1800, "إلكترونيات"),
            ("WEBCAM-001", "كاميرا ويب", 30, 8, 80, 120, "إكسسوارات"),
        ]
        
        for sku, name, qty, reorder, cost, price, category in items:
            item = InventoryItem(
                id=str(uuid.uuid4()),
                sku=sku,
                name=name,
                description=f"{name} - عالي الجودة",
                quantity=qty,
                reorder_point=reorder,
                unit_cost=cost,
                unit_price=price,
                category=category,
                supplier="المورد الرئيسي",
                location="المستودع A"
            )
            self.items[item.id] = item
    
    def get_all_items(self) -> List[InventoryItem]:
        """كل العناصر"""
        return list(self.items.values())
    
    def get_low_stock(self) -> List[InventoryItem]:
        """عناصر مخزون منخفض"""
        return [item for item in self.items.values() if item.quantity <= item.reorder_point]
    
    def get_item(self, item_id: str) -> Optional[InventoryItem]:
        """الحصول على عنصر"""
        return self.items.get(item_id)
    
    def update_stock(self, item_id: str, quantity_change: int) -> bool:
        """تحديث المخزون"""
        if item_id in self.items:
            self.items[item_id].quantity += quantity_change
            self.items[item_id].last_updated = datetime.now()
            return True
        return False
    
    def get_inventory_value(self) -> Dict:
        """قيمة المخزون"""
        total_cost = sum(item.quantity * item.unit_cost for item in self.items.values())
        total_value = sum(item.quantity * item.unit_price for item in self.items.values())
        
        return {
            "total_items": len(self.items),
            "total_quantity": sum(item.quantity for item in self.items.values()),
            "total_cost": total_cost,
            "total_value": total_value,
            "potential_profit": total_value - total_cost,
            "low_stock_count": len(self.get_low_stock())
        }


class HRManager:
    """
    مدير الموارد البشرية
    """
    
    def __init__(self):
        self.employees: Dict[str, Employee] = {}
        self.departments = ["IT", "المحاسبة", "المبيعات", "الموارد البشرية", "الإدارة"]
        self._init_sample_data()
    
    def _init_sample_data(self):
        """بيانات نموذجية"""
        employees = [
            ("EMP-001", "أحمد محمد", "IT", "مطور", 8000),
            ("EMP-002", "سارة علي", "المحاسبة", "محاسب", 6500),
            ("EMP-003", "خالد العمر", "المبيعات", "مندوب مبيعات", 5500),
            ("EMP-004", "نورة سعد", "الموارد البشرية", "مسؤول موارد بشرية", 7000),
            ("EMP-005", "محمد عبدالله", "الإدارة", "مدير", 15000),
        ]
        
        for emp_id, name, dept, pos, salary in employees:
            emp = Employee(
                id=str(uuid.uuid4()),
                employee_id=emp_id,
                name=name,
                email=f"{name.split()[0].lower()}@company.com",
                phone="05xxxxxxxx",
                department=dept,
                position=pos,
                salary=salary,
                join_date=datetime.now() - timedelta(days=365),
                status=EmployeeStatus.ACTIVE
            )
            self.employees[emp.id] = emp
    
    def get_all_employees(self) -> List[Employee]:
        """كل الموظفين"""
        return list(self.employees.values())
    
    def get_employee(self, emp_id: str) -> Optional[Employee]:
        """الحصول على موظف"""
        return self.employees.get(emp_id)
    
    def get_department_stats(self) -> Dict:
        """إحصائيات الأقسام"""
        stats = {}
        for dept in self.departments:
            dept_emps = [e for e in self.employees.values() if e.department == dept]
            stats[dept] = {
                "count": len(dept_emps),
                "total_salary": sum(e.salary for e in dept_emps)
            }
        return stats
    
    def calculate_payroll(self) -> Dict:
        """حساب الرواتب"""
        total_salary = sum(emp.salary for emp in self.employees.values() if emp.status == EmployeeStatus.ACTIVE)
        
        return {
            "total_employees": len(self.employees),
            "active_employees": len([e for e in self.employees.values() if e.status == EmployeeStatus.ACTIVE]),
            "total_payroll": total_salary,
            "average_salary": total_salary / len(self.employees) if self.employees else 0,
            "payroll_date": (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1).strftime("%Y-%m-%d")
        }


class ERPService:
    """
    خدمة ERP الرئيسية
    تجمع كل المكونات
    """
    
    def __init__(self, hierarchy=None):
        self.accounting = AccountingManager()
        self.inventory = InventoryManager()
        self.hr = HRManager()
        self.hierarchy = hierarchy
        print("🏢 ERP Service initialized")
    
    def get_dashboard(self) -> Dict:
        """لوحة تحكم ERP"""
        return {
            "accounting": self.accounting.get_dashboard_stats(),
            "inventory": self.inventory.get_inventory_value(),
            "hr": self.hr.calculate_payroll(),
            "alerts": self._get_alerts()
        }
    
    def _get_alerts(self) -> List[Dict]:
        """تنبيهات النظام"""
        alerts = []
        
        # تنبيهات المخزون المنخفض
        low_stock = self.inventory.get_low_stock()
        if low_stock:
            alerts.append({
                "type": "inventory",
                "severity": "warning",
                "message": f"{len(low_stock)} عناصر مخزون منخفض",
                "items": [item.name for item in low_stock[:3]]
            })
        
        # تنبيهات الفواتير المتأخرة
        overdue = [i for i in self.accounting.invoices.values() if i.status == InvoiceStatus.OVERDUE]
        if overdue:
            alerts.append({
                "type": "accounting",
                "severity": "danger",
                "message": f"{len(overdue)} فواتير متأخرة",
                "amount": sum(i.total for i in overdue)
            })
        
        return alerts
    
    async def get_ai_insights(self) -> Dict:
        """رؤى AI للـ ERP"""
        if not self.hierarchy:
            return {"status": "AI not connected"}
        
        # طلب تحليل من الـ Domain Expert
        try:
            result = await self.hierarchy.experts.route_query(
                "تحليل بيانات ERP",
                {
                    "sales": self.accounting.get_dashboard_stats(),
                    "inventory": self.inventory.get_inventory_value(),
                    "payroll": self.hr.calculate_payroll()
                }
            )
            return result
        except:
            return {
                "recommendations": [
                    "زيادة المخزون للمنتجات الأكثر مبيعاً",
                    "متابعة الفواتير المتأخرة",
                    "مراجعة تكاليف التوظيف"
                ]
            }


# Singleton
erp_service = None

def get_erp_service(hierarchy=None):
    global erp_service
    if erp_service is None:
        erp_service = ERPService(hierarchy)
    return erp_service
