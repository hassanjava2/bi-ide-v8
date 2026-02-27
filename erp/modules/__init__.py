# ERP Modules Package
"""
ERP Modules - وحدات نظام إدارة الموارد المؤسسية

Modules:
- accounting: المحاسبة والتقارير المالية
- inventory: إدارة المخزون والمشتريات
- hr: الموارد البشرية والرواتب
- crm: إدارة العملاء والمبيعات
"""

from .accounting.ledger import GeneralLedger, ChartOfAccounts
from .accounting.accounts import AccountsPayable, AccountsReceivable
from .accounting.reports import FinancialReportGenerator

from .inventory.stock import StockManager
from .inventory.purchase_orders import PurchaseOrder, PurchaseOrderManager
from .inventory.suppliers import Supplier, SupplierManager

from .hr.employees import Employee, EmployeeManager
from .hr.payroll import PayrollProcessor
from .hr.attendance import AttendanceTracker

from .crm.customers import Customer, CustomerManager
from .crm.sales_pipeline import SalesPipeline, Deal
from .crm.support_tickets import SupportTicket, TicketManager

__all__ = [
    # Accounting
    'GeneralLedger', 'ChartOfAccounts',
    'AccountsPayable', 'AccountsReceivable',
    'FinancialReportGenerator',
    # Inventory
    'StockManager',
    'PurchaseOrder', 'PurchaseOrderManager',
    'Supplier', 'SupplierManager',
    # HR
    'Employee', 'EmployeeManager',
    'PayrollProcessor',
    'AttendanceTracker',
    # CRM
    'Customer', 'CustomerManager',
    'SalesPipeline', 'Deal',
    'SupportTicket', 'TicketManager',
]
