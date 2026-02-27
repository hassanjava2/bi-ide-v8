"""
ERP Module - Enterprise Resource Planning
وحدات إدارة الموارد المؤسسية

Modules:
- accounting: Double-entry accounting
- inventory: Stock management
- hr: Human resources and payroll
- invoices: Billing and invoicing
- crm: Customer relationship management
- dashboard: Analytics and reporting
"""

# IMPORTANT:
# Avoid importing ERP submodules at package import time.
# Many submodules define ORM tables on `core.database.Base.metadata`, which
# can cause duplicate table/index definitions when other modules import
# `erp.models.*`. Use lazy loading instead.

from importlib import import_module
from typing import Dict, Tuple


_SYMBOLS: Dict[str, Tuple[str, str]] = {
    # Accounting
    "create_account": ("erp.accounting", "create_account"),
    "get_account": ("erp.accounting", "get_account"),
    "post_transaction": ("erp.accounting", "post_transaction"),
    "get_trial_balance": ("erp.accounting", "get_trial_balance"),
    "get_balance_sheet": ("erp.accounting", "get_balance_sheet"),
    "Account": ("erp.accounting", "Account"),
    "Transaction": ("erp.accounting", "Transaction"),

    # Inventory
    "create_product": ("erp.inventory", "create_product"),
    "get_product": ("erp.inventory", "get_product"),
    "adjust_stock": ("erp.inventory", "adjust_stock"),
    "get_low_stock_items": ("erp.inventory", "get_low_stock_items"),
    "Product": ("erp.inventory", "Product"),
    "StockMovement": ("erp.inventory", "StockMovement"),

    # HR
    "create_employee": ("erp.hr", "create_employee"),
    "get_employee": ("erp.hr", "get_employee"),
    "process_payroll": ("erp.hr", "process_payroll"),
    "Employee": ("erp.hr", "Employee"),
    "PayrollRecord": ("erp.hr", "PayrollRecord"),

    # Invoices
    "create_invoice": ("erp.invoices", "create_invoice"),
    "get_invoice": ("erp.invoices", "get_invoice"),
    "mark_invoice_paid": ("erp.invoices", "mark_invoice_paid"),
    "Invoice": ("erp.invoices", "Invoice"),
    "InvoiceItem": ("erp.invoices", "InvoiceItem"),

    # CRM
    "create_customer": ("erp.crm", "create_customer"),
    "get_customer": ("erp.crm", "get_customer"),
    "calculate_customer_ltv": ("erp.crm", "calculate_customer_ltv"),
    "Customer": ("erp.crm", "Customer"),

    # Dashboard & Reports
    "get_dashboard_metrics": ("erp.dashboard", "get_dashboard_metrics"),
    "generate_monthly_report": ("erp.reports", "generate_monthly_report"),
}

__all__ = [
    # Accounting
    "create_account", "get_account", "post_transaction",
    "get_trial_balance", "get_balance_sheet",
    "Account", "Transaction",
    # Inventory
    "create_product", "get_product", "adjust_stock",
    "get_low_stock_items",
    "Product", "StockMovement",
    # HR
    "create_employee", "get_employee", "process_payroll",
    "Employee", "PayrollRecord",
    # Invoices
    "create_invoice", "get_invoice", "mark_invoice_paid",
    "Invoice", "InvoiceItem",
    # CRM
    "create_customer", "get_customer", "calculate_customer_ltv",
    "Customer",
    # Dashboard & Reports
    "get_dashboard_metrics", "generate_monthly_report"
]


def __getattr__(name: str):
    spec = _SYMBOLS.get(name)
    if spec is None:
        raise AttributeError(f"module 'erp' has no attribute '{name}'")

    module_name, attr_name = spec
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
