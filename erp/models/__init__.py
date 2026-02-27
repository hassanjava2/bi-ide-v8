"""
ERP Models - نماذج قاعدة البيانات لنظام ERP
"""

from .database_models import (
    # Enums
    AccountTypeDB,
    InvoiceStatusDB,
    EmployeeStatusDB,
    # Accounting
    AccountDB,
    TransactionDB,
    InvoiceDB,
    InvoiceItemDB,
    # Inventory
    ProductDB,
    StockMovementDB,
    # HR
    EmployeeDB,
    PayrollRecordDB,
    # CRM
    CustomerDB,
    SupplierDB,
)

__all__ = [
    "AccountTypeDB",
    "InvoiceStatusDB",
    "EmployeeStatusDB",
    "AccountDB",
    "TransactionDB",
    "InvoiceDB",
    "InvoiceItemDB",
    "ProductDB",
    "StockMovementDB",
    "EmployeeDB",
    "PayrollRecordDB",
    "CustomerDB",
    "SupplierDB",
]
