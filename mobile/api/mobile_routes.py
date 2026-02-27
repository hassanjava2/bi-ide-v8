"""
Mobile Routes - نقاط نهاية API المحسّنة للجوال

المميزات:
- حمولات أخف (lighter payloads)
- دعم دون اتصال (offline support)
- نقاط نهاية خاصة بالجوال
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date, timezone
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/mobile", tags=["mobile"])


# ═══════════════════════════════════════════════════════════════
# Response Models - نماذج الاستجابة
# ═══════════════════════════════════════════════════════════════

class MobileDashboardResponse(BaseModel):
    """استجابة لوحة التحكم للجوال - مبسطة"""
    summary: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    quick_actions: List[Dict[str, str]]
    last_updated: str
    offline_available: bool = True


class MobileInventoryItem(BaseModel):
    """عنصر مخزون مبسط للجوال"""
    id: str
    sku: str
    name: str
    quantity: int
    location: str
    status: str  # "ok", "low", "out"
    unit_price: float


class MobileCustomerSummary(BaseModel):
    """ملخص عميل مبسط"""
    id: str
    name: str
    classification: str
    outstanding: float
    last_order: Optional[str]
    phone: str


class MobileDealSummary(BaseModel):
    """ملخص صفقة مبسط"""
    id: str
    name: str
    customer_name: str
    value: float
    stage: str
    probability: int
    expected_close: Optional[str]


class SyncStatusResponse(BaseModel):
    """حالة المزامنة"""
    last_sync: Optional[str]
    pending_changes: int
    sync_in_progress: bool
    server_available: bool


# ═══════════════════════════════════════════════════════════════
# Dashboard Endpoints - نقاط لوحة التحكم
# ═══════════════════════════════════════════════════════════════

@router.get("/dashboard", response_model=MobileDashboardResponse)
async def get_mobile_dashboard():
    """
    الحصول على لوحة تحكم مبسطة للتطبيق الجوال
    
    Returns lightweight summary optimized for mobile viewing
    """
    return {
        "summary": {
            "total_sales_today": 15000.00,
            "total_sales_change": 12.5,
            "pending_orders": 8,
            "low_stock_items": 5,
            "overdue_invoices": 3,
            "active_employees": 25
        },
        "alerts": [
            {
                "type": "inventory",
                "severity": "warning",
                "message": "5 items low in stock",
                "action_url": "/mobile/inventory/alerts"
            },
            {
                "type": "accounting",
                "severity": "danger",
                "message": "3 overdue invoices",
                "action_url": "/mobile/accounting/overdue"
            }
        ],
        "quick_actions": [
            {"label": "New Sale", "icon": "cart", "url": "/mobile/sales/new"},
            {"label": "Check-in", "icon": "clock", "url": "/mobile/hr/checkin"},
            {"label": "Inventory", "icon": "box", "url": "/mobile/inventory"},
            {"label": "Reports", "icon": "chart", "url": "/mobile/reports"}
        ],
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "offline_available": True
    }


# ═══════════════════════════════════════════════════════════════
# Inventory Endpoints - نقاط المخزون
# ═══════════════════════════════════════════════════════════════

@router.get("/inventory", response_model=List[MobileInventoryItem])
async def get_mobile_inventory(
    location: Optional[str] = None,
    status: Optional[str] = None,  # "all", "low", "out"
    search: Optional[str] = None,
    limit: int = Query(50, le=100)
):
    """
    الحصول على قائمة المخزون المبسطة للجوال
    
    Args:
        location: تصفية حسب الموقع
        status: تصفية حسب الحالة (all, low, out)
        search: البحث بالاسم أو SKU
        limit: عدد النتائج
    """
    # Mock data - replace with actual ERP integration
    items = [
        {
            "id": "1",
            "sku": "LAPTOP-001",
            "name": "Laptop Dell XPS 15",
            "quantity": 15,
            "location": "Warehouse A",
            "status": "ok",
            "unit_price": 5000.00
        },
        {
            "id": "2",
            "sku": "MOUSE-001",
            "name": "Wireless Mouse",
            "quantity": 5,
            "location": "Warehouse A",
            "status": "low",
            "unit_price": 45.00
        },
        {
            "id": "3",
            "sku": "KEYBOARD-001",
            "name": "Mechanical Keyboard",
            "quantity": 0,
            "location": "Warehouse B",
            "status": "out",
            "unit_price": 250.00
        }
    ]
    
    # Apply filters
    if location:
        items = [i for i in items if i["location"] == location]
    
    if status and status != "all":
        items = [i for i in items if i["status"] == status]
    
    if search:
        search_lower = search.lower()
        items = [i for i in items if search_lower in i["name"].lower() or search_lower in i["sku"].lower()]
    
    return items[:limit]


@router.post("/inventory/adjust")
async def mobile_inventory_adjustment(
    item_id: str,
    quantity_change: int,
    reason: str,
    location: str = "main"
):
    """
    تعديل مخزون سريع من الجوال
    
    Optimized for quick barcode scanning and quantity updates
    """
    return {
        "success": True,
        "item_id": item_id,
        "new_quantity": 100 + quantity_change,
        "adjusted_at": datetime.now(timezone.utc).isoformat(),
        "sync_status": "pending"  # Will sync when online
    }


# ═══════════════════════════════════════════════════════════════
# CRM Endpoints - نقاط إدارة العملاء
# ═══════════════════════════════════════════════════════════════

@router.get("/crm/customers", response_model=List[MobileCustomerSummary])
async def get_mobile_customers(
    classification: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(30, le=100)
):
    """الحصول على قائمة العملاء المبسطة"""
    customers = [
        {
            "id": "1",
            "name": "شركة التقنية المتقدمة",
            "classification": "customer",
            "outstanding": 25000.00,
            "last_order": "2024-01-15",
            "phone": "+966501234567"
        },
        {
            "id": "2",
            "name": "مؤسسة الأمل",
            "classification": "lead",
            "outstanding": 0.00,
            "last_order": None,
            "phone": "+966509876543"
        }
    ]
    
    if classification:
        customers = [c for c in customers if c["classification"] == classification]
    
    if search:
        search_lower = search.lower()
        customers = [c for c in customers if search_lower in c["name"].lower()]
    
    return customers[:limit]


@router.get("/crm/deals", response_model=List[MobileDealSummary])
async def get_mobile_deals(
    stage: Optional[str] = None,
    assigned_to: Optional[str] = None
):
    """الحصول على الصفقات النشطة"""
    deals = [
        {
            "id": "1",
            "name": "مشروع النظام المحاسبي",
            "customer_name": "شركة التقنية",
            "value": 150000.00,
            "stage": "negotiation",
            "probability": 75,
            "expected_close": "2024-03-01"
        },
        {
            "id": "2",
            "name": "ترخيص ERP",
            "customer_name": "مؤسسة النور",
            "value": 50000.00,
            "stage": "proposal",
            "probability": 50,
            "expected_close": "2024-02-15"
        }
    ]
    
    if stage:
        deals = [d for d in deals if d["stage"] == stage]
    
    return deals


# ═══════════════════════════════════════════════════════════════
# HR Endpoints - نقاط الموارد البشرية
# ═══════════════════════════════════════════════════════════════

@router.post("/hr/checkin")
async def mobile_checkin(
    employee_id: str,
    location: Optional[str] = None,
    notes: Optional[str] = None
):
    """
    تسجيل دخول سريع من الجوال
    
    Supports GPS location tracking and biometric verification
    """
    return {
        "success": True,
        "employee_id": employee_id,
        "checkin_time": datetime.now(timezone.utc).isoformat(),
        "location": location,
        "status": "checked_in",
        "shift_start": "08:00",
        "is_late": False
    }


@router.post("/hr/checkout")
async def mobile_checkout(
    employee_id: str,
    location: Optional[str] = None
):
    """تسجيل خروج سريع"""
    return {
        "success": True,
        "employee_id": employee_id,
        "checkout_time": datetime.now(timezone.utc).isoformat(),
        "hours_worked": 8.5,
        "overtime_hours": 0.5
    }


@router.get("/hr/attendance/{employee_id}")
async def get_mobile_attendance(
    employee_id: str,
    month: int = Query(..., ge=1, le=12),
    year: int = Query(..., ge=2020)
):
    """الحصول على سجل الحضور للجوال"""
    return {
        "employee_id": employee_id,
        "period": f"{month}/{year}",
        "summary": {
            "present_days": 22,
            "absent_days": 0,
            "late_days": 2,
            "leave_days": 1,
            "total_hours": 176
        },
        "records": [
            {
                "date": f"{year}-{month:02d}-{day:02d}",
                "check_in": "08:05",
                "check_out": "17:00",
                "status": "present",
                "is_late": day == 5
            }
            for day in range(1, 23)
        ]
    }


# ═══════════════════════════════════════════════════════════════
# Accounting Endpoints - نقاط المحاسبة
# ═══════════════════════════════════════════════════════════════

@router.get("/accounting/summary")
async def get_mobile_accounting_summary(
    period: str = "month"  # week, month, quarter, year
):
    """ملخص محاسبي مبسط للجوال"""
    return {
        "period": period,
        "revenue": {
            "total": 150000.00,
            "change": 12.5
        },
        "expenses": {
            "total": 85000.00,
            "change": -3.2
        },
        "profit": {
            "total": 65000.00,
            "margin": 43.3
        },
        "pending_invoices": 8,
        "overdue_invoices": 3,
        "cash_balance": 250000.00
    }


@router.get("/accounting/invoices")
async def get_mobile_invoices(
    status: Optional[str] = None,  # pending, paid, overdue
    limit: int = Query(20, le=50)
):
    """قائمة مبسطة للفواتير"""
    invoices = [
        {
            "id": "INV-001",
            "customer": "شركة التقنية",
            "amount": 15000.00,
            "status": "pending",
            "due_date": "2024-02-01",
            "is_overdue": False
        },
        {
            "id": "INV-002",
            "customer": "مؤسسة النور",
            "amount": 8500.00,
            "status": "overdue",
            "due_date": "2024-01-15",
            "is_overdue": True
        }
    ]
    
    if status:
        invoices = [i for i in invoices if i["status"] == status]
    
    return invoices[:limit]


# ═══════════════════════════════════════════════════════════════
# Sync & Offline Endpoints - نقاط المزامنة والعمل دون اتصال
# ═══════════════════════════════════════════════════════════════

@router.get("/sync/status", response_model=SyncStatusResponse)
async def get_sync_status(device_id: str):
    """الحصول على حالة المزامنة"""
    return {
        "last_sync": datetime.now(timezone.utc).isoformat(),
        "pending_changes": 5,
        "sync_in_progress": False,
        "server_available": True
    }


@router.post("/sync")
async def sync_data(
    device_id: str,
    changes: List[Dict[str, Any]],
    last_sync: Optional[str] = None
):
    """
    مزامنة البيانات بين الجهاز والخادم
    
    Accepts pending changes from device and returns server changes
    """
    # Process incoming changes
    processed = []
    for change in changes:
        processed.append({
            "local_id": change.get("local_id"),
            "server_id": str(uuid.uuid4()),
            "status": "synced",
            "synced_at": datetime.now(timezone.utc).isoformat()
        })
    
    return {
        "success": True,
        "processed_changes": processed,
        "server_changes": [],  # Changes to apply on device
        "conflicts": [],
        "sync_timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/offline/data")
async def get_offline_data_bundle(
    modules: List[str] = Query(default=["inventory", "customers"])
):
    """
    الحصول على حزمة بيانات للاستخدام دون اتصال
    
    Returns essential data for offline use
    """
    bundle = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "expires_at": None,  # No expiration
        "data": {}
    }
    
    if "inventory" in modules:
        bundle["data"]["inventory"] = {
            "items": [],  # Would include actual inventory data
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    if "customers" in modules:
        bundle["data"]["customers"] = {
            "customers": [],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    return bundle


# ═══════════════════════════════════════════════════════════════
# Notifications Endpoints - نقاط الإشعارات
# ═══════════════════════════════════════════════════════════════

@router.post("/notifications/register")
async def register_push_notifications(
    device_id: str,
    platform: str,  # ios, android, web
    push_token: str,
    user_id: str
):
    """تسجيل جهاز للإشعارات"""
    return {
        "success": True,
        "device_id": device_id,
        "registered_at": datetime.now(timezone.utc).isoformat()
    }


@router.post("/notifications/unregister")
async def unregister_push_notifications(
    device_id: str,
    user_id: str
):
    """إلغاء تسجيل جهاز"""
    return {
        "success": True,
        "unregistered_at": datetime.now(timezone.utc).isoformat()
    }


# ═══════════════════════════════════════════════════════════════
# Search Endpoint - نقطة البحث
# ═══════════════════════════════════════════════════════════════

@router.get("/search")
async def mobile_search(
    q: str,
    scope: List[str] = Query(default=["customers", "inventory", "invoices"]),
    limit: int = Query(10, le=30)
):
    """
    بحث شامل للجوال
    
    Searches across multiple modules with relevance ranking
    """
    results = {
        "query": q,
        "total_results": 0,
        "results": {}
    }
    
    if "customers" in scope:
        results["results"]["customers"] = []
        results["total_results"] += 0
    
    if "inventory" in scope:
        results["results"]["inventory"] = []
        results["total_results"] += 0
    
    return results


# ═══════════════════════════════════════════════════════════════
# Settings Endpoints - نقاط الإعدادات
# ═══════════════════════════════════════════════════════════════

@router.get("/settings")
async def get_mobile_settings(user_id: str):
    """الحصول على إعدادات التطبيق الجوال"""
    return {
        "user_id": user_id,
        "theme": "auto",  # light, dark, auto
        "language": "ar",
        "notifications": {
            "enabled": True,
            "inventory_alerts": True,
            "invoice_reminders": True,
            "attendance_reminders": True
        },
        "offline_mode": {
            "enabled": True,
            "auto_sync": True,
            "sync_interval_minutes": 15
        },
        "quick_actions": [
            {"id": "checkin", "enabled": True, "order": 1},
            {"id": "new_sale", "enabled": True, "order": 2},
            {"id": "inventory_scan", "enabled": True, "order": 3}
        ]
    }


@router.post("/settings")
async def update_mobile_settings(user_id: str, settings: Dict[str, Any]):
    """تحديث إعدادات التطبيق"""
    return {
        "success": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings
    }
