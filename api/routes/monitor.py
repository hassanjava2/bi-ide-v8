"""
مسارات لوحة المراقبة — Monitor Dashboard Routes
/api/v1/monitor — بيانات JSON
/monitor/dashboard — صفحة HTML
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["monitor"])


@router.get("/api/v1/monitor")
async def get_monitor_data():
    """بيانات مراقبة كل الأجهزة — JSON"""
    try:
        from monitoring.multi_machine_monitor import collect_all
        return await collect_all()
    except Exception as e:
        return {"error": str(e), "machines": [], "total": 0, "online": 0}


@router.get("/monitor/dashboard", response_class=HTMLResponse)
async def monitor_dashboard():
    """صفحة لوحة المراقبة — HTML"""
    try:
        from monitoring.multi_machine_monitor import DASHBOARD_HTML
        return HTMLResponse(content=DASHBOARD_HTML)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {e}</h1>")
