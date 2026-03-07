"""
capsule_api.py — API Endpoints للكبسولات بالـ IDE

يوفّر:
  GET  /api/v1/capsules/status       → حالة الكبسولات
  GET  /api/v1/capsules/search?q=... → بحث بالكبسولات
  GET  /api/v1/capsules/tree         → شجرة الوراثة
  GET  /api/v1/capsules/categories   → إحصائيات الفئات
  POST /api/v1/capsules/sync         → مزامنة كاملة
  GET  /api/v1/capsules/sage/{name}  → كبسولات حكيم محدد
"""
from fastapi import APIRouter, Query
from typing import Optional

router = APIRouter(prefix="/api/v1/capsules", tags=["capsules"])


def _get_bridge():
    from brain.capsule_bridge import bridge
    bridge.initialize()
    return bridge


@router.get("/status")
async def capsule_status():
    """حالة الكبسولات الكاملة"""
    bridge = _get_bridge()
    return bridge.get_full_status()


@router.get("/search")
async def capsule_search(q: str = Query(..., min_length=2), top_k: int = 10):
    """بحث بالكبسولات"""
    bridge = _get_bridge()
    results = bridge.find_capsules_for_query(q, top_k=top_k)
    return {"query": q, "count": len(results), "results": results}


@router.get("/categories")
async def capsule_categories():
    """إحصائيات الفئات"""
    from brain.capsule_500 import count_capsules, get_categories
    cats = get_categories()
    return {
        "total": count_capsules(),
        "categories": dict(sorted(cats.items(), key=lambda x: -x[1])),
    }


@router.get("/tree")
async def capsule_tree(root: Optional[str] = None):
    """شجرة الوراثة"""
    bridge = _get_bridge()
    if bridge.tree:
        stats = bridge.tree.stats()
        tree_view = bridge.tree.get_tree_view(root_id=root)
        return {"stats": stats, "tree": tree_view}
    return {"error": "Tree not available"}


@router.post("/sync")
async def capsule_sync():
    """مزامنة كاملة"""
    bridge = _get_bridge()
    result = bridge.sync_all()
    # Remove non-serializable parts
    result.pop("hierarchy", None)
    result.pop("real_life", None)
    return {"status": "synced", "details": result}


@router.get("/sage/{sage_name}")
async def capsule_by_sage(sage_name: str):
    """كبسولات حكيم محدد"""
    bridge = _get_bridge()
    if "hierarchy" not in bridge.layer_connections:
        bridge.connect_to_hierarchy()
    
    sage_caps = bridge.layer_connections.get("hierarchy", {})
    for name, caps in sage_caps.items():
        if sage_name.lower() in name.lower():
            return {"sage": name, "capsule_count": len(caps), "capsules": caps}
    
    return {"error": f"Sage '{sage_name}' not found", "available": list(sage_caps.keys())}
