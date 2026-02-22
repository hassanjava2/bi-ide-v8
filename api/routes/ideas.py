"""
Ideas Routes - سجل الأفكار
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

from fastapi import APIRouter, HTTPException

from api.schemas import IdeaLedgerUpdateRequest

router = APIRouter(prefix="/api/v1/ideas", tags=["ideas"])

# State
idea_ledger_lock = threading.Lock()
IDEA_LEDGER_FILE = Path("data/knowledge/idea-ledger-v6.json")
idea_ledger_cache: Dict[str, Any] = {
    "version": "1.0.0",
    "generated_at": datetime.now().strftime("%Y-%m-%d"),
    "policy": "code-free-migration",
    "sources": [],
    "ideas": [],
}


def _load_idea_ledger() -> None:
    global idea_ledger_cache
    try:
        IDEA_LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not IDEA_LEDGER_FILE.exists():
            IDEA_LEDGER_FILE.write_text(
                json.dumps(idea_ledger_cache, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            return
        loaded = json.loads(IDEA_LEDGER_FILE.read_text(encoding="utf-8") or "{}")
        if isinstance(loaded, dict):
            loaded.setdefault("version", "1.0.0")
            loaded.setdefault("generated_at", datetime.now().strftime("%Y-%m-%d"))
            loaded.setdefault("policy", "code-free-migration")
            loaded.setdefault("sources", [])
            loaded.setdefault("ideas", [])
            if not isinstance(loaded.get("ideas"), list):
                loaded["ideas"] = []
            idea_ledger_cache = loaded
    except Exception as e:
        print(f"⚠️ Failed to load idea ledger: {e}")


def _persist_idea_ledger() -> None:
    try:
        IDEA_LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
        IDEA_LEDGER_FILE.write_text(
            json.dumps(idea_ledger_cache, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    except Exception as e:
        print(f"⚠️ Failed to persist idea ledger: {e}")


def _get_idea_by_id(idea_id: str):
    for item in idea_ledger_cache.get("ideas", []):
        if str(item.get("idea_id", "")).strip() == idea_id:
            return item
    return None


def init_ideas():
    """Called at startup."""
    _load_idea_ledger()
    print(f"🧾 Idea ledger loaded: {len(idea_ledger_cache.get('ideas', []))} ideas")


@router.get("")
async def ideas_list(status: Optional[str] = None, owner: Optional[str] = None, priority: Optional[str] = None):
    with idea_ledger_lock:
        ideas = list(idea_ledger_cache.get("ideas", []))
        if status:
            ideas = [i for i in ideas if str(i.get("status", "")).lower() == status.lower()]
        if owner:
            ideas = [i for i in ideas if str(i.get("owner", "")).lower() == owner.lower()]
        if priority:
            ideas = [i for i in ideas if str(i.get("priority", "")).lower() == priority.lower()]
        return {"status": "ok", "policy": idea_ledger_cache.get("policy"), "total": len(ideas), "ideas": ideas}


@router.get("/{idea_id}")
async def idea_get(idea_id: str):
    with idea_ledger_lock:
        item = _get_idea_by_id(idea_id)
        if not item:
            raise HTTPException(404, "Idea not found")
        return {"status": "ok", "idea": item}


@router.patch("/{idea_id}")
async def idea_update(idea_id: str, request: IdeaLedgerUpdateRequest):
    with idea_ledger_lock:
        item = _get_idea_by_id(idea_id)
        if not item:
            raise HTTPException(404, "Idea not found")
        updates = request.model_dump(exclude_unset=True)
        for key, value in updates.items():
            if value is not None:
                item[key] = value
        item["updated_at"] = datetime.now().isoformat()
        idea_ledger_cache["generated_at"] = datetime.now().strftime("%Y-%m-%d")
        _persist_idea_ledger()
        return {"status": "ok", "message": "Idea updated", "idea": item}
