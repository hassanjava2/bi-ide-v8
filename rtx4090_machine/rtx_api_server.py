"""
RTX 5090 AI API Server — Lightweight FastAPI on port 8090
Direct LAN access for desktop apps without going through VPS.

Endpoints:
  POST /council/message — Send message to AI hierarchy
  GET  /health          — Health check
"""

import sys
import os
import time
from pathlib import Path

# Add worker path for hierarchy imports
WORKER_DIR = Path("/home/bi/.bi-ide-worker")
if WORKER_DIR.exists():
    sys.path.insert(0, str(WORKER_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

app = FastAPI(title="BI-IDE RTX AI Server", version="1.0.0")

# Allow all origins (LAN only, no public exposure)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MessageResponse(BaseModel):
    response: str
    source: str = "rtx5090"
    confidence: float = 0.85
    evidence: list = Field(default_factory=list)
    response_source: str = "rtx5090-direct"
    wise_man: str = "المجلس"
    processing_time_ms: int = 0
    timestamp: str = ""

# Lazy-load hierarchy (heavy import)
_hierarchy = None

def get_hierarchy():
    global _hierarchy
    if _hierarchy is None:
        try:
            from hierarchy import ai_hierarchy
            _hierarchy = ai_hierarchy
            print("✅ AI Hierarchy loaded successfully")
        except Exception as e:
            print(f"⚠️ Hierarchy import failed: {e}")
            _hierarchy = None
    return _hierarchy


@app.get("/health")
async def health():
    h = get_hierarchy()
    return {
        "status": "healthy",
        "service": "rtx-ai-server",
        "gpu_available": True,
        "hierarchy_loaded": h is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/council/message", response_model=MessageResponse)
async def council_message(request: MessageRequest):
    """Send message to AI hierarchy — direct RTX path."""
    start = time.time()
    hierarchy = get_hierarchy()

    if hierarchy is None:
        return MessageResponse(
            response="عذراً، نظام AI غير متاح حالياً على هذا الجهاز.",
            source="rtx5090-error",
            confidence=0.0,
            response_source="rtx5090-error",
            wise_man="النظام",
            processing_time_ms=0,
            timestamp=datetime.now().isoformat(),
        )

    try:
        result = hierarchy.ask(request.message)
        processing_time = int((time.time() - start) * 1000)

        return MessageResponse(
            response=result.get("response", ""),
            source="rtx5090",
            confidence=result.get("confidence", 0.85),
            evidence=result.get("evidence", []),
            response_source="rtx5090-direct",
            wise_man=result.get("wise_man", "المجلس"),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        return MessageResponse(
            response=f"خطأ في معالجة الطلب: {str(e)}",
            source="rtx5090-error",
            confidence=0.0,
            response_source="rtx5090-error",
            wise_man="النظام",
            processing_time_ms=int((time.time() - start) * 1000),
            timestamp=datetime.now().isoformat(),
        )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RTX AI API Server on 0.0.0.0:8090")
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")
