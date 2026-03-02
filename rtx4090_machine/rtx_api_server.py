"""
RTX 5090 AI API Server — FastAPI on port 8090
Direct LAN access for desktop apps + training data receiver.

Endpoints:
  POST /council/message          — Send message to AI hierarchy
  GET  /health                   — Health check
  POST /api/v1/training-data/ingest — Receive training samples
  GET  /api/v1/training/status   — Training status & GPU info
  POST /api/v1/training/start    — Start training manually
"""

import sys
import os
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add worker path for hierarchy imports
WORKER_DIR = Path("/home/bi/.bi-ide-worker")
if WORKER_DIR.exists():
    sys.path.insert(0, str(WORKER_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime

app = FastAPI(title="BI-IDE RTX AI Server", version="8.0.1")

# Allow all origins (LAN only, no public exposure)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Training Data Storage ───────────────────────────────────────
TRAINING_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/home/bi/.bi-ide-worker/training_data"))
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
AUTO_TRAIN_THRESHOLD = 50  # start training after N samples

# ─── Models ──────────────────────────────────────────────────────
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

class TrainingSample(BaseModel):
    source: str = "desktop"
    kind: str = "chat_pair"
    input_text: str
    output_text: Optional[str] = None
    workspace_path: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp_ms: Optional[int] = None

class TrainingIngestRequest(BaseModel):
    samples: List[TrainingSample] = Field(default_factory=list)
    relay: bool = False  # RTX is the final destination
    store_local: bool = True

# ─── Lazy-load hierarchy ─────────────────────────────────────────
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

# ─── Training state ──────────────────────────────────────────────
_training_active = False
_training_stats = {
    "total_samples": 0,
    "last_ingest": None,
    "training_runs": 0,
    "last_training": None,
}

def _count_samples() -> int:
    f = TRAINING_DIR / "samples.jsonl"
    if not f.exists():
        return 0
    with f.open() as fh:
        return sum(1 for _ in fh)

def _auto_train_check():
    """Start training if enough samples accumulated."""
    global _training_active
    count = _count_samples()
    if count >= AUTO_TRAIN_THRESHOLD and not _training_active:
        print(f"🚀 Auto-training triggered ({count} samples >= {AUTO_TRAIN_THRESHOLD})")
        _start_training()

def _start_training():
    """Launch training in background thread."""
    global _training_active
    if _training_active:
        return
    _training_active = True
    t = threading.Thread(target=_run_training, daemon=True)
    t.start()

def _run_training():
    """Actual training loop — uses real_training_system if available."""
    global _training_active, _training_stats
    try:
        print("🧠 Training started on RTX 5090...")
        try:
            from hierarchy.real_training_system import training_system
            training_system.start_all()
            # Train for 5 minutes then pause
            time.sleep(300)
            training_system.stop_all()
            status = training_system.get_status()
            _training_stats["last_training"] = datetime.now().isoformat()
            _training_stats["training_runs"] += 1
            print(f"✅ Training completed: {status}")
        except ImportError:
            print("⚠️ real_training_system not available, training skipped")
            time.sleep(1)
    except Exception as e:
        print(f"❌ Training error: {e}")
    finally:
        _training_active = False

# ─── Endpoints ───────────────────────────────────────────────────

@app.get("/health")
async def health():
    h = get_hierarchy()
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            }
    except ImportError:
        pass

    return {
        "status": "healthy",
        "service": "rtx-ai-server",
        "version": "8.0.1",
        "gpu": gpu_info,
        "hierarchy_loaded": h is not None,
        "training_samples": _count_samples(),
        "training_active": _training_active,
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


@app.post("/api/v1/training-data/ingest")
async def ingest_training_data(request: TrainingIngestRequest):
    """Receive training samples from VPS or desktop."""
    if not request.samples:
        raise HTTPException(status_code=400, detail="No samples provided")

    samples_file = TRAINING_DIR / "samples.jsonl"
    count = 0
    with samples_file.open("a", encoding="utf-8") as f:
        for sample in request.samples:
            f.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
            count += 1

    _training_stats["total_samples"] += count
    _training_stats["last_ingest"] = datetime.now().isoformat()

    # Check if auto-training should start
    _auto_train_check()

    return {
        "status": "ok",
        "accepted": count,
        "total_samples": _count_samples(),
        "training_active": _training_active,
        "auto_train_threshold": AUTO_TRAIN_THRESHOLD,
    }


@app.get("/api/v1/training/status")
async def training_status():
    """Get training status and GPU info."""
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0,
                "memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            }
    except ImportError:
        pass

    return {
        "status": "ok",
        "training_active": _training_active,
        "total_samples": _count_samples(),
        "auto_train_threshold": AUTO_TRAIN_THRESHOLD,
        "stats": _training_stats,
        "gpu": gpu_info,
        "data_dir": str(TRAINING_DIR),
    }


@app.post("/api/v1/training/start")
async def start_training():
    """Manually trigger training."""
    if _training_active:
        return {"status": "already_running"}
    _start_training()
    return {"status": "started", "samples": _count_samples()}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RTX AI API Server v8.0.1 on 0.0.0.0:8090")
    print(f"📦 Training data dir: {TRAINING_DIR}")
    print(f"🎯 Auto-train threshold: {AUTO_TRAIN_THRESHOLD} samples")
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")

