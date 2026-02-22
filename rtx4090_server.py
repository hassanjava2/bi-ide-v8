"""
🌐 RTX 4090 - REAL API LEARNING SERVER
بيانات حقيقية 100% من GitHub, CoinGecko, HackerNews, Reddit, arXiv, CVE
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hierarchy.real_api_learning import real_api_learning_system, DEVICE
import torch
import requests

# Windows API endpoint for auto-sync
WINDOWS_API = os.getenv("WINDOWS_API", "http://192.168.68.117:8000")

app = FastAPI(title="RTX 4090 REAL API LEARNING", version="7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "name": "RTX 4090 REAL API LEARNING",
        "version": "7.0",
        "mode": "REAL_API_LEARNING",
        "layers": 15,
        "data_source": "100% REAL APIs",
        "apis": [
            "GitHub API - Trending Repos",
            "CoinGecko API - Live Prices",
            "HackerNews API - Tech News",
            "Reddit API - Discussions",
            "StackOverflow API - Questions",
            "CVE API - Security Alerts",
            "arXiv API - Research Papers"
        ],
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "status": "🌐 Connected to Real Internet APIs"
    }

@app.get("/status")
def get_status():
    return real_api_learning_system.get_status()

@app.post("/start")
def start():
    if real_api_learning_system.running:
        return {"status": "already_running"}
    real_api_learning_system.start_all()
    return {
        "status": "started",
        "message": "🌐 Started learning from REAL APIs: GitHub, CoinGecko, HackerNews, Reddit, arXiv, CVE"
    }

@app.post("/stop")
def stop():
    if not real_api_learning_system.running:
        return {"status": "not_running"}
    real_api_learning_system.stop_all()
    return {"status": "stopped"}


# ========== Checkpoint Management ==========

@app.get("/checkpoints/list")
def list_checkpoints():
    """قائمة جميع الـ checkpoints"""
    checkpoints = []
    base_dir = Path("learning_data/checkpoints")
    if base_dir.exists():
        for layer_dir in base_dir.iterdir():
            if layer_dir.is_dir():
                for ckpt in layer_dir.glob("*.pt"):
                    checkpoints.append({
                        "layer": layer_dir.name,
                        "file": ckpt.name,
                        "size_mb": round(ckpt.stat().st_size / (1024*1024), 2),
                        "modified": ckpt.stat().st_mtime
                    })
    return {
        "total": len(checkpoints),
        "location": str(base_dir),
        "checkpoints": checkpoints
    }

@app.get("/checkpoints/download/{layer_name}/{filename}")
def download_checkpoint(layer_name: str, filename: str):
    """تحميل checkpoint معين"""
    file_path = Path("learning_data/checkpoints") / layer_name / filename
    if not file_path.exists():
        raise HTTPException(404, "Checkpoint not found")
    return FileResponse(file_path, filename=filename)

@app.post("/checkpoints/sync-to-windows")
def sync_all_to_windows():
    """مزامنة جميع الـ checkpoints للـ Windows"""
    synced = 0
    failed = 0
    base_dir = Path("learning_data/checkpoints")
    
    if not base_dir.exists():
        return {"status": "no_checkpoints", "synced": 0}
    
    for layer_dir in base_dir.iterdir():
        if layer_dir.is_dir():
            for ckpt in layer_dir.glob("*.pt"):
                try:
                    with open(ckpt, "rb") as f:
                        files = {"file": (ckpt.name, f, "application/octet-stream")}
                        response = requests.post(
                            f"{WINDOWS_API}/api/v1/checkpoints/upload/{layer_dir.name}",
                            files=files,
                            timeout=30
                        )
                        if response.status_code == 200:
                            synced += 1
                        else:
                            failed += 1
                except Exception as e:
                    print(f"Failed to sync {ckpt}: {e}")
                    failed += 1
    
    return {"status": "completed", "synced": synced, "failed": failed}


# Function to auto-sync checkpoints
def auto_sync_checkpoint(layer_name: str, checkpoint_path: Path):
    """إرسال checkpoint للـ Windows تلقائياً"""
    try:
        with open(checkpoint_path, "rb") as f:
            files = {"file": (checkpoint_path.name, f, "application/octet-stream")}
            response = requests.post(
                f"{WINDOWS_API}/api/v1/checkpoints/upload/{layer_name}",
                files=files,
                timeout=30
            )
            if response.status_code == 200:
                print(f"  🔄 Auto-synced to Windows: {layer_name}/{checkpoint_path.name}")
            else:
                print(f"  ⚠️ Auto-sync failed: {response.status_code}")
    except Exception as e:
        print(f"  ⚠️ Auto-sync error: {e}")

if __name__ == "__main__":
    print("🌐" * 40)
    print("RTX 4090 REAL API LEARNING v7.0")
    print("🌐" * 40)
    print("100% REAL DATA FROM:")
    print("  • GitHub API")
    print("  • CoinGecko API")
    print("  • HackerNews API")
    print("  • Reddit API")
    print("  • StackOverflow API")
    print("  • CVE API")
    print("  • arXiv API")
    print("🌐" * 40)
    uvicorn.run(app, host="0.0.0.0", port=8080)
