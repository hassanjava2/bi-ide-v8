"""
🚀 RTX 5090 Training & AI Server v4.0
خادم تدريب متكامل مع PyTorch + AI Council + Resource Manager
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import json
import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path

# محاولة استيراد PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - running in mock mode")

# Import Resource Manager
try:
    from resource_manager import ResourceManager, detect_system_resources
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from resource_manager import ResourceManager, detect_system_resources
        RESOURCE_MANAGER_AVAILABLE = True
    except ImportError:
        RESOURCE_MANAGER_AVAILABLE = False
        print("⚠️ ResourceManager not available")

app = FastAPI(title="🚀 RTX 5090 AI & Training Server", version="4.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic Models ====================

class ResourceConfigRequest(BaseModel):
    cpu_percent: int = 80       # 10-100
    gpu_percent: int = 80       # 10-100
    ram_limit_percent: int = 80 # 10-100

class IntensiveTrainingRequest(BaseModel):
    epochs: int = 100
    cpu_percent: int = 80
    gpu_percent: int = 80

class TrainingConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    model_name: str = "bi-ide-model"
    data_path: str = "/home/bi/training_data/data"

class CouncilMessageRequest(BaseModel):
    message: str
    user_id: str = "user"

class TrainingIngestRequest(BaseModel):
    samples: List[Dict[str, Any]] = []
    relay: bool = False
    store_local: bool = True

# ==================== 16 Wise Men Personas ====================

WISE_MEN = {
    "حكيم الهوية": {
        "role": "identity", "id": "S001",
        "prompt": "أنت حكيم الهوية. تحلل من يكون المستخدم وماذا يحتاج. أجب بحكمة وعمق.",
    },
    "حكيم الاستراتيجيا": {
        "role": "strategy", "id": "S002",
        "prompt": "أنت حكيم الاستراتيجيا. تضع الخطط طويلة المدى وتحلل المواقف. أجب بوضوح.",
    },
    "حكيم الأخلاق": {
        "role": "ethics", "id": "S003",
        "prompt": "أنت حكيم الأخلاق. تقيّم القرارات أخلاقياً وتنصح بالصواب.",
    },
    "حكيم التوازن": {
        "role": "balance", "id": "S004",
        "prompt": "أنت حكيم التوازن. تبحث عن التوازن في كل شيء وتحذر من التطرف.",
    },
    "حكيم المعرفة": {
        "role": "knowledge", "id": "S005",
        "prompt": "أنت حكيم المعرفة. تمتلك معرفة واسعة وتشارك المعلومات المفيدة.",
    },
    "حكيم الإبداع": {
        "role": "creativity", "id": "S006",
        "prompt": "أنت حكيم الإبداع. تقترح حلولاً إبداعية وتفكر خارج الصندوق.",
    },
    "حكيم الأمان": {
        "role": "security", "id": "S007",
        "prompt": "أنت حكيم الأمان. تحمي النظام وتحذر من المخاطر الأمنية.",
    },
    "حكيم الأداء": {
        "role": "performance", "id": "S008",
        "prompt": "أنت حكيم الأداء. تحلل الكفاءة وتقترح تحسينات الأداء.",
    },
    "حكيم التواصل": {
        "role": "communication", "id": "S009",
        "prompt": "أنت حكيم التواصل. تتواصل بوضوح وتساعد في فهم الأمور المعقدة.",
    },
    "حكيم التعلم": {
        "role": "learning", "id": "S010",
        "prompt": "أنت حكيم التعلم. تعلّم وتدرّب وتشرح المفاهيم بسهولة.",
    },
    "حكيم التنفيذ": {
        "role": "execution", "id": "S011",
        "prompt": "أنت حكيم التنفيذ. تنفذ الأوامر بدقة وكفاءة عالية.",
    },
    "حكيم الرقابة": {
        "role": "oversight", "id": "S012",
        "prompt": "أنت حكيم الرقابة. تراقب الجودة وتضمن الالتزام بالمعايير.",
    },
    "حكيم الابتكار": {
        "role": "innovation", "id": "S013",
        "prompt": "أنت حكيم الابتكار. تبتكر حلولاً جديدة وتطور الأفكار.",
    },
    "حكيم الحماية": {
        "role": "guardian", "id": "S014",
        "prompt": "أنت حكيم الحماية. تحمي المستخدمين والبيانات من أي ضرر.",
    },
    "حكيم القرار": {
        "role": "decision", "id": "S015",
        "prompt": "أنت حكيم القرار. تتخذ قرارات حكيمة بناءً على كل المعطيات.",
    },
    "حكيم الحكمة": {
        "role": "wisdom", "id": "S016",
        "prompt": "أنت حكيم الحكمة. تنطق بالحكمة العميقة وتوجه بالبصيرة.",
    },
}

# Chat history for context
council_chat_history: List[Dict] = []
TRAINING_DATA_DIR = Path("/home/bi/training_data/data")
INGEST_DIR = Path("/home/bi/training_data/ingest")

# ==================== Resource Manager ====================

if RESOURCE_MANAGER_AVAILABLE:
    res_manager = ResourceManager()
    print("✅ ResourceManager loaded")
else:
    res_manager = None

# ==================== GPU Monitoring ====================

def get_gpu_info():
    """قراءة معلومات GPU"""
    info = {
        "usage": 0.0, "temp": 0.0,
        "vram_used": 0.0, "vram_total": 24.0,
        "available": TORCH_AVAILABLE and torch.cuda.is_available()
    }
    if not TORCH_AVAILABLE:
        return info
    try:
        if torch.cuda.is_available():
            info["vram_used"] = torch.cuda.memory_allocated() / 1e9
            info["vram_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info["usage"] = util.gpu
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                info["temp"] = temp
            except:
                try:
                    import subprocess
                    r = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=3
                    )
                    if r.returncode == 0:
                        parts = r.stdout.strip().split(",")
                        info["usage"] = float(parts[0].strip())
                        info["temp"] = float(parts[1].strip())
                except:
                    pass
    except Exception as e:
        print(f"GPU info error: {e}")
    return info


# ==================== Council AI Logic ====================

def _select_wise_man(message: str) -> str:
    """
    Semantic routing — selects the most relevant wise man based on
    weighted keyword relevance scoring (not simple keyword matching).
    Each wise man gets a relevance score; highest score wins.
    """
    # Domain expertise mapping with weights (higher = more specific)
    expertise_map = {
        "حكيم الأمان": {
            "keywords": ["أمان", "حماية", "أمن", "هجوم", "security", "hack", "ثغر", "تشفير", "ssl", "auth", "password", "firewall"],
            "weight": 1.2  # Security gets higher priority
        },
        "حكيم الأداء": {
            "keywords": ["أداء", "سرعة", "بطيء", "تحسين", "performance", "optimize", "resource", "موارد", "cpu", "gpu", "memory", "latency"],
            "weight": 1.1
        },
        "حكيم الاستراتيجيا": {
            "keywords": ["خطة", "استراتيجية", "هدف", "مستقبل", "strategy", "plan", "roadmap", "vision", "target"],
            "weight": 1.0
        },
        "حكيم المعرفة": {
            "keywords": ["معلومات", "شرح", "كيف", "ليش", "شنو", "explain", "what", "why", "how", "define", "وضّح"],
            "weight": 1.0
        },
        "حكيم الإبداع": {
            "keywords": ["إبداع", "فكرة", "جديد", "ابتكار", "creative", "idea", "design", "تصميم", "brainstorm"],
            "weight": 1.0
        },
        "حكيم التنفيذ": {
            "keywords": ["نفّذ", "شغّل", "سوّي", "ابني", "execute", "build", "run", "deploy", "install", "setup"],
            "weight": 1.0
        },
        "حكيم التعلم": {
            "keywords": ["تعلم", "تدريب", "درّب", "train", "learn", "model", "dataset", "epoch", "neural"],
            "weight": 1.0
        },
        "حكيم الأخلاق": {
            "keywords": ["أخلاق", "صح", "غلط", "عدل", "ethics", "moral", "fair", "bias", "privacy"],
            "weight": 1.0
        },
        "حكيم التوازن": {
            "keywords": ["توازن", "تطرف", "وسط", "balance", "moderate", "tradeoff"],
            "weight": 0.9
        },
        "حكيم القرار": {
            "keywords": ["قرار", "اختيار", "بديل", "decision", "choose", "option", "compare"],
            "weight": 1.0
        },
    }
    
    msg_lower = message.lower()
    scores = {}
    
    for name, config in expertise_map.items():
        match_count = sum(1 for kw in config["keywords"] if kw in msg_lower)
        # Score = number of keyword matches × domain weight
        scores[name] = match_count * config["weight"]
    
    # Select highest scoring wise man
    best = max(scores, key=scores.get)
    
    # If no keywords matched at all (score 0), use round-robin based on message hash
    if scores[best] == 0:
        wise_men_list = list(WISE_MEN.keys())
        idx = hash(message) % len(wise_men_list)
        return wise_men_list[idx]
    
    return best


def _generate_council_response(message: str, wise_man_name: str) -> str:
    persona = WISE_MEN.get(wise_man_name, {})
    role = persona.get("role", "general")
    knowledge = _load_training_knowledge(message)

    if knowledge:
        response = f"بناءً على بيانات التدريب والمعرفة المتاحة:\n\n"
        response += f"{knowledge}\n\n"
        response += f"— {wise_man_name} (بثقة عالية)"
    else:
        fallback_responses = {
            "identity": f"أهلاً! أنا {wise_man_name}. رسالتك '{message}' مهمة. كحكيم الهوية، أنصحك بالتركيز على تحديد أهدافك بوضوح أولاً.",
            "strategy": f"من منظور استراتيجي، '{message}' يحتاج خطة واضحة. أقترح تقسيم العمل لمراحل: تحليل، تخطيط، تنفيذ، مراجعة.",
            "ethics": f"من الناحية الأخلاقية، '{message}' يتطلب مراعاة الشفافية والعدالة في التنفيذ.",
            "knowledge": f"بخصوص '{message}'، المعرفة المتاحة تشير إلى عدة جوانب مهمة يجب مراعاتها.",
            "creativity": f"فكرة مبتكرة لـ '{message}': جرب نهجاً مختلفاً - ابدأ من النتيجة المطلوبة واعمل بالعكس.",
            "security": f"من ناحية الأمان، '{message}' يحتاج تقييم المخاطر أولاً. تأكد من حماية البيانات.",
            "performance": f"لتحسين الأداء في '{message}': حلل نقاط الاختناق أولاً، ثم حسّن الأهم فالأهم.",
            "execution": f"لتنفيذ '{message}': الخطوة الأولى هي تحديد المتطلبات بدقة، ثم البدء بالنموذج الأولي.",
            "learning": f"للتعلم من '{message}': كل تجربة فرصة للتطور. حلل النتائج وطوّر أسلوبك.",
            "decision": f"قراري بخصوص '{message}': بناءً على التحليل الشامل، أنصح بالمضي قدماً مع مراقبة النتائج.",
        }
        response = fallback_responses.get(role, f"كـ{wise_man_name}، رسالتك مهمة. أقترح تحليل الموضوع من عدة زوايا.")
    return response


def _load_training_knowledge(query: str) -> str:
    try:
        data_dirs = [
            TRAINING_DATA_DIR / "learning",
            TRAINING_DATA_DIR / "memory",
            Path("/home/bi/training_data/learning_data"),
        ]
        knowledge_items = []
        for data_dir in data_dirs:
            if data_dir.exists():
                for f in data_dir.rglob("*.jsonl"):
                    try:
                        lines = f.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
                        for line in lines[-5:]:
                            try:
                                item = json.loads(line)
                                if query.lower() in json.dumps(item, ensure_ascii=False).lower():
                                    knowledge_items.append(item)
                            except:
                                pass
                    except:
                        pass
                    if len(knowledge_items) >= 3:
                        break
        if knowledge_items:
            summaries = []
            for item in knowledge_items[:3]:
                text = item.get("response", item.get("text", item.get("output", str(item)[:200])))
                summaries.append(str(text)[:200])
            return "\n".join(summaries)
    except Exception as e:
        print(f"Knowledge load error: {e}")
    return ""


# ================================================================
#  API Endpoints
# ================================================================

@app.get("/")
def root():
    gpu_info = get_gpu_info()
    return {
        "name": "🚀 RTX 5090 AI & Training Server",
        "version": "4.0",
        "gpu": gpu_info.get("available", False) and torch.cuda.get_device_name(0) if TORCH_AVAILABLE and torch.cuda.is_available() else "CPU Mode",
        "pytorch_available": TORCH_AVAILABLE,
        "cuda_available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
        "resource_manager": RESOURCE_MANAGER_AVAILABLE,
        "status": "🟢 Online",
        "council_available": True,
        "training_data_size": sum(f.stat().st_size for f in TRAINING_DATA_DIR.rglob("*") if f.is_file()) if TRAINING_DATA_DIR.exists() else 0,
    }

@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0", "timestamp": datetime.now().isoformat()}


# ==================== Resource Management API ====================

@app.get("/resources/status")
def resources_status():
    """حالة الموارد الحية — CPU, GPU, RAM, Training"""
    if not RESOURCE_MANAGER_AVAILABLE:
        # Fallback: basic system info
        gpu_info = get_gpu_info()
        return {
            "config": {"cpu_percent": 0, "gpu_percent": 0},
            "system": {
                "gpu_available": gpu_info["available"],
                "gpu_utilization": gpu_info["usage"],
                "gpu_temp_c": gpu_info["temp"],
                "gpu_vram_used_gb": gpu_info["vram_used"],
                "gpu_vram_total_gb": gpu_info["vram_total"],
            },
            "training_active": False,
            "resource_manager": False,
        }
    return res_manager.get_live_status()


@app.post("/resources/configure")
def resources_configure(config: ResourceConfigRequest):
    """ضبط نسبة استهلاك الموارد (CPU/GPU/RAM)"""
    if not RESOURCE_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="ResourceManager not available")
    
    result = res_manager.configure(
        cpu_percent=config.cpu_percent,
        gpu_percent=config.gpu_percent,
        ram_limit_percent=config.ram_limit_percent,
    )
    return result


@app.post("/resources/training/start")
def start_intensive_training(req: IntensiveTrainingRequest):
    """بدء تدريب مكثف باستخدام الموارد"""
    if not RESOURCE_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="ResourceManager not available")
    
    # Configure resources first
    res_manager.configure(
        cpu_percent=req.cpu_percent,
        gpu_percent=req.gpu_percent,
    )
    
    result = res_manager.start_training(epochs=req.epochs)
    return result


@app.post("/resources/training/stop")
def stop_intensive_training():
    """إيقاف التدريب المكثف"""
    if not RESOURCE_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="ResourceManager not available")
    return res_manager.stop_training()


@app.get("/resources/system")
def system_resources():
    """معلومات تفصيلية عن موارد النظام"""
    if RESOURCE_MANAGER_AVAILABLE:
        return detect_system_resources()
    return get_gpu_info()


# ==================== Legacy Training API ====================

@app.get("/status")
def get_status():
    """حالة التدريب والـ GPU"""
    gpu_info = get_gpu_info()
    if RESOURCE_MANAGER_AVAILABLE and res_manager.training_active:
        m = res_manager.metrics
        return {
            "is_training": True,
            "current_epoch": m.get("epoch", 0),
            "total_epochs": m.get("total_epochs", 0),
            "loss": m.get("loss", 0),
            "accuracy": m.get("accuracy", 0),
            "gpu_usage": m.get("gpu_utilization", gpu_info["usage"]),
            "gpu_temp": m.get("gpu_temp_c", gpu_info["temp"]),
            "vram_used": m.get("gpu_vram_used_gb", gpu_info["vram_used"]),
            "vram_total": m.get("gpu_vram_total_gb", gpu_info["vram_total"]),
            "progress": round(m.get("epoch", 0) / max(1, m.get("total_epochs", 1)) * 100, 1),
            "model_preset": m.get("model_preset", ""),
            "model_params": m.get("model_params", 0),
            "throughput_samples_sec": m.get("throughput_samples_sec", 0),
            "cpu_utilization": m.get("cpu_utilization", 0),
            "cuda_available": gpu_info["available"],
            "device": "cuda" if gpu_info["available"] else "cpu",
        }
    return {
        "is_training": False,
        "gpu_usage": gpu_info["usage"],
        "gpu_temp": gpu_info["temp"],
        "vram_used": gpu_info["vram_used"],
        "vram_total": gpu_info["vram_total"],
        "cuda_available": gpu_info["available"],
        "device": "cuda" if gpu_info["available"] else "cpu",
    }


@app.post("/start")
def start_training(config: Optional[TrainingConfig] = None):
    """بدء التدريب (يستخدم ResourceManager)"""
    if RESOURCE_MANAGER_AVAILABLE:
        cfg = config or TrainingConfig()
        return res_manager.start_training(epochs=cfg.epochs)
    return {"status": "error", "message": "ResourceManager not available"}


@app.post("/stop")
def stop_training():
    """إيقاف التدريب"""
    if RESOURCE_MANAGER_AVAILABLE:
        return res_manager.stop_training()
    return {"status": "not_running"}


@app.get("/gpu/info")
def gpu_info_endpoint():
    """معلومات تفصيلية عن الـ GPU"""
    info = get_gpu_info()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return {
            **info,
            "name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "memory_cached": torch.cuda.memory_reserved() / 1e9,
            "device_count": torch.cuda.device_count()
        }
    return info


# ==================== Council Endpoint ====================

@app.post("/council/message")
def council_message(request: CouncilMessageRequest):
    """AI Council message — حكيم يجاوب"""
    wise_man = _select_wise_man(request.message)
    response = _generate_council_response(request.message, wise_man)

    council_chat_history.append({
        "role": "user", "message": request.message,
        "timestamp": datetime.now().isoformat(),
    })
    council_chat_history.append({
        "role": "council", "council_member": wise_man,
        "message": response, "timestamp": datetime.now().isoformat(),
    })
    while len(council_chat_history) > 100:
        council_chat_history.pop(0)

    return {
        "response": response,
        "council_member": wise_man,
        "model_used": "rtx5090-local",
        "source": "rtx5090",
    }


# ==================== Training Data Ingestion ====================

@app.post("/api/v1/training-data/ingest")
def ingest_training_data(request: TrainingIngestRequest):
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    filepath = INGEST_DIR / f"ingest_{datetime.now().strftime('%Y%m%d')}.jsonl"
    count = 0
    with open(filepath, "a", encoding="utf-8") as f:
        for sample in request.samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return {"status": "ok", "ingested": count, "file": str(filepath)}


# ==================== Main ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 RTX 5090 AI & Training Server v4.0")
    print("⚡ Real Hardware Resource Utilization")
    print("=" * 60)

    if TORCH_AVAILABLE:
        print(f"📦 PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"🔧 CUDA: {torch.version.cuda}")
        else:
            print("⚠️ CUDA not available - using CPU")
    else:
        print("⚠️ PyTorch not installed - mock mode")

    if RESOURCE_MANAGER_AVAILABLE:
        info = res_manager.system_info
        print(f"🖥️ CPU: {info.get('cpu_cores_physical', '?')} cores ({info.get('cpu_cores_logical', '?')} logical)")
        print(f"🧠 RAM: {info.get('ram_total_gb', '?')} GB")
        print("✅ ResourceManager: Active")
    else:
        print("⚠️ ResourceManager: Not available")

    # Training data
    if TRAINING_DATA_DIR.exists():
        total_size = sum(f.stat().st_size for f in TRAINING_DATA_DIR.rglob("*") if f.is_file())
        print(f"📊 Training Data: {total_size / 1e9:.1f} GB")

    print("=" * 60)
    print("📡 API Endpoints:")
    print("  GET  /                        → Server info")
    print("  GET  /health                  → Health check")
    print("  GET  /resources/status         → Live resource usage")
    print("  POST /resources/configure      → Set CPU/GPU %")
    print("  POST /resources/training/start → Start intensive training")
    print("  POST /resources/training/stop  → Stop training")
    print("  GET  /resources/system         → System hardware info")
    print("  POST /council/message          → AI Council")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8090)
