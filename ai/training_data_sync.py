"""
Training Data Sync - مزامنة بيانات التدريب مع RTX5090

يدير بيانات التدريب الـ 45GB على RTX5090
"""

import os
import json
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


RTX5090_URL = os.getenv("RTX5090_URL", "http://192.168.1.164:8090")


@dataclass
class TrainingDataInfo:
    """معلومات بيانات التدريب"""
    total_size_gb: float
    total_samples: int
    models_count: int
    latest_model: Optional[str]
    last_updated: datetime


class TrainingDataSync:
    """
    مدير بيانات التدريب على RTX5090
    
    يوفر:
    - معلومات عن بيانات التدريب (45GB)
    - حالة النماذج المدربة
    - مزامنة الإحصائيات
    """
    
    def __init__(self, rtx_url: str = None):
        self.rtx_url = rtx_url or RTX5090_URL
        self._cache: Dict[str, Any] = {}
        self._last_sync: Optional[datetime] = None
    
    async def get_training_status(self) -> Dict[str, Any]:
        """الحصول على حالة التدريب من RTX5090"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.rtx_url}/api/v1/training/status",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._last_sync = datetime.now()
                        return {
                            "status": "connected",
                            "rtx5090": data,
                            "synced_at": self._last_sync.isoformat(),
                        }
                    else:
                        return {"status": "error", "code": resp.status}
        except Exception as e:
            return {"status": "disconnected", "error": str(e)}
    
    async def get_models_info(self) -> Dict[str, Any]:
        """معلومات عن النماذج المدربة"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.rtx_url}/api/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {"models": [], "error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"models": [], "error": str(e)}
    
    async def ingest_samples(self, samples: List[Dict]) -> Dict[str, Any]:
        """إرسال عينات تدريب جديدة لـ RTX5090"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rtx_url}/api/v1/training-data/ingest",
                    json={"samples": samples, "store_local": True},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {"status": "error", "code": resp.status}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def sync_all(self) -> Dict[str, Any]:
        """مزامنة كاملة مع RTX5090"""
        status = await self.get_training_status()
        models = await self.get_models_info()
        
        result = {
            "rtx5090_connection": status.get("status"),
            "training_data": status.get("rtx5090", {}),
            "models": models.get("models", []),
            "synced_at": datetime.now().isoformat(),
        }
        
        self._cache = result
        return result
    
    def get_cached_info(self) -> Dict[str, Any]:
        """الحصول على آخر معلومات مخزنة"""
        return self._cache


# Singleton
_sync_manager: Optional[TrainingDataSync] = None


def get_training_sync() -> TrainingDataSync:
    """الحصول على مدير المزامنة الموحد"""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = TrainingDataSync()
    return _sync_manager
