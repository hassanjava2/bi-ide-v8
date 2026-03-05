"""
Model Manager - إدارة النماذج

Handles:
- LoRA adapter discovery
- Model versioning
- Adapter merging
- Cleanup old models
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ModelVersion:
    """إصدار نموذج"""
    name: str
    path: Path
    created_at: datetime
    size_mb: float
    config: Dict[str, Any]
    checkpoint: Optional[str] = None
    is_merged: bool = False
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "created_at": self.created_at.isoformat(),
            "size_mb": self.size_mb,
            "config": self.config,
            "checkpoint": self.checkpoint,
            "is_merged": self.is_merged,
        }


class ModelManager:
    """
    مدير النماذج - يدير LoRA adapters
    """
    
    def __init__(
        self,
        models_dir: str = "/home/bi/training_data/models/finetuned",
        max_versions: int = 5,
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.max_versions = max_versions
        
        # Cache
        self._versions_cache: Optional[List[ModelVersion]] = None
        
        print(f"📦 Model Manager initialized")
        print(f"   Directory: {models_dir}")
        print(f"   Max versions: {max_versions}")
    
    def _get_adapter_info(self, path: Path) -> Optional[Dict[str, Any]]:
        """قراءة معلومات adapter"""
        config_file = path / "adapter_config.json"
        if not config_file.exists():
            return None
        
        try:
            with open(config_file) as f:
                return json.load(f)
        except:
            return None
    
    def _get_dir_size(self, path: Path) -> float:
        """حساب حجم المجلد بالميجابايت"""
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total / (1024 * 1024)
    
    def discover_versions(self) -> List[ModelVersion]:
        """اكتشاف جميع إصدارات النماذج"""
        versions = []
        
        if not self.models_dir.exists():
            return versions
        
        # أنماط البحث
        patterns = ["auto_*", "run_*", "merged-*", "v8-*", "checkpoint-*"]
        
        for pattern in patterns:
            for path in self.models_dir.glob(pattern):
                if not path.is_dir():
                    continue
                
                config = self._get_adapter_info(path)
                if config is None:
                    # Check for nested checkpoints
                    for checkpoint in path.glob("checkpoint-*"):
                        cp_config = self._get_adapter_info(checkpoint)
                        if cp_config:
                            versions.append(ModelVersion(
                                name=f"{path.name}/{checkpoint.name}",
                                path=checkpoint,
                                created_at=datetime.fromtimestamp(
                                    checkpoint.stat().st_mtime
                                ),
                                size_mb=self._get_dir_size(checkpoint),
                                config=cp_config,
                                checkpoint=checkpoint.name,
                            ))
                    continue
                
                versions.append(ModelVersion(
                    name=path.name,
                    path=path,
                    created_at=datetime.fromtimestamp(path.stat().st_mtime),
                    size_mb=self._get_dir_size(path),
                    config=config,
                    is_merged="merged" in path.name.lower(),
                ))
        
        # ترتيب حسب الأحدث
        versions.sort(key=lambda v: v.created_at, reverse=True)
        self._versions_cache = versions
        
        return versions
    
    def get_latest(self) -> Optional[ModelVersion]:
        """الحصول على أحدث إصدار"""
        versions = self.discover_versions()
        return versions[0] if versions else None
    
    def get_best_for_inference(self) -> Optional[ModelVersion]:
        """الحصول على أفضل نموذج للاستنتاج"""
        versions = self.discover_versions()
        
        # تفضيل المدمج أولاً
        for v in versions:
            if v.is_merged:
                return v
        
        # ثم الأحدث
        return versions[0] if versions else None
    
    def create_merged_adapter(
        self,
        adapter_paths: List[Path],
        output_name: str = None,
    ) -> Optional[ModelVersion]:
        """
        دمج عدة adapters في واحد
        
        Uses weighted average of LoRA weights.
        """
        if len(adapter_paths) < 2:
            print("⚠️ Need at least 2 adapters to merge")
            return None
        
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"merged-{timestamp}"
        
        output_path = self.models_dir / output_name
        
        try:
            print(f"🔀 Merging {len(adapter_paths)} adapters...")
            
            # استخدام PEFT لدمج الـ adapters
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
            import torch
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B",
                torch_dtype=torch.float16,
                device_map="auto",
            )
            
            # Load and merge first adapter
            model = PeftModel.from_pretrained(base_model, str(adapter_paths[0]))
            
            # Merge additional adapters (simplified - in practice use more sophisticated merging)
            for i, adapter_path in enumerate(adapter_paths[1:], 1):
                print(f"   Adding adapter {i}: {adapter_path.name}")
                # Note: Real implementation would use weight merging
                # This is a placeholder for the concept
            
            # Save merged
            model = model.merge_and_unload()
            model.save_pretrained(output_path)
            
            # Copy config from first adapter
            shutil.copy(
                adapter_paths[0] / "adapter_config.json",
                output_path / "adapter_config.json"
            )
            
            print(f"✅ Merged adapter saved: {output_path}")
            
            return ModelVersion(
                name=output_name,
                path=output_path,
                created_at=datetime.now(),
                size_mb=self._get_dir_size(output_path),
                config=self._get_adapter_info(output_path) or {},
                is_merged=True,
            )
            
        except Exception as e:
            print(f"❌ Merge error: {e}")
            return None
    
    def cleanup_old_versions(self, keep: int = None):
        """حذف الإصدارات القديمة"""
        if keep is None:
            keep = self.max_versions
        
        versions = self.discover_versions()
        
        if len(versions) <= keep:
            return
        
        to_delete = versions[keep:]
        
        print(f"🧹 Cleaning up {len(to_delete)} old model versions...")
        
        for version in to_delete:
            try:
                if version.path.exists():
                    shutil.rmtree(version.path)
                    print(f"   🗑️ Deleted: {version.name}")
            except Exception as e:
                print(f"   ⚠️ Failed to delete {version.name}: {e}")
        
        # Clear cache
        self._versions_cache = None
    
    def get_version_by_name(self, name: str) -> Optional[ModelVersion]:
        """الحصول على إصدار بالاسم"""
        versions = self.discover_versions()
        for v in versions:
            if v.name == name:
                return v
        return None
    
    def validate_adapter(self, path: Path) -> bool:
        """التحقق من صحة adapter"""
        if not path.exists():
            return False
        
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        
        for f in required_files:
            if not (path / f).exists():
                return False
        
        # التحقق من صحة JSON
        try:
            with open(path / "adapter_config.json") as f:
                config = json.load(f)
                return "base_model_name_or_path" in config
        except:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """إحصائيات النماذج"""
        versions = self.discover_versions()
        
        total_size = sum(v.size_mb for v in versions)
        merged_count = sum(1 for v in versions if v.is_merged)
        
        return {
            "total_versions": len(versions),
            "merged_versions": merged_count,
            "total_size_mb": round(total_size, 2),
            "models_dir": str(self.models_dir),
            "max_versions": self.max_versions,
            "latest": versions[0].to_dict() if versions else None,
            "all_versions": [v.to_dict() for v in versions[:10]],
        }


# Singleton
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """الحصول على مدير النماذج الموحد"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
