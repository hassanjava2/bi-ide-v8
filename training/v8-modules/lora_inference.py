"""
LoRA Inference Engine - محرك استنتاج LoRA

قوانين RTX5090:
✅ مسموح: LoRA trained model
❌ ممنوع: Ollama للاستنتاج
"""

import os
import sys
import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """نتيجة الاستنتاج"""
    text: str
    confidence: float
    source: str
    evidence: List[str]
    processing_time_ms: int
    model_version: str


class LoRAInferenceEngine:
    """
    محرك استنتاج LoRA - الإنتاج فقط
    
    Rules:
    - LoRA model only
    - No Ollama fallback for inference
    - GPU accelerated (Blackwell sm_120 supported)
    """
    
    # القيود الحرارية
    TEMP_WARNING = 85   # °C
    TEMP_THROTTLE = 90  # °C
    TEMP_STOP = 97      # °C - EMERGENCY STOP
    
    def __init__(
        self,
        models_dir: str = "/home/bi/training_data/models/finetuned",
        base_model: str = "Qwen/Qwen2.5-1.5B",
        device: str = "cuda",
        max_seq_length: int = 2048,
    ):
        self.models_dir = Path(models_dir)
        self.base_model_name = base_model
        self.device = device
        self.max_seq_length = max_seq_length
        
        # Model cache
        self._model = None
        self._tokenizer = None
        self._current_adapter = None
        self._lock = threading.RLock()
        
        # Stats
        self._inference_count = 0
        self._total_time_ms = 0
        
        print(f"🧠 LoRA Inference Engine initialized")
        print(f"   Base: {base_model}")
        print(f"   Device: {device}")
        print(f"   Models dir: {models_dir}")
    
    def _get_latest_adapter(self) -> Optional[Path]:
        """الحصول على أحدث LoRA adapter"""
        if not self.models_dir.exists():
            return None
        
        # البحث عن auto_* أو run_* أو merged-*
        patterns = ["auto_*", "run_*", "merged-*", "v8-*"]
        all_dirs = []
        
        for pattern in patterns:
            all_dirs.extend(self.models_dir.glob(pattern))
        
        # ترتيب حسب وقت التعديل (الأحدث أولاً)
        all_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        for d in all_dirs:
            if (d / "adapter_config.json").exists():
                return d
            # Check checkpoints
            checkpoints = sorted(d.glob("checkpoint-*"), 
                               key=lambda p: p.stat().st_mtime, 
                               reverse=True)
            for cp in checkpoints:
                if (cp / "adapter_config.json").exists():
                    return cp
        
        return None
    
    def _load_model(self, adapter_path: Path):
        """تحميل النموذج"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            # تحميل التوكنيزر
            print(f"📥 Loading tokenizer from {self.base_model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # تحميل النموذج الأساسي
            print(f"📥 Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            
            if self.device == "cpu":
                base_model = base_model.to("cpu")
            
            # تحميل LoRA adapter
            print(f"📥 Loading LoRA adapter from {adapter_path}")
            self._model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self._model.eval()
            
            self._current_adapter = adapter_path
            print(f"✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self._model = None
            self._tokenizer = None
            raise
    
    def _check_thermal(self) -> bool:
        """التحقق من الحرارة - قانون 97°C"""
        try:
            temp = 0
            hwmon_path = Path("/sys/class/hwmon")
            if hwmon_path.exists():
                for hwmon in hwmon_path.glob("hwmon*"):
                    name_file = hwmon / "name"
                    if name_file.exists():
                        name = name_file.read_text().strip()
                        if name in ("coretemp", "k10temp", "nvme"):
                            for t_file in hwmon.glob("temp*_input"):
                                t = int(t_file.read_text().strip()) / 1000
                                temp = max(temp, t)
            
            if temp >= self.TEMP_STOP:
                print(f"🚨 THERMAL STOP: {temp}°C >= {self.TEMP_STOP}°C")
                return False
            elif temp >= self.TEMP_THROTTLE:
                print(f"⚠️ THERMAL THROTTLE: {temp}°C")
            
            return True
            
        except Exception:
            return True  # افتراضي آمن
    
    async def infer(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout_sec: float = 30.0,
    ) -> InferenceResult:
        """
        استنتاج LoRA - القناة الوحيدة المسموحة
        
        Rules:
        - No Ollama fallback
        - Thermal protection active
        - Returns empty if model unavailable
        """
        start_time = time.time()
        
        # التحقق الحراري
        if not self._check_thermal():
            return InferenceResult(
                text="AI غير متاح حالياً بسبب درجة الحرارة العالية.",
                confidence=0.0,
                source="rtx5090-thermal-stop",
                evidence=["thermal-protection"],
                processing_time_ms=int((time.time() - start_time) * 1000),
                model_version="N/A"
            )
        
        with self._lock:
            try:
                # التحقق/تحميل النموذج
                adapter_path = self._get_latest_adapter()
                
                if adapter_path is None:
                    return InferenceResult(
                        text="",
                        confidence=0.0,
                        source="rtx5090-no-model",
                        evidence=["no-adapter-found"],
                        processing_time_ms=int((time.time() - start_time) * 1000),
                        model_version="N/A"
                    )
                
                if self._model is None or self._current_adapter != adapter_path:
                    self._load_model(adapter_path)
                
                if self._model is None or self._tokenizer is None:
                    return InferenceResult(
                        text="",
                        confidence=0.0,
                        source="rtx5090-load-failed",
                        evidence=["model-load-error"],
                        processing_time_ms=int((time.time() - start_time) * 1000),
                        model_version="N/A"
                    )
                
                # تنفيذ الاستنتاج
                import torch
                
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=True
                )
                
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self._tokenizer.pad_token_id,
                        eos_token_id=self._tokenizer.eos_token_id,
                    )
                
                # فك التشفير
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                response_text = self._tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                ).strip()
                
                processing_time = int((time.time() - start_time) * 1000)
                
                # تحديث الإحصائيات
                self._inference_count += 1
                self._total_time_ms += processing_time
                
                return InferenceResult(
                    text=response_text,
                    confidence=0.92,
                    source="rtx5090-lora-trained",
                    evidence=["lora-adapter", str(adapter_path.name)],
                    processing_time_ms=processing_time,
                    model_version=adapter_path.name
                )
                
            except asyncio.TimeoutError:
                return InferenceResult(
                    text="",
                    confidence=0.0,
                    source="rtx5090-timeout",
                    evidence=["inference-timeout"],
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    model_version="N/A"
                )
                
            except Exception as e:
                print(f"❌ Inference error: {e}")
                return InferenceResult(
                    text="",
                    confidence=0.0,
                    source="rtx5090-error",
                    evidence=[str(e)],
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    model_version="N/A"
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """إحصائيات الاستنتاج"""
        avg_time = 0
        if self._inference_count > 0:
            avg_time = self._total_time_ms / self._inference_count
        
        return {
            "inference_count": self._inference_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": round(avg_time, 2),
            "current_adapter": str(self._current_adapter) if self._current_adapter else None,
            "device": self.device,
        }


# Singleton instance
_inference_engine: Optional[LoRAInferenceEngine] = None


def get_inference_engine() -> LoRAInferenceEngine:
    """الحصول على محرك الاستنتاج الموحد"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = LoRAInferenceEngine()
    return _inference_engine


async def lora_infer(prompt: str, **kwargs) -> InferenceResult:
    """دالة مساعدة للاستنتاج السريع"""
    engine = get_inference_engine()
    return await engine.infer(prompt, **kwargs)
