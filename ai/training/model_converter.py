"""
Model Converter - محول النماذج
Merged from: convert-to-gguf.py, convert-to-onnx.py

Features / المميزات:
  • Convert to GGUF (llama.cpp)
  • Convert to ONNX
  • Quantization options
  • Format validation
  • Batch conversion
  • LoRA merging
  • Model versioning

PyTorch 2.x + CUDA 12.x Compatible
"""

import os
import sys
import json
import shutil
import subprocess
import urllib.request
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class ConversionFormat(Enum):
    """صيغة التحويل - Conversion format"""
    GGUF = "gguf"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"


class QuantizationType(Enum):
    """نوع التكميم - Quantization type"""
    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_K_M = "q4_k_m"
    Q5_K_M = "q5_k_m"


@dataclass
class ConversionResult:
    """نتيجة التحويل - Conversion result"""
    success: bool
    format: ConversionFormat
    output_path: Path
    size_mb: float
    duration_seconds: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionConfig:
    """إعدادات التحويل - Conversion configuration"""
    quantize: bool = True
    quantization_type: QuantizationType = QuantizationType.Q4_K_M
    merge_lora: bool = True
    validate: bool = True
    save_merged: bool = True
    cleanup_temp: bool = True


class ModelConverter:
    """
    محول النماذج - Model Converter
    
    يدعم تحويل النماذج المدربة إلى صيغ متعددة:
    - GGUF: لتشغيل بـ llama.cpp
    - ONNX: للتشغيل السريع مع optimum
    - PyTorch: للتدريب المستمر
    
    Supports converting trained models to multiple formats:
    - GGUF: for llama.cpp inference
    - ONNX: for fast inference with optimum
    - PyTorch: for continued training
    """
    
    # روابط السكربتات - Script URLs
    GGUF_CONVERT_URL = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        config: Optional[ConversionConfig] = None
    ):
        """
        Initialize model converter
        
        Args:
            base_dir: Base project directory
            config: Conversion configuration
        """
        self.base_dir = base_dir or Path(__file__).parent.parent.parent
        self.config = config or ConversionConfig()
        
        self.models_dir = self.base_dir / "models"
        self.finetuned_dir = self.models_dir / "finetuned"
        self.finetuned_ext_dir = self.models_dir / "finetuned-extended"
        self.finetuned_chat_dir = self.models_dir / "finetuned-chat"
        self.merged_dir = self.models_dir / "merged"
        self.onnx_dir = self.models_dir / "bi-ai-onnx"
        self.gguf_dir = self.models_dir / "bi-chat-gguf"
        
        self.registry_path = self.models_dir / "model-registry.json"
        
        logger.info("=" * 60)
        logger.info("Model Converter - محول النماذج")
        logger.info("=" * 60)
        logger.info(f"   Models dir: {self.models_dir}")
    
    def _find_finetuned_model(self) -> Optional[Path]:
        """
        البحث عن النموذج المُدرب
        Find finetuned model
        
        Returns:
            Path to model or None
        """
        # تفضيل الممتد أولاً
        if self.finetuned_ext_dir.exists():
            if any(self.finetuned_ext_dir.glob("*.safetensors")) or \
               any(self.finetuned_ext_dir.glob("*.bin")):
                return self.finetuned_ext_dir
        
        # ثم النموذج الأساسي
        if self.finetuned_dir.exists():
            if any(self.finetuned_dir.glob("*.safetensors")) or \
               any(self.finetuned_dir.glob("*.bin")):
                return self.finetuned_dir
        
        # نموذج المحادثة
        if self.finetuned_chat_dir.exists():
            if any(self.finetuned_chat_dir.glob("*.safetensors")) or \
               any(self.finetuned_chat_dir.glob("*.bin")):
                return self.finetuned_chat_dir
        
        return None
    
    def _get_base_model_name(self, model_dir: Path) -> str:
        """
        الحصول على اسم النموذج الأساسي
        Get base model name from adapter config
        """
        config_path = model_dir / "adapter_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get("base_model_name_or_path", "Qwen/Qwen2-0.5B")
        
        # افتراضي
        if "chat" in str(model_dir).lower():
            return "Qwen/Qwen2.5-3B-Instruct"
        return "Qwen/Qwen2-0.5B"
    
    def merge_lora(
        self,
        model_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        base_model_name: Optional[str] = None
    ) -> Tuple[bool, Path]:
        """
        دمج LoRA مع النموذج الأساسي
        Merge LoRA weights with base model
        
        Args:
            model_dir: Directory with LoRA weights
            output_dir: Output directory
            base_model_name: Base model name
            
        Returns:
            (success, output_path)
        """
        start_time = time.time()
        
        if model_dir is None:
            model_dir = self._find_finetuned_model()
        
        if model_dir is None or not model_dir.exists():
            logger.error("Finetuned model not found")
            return False, Path()
        
        if output_dir is None:
            output_dir = self.merged_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # التحقق من وجود LoRA
        adapter_config = model_dir / "adapter_config.json"
        if not adapter_config.exists():
            logger.info("No LoRA adapter found, copying as-is")
            for f in model_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, output_dir)
            return True, output_dir
        
        logger.info("Merging LoRA weights...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            base_model_name = base_model_name or self._get_base_model_name(model_dir)
            
            logger.info(f"   Base model: {base_model_name}")
            
            # تحميل النموذج الأساسي
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            # تحميل ودمج LoRA
            model = PeftModel.from_pretrained(model, str(model_dir))
            model = model.merge_and_unload()
            
            # حفظ النموذج المدمج
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
            
            duration = time.time() - start_time
            logger.info(f"✅ Merged model saved: {output_dir} ({duration:.1f}s)")
            
            return True, output_dir
            
        except Exception as e:
            logger.exception("LoRA merge failed")
            return False, Path()
    
    def convert_to_onnx(
        self,
        model_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        task: str = "text-generation"
    ) -> Dict[str, Any]:
        """
        تحويل إلى ONNX
        Convert to ONNX format
        
        Args:
            model_dir: Model directory
            output_dir: Output directory
            task: Task type
            
        Returns:
            Conversion result
        """
        start_time = time.time()
        
        if model_dir is None:
            model_dir = self._find_finetuned_model()
        
        if output_dir is None:
            output_dir = self.onnx_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 50)
        logger.info("Converting to ONNX...")
        logger.info("=" * 50)
        
        # دمج LoRA أولاً
        if self.config.merge_lora:
            success, merged_path = self.merge_lora(model_dir)
            if not success:
                return {'success': False, 'error': 'LoRA merge failed'}
            model_dir = merged_path
        
        try:
            # المحاولة بـ optimum
            from optimum.onnxruntime import ORTModelForCausalLM
            from transformers import AutoTokenizer
            
            logger.info("Using optimum for conversion...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            ort_model = ORTModelForCausalLM.from_pretrained(
                model_dir,
                export=True
            )
            ort_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            logger.info(f"✅ ONNX export successful: {output_dir}")
            
        except Exception as e1:
            logger.warning(f"Optimum export failed: {e1}")
            logger.info("Trying CLI method...")
            
            # طريقة CLI
            cmd = [
                sys.executable, "-m", "optimum.exporters.onnx",
                "--model", str(model_dir),
                "--task", task,
                str(output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"ONNX conversion failed: {result.stderr[:200]}")
                return {
                    'success': False,
                    'error': f'ONNX conversion failed: {result.stderr[:500]}'
                }
            
            logger.info(f"✅ ONNX export successful (CLI): {output_dir}")
        
        # حساب الحجم
        total_size = sum(
            f.stat().st_size for f in output_dir.rglob('*') if f.is_file()
        )
        size_mb = total_size / 1024 / 1024
        
        duration = time.time() - start_time
        
        # التسجيل
        self._register_version("onnx", output_dir, size_mb)
        
        return {
            'success': True,
            'output_dir': str(output_dir),
            'size_mb': round(size_mb, 2),
            'duration_seconds': round(duration, 2)
        }
    
    def _ensure_gguf_script(self) -> Optional[Path]:
        """
        التأكد من وجود سكربت GGUF
        Ensure GGUF conversion script exists
        
        Returns:
            Path to script or None
        """
        script_path = self.base_dir / "training" / "convert_hf_to_gguf.py"
        
        if script_path.exists():
            return script_path
        
        logger.info("Downloading GGUF conversion script...")
        
        try:
            urllib.request.urlretrieve(self.GGUF_CONVERT_URL, script_path)
            logger.info(f"✅ Downloaded: {script_path}")
            return script_path
        except Exception as e:
            logger.error(f"Failed to download GGUF script: {e}")
            return None
    
    def convert_to_gguf(
        self,
        model_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        quantize: bool = True,
        quantization_type: Optional[QuantizationType] = None
    ) -> Dict[str, Any]:
        """
        تحويل إلى GGUF
        Convert to GGUF format
        
        Args:
            model_dir: Model directory (merged)
            output_dir: Output directory
            quantize: Whether to quantize
            quantization_type: Quantization type
            
        Returns:
            Conversion result
        """
        start_time = time.time()
        
        quantization_type = quantization_type or self.config.quantization_type
        
        if model_dir is None:
            # دمج أولاً إذا لم يكن النموذج مدمجاً
            success, merged_path = self.merge_lora(
                self.finetuned_chat_dir,
                self.models_dir / "finetuned-chat-merged"
            )
            if not success:
                return {'success': False, 'error': 'LoRA merge failed'}
            model_dir = merged_path
        
        if output_dir is None:
            output_dir = self.gguf_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 50)
        logger.info("Converting to GGUF...")
        logger.info("=" * 50)
        
        # التأكد من وجود السكربت
        script = self._ensure_gguf_script()
        if script is None:
            return {'success': False, 'error': 'GGUF script not available'}
        
        # تحويل إلى F16 أولاً
        f16_path = output_dir / "model-f16.gguf"
        
        cmd = [
            sys.executable, str(script),
            str(model_dir),
            "--outfile", str(f16_path),
            "--outtype", "f16"
        ]
        
        logger.info(f"Converting to F16...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"GGUF conversion failed: {result.stderr[:500]}")
            return {
                'success': False,
                'error': f'GGUF conversion failed: {result.stderr[:500]}'
            }
        
        logger.info(f"✅ F16 model: {f16_path}")
        
        # التكميم إذا طُلب
        if quantize:
            quantize_bin = shutil.which("quantize") or shutil.which("llama-quantize")
            
            if quantize_bin:
                q_type = quantization_type.value
                q_path = output_dir / f"model-{q_type}.gguf"
                
                cmd = [
                    quantize_bin,
                    str(f16_path),
                    str(q_path),
                    q_type.upper()
                ]
                
                logger.info(f"Quantizing to {q_type.upper()}...")
                result = subprocess.run(cmd)
                
                if result.returncode == 0:
                    logger.info(f"✅ Quantized model: {q_path}")
                    
                    # حذف F16 إذا كان التكميم ناجحاً
                    if f16_path.exists():
                        f16_path.unlink()
                else:
                    logger.warning("Quantization failed, keeping F16")
            else:
                logger.warning("quantize binary not found, keeping F16")
                logger.info("Install llama.cpp to enable quantization")
        
        duration = time.time() - start_time
        
        # حساب الحجم
        total_size = sum(
            f.stat().st_size for f in output_dir.glob("*.gguf")
        )
        size_mb = total_size / 1024 / 1024
        
        # التسجيل
        self._register_version("gguf", output_dir, size_mb)
        
        return {
            'success': True,
            'output_dir': str(output_dir),
            'size_mb': round(size_mb, 2),
            'duration_seconds': round(duration, 2)
        }
    
    def _register_version(
        self,
        format_type: str,
        path: Path,
        size_mb: float
    ):
        """
        تسجيل إصدار النموذج
        Register model version
        """
        registry = {'versions': [], 'current': None}
        
        if self.registry_path.exists():
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
        
        version = len(registry.get('versions', [])) + 1
        
        # نسخ للإصدار
        version_dir = path.parent / f"v{version}_{format_type}"
        if version_dir.exists():
            shutil.rmtree(version_dir)
        shutil.copytree(path, version_dir)
        
        registry.setdefault('versions', []).append({
            'version': version,
            'format': format_type,
            'path': str(version_dir),
            'timestamp': datetime.now().isoformat(),
            'size_mb': round(size_mb, 1)
        })
        
        registry['current'] = version
        
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Registered version v{version} ({format_type})")
    
    def batch_convert(
        self,
        formats: List[ConversionFormat],
        model_dir: Optional[Path] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        تحويل دفعي
        Batch convert to multiple formats
        
        Args:
            formats: List of formats to convert
            model_dir: Model directory
            
        Returns:
            Dict of conversion results
        """
        results = {}
        
        logger.info("=" * 60)
        logger.info("Batch Conversion")
        logger.info("=" * 60)
        
        for fmt in formats:
            logger.info(f"\nConverting to {fmt.value}...")
            
            if fmt == ConversionFormat.ONNX:
                result = self.convert_to_onnx(model_dir)
            elif fmt == ConversionFormat.GGUF:
                result = self.convert_to_gguf(model_dir)
            else:
                result = {'success': False, 'error': f'Format {fmt.value} not supported'}
            
            results[fmt.value] = result
            
            if result.get('success'):
                logger.info(f"✅ {fmt.value}: Success")
            else:
                logger.error(f"❌ {fmt.value}: {result.get('error', 'Unknown error')}")
        
        # ملخص
        successful = sum(1 for r in results.values() if r.get('success'))
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Results: {successful}/{len(formats)} successful")
        logger.info(f"{'=' * 60}")
        
        return results
    
    def validate_conversion(
        self,
        path: Path,
        format_type: ConversionFormat
    ) -> bool:
        """
        التحقق من صحة التحويل
        Validate converted model
        
        Args:
            path: Model path
            format_type: Expected format
            
        Returns:
            True if valid
        """
        if not path.exists():
            return False
        
        if format_type == ConversionFormat.ONNX:
            # التحقق من وجود ملفات ONNX
            return any(path.rglob("*.onnx"))
        
        elif format_type == ConversionFormat.GGUF:
            # التحقق من وجود ملفات GGUF
            return any(path.glob("*.gguf"))
        
        elif format_type == ConversionFormat.SAFETENSORS:
            # التحقق من وجود safetensors
            return any(path.glob("*.safetensors"))
        
        return True


import time


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Converter")
    parser.add_argument('--onnx', action='store_true', help='Convert to ONNX')
    parser.add_argument('--gguf', action='store_true', help='Convert to GGUF')
    parser.add_argument('--all', action='store_true', help='Convert to all formats')
    parser.add_argument('--merge-only', action='store_true', help='Only merge LoRA')
    parser.add_argument('--model-dir', type=str, help='Model directory')
    
    args = parser.parse_args()
    
    converter = ModelConverter()
    
    if args.merge_only:
        model_dir = Path(args.model_dir) if args.model_dir else None
        success, path = converter.merge_lora(model_dir)
        print(f"Merge: {'Success' if success else 'Failed'} -> {path}")
    
    elif args.all:
        formats = [ConversionFormat.ONNX]
        if converter._ensure_gguf_script():
            formats.append(ConversionFormat.GGUF)
        results = converter.batch_convert(formats)
        print(json.dumps(results, indent=2))
    
    elif args.onnx:
        result = converter.convert_to_onnx(
            Path(args.model_dir) if args.model_dir else None
        )
        print(json.dumps(result, indent=2))
    
    elif args.gguf:
        result = converter.convert_to_gguf(
            Path(args.model_dir) if args.model_dir else None
        )
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
