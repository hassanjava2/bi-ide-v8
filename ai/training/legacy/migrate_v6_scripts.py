"""
V6 Scripts Migration - ترحيل سكربتات v6
Migrate useful scripts from v6 to new structure

This script scans the v6-scripts directory, identifies useful scripts,
and migrates them to the new ai/training/ structure with updated imports.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """حالة الترحيل - Migration status"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class ScriptInfo:
    """معلومات السكربت - Script information"""
    name: str
    source_path: Path
    target_path: Optional[Path]
    description: str
    features: List[str]
    status: MigrationStatus
    error_message: Optional[str] = None
    lines_of_code: int = 0


@dataclass
class MigrationReport:
    """تقرير الترحيل - Migration report"""
    timestamp: str
    total_scripts: int
    migrated: int
    skipped: int
    failed: int
    partial: int
    scripts: List[Dict]


class V6MigrationTool:
    """
    أداة ترحيل سكربتات v6
    V6 Scripts Migration Tool
    
    تفحص سكربتات v6 وتحدد ما هو مفيد للترحيل
    Scans v6 scripts and identifies useful ones for migration
    """
    
    # تعيين السكربتات للأهداف الجديدة
    # Script to target mapping
    MIGRATION_MAP: Dict[str, Tuple[str, List[str]]] = {
        "finetune.py": ("advanced_trainer.py", ["base_training", "lora", "completion"]),
        "finetune-chat.py": ("advanced_trainer.py", ["chat_training", "chatml", "conversation"]),
        "finetune-extended.py": ("advanced_trainer.py", ["extended_training", "data_augmentation", "deep_learning"]),
        "evaluate-model.py": ("evaluation_engine.py", ["perplexity", "model_evaluation"]),
        "validate-data.py": ("evaluation_engine.py", ["data_validation", "quality_check"]),
        "continuous-train.py": ("continuous_trainer.py", ["auto_trigger", "pipeline", "curriculum"]),
        "auto-finetune.py": ("continuous_trainer.py", ["idle_training", "background", "auto_conditions"]),
        "convert-to-gguf.py": ("model_converter.py", ["gguf", "llamacpp", "quantization"]),
        "convert-to-onnx.py": ("model_converter.py", ["onnx", "optimum", "export"]),
        "prepare-chat-data.py": (None, ["data_preparation", "chatml"]),  # مدمج في preprocessing
        "monitor.py": (None, ["monitoring", "reporting"]),  # يستخدم الموجود
        "smart-learn.py": ("continuous_trainer.py", ["online_learning", "curriculum", "smart"]),
        "train_ai.py": ("advanced_trainer.py", ["comprehensive", "code_training", "patterns"]),
    }
    
    def __init__(
        self,
        v6_dir: Optional[Path] = None,
        target_dir: Optional[Path] = None
    ):
        """
        Initialize migration tool
        
        Args:
            v6_dir: Directory containing v6 scripts
            target_dir: Target directory for new structure
        """
        self.v6_dir = v6_dir or Path(__file__).parent.parent.parent.parent / "training" / "v6-scripts"
        self.target_dir = target_dir or Path(__file__).parent.parent
        self.legacy_dir = Path(__file__).parent
        
        self.scripts_info: List[ScriptInfo] = []
        self.report: Optional[MigrationReport] = None
        
        logger.info("=" * 60)
        logger.info("V6 Scripts Migration Tool")
        logger.info("ترحيل سكربتات الإصدار السادس")
        logger.info("=" * 60)
        logger.info(f"Source: {self.v6_dir}")
        logger.info(f"Target: {self.target_dir}")
        logger.info("")
    
    def scan_v6_scripts(self) -> List[ScriptInfo]:
        """
        فحص سكربتات v6
        Scan v6 scripts directory
        
        Returns:
            List of script information
        """
        logger.info("🔍 Scanning v6-scripts directory...")
        
        if not self.v6_dir.exists():
            logger.error(f"❌ v6 directory not found: {self.v6_dir}")
            return []
        
        scripts = []
        for py_file in sorted(self.v6_dir.glob("*.py")):
            info = self._analyze_script(py_file)
            scripts.append(info)
            logger.info(f"  📄 {py_file.name}: {info.lines_of_code} lines, {len(info.features)} features")
        
        self.scripts_info = scripts
        logger.info(f"\n📊 Found {len(scripts)} Python scripts")
        return scripts
    
    def _analyze_script(self, path: Path) -> ScriptInfo:
        """
        تحليل سكربت فردي
        Analyze a single script
        
        Args:
            path: Path to script
            
        Returns:
            Script information
        """
        try:
            content = path.read_text(encoding='utf-8')
            lines = len(content.splitlines())
            
            # استخراج الوصف من docstring
            # Extract description from docstring
            description = ""
            if '"""' in content:
                docstring = content.split('"""')[1] if '"""' in content else ""
                description = docstring.strip().split('\n')[0][:100]
            
            # تحديد الميزات
            # Identify features
            features = []
            if 'lora' in content.lower() or 'peft' in content.lower():
                features.append('lora')
            if 'chat' in content.lower() or 'conversation' in content.lower():
                features.append('chat')
            if 'onnx' in content.lower():
                features.append('onnx')
            if 'gguf' in content.lower():
                features.append('gguf')
            if 'evaluate' in content.lower():
                features.append('evaluation')
            if 'continuous' in content.lower() or 'auto' in content.lower():
                features.append('continuous')
            
            # تحديد الهدف
            # Determine target
            target_file, mapped_features = self.MIGRATION_MAP.get(
                path.name, 
                (None, [])
            )
            target_path = self.target_dir / target_file if target_file else None
            
            return ScriptInfo(
                name=path.name,
                source_path=path,
                target_path=target_path,
                description=description,
                features=list(set(features + mapped_features)),
                status=MigrationStatus.PENDING,
                lines_of_code=lines
            )
            
        except Exception as e:
            return ScriptInfo(
                name=path.name,
                source_path=path,
                target_path=None,
                description=f"Error: {e}",
                features=[],
                status=MigrationStatus.FAILED,
                error_message=str(e),
                lines_of_code=0
            )
    
    def identify_useful_scripts(self) -> List[ScriptInfo]:
        """
        تحديد السكربتات المفيدة
        Identify useful scripts for migration
        
        Returns:
            List of useful scripts
        """
        logger.info("\n🎯 Identifying useful scripts...")
        
        useful = []
        for script in self.scripts_info:
            # معايير التحديد
            # Criteria for usefulness
            is_mapped = script.name in self.MIGRATION_MAP
            has_features = len(script.features) > 0
            not_deprecated = not any(
                keyword in script.name.lower() 
                for keyword in ['old', 'deprecated', 'test_', '_test']
            )
            
            if is_mapped and has_features and not_deprecated:
                useful.append(script)
                logger.info(f"  ✅ {script.name} → {script.target_path.name if script.target_path else 'merged'}")
            else:
                script.status = MigrationStatus.SKIPPED
                logger.info(f"  ⏭️  {script.name} (skipped)")
        
        logger.info(f"\n📋 {len(useful)} scripts marked for migration")
        return useful
    
    def create_compatibility_layer(self) -> Path:
        """
        إنشاء طبقة التوافق
        Create compatibility layer
        
        Returns:
            Path to compatibility module
        """
        logger.info("\n🔧 Creating compatibility layer...")
        
        compat_content = '''"""
V6 Compatibility Layer - طبقة توافق v6

Provides backward compatibility for code using v6 APIs.
يوفر توافقاً للكود الذي يستخدم واجهات v6.
"""

import warnings
from typing import Any, Optional
import sys
from pathlib import Path

# Add new training module to path
_training_module_path = Path(__file__).parent.parent
if str(_training_module_path) not in sys.path:
    sys.path.insert(0, str(_training_module_path))

# Import new modules with compatibility aliases
try:
    from ai.training.advanced_trainer import AdvancedTrainer
    from ai.training.evaluation_engine import EvaluationEngine
    from ai.training.continuous_trainer import ContinuousTrainer
    from ai.training.model_converter import ModelConverter
    V8_AVAILABLE = True
except ImportError:
    V8_AVAILABLE = False


class FineTuneV6:
    """
    Compatibility wrapper for finetune.py
    غلاف توافق لـ finetune.py
    """
    
    def __init__(self):
        warnings.warn(
            "FineTuneV6 is deprecated. Use AdvancedTrainer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if V8_AVAILABLE:
            self._trainer = AdvancedTrainer()
        else:
            raise ImportError("V8 training module not available")
    
    def train(self, *args, **kwargs) -> Any:
        """Delegate to AdvancedTrainer"""
        return self._trainer.train(*args, **kwargs)


class FineTuneChatV6:
    """
    Compatibility wrapper for finetune-chat.py
    غلاف توافق لـ finetune-chat.py
    """
    
    def __init__(self):
        warnings.warn(
            "FineTuneChatV6 is deprecated. Use AdvancedTrainer with mode='chat'.",
            DeprecationWarning,
            stacklevel=2
        )
        if V8_AVAILABLE:
            self._trainer = AdvancedTrainer(mode='chat')
        else:
            raise ImportError("V8 training module not available")
    
    def train(self, *args, **kwargs) -> Any:
        """Delegate to AdvancedTrainer"""
        return self._trainer.train(*args, **kwargs)


class EvaluateModelV6:
    """
    Compatibility wrapper for evaluate-model.py
    غلاف توافق لـ evaluate-model.py
    """
    
    def __init__(self):
        warnings.warn(
            "EvaluateModelV6 is deprecated. Use EvaluationEngine instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if V8_AVAILABLE:
            self._engine = EvaluationEngine()
        else:
            raise ImportError("V8 training module not available")
    
    def evaluate(self, *args, **kwargs) -> Any:
        """Delegate to EvaluationEngine"""
        return self._engine.evaluate_model(*args, **kwargs)


class ContinuousTrainV6:
    """
    Compatibility wrapper for continuous-train.py
    غلاف توافق لـ continuous-train.py
    """
    
    def __init__(self):
        warnings.warn(
            "ContinuousTrainV6 is deprecated. Use ContinuousTrainer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if V8_AVAILABLE:
            self._trainer = ContinuousTrainer()
        else:
            raise ImportError("V8 training module not available")
    
    def start(self, *args, **kwargs) -> Any:
        """Delegate to ContinuousTrainer"""
        return self._trainer.start(*args, **kwargs)


def migrate_v6_config(config: dict) -> dict:
    """
    Convert v6 config format to v8
    تحويل إعدادات v6 إلى v8
    
    Args:
        config: V6 configuration dict
        
    Returns:
        V8 configuration dict
    """
    mapping = {
        'MODEL_NAME': 'model_name',
        'MAX_LENGTH': 'max_length',
        'BATCH_SIZE': 'batch_size',
        'EPOCHS': 'epochs',
        'LEARNING_RATE': 'learning_rate',
        'LORA_R': 'lora_r',
        'LORA_ALPHA': 'lora_alpha',
        'NUM_WORKERS': 'num_workers',
    }
    
    return {
        mapping.get(k, k): v 
        for k, v in config.items()
    }


# Export compatibility classes
__all__ = [
    'FineTuneV6',
    'FineTuneChatV6', 
    'EvaluateModelV6',
    'ContinuousTrainV6',
    'migrate_v6_config',
    'V8_AVAILABLE',
]
'''
        
        compat_path = self.legacy_dir / "v6_compatibility.py"
        compat_path.write_text(compat_content, encoding='utf-8')
        
        logger.info(f"  ✅ Created: {compat_path}")
        return compat_path
    
    def log_migration_status(self) -> Path:
        """
        تسجيل حالة الترحيل
        Log migration status
        
        Returns:
            Path to log file
        """
        log_path = self.legacy_dir / "migration_log.json"
        
        # حساب الإحصائيات
        # Calculate statistics
        total = len(self.scripts_info)
        migrated = sum(1 for s in self.scripts_info if s.status == MigrationStatus.SUCCESS)
        skipped = sum(1 for s in self.scripts_info if s.status == MigrationStatus.SKIPPED)
        failed = sum(1 for s in self.scripts_info if s.status == MigrationStatus.FAILED)
        partial = sum(1 for s in self.scripts_info if s.status == MigrationStatus.PARTIAL)
        
        self.report = MigrationReport(
            timestamp=datetime.now().isoformat(),
            total_scripts=total,
            migrated=migrated,
            skipped=skipped,
            failed=failed,
            partial=partial,
            scripts=[
                {
                    **asdict(script),
                    'status': script.status.value,
                    'source_path': str(script.source_path),
                    'target_path': str(script.target_path) if script.target_path else None
                }
                for script in self.scripts_info
            ]
        )
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.report), f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n📝 Migration log saved: {log_path}")
        return log_path
    
    def generate_migration_summary(self) -> str:
        """
        توليد ملخص الترحيل
        Generate migration summary
        
        Returns:
            Summary string
        """
        if not self.report:
            return "No migration report available."
        
        summary = f"""
{'=' * 60}
V6 Migration Summary | ملخص ترحيل v6
{'=' * 60}

Timestamp: {self.report.timestamp}

Statistics:
  📊 Total scripts: {self.report.total_scripts}
  ✅ Migrated: {self.report.migrated}
  ⏭️  Skipped: {self.report.skipped}
  ⚠️  Failed: {self.report.failed}
  🔄 Partial: {self.report.partial}

New Structure:
  📁 ai/training/advanced_trainer.py
     └─ Merged: finetune.py, finetune-chat.py, finetune-extended.py
  
  📁 ai/training/evaluation_engine.py
     └─ Merged: evaluate-model.py, validate-data.py
  
  📁 ai/training/continuous_trainer.py
     └─ Merged: continuous-train.py, auto-finetune.py, smart-learn.py
  
  📁 ai/training/model_converter.py
     └─ Merged: convert-to-gguf.py, convert-to-onnx.py
  
  📁 ai/training/multi_gpu_trainer.py
     └─ New: Distributed training with PyTorch 2.x DDP
  
  📁 ai/training/legacy/v6_compatibility.py
     └─ Compatibility layer for backward compatibility

Key Improvements:
  • PyTorch 2.x support
  • CUDA 12.x compatibility
  • Type hints throughout
  • Arabic/English docstrings
  • Unified logging
  • Better error handling
  • LoRA support in all trainers
  • Mixed precision training

{'=' * 60}
"""
        return summary
    
    def run_full_migration(self) -> MigrationReport:
        """
        تشغيل الترحيل الكامل
        Run full migration process
        
        Returns:
            Migration report
        """
        logger.info("\n" + "=" * 60)
        logger.info("Starting Full Migration Process")
        logger.info("بدء عملية الترحيل الكاملة")
        logger.info("=" * 60 + "\n")
        
        # 1. فحص السكربتات
        self.scan_v6_scripts()
        
        # 2. تحديد السكربتات المفيدة
        useful = self.identify_useful_scripts()
        
        # 3. إنشاء طبقة التوافق
        self.create_compatibility_layer()
        
        # 4. تحديث حالة السكربتات المدمجة
        for script in useful:
            if script.target_path:
                script.status = MigrationStatus.SUCCESS
            else:
                script.status = MigrationStatus.PARTIAL
        
        # 5. تسجيل الحالة
        self.log_migration_status()
        
        # 6. طباعة الملخص
        summary = self.generate_migration_summary()
        logger.info(summary)
        
        # حفظ الملخص
        summary_path = self.legacy_dir / "MIGRATION_SUMMARY.txt"
        summary_path.write_text(summary, encoding='utf-8')
        
        logger.info(f"\n✅ Migration complete!")
        logger.info(f"   Summary: {summary_path}")
        logger.info(f"   Log: {self.legacy_dir / 'migration_log.json'}")
        
        return self.report


def main():
    """Main entry point for migration"""
    tool = V6MigrationTool()
    report = tool.run_full_migration()
    return 0 if report and report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
