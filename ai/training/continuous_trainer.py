"""
Continuous Trainer - التدريب المستمر
Merged from: continuous-train.py, auto-finetune.py

Features / المميزات:
  • Watch for new data
  • Auto-trigger training
  • Curriculum learning
  • Progressive training
  • Model versioning
  • Idle detection
  • Background training

PyTorch 2.x + CUDA 12.x Compatible
"""

import json
import os
import sys
import time
import psutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from threading import Thread, Event
import asyncio

logger = logging.getLogger(__name__)


class TrainingState(Enum):
    """حالة التدريب - Training state"""
    IDLE = "idle"
    WATCHING = "watching"
    COLLECTING = "collecting"
    VALIDATING = "validating"
    TRAINING = "training"
    CONVERTING = "converting"
    COMPLETED = "completed"
    FAILED = "failed"


class CurriculumType(Enum):
    """نوع المنهج - Curriculum type"""
    JAVASCRIPT = "js"
    PYTHON = "python"
    WEB = "web"
    LARAVEL = "laravel"
    SECURITY = "security"
    AI = "ai"
    SURVIVAL = "survival"
    FACTORY = "factory"
    ALL = "all"


# تعريف المناهج - Curriculum definitions
CURRICULA: Dict[str, Dict] = {
    "js": {
        "name": "JavaScript Mastery",
        "topics": [
            "JavaScript variables let const var scope",
            "JavaScript functions arrow closures hoisting",
            "JavaScript promises async await error handling",
            "JavaScript classes inheritance prototypes OOP",
            "JavaScript modules import export ES6",
            "JavaScript array methods map filter reduce",
            "Node.js express REST API middleware",
            "Node.js file system streams buffers",
        ]
    },
    "python": {
        "name": "Python Complete",
        "topics": [
            "Python data types variables strings lists dicts",
            "Python functions lambda decorators generators",
            "Python OOP classes inheritance polymorphism",
            "Python exception handling try except finally",
            "Python async programming asyncio await",
            "Python Flask web framework routing templates",
            "Python FastAPI REST API async endpoints",
            "PyTorch training pipeline DataLoader optimizer",
        ]
    },
    "web": {
        "name": "Full Stack Web Development",
        "topics": [
            "HTML5 semantic elements forms accessibility",
            "CSS3 flexbox grid layout responsive design",
            "React components JSX props state hooks",
            "React useState useEffect useContext",
            "TypeScript types interfaces generics enums",
            "REST API design best practices versioning",
            "GraphQL queries mutations subscriptions",
            "WebSocket real-time Socket.io implementation",
        ]
    },
    "laravel": {
        "name": "Laravel PHP Framework",
        "topics": [
            "Laravel routing controllers middleware groups",
            "Laravel Blade templates components layouts",
            "Laravel Eloquent ORM models relationships",
            "Laravel authentication Breeze Sanctum",
            "Laravel validation form requests rules",
            "Laravel API resources JSON responses",
            "Laravel queues jobs workers scheduling",
        ]
    },
    "security": {
        "name": "Security & DevOps",
        "topics": [
            "Linux command line bash scripting permissions",
            "Network security TCP/IP DNS firewalls",
            "Web security OWASP top 10 vulnerabilities",
            "Docker security best practices scanning",
            "Kubernetes basics pods services deployments",
            "SSL TLS HTTPS certificate management",
        ]
    },
    "ai": {
        "name": "AI & Machine Learning",
        "topics": [
            "Machine learning supervised unsupervised basics",
            "Neural networks perceptron backpropagation",
            "Transformer architecture attention mechanism",
            "NLP tokenization embeddings word2vec",
            "Large Language Models GPT BERT training",
            "Fine-tuning LLM LoRA QLoRA PEFT",
            "RAG retrieval augmented generation",
        ]
    },
    "survival": {
        "name": "البقاء بعد الكارثة",
        "topics": [
            "emergency survival shelter building techniques",
            "water purification filtration methods survival",
            "food preservation drying salting fermentation",
            "fire starting methods primitive tools survival",
            "solar power DIY panels battery systems",
            "basic surgery wound care field medicine",
            "food storage long term preservation techniques",
        ]
    },
    "factory": {
        "name": "التصنيع من الصفر",
        "topics": [
            "PCB design manufacturing etching soldering",
            "battery lithium ion cell assembly pack",
            "thermal management heatsink cooling design",
            "quality testing reliability burn-in testing",
            "Lean manufacturing Six Sigma quality",
            "factory ERP MES PLM WMS systems",
        ]
    }
}


@dataclass
class ContinuousTrainingConfig:
    """
    إعدادات التدريب المستمر - Continuous training configuration
    
    Attributes:
        min_samples: الحد الأدنى من العينات للبدء
        check_interval: فترة الفحص (ثواني)
        min_idle_cpu: أقصى استخدام CPU للتدريب (%)
        min_hours_between: الحد الأدنى بين التدريبات (ساعات)
        enable_curriculum: تفعيل التعلم بالمنهج
        curriculum_cycle: قائمة المناهج للتناوب
        max_pulse_minutes: وقت التدريب الأقصى (0 = غير محدود)
        auto_convert: تحويل تلقائي بعد التدريب
        train_chat_model: تدريب نموذج المحادثة
    """
    min_samples: int = 200
    check_interval: int = 3600  # 1 hour
    min_idle_cpu: float = 40.0
    min_hours_between: float = 12.0
    enable_curriculum: bool = False
    curriculum_cycle: List[str] = field(default_factory=lambda: [
        "js", "python", "web", "laravel", "security", "ai"
    ])
    max_pulse_minutes: int = 0
    auto_convert: bool = True
    train_chat_model: bool = False


@dataclass
class TrainingSession:
    """جلسة تدريب - Training session"""
    id: str
    started_at: datetime
    curriculum: str
    sample_count: int
    state: TrainingState
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class ModelVersion:
    """إصدار النموذج - Model version"""
    version: int
    timestamp: str
    sample_count: int
    curriculum: str
    path: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class ContinuousTrainer:
    """
    المدرب المستمر - Continuous Trainer
    
    يدير التدريب المستمر:
    - مراقبة البيانات الجديدة
    - التدريب التلقائي عند توفر الموارد
    - التعلم بالمنهج (Curriculum Learning)
    - إدارة إصدارات النماذج
    
    Manages continuous training:
    - Watch for new data
    - Auto-train when resources available
    - Curriculum learning
    - Model versioning
    """
    
    def __init__(
        self,
        config: Optional[ContinuousTrainingConfig] = None,
        base_dir: Optional[Path] = None,
        advanced_trainer=None
    ):
        """
        Initialize continuous trainer
        
        Args:
            config: Training configuration
            base_dir: Base project directory
            advanced_trainer: AdvancedTrainer instance
        """
        self.config = config or ContinuousTrainingConfig()
        self.base_dir = base_dir or Path(__file__).parent.parent.parent
        
        self.training_dir = self.base_dir / "training" / "output"
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        self.knowledge_dir = self.data_dir / "knowledge"
        self.learning_dir = self.data_dir / "learning"
        
        # Registry
        self.registry_path = self.models_dir / "model-registry.json"
        self.state_path = self.learning_dir / "continuous-trainer-state.json"
        
        # State
        self.state = TrainingState.IDLE
        self.current_session: Optional[TrainingSession] = None
        self.sessions: List[TrainingSession] = []
        self._stop_event = Event()
        self._watch_thread: Optional[Thread] = None
        
        # Advanced trainer (imported lazily)
        self._advanced_trainer_class = advanced_trainer
        
        # Curriculum state
        self._curriculum_index = 0
        
        # Create directories
        for d in [self.training_dir, self.models_dir, self.learning_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load state
        self._load_state()
        
        logger.info("=" * 60)
        logger.info("Continuous Trainer - المدرب المستمر")
        logger.info("=" * 60)
        logger.info(f"   Min samples: {self.config.min_samples}")
        logger.info(f"   Check interval: {self.config.check_interval}s")
        logger.info(f"   Curriculum: {self.config.enable_curriculum}")
    
    def _load_state(self):
        """تحميل الحالة - Load state"""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._curriculum_index = data.get('curriculum_index', 0)
                logger.info(f"Loaded state: curriculum_index={self._curriculum_index}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
    
    def _save_state(self):
        """حفظ الحالة - Save state"""
        data = {
            'curriculum_index': self._curriculum_index,
            'last_updated': datetime.now().isoformat(),
            'total_sessions': len(self.sessions)
        }
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_registry(self) -> Dict:
        """تحميل سجل النماذج - Load model registry"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'versions': [],
            'last_training': None,
            'last_sample_count': 0
        }
    
    def _save_registry(self, registry: Dict):
        """حفظ سجل النماذج - Save model registry"""
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
    
    def count_samples(self) -> int:
        """
        عد العينات المتاحة
        Count available training samples
        
        Returns:
            Total sample count
        """
        total = 0
        
        # ملفات training/output
        for json_file in self.training_dir.glob("*.json"):
            if json_file.name in ['training_report.json', 'validation_report.json']:
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    total += len(data)
                elif isinstance(data, dict):
                    total += len(data.get('samples', data.get('examples', [])))
            except:
                pass
        
        # بيانات المعرفة
        for kb_file in ['rag-knowledge-base.json', 'smart-learned-data.json']:
            kb_path = self.knowledge_dir / kb_file
            if kb_path.exists():
                try:
                    with open(kb_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        total += len(data)
                except:
                    pass
        
        return total
    
    def check_conditions(self) -> Tuple[bool, List[str]]:
        """
        فحص شروط التدريب
        Check training conditions
        
        Returns:
            (can_train, list_of_issues)
        """
        issues = []
        
        # فحص CPU
        cpu = psutil.cpu_percent(interval=2)
        if cpu > self.config.min_idle_cpu:
            issues.append(f"CPU high: {cpu}% (limit: {self.config.min_idle_cpu}%)")
        
        # فحص آخر تدريب
        registry = self._load_registry()
        last_training = registry.get('last_training')
        if last_training:
            last = datetime.fromisoformat(last_training)
            hours_since = (datetime.now() - last).total_seconds() / 3600
            if hours_since < self.config.min_hours_between:
                issues.append(f"Last training {hours_since:.1f}h ago (min: {self.config.min_hours_between}h)")
        
        # فحص البيانات
        sample_count = self.count_samples()
        if sample_count < self.config.min_samples:
            issues.append(f"Insufficient data: {sample_count} (min: {self.config.min_samples})")
        
        return len(issues) == 0, issues
    
    def get_next_curriculum(self) -> Tuple[int, str]:
        """
        الحصول على المنهج التالي
        Get next curriculum
        
        Returns:
            (index, curriculum_name)
        """
        if not self.config.enable_curriculum:
            return -1, "all"
        
        cycle = self.config.curriculum_cycle
        next_idx = self._curriculum_index % len(cycle)
        self._curriculum_index = next_idx
        return next_idx, cycle[next_idx]
    
    def collect_data(self, curriculum: str = "all") -> int:
        """
        جمع البيانات (يمكن توسيعه بـ smart-learn)
        Collect training data
        
        Args:
            curriculum: Curriculum type
            
        Returns:
            Number of samples collected
        """
        logger.info(f"Collecting data (curriculum: {curriculum})...")
        
        # هنا يمكن استدعاء smart-learn أو مصادر أخرى
        # For now, just count existing data
        count = self.count_samples()
        logger.info(f"  Found {count} samples")
        return count
    
    def validate_data(self) -> bool:
        """
        التحقق من البيانات
        Validate training data
        
        Returns:
            True if validation passed
        """
        logger.info("Validating data...")
        
        try:
            from ai.training.evaluation_engine import EvaluationEngine
            
            engine = EvaluationEngine(base_dir=self.base_dir)
            report = engine.load_and_validate_directory(fix=True)
            
            logger.info(f"  Valid samples: {report.valid_samples}/{report.total_samples}")
            return report.valid_samples >= self.config.min_samples
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def train_model(
        self,
        mode: str = "completion",
        resume_from: Optional[str] = None
    ) -> bool:
        """
        تدريب النموذج
        Train model
        
        Args:
            mode: Training mode (completion/chat/extended)
            resume_from: Checkpoint to resume from
            
        Returns:
            True if training succeeded
        """
        logger.info(f"Starting training (mode: {mode})...")
        
        try:
            from ai.training.advanced_trainer import (
                AdvancedTrainer, TrainingConfig, TrainingMode
            )
            
            config = TrainingConfig(
                pulse_max_minutes=self.config.max_pulse_minutes
            )
            
            mode_map = {
                "completion": TrainingMode.COMPLETION,
                "chat": TrainingMode.CHAT,
                "extended": TrainingMode.EXTENDED
            }
            
            trainer = AdvancedTrainer(
                config=config,
                mode=mode_map.get(mode, TrainingMode.COMPLETION),
                base_dir=self.base_dir
            )
            
            result = trainer.train(resume_from_checkpoint=resume_from)
            
            if result.success:
                logger.info(f"Training completed: {result.output_dir}")
                return True
            else:
                logger.error(f"Training failed: {result.error_message}")
                return False
                
        except Exception as e:
            logger.exception("Training failed")
            return False
    
    def convert_model(self) -> bool:
        """
        تحويل النموذج
        Convert model to deployment formats
        
        Returns:
            True if conversion succeeded
        """
        if not self.config.auto_convert:
            return True
        
        logger.info("Converting model...")
        
        try:
            from ai.training.model_converter import ModelConverter
            
            converter = ModelConverter(base_dir=self.base_dir)
            
            # Convert to ONNX
            onnx_result = converter.convert_to_onnx()
            if onnx_result.get('success'):
                logger.info("ONNX conversion successful")
            
            # Convert to GGUF if chat model
            if self.config.train_chat_model:
                gguf_result = converter.convert_to_gguf()
                if gguf_result.get('success'):
                    logger.info("GGUF conversion successful")
            
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False
    
    def run_training_cycle(self) -> bool:
        """
        تشغيل دورة تدريب كاملة
        Run complete training cycle
        
        Returns:
            True if cycle succeeded
        """
        self.state = TrainingState.WATCHING
        
        # فحص الشروط
        can_train, issues = self.check_conditions()
        if not can_train:
            logger.info("Cannot train:")
            for issue in issues:
                logger.info(f"  - {issue}")
            return False
        
        # الحصول على المنهج
        cur_idx, cur_name = self.get_next_curriculum()
        if self.config.enable_curriculum:
            logger.info(f"Curriculum [{cur_idx + 1}/{len(self.config.curriculum_cycle)}]: {cur_name}")
        
        # إنشاء جلسة
        session = TrainingSession(
            id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            started_at=datetime.now(),
            curriculum=cur_name,
            sample_count=self.count_samples(),
            state=TrainingState.COLLECTING
        )
        self.current_session = session
        
        # جمع البيانات
        self.collect_data(cur_name)
        
        # التحقق
        self.state = TrainingState.VALIDATING
        if not self.validate_data():
            session.state = TrainingState.FAILED
            session.error = "Validation failed"
            return False
        
        # التدريب
        self.state = TrainingState.TRAINING
        if not self.train_model():
            session.state = TrainingState.FAILED
            session.error = "Training failed"
            return False
        
        # التحويل
        self.state = TrainingState.CONVERTING
        self.convert_model()
        
        # تدريب نموذج المحادثة
        if self.config.train_chat_model:
            logger.info("Training chat model...")
            self.train_model(mode="chat")
            try:
                from ai.training.model_converter import ModelConverter
                converter = ModelConverter(base_dir=self.base_dir)
                converter.convert_to_gguf()
            except:
                pass
        
        # تحديث السجل
        registry = self._load_registry()
        version = len(registry.get('versions', [])) + 1
        
        registry['last_training'] = datetime.now().isoformat()
        registry['last_sample_count'] = session.sample_count
        registry['last_curriculum'] = cur_name if self.config.enable_curriculum else 'all'
        
        registry.setdefault('versions', []).append({
            'version': version,
            'timestamp': registry['last_training'],
            'sample_count': session.sample_count,
            'curriculum': cur_name if self.config.enable_curriculum else 'all'
        })
        
        self._save_registry(registry)
        
        # إكمال الجلسة
        session.state = TrainingState.COMPLETED
        session.completed_at = datetime.now()
        self.sessions.append(session)
        
        # تحديث المنهج
        if self.config.enable_curriculum:
            self._curriculum_index += 1
            self._save_state()
        
        self.state = TrainingState.IDLE
        logger.info(f"Cycle complete. Version: {version}")
        
        return True
    
    def start_watching(self):
        """
        بدء المراقبة في الخلفية
        Start watching in background
        """
        logger.info(f"Starting watch mode (interval: {self.config.check_interval}s)")
        logger.info(f"Curricula: {', '.join(self.config.curriculum_cycle)}")
        
        self._stop_event.clear()
        
        def watch_loop():
            cycle_num = 0
            while not self._stop_event.is_set():
                cycle_num += 1
                logger.info(f"\n{'=' * 50}")
                logger.info(f"Cycle {cycle_num} at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                logger.info(f"{'=' * 50}")
                
                try:
                    self.run_training_cycle()
                except Exception as e:
                    logger.error(f"Cycle {cycle_num} error: {e}")
                
                logger.info(f"Sleeping {self.config.check_interval}s...")
                self._stop_event.wait(self.config.check_interval)
        
        self._watch_thread = Thread(target=watch_loop, daemon=True)
        self._watch_thread.start()
        logger.info("Watch thread started")
    
    def stop_watching(self):
        """إيقاف المراقبة - Stop watching"""
        logger.info("Stopping watch mode...")
        self._stop_event.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=5)
        logger.info("Watch mode stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        الحصول على الحالة
        Get current status
        
        Returns:
            Status dict
        """
        registry = self._load_registry()
        
        return {
            'state': self.state.value,
            'current_session': {
                'id': self.current_session.id if self.current_session else None,
                'curriculum': self.current_session.curriculum if self.current_session else None,
                'started_at': self.current_session.started_at.isoformat() if self.current_session else None
            },
            'samples': self.count_samples(),
            'min_samples': self.config.min_samples,
            'total_sessions': len(self.sessions),
            'registry': registry,
            'watching': self._watch_thread is not None and self._watch_thread.is_alive()
        }


def main():
    """Main entry point for continuous training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Trainer")
    parser.add_argument('--watch', action='store_true', help='Watch mode')
    parser.add_argument('--once', action='store_true', help='Run once')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--interval', type=int, default=3600, help='Check interval (seconds)')
    parser.add_argument('--min-samples', type=int, default=200, help='Minimum samples')
    
    args = parser.parse_args()
    
    config = ContinuousTrainingConfig(
        check_interval=args.interval,
        min_samples=args.min_samples,
        enable_curriculum=args.curriculum
    )
    
    trainer = ContinuousTrainer(config=config)
    
    if args.status:
        status = trainer.get_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
    elif args.watch:
        trainer.start_watching()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            trainer.stop_watching()
    elif args.once:
        trainer.run_training_cycle()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
