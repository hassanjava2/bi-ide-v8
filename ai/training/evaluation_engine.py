"""
Evaluation Engine - محرك التقييم
Merged from: evaluate-model.py, validate-data.py

Features / المميزات:
  • Model evaluation with perplexity calculation
  • Data validation and quality checks
  • Benchmark suite support
  • BLEU/ROUGE scores
  • Custom metrics
  • Report generation

PyTorch 2.x + CUDA 12.x Compatible
"""

import json
import os
import re
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """حالة التحقق - Validation status"""
    VALID = "valid"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    DUPLICATE = "duplicate"
    INVALID_FORMAT = "invalid_format"
    LOW_QUALITY = "low_quality"


class Language(Enum):
    """اللغة - Language"""
    ARABIC = "ar"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """نتيجة التحقق - Validation result"""
    status: ValidationStatus
    sample: Dict[str, Any]
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """تقرير جودة البيانات - Data quality report"""
    timestamp: str
    total_samples: int
    valid_samples: int
    invalid_samples: int
    duplicates_removed: int
    by_language: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    samples: List[Dict] = field(default_factory=list)
    invalid_examples: List[Dict] = field(default_factory=list)
    
    @property
    def valid_ratio(self) -> float:
        """نسبة الصلاحية - Valid ratio"""
        if self.total_samples == 0:
            return 0.0
        return self.valid_samples / self.total_samples


@dataclass
class EvaluationMetrics:
    """مقاييس التقييم - Evaluation metrics"""
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_l: Optional[float] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'perplexity': self.perplexity,
            'bleu_score': self.bleu_score,
            'rouge_l': self.rouge_l,
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            **self.custom_metrics
        }


@dataclass
class ModelEvaluationReport:
    """تقرير تقييم النموذج - Model evaluation report"""
    timestamp: str
    model_path: str
    metrics: EvaluationMetrics
    samples_evaluated: int
    duration_seconds: float
    error: Optional[str] = None
    comparison_with_previous: Optional[Dict] = None


class EvaluationEngine:
    """
    محرك التقييم - Evaluation Engine
    
    يوفر وظائف التقييم الشاملة:
    - تقييم النماذج (perplexity, BLEU, ROUGE)
    - التحقق من جودة البيانات
    - اكتشاف التكرار
    - توليد التقارير
    
    Provides comprehensive evaluation:
    - Model evaluation (perplexity, BLEU, ROUGE)
    - Data quality validation
    - Duplicate detection
    - Report generation
    """
    
    # الثوابت - Constants
    MIN_LENGTH = 10
    MAX_LENGTH = 8000
    ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize evaluation engine
        
        Args:
            base_dir: Base project directory
            output_dir: Output directory for reports
        """
        self.base_dir = base_dir or Path(__file__).parent.parent.parent
        
        if output_dir is None:
            self.output_dir = self.base_dir / "training" / "output"
        else:
            self.output_dir = output_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔍 EvaluationEngine initialized")
        logger.info(f"   Output: {self.output_dir}")
    
    # ═══════════════════════════════════════════════════════════════
    # وظائف التحقق من البيانات - Data Validation Functions
    # ═══════════════════════════════════════════════════════════════
    
    def count_arabic(self, text: str) -> int:
        """عد الحروف العربية - Count Arabic characters"""
        return len(self.ARABIC_PATTERN.findall(text)) if text else 0
    
    def count_english(self, text: str) -> int:
        """عد الكلمات الإنجليزية - Count English words"""
        return len(re.findall(r'[a-zA-Z]+', text or ''))
    
    def detect_language(self, text: str) -> Language:
        """اكتشاف اللغة - Detect language"""
        arabic_count = self.count_arabic(text)
        english_count = self.count_english(text)
        
        if arabic_count > english_count:
            return Language.ARABIC
        elif english_count > arabic_count:
            return Language.ENGLISH
        elif arabic_count > 0 and english_count > 0:
            return Language.MIXED
        return Language.UNKNOWN
    
    def get_sample_signature(self, sample: Dict[str, Any]) -> str:
        """
        توليد توقيع للعينة (للكشف عن التكرار)
        Generate sample signature (for duplicate detection)
        
        Args:
            sample: Data sample
            
        Returns:
            Hash signature
        """
        if isinstance(sample, dict):
            inp = sample.get('input', sample.get('prompt', sample.get('question', sample.get('instruction', ''))))
            out = sample.get('output', sample.get('response', sample.get('completion', sample.get('answer', ''))))
            key = f"{inp}|{out}"
        else:
            key = str(sample)
        return hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]
    
    def get_text_length(self, sample: Dict[str, Any]) -> int:
        """حساب طول النص - Calculate text length"""
        d = sample.get('data', sample) if isinstance(sample, dict) and 'data' in sample else sample
        
        if not isinstance(d, dict):
            return len(str(d))
        
        inp = d.get('input', d.get('prompt', d.get('question', d.get('instruction', '')))) or ''
        out = d.get('output', d.get('response', d.get('completion', d.get('answer', '')))) or ''
        return len(inp) + len(out)
    
    def validate_sample(
        self,
        sample: Dict[str, Any],
        seen_signatures: Optional[set] = None
    ) -> ValidationResult:
        """
        التحقق من عينة واحدة
        Validate a single sample
        
        Args:
            sample: Data sample
            seen_signatures: Set of seen signatures for duplicate detection
            
        Returns:
            Validation result
        """
        # التحقق من التكرار
        if seen_signatures is not None:
            sig = self.get_sample_signature(sample)
            if sig in seen_signatures:
                return ValidationResult(
                    status=ValidationStatus.DUPLICATE,
                    sample=sample,
                    message="Duplicate sample detected"
                )
            seen_signatures.add(sig)
        
        # التحقق من الطول
        length = self.get_text_length(sample)
        if length < self.MIN_LENGTH:
            return ValidationResult(
                status=ValidationStatus.TOO_SHORT,
                sample=sample,
                message=f"Sample too short ({length} < {self.MIN_LENGTH})",
                details={'length': length}
            )
        
        if length > self.MAX_LENGTH:
            return ValidationResult(
                status=ValidationStatus.TOO_LONG,
                sample=sample,
                message=f"Sample too long ({length} > {self.MAX_LENGTH})",
                details={'length': length}
            )
        
        # التحقق من المحتوى
        d = sample.get('data', sample) if isinstance(sample, dict) and 'data' in sample else sample
        if isinstance(d, dict):
            inp = d.get('input', d.get('prompt', d.get('question', d.get('instruction', ''))))
            out = d.get('output', d.get('response', d.get('completion', d.get('answer', ''))))
            
            if not inp or not out:
                return ValidationResult(
                    status=ValidationStatus.INVALID_FORMAT,
                    sample=sample,
                    message="Missing input or output"
                )
        
        return ValidationResult(
            status=ValidationStatus.VALID,
            sample=sample,
            message="Valid sample"
        )
    
    def validate_data(
        self,
        data: List[Dict[str, Any]],
        remove_duplicates: bool = True,
        fix: bool = False
    ) -> DataQualityReport:
        """
        التحقق من مجموعة بيانات
        Validate a dataset
        
        Args:
            data: List of data samples
            remove_duplicates: Whether to remove duplicates
            fix: Whether to fix issues automatically
            
        Returns:
            Data quality report
        """
        logger.info(f"🔍 Validating {len(data)} samples...")
        
        seen = set() if remove_duplicates else None
        results = []
        valid_list = []
        by_language = defaultdict(int)
        by_status = defaultdict(int)
        
        for sample in data:
            result = self.validate_sample(sample, seen)
            results.append(result)
            by_status[result.status.value] += 1
            
            if result.status == ValidationStatus.VALID:
                valid_list.append(sample)
                # تحديد اللغة
                d = sample.get('data', sample) if isinstance(sample, dict) and 'data' in sample else sample
                if isinstance(d, dict):
                    inp = d.get('input', d.get('prompt', d.get('question', d.get('instruction', '')))) or ''
                    lang = self.detect_language(str(inp))
                    by_language[lang.value] += 1
        
        duplicates = by_status.get(ValidationStatus.DUPLICATE.value, 0)
        
        report = DataQualityReport(
            timestamp=datetime.now().isoformat(),
            total_samples=len(data),
            valid_samples=len(valid_list),
            invalid_samples=len(data) - len(valid_list) - duplicates,
            duplicates_removed=duplicates,
            by_language=dict(by_language),
            by_status=dict(by_status),
            samples=valid_list if fix else [],
            invalid_examples=[
                {'status': r.status.value, 'message': r.message, 'details': r.details}
                for r in results if r.status != ValidationStatus.VALID
            ][:100]
        )
        
        logger.info(f"   Total: {report.total_samples}")
        logger.info(f"   Valid: {report.valid_samples}")
        logger.info(f"   Duplicates: {report.duplicates_removed}")
        logger.info(f"   Valid ratio: {report.valid_ratio:.1%}")
        
        return report
    
    def load_and_validate_directory(
        self,
        directory: Optional[Path] = None,
        pattern: str = "*.json",
        fix: bool = False
    ) -> DataQualityReport:
        """
        تحميل والتحقق من مجلد
        Load and validate directory
        
        Args:
            directory: Directory to load from
            pattern: File pattern
            fix: Whether to save fixed data
            
        Returns:
            Data quality report
        """
        if directory is None:
            directory = self.base_dir / "training" / "output"
        
        all_samples = []
        
        for json_file in directory.rglob(pattern):
            if json_file.name.startswith('.'):
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        all_samples.append({
                            'source': str(json_file),
                            'index': i,
                            'data': item
                        })
                elif isinstance(data, dict):
                    if 'samples' in data:
                        for i, item in enumerate(data['samples']):
                            all_samples.append({
                                'source': str(json_file),
                                'index': i,
                                'data': item
                            })
            except Exception as e:
                logger.warning(f"   ⚠️ {json_file}: {e}")
        
        logger.info(f"📂 Loaded {len(all_samples)} raw samples from {directory}")
        
        report = self.validate_data(all_samples, fix=fix)
        
        # حفظ البيانات المُصححة
        if fix and report.samples:
            output_path = self.output_dir / "validated_training_data.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.samples, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved validated data: {output_path}")
        
        # حفظ التقرير
        report_path = self.output_dir / "validation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': report.timestamp,
                'total': report.total_samples,
                'valid': report.valid_samples,
                'duplicates_removed': report.duplicates_removed,
                'by_language': report.by_language,
                'by_status': report.by_status,
                'valid_ratio': report.valid_ratio
            }, f, ensure_ascii=False, indent=2)
        
        return report
    
    # ═══════════════════════════════════════════════════════════════
    # وظائف تقييم النموذج - Model Evaluation Functions
    # ═══════════════════════════════════════════════════════════════
    
    def compute_perplexity(
        self,
        model_path: Union[str, Path],
        samples: List[Dict[str, Any]],
        max_samples: int = 50,
        max_length: int = 512
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        حساب الـ perplexity
        Compute perplexity
        
        Args:
            model_path: Path to model
            samples: Validation samples
            max_samples: Maximum samples to evaluate
            max_length: Maximum sequence length
            
        Returns:
            (perplexity, error_message)
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            return None, f"Missing dependencies: {e}"
        
        try:
            logger.info(f"🤖 Loading model: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            model.eval()
            
            total_loss = 0.0
            count = 0
            
            for sample in samples[:max_samples]:
                text = self._get_text_from_sample(sample)
                if len(text) < 20:
                    continue
                
                inputs = tokenizer(
                    text[:max_length],
                    return_tensors='pt',
                    truncation=True
                )
                
                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        labels=inputs['input_ids']
                    )
                    total_loss += outputs.loss.item()
                    count += 1
            
            if count == 0:
                return None, "No valid samples"
            
            avg_loss = total_loss / count
            perplexity = float(torch.exp(torch.tensor(avg_loss)).item())
            
            logger.info(f"   Perplexity: {perplexity:.2f} (evaluated on {count} samples)")
            return perplexity, None
            
        except Exception as e:
            return None, str(e)
    
    def _get_text_from_sample(self, sample: Any) -> str:
        """استخراج النص من العينة - Extract text from sample"""
        if isinstance(sample, dict):
            # Handle different formats
            inp = sample.get('input', sample.get('prompt', sample.get('question', sample.get('instruction', '')))) or ''
            out = sample.get('output', sample.get('response', sample.get('completion', sample.get('answer', '')))) or ''
            
            # Handle nested 'data' key
            if not inp and not out and 'data' in sample:
                return self._get_text_from_sample(sample['data'])
            
            return str(inp) + '\n' + str(out)
        return str(sample)
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Optional[float]:
        """
        حساب BLEU score
        Compute BLEU score
        
        Args:
            predictions: Predicted texts
            references: Reference texts
            
        Returns:
            BLEU score
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            scores = []
            smooth = SmoothingFunction()
            
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    smoothing_function=smooth.method1
                )
                scores.append(score)
            
            return float(np.mean(scores)) if scores else None
            
        except ImportError:
            logger.warning("nltk not installed, skipping BLEU")
            return None
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            return None
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        حساب ROUGE scores
        Compute ROUGE scores
        
        Args:
            predictions: Predicted texts
            references: Reference texts
            
        Returns:
            Dict with ROUGE scores
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                for key in scores:
                    scores[key].append(score[key].fmeasure)
            
            return {
                key: float(np.mean(values))
                for key, values in scores.items()
            }
            
        except ImportError:
            logger.warning("rouge_score not installed, skipping ROUGE")
            return None
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            return None
    
    def evaluate_model(
        self,
        model_path: Union[str, Path],
        validation_data: Optional[List[Dict]] = None,
        compute_bleu_rouge: bool = False
    ) -> ModelEvaluationReport:
        """
        تقييم النموذج الشامل
        Comprehensive model evaluation
        
        Args:
            model_path: Path to model
            validation_data: Validation samples
            compute_bleu_rouge: Whether to compute BLEU/ROUGE
            
        Returns:
            Evaluation report
        """
        start_time = time.time()
        model_path = Path(model_path)
        
        logger.info("=" * 50)
        logger.info("Model Evaluation - تقييم النموذج")
        logger.info("=" * 50)
        
        # البحث عن النموذج
        if not model_path.exists():
            # محاولة العثور على النموذج
            candidates = [
                self.base_dir / "models" / "merged",
                self.base_dir / "models" / "finetuned-extended",
                self.base_dir / "models" / "finetuned",
            ]
            for candidate in candidates:
                if candidate.exists():
                    model_path = candidate
                    break
        
        if not model_path.exists():
            return ModelEvaluationReport(
                timestamp=datetime.now().isoformat(),
                model_path=str(model_path),
                metrics=EvaluationMetrics(),
                samples_evaluated=0,
                duration_seconds=0,
                error=f"Model not found: {model_path}"
            )
        
        # تحميل بيانات التحقق
        if validation_data is None:
            validation_path = self.output_dir / "validated_training_data.json"
            if validation_path.exists():
                with open(validation_path, 'r', encoding='utf-8') as f:
                    validation_data = json.load(f)
            else:
                all_data_path = self.output_dir / "all_training_data.json"
                if all_data_path.exists():
                    with open(all_data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        validation_data = data if isinstance(data, list) else data.get('samples', [])
        
        validation_data = validation_data or []
        
        # حساب Perplexity
        perplexity, error = self.compute_perplexity(model_path, validation_data)
        
        metrics = EvaluationMetrics(perplexity=perplexity)
        
        # حساب BLEU/ROUGE إذا طُلب
        if compute_bleu_rouge and validation_data:
            # هذا يتطلب توليد تنبؤات من النموذج
            # يمكن توسيعه لاحقاً
            pass
        
        duration = time.time() - start_time
        
        # تحميل التقرير السابق للمقارنة
        prev_report = None
        prev_report_path = self.output_dir / "evaluation_report_previous.json"
        current_report_path = self.output_dir / "evaluation_report.json"
        
        if current_report_path.exists():
            try:
                with open(current_report_path, 'r', encoding='utf-8') as f:
                    prev_report = json.load(f)
                shutil.copy2(current_report_path, prev_report_path)
            except:
                pass
        
        # المقارنة
        comparison = None
        if prev_report and prev_report.get('metrics', {}).get('perplexity'):
            prev_ppl = prev_report['metrics']['perplexity']
            if perplexity:
                diff = perplexity - prev_ppl
                comparison = {
                    'previous_perplexity': prev_ppl,
                    'current_perplexity': perplexity,
                    'difference': diff,
                    'improved': diff < 0
                }
                logger.info(f"   Comparison: {prev_ppl:.2f} → {perplexity:.2f} ({diff:+.2f})")
        
        report = ModelEvaluationReport(
            timestamp=datetime.now().isoformat(),
            model_path=str(model_path),
            metrics=metrics,
            samples_evaluated=len(validation_data),
            duration_seconds=duration,
            error=error,
            comparison_with_previous=comparison
        )
        
        # حفظ التقرير
        with open(current_report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': report.timestamp,
                'model_path': report.model_path,
                'metrics': metrics.to_dict(),
                'samples_evaluated': report.samples_evaluated,
                'duration_seconds': report.duration_seconds,
                'error': report.error,
                'comparison': report.comparison_with_previous
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ Evaluation complete!")
        logger.info(f"   Report: {current_report_path}")
        
        return report
    
    def run_benchmark_suite(
        self,
        model_path: Union[str, Path],
        benchmarks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        تشغيل مجموعة اختبارات
        Run benchmark suite
        
        Args:
            model_path: Path to model
            benchmarks: List of benchmark names
            
        Returns:
            Benchmark results
        """
        benchmarks = benchmarks or ['perplexity', 'validation']
        results = {}
        
        logger.info("=" * 50)
        logger.info("Benchmark Suite - مجموعة الاختبارات")
        logger.info("=" * 50)
        
        if 'validation' in benchmarks:
            report = self.load_and_validate_directory(fix=True)
            results['validation'] = {
                'valid_ratio': report.valid_ratio,
                'total_samples': report.total_samples,
                'valid_samples': report.valid_samples
            }
        
        if 'perplexity' in benchmarks:
            eval_report = self.evaluate_model(model_path)
            results['perplexity'] = {
                'score': eval_report.metrics.perplexity,
                'samples': eval_report.samples_evaluated
            }
        
        # حفظ النتائج
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model_path': str(model_path),
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n💾 Benchmark results: {results_path}")
        
        return results


# Import shutil for file operations
import shutil


if __name__ == "__main__":
    # اختبار
    print("=" * 50)
    print("Evaluation Engine - Test")
    print("=" * 50)
    
    engine = EvaluationEngine()
    
    # اختبار التحقق
    test_data = [
        {'instruction': 'What is Python?', 'output': 'A programming language'},
        {'instruction': 'Hi', 'output': 'Hello!'},
        {'instruction': 'A' * 10000, 'output': 'Too long'},
    ]
    
    report = engine.validate_data(test_data)
    print(f"\nValidation: {report.valid_ratio:.1%} valid")
    
    print("\n✅ EvaluationEngine ready!")
