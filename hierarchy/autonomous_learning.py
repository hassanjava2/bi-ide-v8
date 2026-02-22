"""
التعلم الذاتي - Autonomous Learning
النظام يراقب شغلك ويتعلم منه أوتماتيكياً
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class ObservationalLearning:
    """
    نظام المراقبة والتعلم الذاتي
    يتعلم من أفعال المستخدم مباشرة
    """
    
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy
        self.learning_buffer = []
        self.last_save = time.time()
        self.save_interval = 300  # حفظ كل 5 دقايق
        
        # ملفات التعلم
        self.learning_dir = Path("learning_data")
        self.learning_dir.mkdir(exist_ok=True)
        
    # ==================== تعلم الـ IDE ====================
    
    def observe_code_written(self, file_path: str, code: str, language: str):
        """
        يراقب الكود اللي تكتبه
        """
        # استخراج patterns
        patterns = self._extract_code_patterns(code, language)
        
        experience = {
            "type": "code_written",
            "timestamp": datetime.now().isoformat(),
            "file": file_path,
            "language": language,
            "patterns": patterns,
            "complexity": len(code.split('\n')),
            "learning_value": "high" if len(patterns) > 3 else "medium"
        }
        
        self._distribute_to_experts(experience, "tech")
        self._add_to_buffer(experience)
        
    def observe_code_error(self, code: str, error: str, solution: str = None):
        """
        يراقب الأخطاء والحلول
        """
        experience = {
            "type": "error_learning",
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "code_snippet": code[:500],
            "solution": solution,
            "learned": solution is not None
        }
        
        # Shadow Team يسجل الخطأ
        self._notify_shadow_team(experience)
        # Light Team يسجل الحل
        if solution:
            self._notify_light_team(experience)
            
        self._add_to_buffer(experience)
    
    # ==================== تعلم الـ ERP ====================
    
    def observe_erp_transaction(self, transaction_type: str, data: Dict, result: str):
        """
        يراقب معاملات الـ ERP
        """
        experience = {
            "type": "erp_transaction",
            "timestamp": datetime.now().isoformat(),
            "transaction": transaction_type,
            "data_summary": {k: type(v).__name__ for k, v in data.items()},
            "result": result,
            "business_value": self._calculate_business_value(transaction_type, data)
        }
        
        self._distribute_to_experts(experience, "business")
        self._add_to_buffer(experience)
    
    def observe_decision(self, context: str, decision: str, outcome: str, confidence: float):
        """
        يراقب القرارات وتجنب الأخطاء
        """
        experience = {
            "type": "decision",
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "decision": decision,
            "outcome": outcome,
            "confidence": confidence,
            "success": outcome == "success"
        }
        
        # High Council يراجع القرار
        self._notify_council(experience)
        self._add_to_buffer(experience)
    
    # ==================== تعلم الأوامر ====================
    
    def observe_command(self, command: str, alert_level: str, result: Dict):
        """
        يراقب أوامرك للـ AI
        """
        experience = {
            "type": "command",
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "urgency": alert_level,
            "result_type": result.get("status", "unknown"),
            "council_consensus": result.get("council_consensus", {}),
            "execution_success": result.get("execution", {}).get("success", False)
        }
        
        # Meta Team يحلل الأداء
        self._notify_meta_team(experience)
        self._add_to_buffer(experience)
    
    # ==================== توزيع المعرفة ====================
    
    def _distribute_to_experts(self, experience: Dict, domain: str):
        """
        يوزع الخبرة على الخبراء المعنيين
        """
        if hasattr(self.hierarchy, 'experts') and domain in self.hierarchy.experts.experts:
            expert = self.hierarchy.experts.experts[domain]
            if not hasattr(expert, 'observations'):
                expert.observations = []
            expert.observations.append(experience)
            
            # إذا جمع خبرة كثيرة، يصير "أكثر خبرة"
            if len(expert.observations) % 100 == 0:
                print(f"🧠 {domain} Expert: جمع {len(expert.observations)} ملاحظة")
    
    def _notify_shadow_team(self, experience: Dict):
        """يبلغ فريق الظل بالمشاكل"""
        if hasattr(self.hierarchy, 'balance'):
            shadow = self.hierarchy.balance.shadow
            if not hasattr(shadow, 'failure_memory'):
                shadow.failure_memory = []
            shadow.failure_memory.append(experience)
    
    def _notify_light_team(self, experience: Dict):
        """يبلغ فريق النور بالنجاحات"""
        if hasattr(self.hierarchy, 'balance'):
            light = self.hierarchy.balance.light
            if not hasattr(light, 'success_memory'):
                light.success_memory = []
            light.success_memory.append(experience)
    
    def _notify_council(self, experience: Dict):
        """يبلغ المجلس بالقرارات المهمة"""
        if hasattr(self.hierarchy, 'council'):
            council = self.hierarchy.council
            if not hasattr(council, 'decision_history'):
                council.decision_history = []
            council.decision_history.append(experience)
    
    def _notify_meta_team(self, experience: Dict):
        """يبلغ Meta Team للتحليل"""
        if hasattr(self.hierarchy, 'meta'):
            meta = self.hierarchy.meta
            if hasattr(meta, 'performance_manager'):
                pm = meta.performance_manager
                if not hasattr(pm, 'execution_log'):
                    pm.execution_log = []
                pm.execution_log.append(experience)
    
    # ==================== Helpers ====================
    
    def _extract_code_patterns(self, code: str, language: str) -> List[str]:
        """يستخرج patterns من الكود"""
        patterns = []
        
        if language == "python":
            if "def " in code:
                patterns.append("function_definition")
            if "class " in code:
                patterns.append("class_definition")
            if "import " in code or "from " in code:
                patterns.append("import_pattern")
            if "async def" in code:
                patterns.append("async_pattern")
            if "@" in code:
                patterns.append("decorator_pattern")
                
        return patterns
    
    def _calculate_business_value(self, transaction_type: str, data: Dict) -> str:
        """يحسب قيمة المعاملة"""
        amount = data.get('amount', 0)
        if amount > 10000:
            return "high"
        elif amount > 1000:
            return "medium"
        return "low"
    
    def _add_to_buffer(self, experience: Dict):
        """يضيف للـ buffer"""
        self.learning_buffer.append(experience)
        
        # حفظ دوري
        if time.time() - self.last_save > self.save_interval:
            self._save_learning()
    
    def _save_learning(self):
        """يحفظ التعلم للقراءة لاحقاً"""
        if not self.learning_buffer:
            return
            
        filename = self.learning_dir / f"learning_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.learning_buffer, f, ensure_ascii=False, indent=2)
        
        print(f"💾 حفظ {len(self.learning_buffer)} تجربة في {filename}")
        self.learning_buffer = []
        self.last_save = time.time()
    
    def get_learning_stats(self) -> Dict:
        """إحصائيات التعلم"""
        stats = {
            "buffer_size": len(self.learning_buffer),
            "saved_files": len(list(self.learning_dir.glob("learning_*.json"))),
        }
        
        # عدد ملاحظات كل خبير
        if hasattr(self.hierarchy, 'experts'):
            for name, expert in self.hierarchy.experts.experts.items():
                stats[f"{name}_observations"] = len(getattr(expert, 'observations', []))
        
        return stats


# Singleton
_learning_system = None

def get_learning_system(hierarchy=None):
    global _learning_system
    if _learning_system is None and hierarchy:
        _learning_system = ObservationalLearning(hierarchy)
    return _learning_system
