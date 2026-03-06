#!/usr/bin/env python3
"""
curriculum_learning.py — تدريب مرتب 📚🎓

بدل تدريب عشوائي → تدريب مرتب بترتيب صعوبة:
  Level 1: لغة أساسية (كلمات، جمل)
  Level 2: منطق (if/else، loops)
  Level 3: علوم (رياضيات، فيزياء)
  Level 4: هندسة (تصميم، معمارية)
  Level 5: تخصص (أمن، AI، مصانع)
  Level 6: إبداع (خيال، اختراع، دمج)

كل كبسولة تتقدم حسب أدائها — مثل طالب بالمدرسة.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("curriculum")

PROJECT_ROOT = Path(__file__).parent.parent
CAPSULES_ROOT = PROJECT_ROOT / "brain" / "capsules"

try:
    from brain.memory_system import memory
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory


# ═══════════════════════════════════════════════════════════
# Curriculum Levels
# ═══════════════════════════════════════════════════════════

@dataclass
class CurriculumLevel:
    """مستوى تعليمي"""
    level: int
    name: str
    name_ar: str
    description: str
    required_samples: int       # عينات مطلوبة للنجاح
    required_accuracy: float    # دقة مطلوبة (0-1)
    data_sources: List[str]     # مصادر البيانات
    topics: List[str]           # المواضيع


LEVELS = [
    CurriculumLevel(
        level=1, name="Language Basics", name_ar="لغة أساسية",
        description="كلمات، جمل، قواعد أساسية",
        required_samples=500, required_accuracy=0.3,
        data_sources=["wikipedia", "basic_docs"],
        topics=["vocabulary", "grammar", "syntax", "simple_patterns"],
    ),
    CurriculumLevel(
        level=2, name="Logic & Control", name_ar="منطق وتحكم",
        description="شروط، حلقات، دوال، هياكل بيانات",
        required_samples=1000, required_accuracy=0.4,
        data_sources=["code_examples", "tutorials"],
        topics=["if_else", "loops", "functions", "data_structures", "algorithms"],
    ),
    CurriculumLevel(
        level=3, name="Science & Math", name_ar="علوم ورياضيات",
        description="رياضيات، فيزياء، كيمياء أساسية",
        required_samples=2000, required_accuracy=0.5,
        data_sources=["textbooks", "khan_academy", "research_papers"],
        topics=["calculus", "linear_algebra", "physics", "chemistry", "statistics"],
    ),
    CurriculumLevel(
        level=4, name="Engineering", name_ar="هندسة",
        description="تصميم أنظمة، معمارية، أنماط",
        required_samples=3000, required_accuracy=0.6,
        data_sources=["github", "architecture_docs", "design_patterns"],
        topics=["system_design", "microservices", "databases", "networking", "security"],
    ),
    CurriculumLevel(
        level=5, name="Specialization", name_ar="تخصص",
        description="تخصص عميق بالمجال",
        required_samples=5000, required_accuracy=0.7,
        data_sources=["domain_specific", "expert_knowledge", "real_projects"],
        topics=["advanced_ai", "cybersecurity", "factory_design", "os_development"],
    ),
    CurriculumLevel(
        level=6, name="Innovation", name_ar="إبداع",
        description="اختراع وابتكار — دمج مجالات",
        required_samples=10000, required_accuracy=0.8,
        data_sources=["cross_domain", "imagination", "research_frontier"],
        topics=["novel_ideas", "cross_domain_fusion", "self_improvement", "breakthroughs"],
    ),
]


@dataclass
class CapsuleProgress:
    """تقدم كبسولة"""
    capsule_id: str
    current_level: int = 1
    samples_completed: int = 0
    current_accuracy: float = 0.0
    cycles_at_level: int = 0
    promoted_at: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class CurriculumManager:
    """
    مدير المنهج التعليمي

    يتابع تقدم كل كبسولة ← يرقّيها لما تنجح
    """

    def __init__(self):
        self.progress: Dict[str, CapsuleProgress] = {}
        self.progress_file = CAPSULES_ROOT / ".curriculum_progress.json"
        self._load_progress()
        logger.info(f"📚 Curriculum Manager: {len(self.progress)} capsules tracked")

    def _load_progress(self):
        """تحميل التقدم"""
        if self.progress_file.exists():
            try:
                data = json.loads(self.progress_file.read_text())
                for cid, p in data.items():
                    self.progress[cid] = CapsuleProgress(**p)
            except Exception:
                pass

        # تهيئة كبسولات جديدة
        for cap_dir in CAPSULES_ROOT.iterdir():
            if cap_dir.is_dir() and not cap_dir.name.startswith("."):
                if cap_dir.name not in self.progress:
                    self.progress[cap_dir.name] = CapsuleProgress(capsule_id=cap_dir.name)

    def _save_progress(self):
        """حفظ التقدم"""
        try:
            data = {}
            for cid, p in self.progress.items():
                data[cid] = {
                    "capsule_id": p.capsule_id,
                    "current_level": p.current_level,
                    "samples_completed": p.samples_completed,
                    "current_accuracy": p.current_accuracy,
                    "cycles_at_level": p.cycles_at_level,
                    "promoted_at": p.promoted_at,
                    "last_updated": p.last_updated,
                }
            self.progress_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Save error: {e}")

    def get_level(self, level_num: int) -> Optional[CurriculumLevel]:
        """جلب مستوى"""
        for lv in LEVELS:
            if lv.level == level_num:
                return lv
        return None

    def report_training(self, capsule_id: str, samples: int,
                        accuracy: float, cycle: int = 0):
        """تقرير تدريب — يتم استدعاؤه بعد كل دورة"""
        if capsule_id not in self.progress:
            self.progress[capsule_id] = CapsuleProgress(capsule_id=capsule_id)

        prog = self.progress[capsule_id]
        prog.samples_completed += samples
        prog.current_accuracy = accuracy
        prog.cycles_at_level += 1
        prog.last_updated = datetime.now().isoformat()

        # فحص الترقية
        level = self.get_level(prog.current_level)
        if level and self._should_promote(prog, level):
            self._promote(prog)

        self._save_progress()

    def _should_promote(self, prog: CapsuleProgress, level: CurriculumLevel) -> bool:
        """هل يستحق ترقية؟"""
        return (
            prog.samples_completed >= level.required_samples and
            prog.current_accuracy >= level.required_accuracy and
            prog.current_level < 6
        )

    def _promote(self, prog: CapsuleProgress):
        """ترقية كبسولة للمستوى التالي"""
        old_level = prog.current_level
        prog.current_level += 1
        prog.cycles_at_level = 0
        prog.promoted_at.append(datetime.now().isoformat())

        new_level = self.get_level(prog.current_level)
        logger.info(f"🎓 {prog.capsule_id}: Level {old_level} → {prog.current_level} ({new_level.name_ar if new_level else '?'})")

        memory.save_knowledge(
            topic=f"Promotion: {prog.capsule_id}",
            content=f"Promoted to Level {prog.current_level} ({new_level.name_ar if new_level else '?'})",
            source="curriculum",
            confidence=prog.current_accuracy,
            capsule_id=prog.capsule_id,
        )

    def get_next_training(self, capsule_id: str) -> Dict:
        """ماذا يجب أن يتدرب عليه بعد؟"""
        prog = self.progress.get(capsule_id)
        if not prog:
            return {"error": f"Unknown capsule: {capsule_id}"}

        level = self.get_level(prog.current_level)
        if not level:
            return {"capsule": capsule_id, "status": "max_level"}

        remaining_samples = max(0, level.required_samples - prog.samples_completed)
        accuracy_gap = max(0, level.required_accuracy - prog.current_accuracy)

        return {
            "capsule": capsule_id,
            "current_level": prog.current_level,
            "level_name": level.name_ar,
            "topics": level.topics,
            "data_sources": level.data_sources,
            "samples_remaining": remaining_samples,
            "accuracy_needed": round(level.required_accuracy, 2),
            "current_accuracy": round(prog.current_accuracy, 2),
            "accuracy_gap": round(accuracy_gap, 2),
        }

    def get_all_status(self) -> List[Dict]:
        """حالة كل الكبسولات"""
        result = []
        for cid, prog in sorted(self.progress.items()):
            level = self.get_level(prog.current_level)
            result.append({
                "capsule": cid,
                "level": prog.current_level,
                "level_name": level.name_ar if level else "?",
                "samples": prog.samples_completed,
                "accuracy": round(prog.current_accuracy, 2),
                "cycles": prog.cycles_at_level,
                "promotions": len(prog.promoted_at),
            })
        return result

    def get_class_report(self) -> str:
        """تقرير الصف — كل الطلاب"""
        lines = ["# 📚 تقرير المنهج التعليمي\n"]

        for lv in LEVELS:
            students = [p for p in self.progress.values() if p.current_level == lv.level]
            if students:
                lines.append(f"## المستوى {lv.level}: {lv.name_ar} ({lv.name})")
                for s in students:
                    pct = min(100, s.samples_completed / max(lv.required_samples, 1) * 100)
                    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                    lines.append(f"  {s.capsule_id:25s} [{bar}] {pct:.0f}% ({s.samples_completed}/{lv.required_samples})")
                lines.append("")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════

curriculum = CurriculumManager()


if __name__ == "__main__":
    print("📚🎓 Curriculum Learning — Test\n")

    # عرض المستويات
    print("المستويات التعليمية:")
    for lv in LEVELS:
        print(f"  Level {lv.level}: {lv.name_ar} ({lv.name})")
        print(f"    عينات: {lv.required_samples}, دقة: {lv.required_accuracy:.0%}")
        print(f"    مواضيع: {', '.join(lv.topics[:3])}...")
        print()

    # محاكاة تدريب
    print("═" * 50)
    print("محاكاة تدريب code_python:\n")

    curriculum.report_training("code_python", samples=200, accuracy=0.25, cycle=1)
    print(f"  Cycle 1: {curriculum.get_next_training('code_python')}")

    curriculum.report_training("code_python", samples=400, accuracy=0.35, cycle=2)
    print(f"  Cycle 2: {curriculum.get_next_training('code_python')}")

    # الآن عنده 600 عينة و 0.35 دقة → ترقية (500 عينة + 0.3 دقة مطلوبة)
    print()

    # تقرير الصف
    print(curriculum.get_class_report())
