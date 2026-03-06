#!/usr/bin/env python3
"""
project_orchestrator.py — منسق المشاريع 🎯

الجزء المفقود الأساسي:
  المستخدم يكتب "سوولي ERP" ← شنو يصير؟

المسار الكامل (من القوانين سطر 187-199):
  1. المستخدم يكتب أمر بالـ IDE
  2. المنسق يحلل: شنو المشروع؟ شنو يحتاج؟
  3. يوزّع على الكبسولات حسب التخصص
  4. كل كبسولة تنفذ جزئها
  5. طبقة الفحص تفحص
  6. يبقى يشتغل حتى يكتمل 100%

مثال:
  "سوولي ERP"
    ↓
  المجلس يحلل → يحتاج: محاسبة، مبيعات، مخازن، HR
    ↓
  code_python: يبني الـ backend
  code_typescript: يبني الـ frontend
  erp_accounting: يصمم المحاسبة
  erp_sales: يصمم المبيعات
  database_design: يصمم قاعدة البيانات
  security: يفحص الأمان
    ↓
  كل واحد ينفذ جزئه → المنسق يجمع
    ↓
  الفحص النهائي → تسليم
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger("orchestrator")

CAPSULES_ROOT = Path(__file__).parent.parent / "capsules"
PROJECTS_DIR = Path(__file__).parent.parent / ".projects"


# خارطة: نوع المشروع → كبسولات مطلوبة + مهام
PROJECT_TEMPLATES = {
    "erp": {
        "name": "نظام ERP",
        "capsules": {
            "erp_accounting": "تصميم نظام المحاسبة (قيود، حسابات، تقارير مالية)",
            "erp_sales": "تصميم نظام المبيعات (فواتير، عملاء، تحصيل)",
            "erp_inventory": "تصميم نظام المخزون (مستودعات، حركة بضاعة)",
            "erp_hr": "تصميم نظام الموارد البشرية (رواتب، موظفين)",
            "erp_purchasing": "تصميم نظام المشتريات (أوامر شراء، موردين)",
            "code_python": "بناء الـ Backend API (FastAPI + endpoints)",
            "code_typescript": "بناء الـ Frontend (React + UI components)",
            "code_sql": "تصميم قاعدة البيانات (tables + relations + indexes)",
            "database_design": "ERD + migrations + schema optimization",
            "security": "فحص أمني شامل (auth, encryption, input validation)",
            "code_css": "تصميم الواجهة (responsive, professional UI)",
            "code_testing": "كتابة اختبارات شاملة (unit + integration)",
        },
    },
    "web_app": {
        "name": "تطبيق ويب",
        "capsules": {
            "code_python": "Backend API",
            "code_typescript": "Frontend React/Next.js",
            "code_sql": "Database design",
            "code_css": "UI/UX design",
            "security": "Security audit",
            "code_testing": "Test suite",
            "devops": "Docker + deployment",
        },
    },
    "mobile_app": {
        "name": "تطبيق موبايل",
        "capsules": {
            "code_typescript": "React Native / Flutter UI",
            "code_python": "Backend API",
            "code_sql": "Database",
            "security": "Mobile security",
        },
    },
    "ai_model": {
        "name": "نموذج ذكاء اصطناعي",
        "capsules": {
            "code_python": "Model architecture + training code",
            "code_sql": "Data pipeline",
            "devops": "Training infrastructure",
        },
    },
}

# كلمات مفتاحية لتصنيف المشروع
PROJECT_KEYWORDS = {
    "erp": ["erp", "محاسبة", "فواتير", "مخزون", "رواتب", "مبيعات", "مشتريات",
            "accounting", "inventory", "hr", "sales", "purchase"],
    "web_app": ["موقع", "ويب", "website", "web app", "dashboard", "لوحة"],
    "mobile_app": ["موبايل", "تطبيق", "mobile", "app", "أندرويد", "آيفون"],
    "ai_model": ["ذكاء اصطناعي", "ai", "model", "تدريب", "machine learning"],
}


class ProjectOrchestrator:
    """منسق المشاريع — يأخذ أمر ← يحلل ← يوزّع ← ينفذ ← 100%"""

    def __init__(self, capsules_dir: Path = None):
        self.capsules_dir = capsules_dir or CAPSULES_ROOT
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    def analyze(self, command: str) -> Dict:
        """
        الخطوة 1: تحليل الأمر
        "سوولي ERP" → {type: "erp", tasks: [...]}
        """
        command_lower = command.lower()

        # تصنيف المشروع
        project_type = "web_app"  # default
        best_score = 0
        for ptype, keywords in PROJECT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in command_lower)
            if score > best_score:
                best_score = score
                project_type = ptype

        template = PROJECT_TEMPLATES.get(project_type, PROJECT_TEMPLATES["web_app"])

        # بناء المهام
        tasks = []
        for capsule_id, task_desc in template["capsules"].items():
            capsule_dir = self.capsules_dir / capsule_id
            has_model = (capsule_dir / "model" / "config.json").exists()

            tasks.append({
                "capsule_id": capsule_id,
                "task": f"{task_desc} — بناءً على: {command}",
                "status": "pending",
                "has_trained_model": has_model,
                "output": None,
            })

        return {
            "project_type": project_type,
            "project_name": template["name"],
            "command": command,
            "tasks": tasks,
            "total_capsules": len(tasks),
            "ready_capsules": sum(1 for t in tasks if t["has_trained_model"]),
        }

    def create_project(self, command: str) -> Dict:
        """
        الخطوة 2: إنشاء مشروع كامل
        """
        analysis = self.analyze(command)
        project_id = f"proj_{int(time.time())}"

        project = {
            "id": project_id,
            "command": command,
            "type": analysis["project_type"],
            "name": analysis["project_name"],
            "status": "created",
            "progress": 0,
            "tasks": analysis["tasks"],
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "results": {},
        }

        # حفظ
        project_file = PROJECTS_DIR / f"{project_id}.json"
        project_file.write_text(json.dumps(project, indent=2, ensure_ascii=False))

        logger.info(f"📋 Project created: {project_id} — {analysis['project_name']}")
        logger.info(f"   Tasks: {len(analysis['tasks'])} "
                    f"({analysis['ready_capsules']} ready)")

        return project

    def execute_project(self, project_id: str) -> Dict:
        """
        الخطوة 3: تنفيذ المشروع — كل كبسولة تنفذ مهمتها
        """
        project_file = PROJECTS_DIR / f"{project_id}.json"
        if not project_file.exists():
            return {"error": "project not found"}

        project = json.loads(project_file.read_text())
        project["status"] = "executing"

        completed = 0
        total = len(project["tasks"])

        for i, task in enumerate(project["tasks"]):
            if task["status"] == "completed":
                completed += 1
                continue

            capsule_id = task["capsule_id"]
            capsule_dir = self.capsules_dir / capsule_id
            model_dir = capsule_dir / "model"

            task["status"] = "running"
            project["progress"] = int((completed / total) * 100)
            self._save_project(project)

            logger.info(f"🔄 [{capsule_id}] Executing: {task['task'][:60]}...")

            if model_dir.exists() and (model_dir / "config.json").exists():
                # كبسولة مدربة — استدلال حقيقي
                try:
                    output = self._capsule_inference(model_dir, task["task"])
                    task["output"] = output
                    task["status"] = "completed"
                    completed += 1
                except Exception as e:
                    task["status"] = "error"
                    task["output"] = f"Error: {e}"
                    logger.error(f"❌ {capsule_id}: {e}")
            else:
                # كبسولة ما اتدربت — ollama fallback أو تخطي
                task["status"] = "skipped"
                task["output"] = f"الكبسولة {capsule_id} ما اتدربت بعد"
                logger.warning(f"⚠️ {capsule_id}: untrained, skipped")

            project["progress"] = int(((completed) / total) * 100)
            self._save_project(project)

        # تسليم
        if completed == total:
            project["status"] = "completed"
            project["progress"] = 100
        else:
            project["status"] = "partial"
            project["progress"] = int((completed / total) * 100)

        project["completed_at"] = datetime.now().isoformat()
        self._save_project(project)

        logger.info(f"📋 Project {project_id}: {project['status']} "
                    f"({completed}/{total} tasks)")

        return project

    def execute_full(self, command: str) -> Dict:
        """المسار الكامل: أمر → تحليل → إنشاء → تنفيذ → نتيجة"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 PROJECT: {command}")
        logger.info(f"{'='*60}")

        # 1. تحليل
        project = self.create_project(command)

        # 2. مجلس (إذا متوفر)
        try:
            from brain.council_brain import CouncilBrain
            council = CouncilBrain(self.capsules_dir)
            decisions = council.decide()
            project["council_decisions"] = len(decisions.get("actions", []))
        except:
            pass

        # 3. تنفيذ
        result = self.execute_project(project["id"])

        # 4. جمع النتائج
        outputs = {}
        for task in result.get("tasks", []):
            if task.get("output") and task["status"] == "completed":
                outputs[task["capsule_id"]] = task["output"]

        result["combined_output"] = self._combine_outputs(outputs, command)
        self._save_project(result)

        return result

    def _combine_outputs(self, outputs: Dict[str, str], command: str) -> str:
        """جمع مخرجات كل الكبسولات بنتيجة واحدة"""
        if not outputs:
            return "لا توجد نتائج — الكبسولات لم تتدرب بعد"

        combined = f"# نتيجة: {command}\n\n"
        for capsule_id, output in outputs.items():
            combined += f"## {capsule_id}\n{output}\n\n---\n\n"

        return combined

    def _capsule_inference(self, model_dir: Path, prompt: str) -> str:
        """استدلال من كبسولة مدربة"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import gc

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )

        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:],
                                     skip_special_tokens=True)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

    def _save_project(self, project: Dict):
        """حفظ حالة المشروع"""
        f = PROJECTS_DIR / f"{project['id']}.json"
        f.write_text(json.dumps(project, indent=2, ensure_ascii=False, default=str))

    def get_project(self, project_id: str) -> Dict:
        """جلب مشروع بالـ ID"""
        f = PROJECTS_DIR / f"{project_id}.json"
        if f.exists():
            return json.loads(f.read_text())
        return {"error": "not found"}

    def list_projects(self) -> List[Dict]:
        """كل المشاريع"""
        projects = []
        for f in sorted(PROJECTS_DIR.glob("proj_*.json"), reverse=True):
            try:
                p = json.loads(f.read_text())
                projects.append({
                    "id": p["id"], "command": p["command"],
                    "status": p["status"], "progress": p["progress"],
                    "created": p.get("created_at"),
                })
            except:
                pass
        return projects


# Singleton
orchestrator = ProjectOrchestrator()
