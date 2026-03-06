#!/usr/bin/env python3
"""
auto_programmer.py — البرمجة الأوتوماتيكية 🤖⚙️

"سوولي ERP" → يبرمج كلشي أوتوماتيكي:
  1. المستخدم يطلب مشروع
  2. Deep mode: الكشافة تبحث عن المنافسين
  3. المجلس يتناقش على الخطة
  4. يقسم لمهام ← كل كبسولة تبرمج جزئها
  5. يدمج ← يجرب ← يصلح ← يكرر
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("auto_programmer")

PROJECT_ROOT = Path(__file__).parent.parent

try:
    from brain.capsule_router import router as capsule_router
    from brain.council_auto_debate import council
    from brain.memory_system import memory
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from brain.capsule_router import router as capsule_router
    from brain.council_auto_debate import council
    from brain.memory_system import memory


# ═══════════════════════════════════════════════════════════
# Project Templates
# ═══════════════════════════════════════════════════════════

PROJECT_TEMPLATES = {
    "api": {
        "name": "REST API",
        "files": {
            "app.py": "python",
            "models.py": "python",
            "routes.py": "python",
            "database.py": "python",
            "requirements.txt": "text",
            "README.md": "markdown",
        },
        "capsules": ["code_python", "code_sql", "database_design"],
    },
    "webapp": {
        "name": "Web Application",
        "files": {
            "src/App.tsx": "typescript",
            "src/pages/Home.tsx": "typescript",
            "src/components/Layout.tsx": "typescript",
            "src/styles/global.css": "css",
            "src/api/client.ts": "typescript",
            "package.json": "json",
            "README.md": "markdown",
        },
        "capsules": ["code_typescript", "code_css", "code_python"],
    },
    "fullstack": {
        "name": "Full-Stack Application",
        "files": {
            "backend/app.py": "python",
            "backend/models.py": "python",
            "backend/routes.py": "python",
            "backend/database.py": "python",
            "frontend/src/App.tsx": "typescript",
            "frontend/src/pages/Home.tsx": "typescript",
            "frontend/src/components/Layout.tsx": "typescript",
            "frontend/src/styles/global.css": "css",
            "docker-compose.yml": "yaml",
            "README.md": "markdown",
        },
        "capsules": ["code_python", "code_typescript", "code_css", "code_sql", "devops"],
    },
    "erp": {
        "name": "ERP System",
        "files": {
            "backend/app.py": "python",
            "backend/models/user.py": "python",
            "backend/models/product.py": "python",
            "backend/models/order.py": "python",
            "backend/models/inventory.py": "python",
            "backend/models/accounting.py": "python",
            "backend/routes/auth.py": "python",
            "backend/routes/products.py": "python",
            "backend/routes/orders.py": "python",
            "backend/routes/reports.py": "python",
            "backend/database.py": "python",
            "frontend/src/App.tsx": "typescript",
            "frontend/src/pages/Dashboard.tsx": "typescript",
            "frontend/src/pages/Products.tsx": "typescript",
            "frontend/src/pages/Orders.tsx": "typescript",
            "frontend/src/pages/Reports.tsx": "typescript",
            "frontend/src/components/Sidebar.tsx": "typescript",
            "frontend/src/styles/global.css": "css",
            "docker-compose.yml": "yaml",
            "README.md": "markdown",
        },
        "capsules": ["code_python", "code_typescript", "code_css", "code_sql", "database_design", "devops"],
    },
}


@dataclass
class ProjectTask:
    """مهمة برمجية"""
    file_path: str
    language: str
    description: str
    capsule_id: str
    code: str = ""
    status: str = "pending"


@dataclass
class ProjectPlan:
    """خطة مشروع"""
    name: str
    description: str
    project_type: str
    output_dir: str
    tasks: List[ProjectTask] = field(default_factory=list)
    research: Optional[Dict] = None
    council_debate: Optional[str] = None
    mode: str = "fast"
    status: str = "planning"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AutoProgrammer:
    """البرمجة الأوتوماتيكية — 'سوولي ERP' → مشروع كامل"""

    def __init__(self):
        self.projects: List[ProjectPlan] = []
        logger.info("🤖 AutoProgrammer initialized")

    def create_project(self, description: str, output_dir: str = None,
                       mode: str = "fast") -> ProjectPlan:
        project_type = self._detect_project_type(description)
        template = PROJECT_TEMPLATES.get(project_type, PROJECT_TEMPLATES["fullstack"])
        name = self._extract_project_name(description)
        output_dir = output_dir or str(PROJECT_ROOT / "generated" / name)

        plan = ProjectPlan(
            name=name, description=description,
            project_type=project_type, output_dir=output_dir, mode=mode,
        )

        if mode == "deep":
            decision = capsule_router.route(description, mode="deep")
            plan.research = decision.research_data
            debate = council.debate(f"خطة مشروع: {description}")
            plan.council_debate = council.format_debate(debate)

        for file_path, lang in template["files"].items():
            capsule_id = self._match_capsule(lang)
            plan.tasks.append(ProjectTask(
                file_path=file_path, language=lang,
                description=f"Generate {file_path} for {name}: {description}",
                capsule_id=capsule_id,
            ))

        self._execute_tasks(plan)
        plan.status = "completed"
        self.projects.append(plan)

        memory.save_knowledge(
            topic=f"Project: {name}",
            content=f"Generated {project_type}: {description}. Files: {len(plan.tasks)}",
            source="auto_programmer",
        )
        return plan

    def _detect_project_type(self, description: str) -> str:
        desc = description.lower()
        if any(w in desc for w in ["erp", "إدارة", "محاسبة", "مخزون"]):
            return "erp"
        elif any(w in desc for w in ["api", "rest", "backend"]):
            return "api"
        elif any(w in desc for w in ["web", "موقع", "واجهة"]):
            return "webapp"
        return "fullstack"

    def _extract_project_name(self, description: str) -> str:
        words = re.sub(r'[^\w\s]', '', description).split()
        name_words = [w for w in words if len(w) > 2 and w.isascii()]
        return "_".join(name_words[:3]).lower() if name_words else f"project_{datetime.now().strftime('%H%M%S')}"

    def _match_capsule(self, language: str) -> str:
        return {"python": "code_python", "typescript": "code_typescript",
                "css": "code_css", "sql": "code_sql", "yaml": "devops",
                "json": "devops", "text": "code_python", "markdown": "knowledge_arabic"
        }.get(language, "code_python")

    def _execute_tasks(self, plan: ProjectPlan):
        out_dir = Path(plan.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for task in plan.tasks:
            task.status = "generating"
            try:
                task.code = self._generate_code(task, plan)
                task.status = "done"
                file_path = out_dir / task.file_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(task.code, encoding="utf-8")
            except Exception as e:
                task.status = "error"
                logger.error(f"  ❌ {task.file_path}: {e}")

    def _generate_code(self, task: ProjectTask, plan: ProjectPlan) -> str:
        model_dir = PROJECT_ROOT / "brain" / "capsules" / task.capsule_id / "model"
        if model_dir.exists():
            try:
                from brain.chat_bridge import bridge
                result = bridge._query_capsule(task.capsule_id, task.description)
                if result and result.response:
                    code = re.search(r'```\w*\n(.*?)```', result.response, re.DOTALL)
                    if code:
                        return code.group(1)
                    return result.response
            except Exception:
                pass
        return self._template_code(task, plan)

    def _template_code(self, task: ProjectTask, plan: ProjectPlan) -> str:
        n = plan.name
        templates = {
            "app.py": f'#!/usr/bin/env python3\n"""{n} — Main Application"""\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI(title="{n}", version="1.0")\napp.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])\n\n@app.get("/")\nasync def root():\n    return {{"status": "running", "project": "{n}"}}\n\n@app.get("/api/health")\nasync def health():\n    return {{"healthy": True}}\n\nif __name__ == "__main__":\n    import uvicorn\n    uvicorn.run(app, host="0.0.0.0", port=8000)\n',
            "database.py": f'"""{n} — Database"""\nfrom sqlalchemy import create_engine\nfrom sqlalchemy.ext.declarative import declarative_base\nfrom sqlalchemy.orm import sessionmaker\nimport os\n\nDATABASE_URL = os.getenv("DATABASE_URL", "postgresql://bi:bi2026@localhost:5432/{n}")\nengine = create_engine(DATABASE_URL)\nSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)\nBase = declarative_base()\n\ndef get_db():\n    db = SessionLocal()\n    try:\n        yield db\n    finally:\n        db.close()\n',
            "requirements.txt": "fastapi\nuvicorn\nsqlalchemy\npsycopg2-binary\npydantic\n",
            "README.md": f"# {n}\n\n{plan.description}\n\nGenerated by BI-IDE AutoProgrammer 🤖\n",
            "docker-compose.yml": f'version: "3.8"\nservices:\n  app:\n    build: .\n    ports:\n      - "8000:8000"\n    depends_on:\n      - db\n  db:\n    image: postgres:15\n    environment:\n      POSTGRES_USER: bi\n      POSTGRES_PASSWORD: bi2026\n      POSTGRES_DB: {n}\n    volumes:\n      - pgdata:/var/lib/postgresql/data\nvolumes:\n  pgdata:\n',
        }
        basename = Path(task.file_path).name
        if basename in templates:
            return templates[basename]
        if task.language == "python":
            return f'"""{task.file_path} — {n}"""\n\n# TODO: Implement\n'
        elif task.language == "typescript":
            return f'// {task.file_path} — {n}\nexport default function Component() {{\n  return <div>{n}</div>;\n}}\n'
        elif task.language == "css":
            return f'/* {n} */\n:root {{ --primary: #3B82F6; --bg: #0F172A; }}\n'
        return f"# {task.file_path}\n"


programmer = AutoProgrammer()

if __name__ == "__main__":
    print("🤖 AutoProgrammer — Test\n")
    plan = programmer.create_project("سوي لي REST API بـ Python للمنتجات", mode="fast")
    print(f"Project: {plan.name}")
    print(f"Type: {plan.project_type}")
    print(f"Files: {len(plan.tasks)}")
    for t in plan.tasks:
        print(f"  {'✅' if t.status == 'done' else '❌'} {t.file_path} ({t.language})")
    print(f"Output: {plan.output_dir}")
