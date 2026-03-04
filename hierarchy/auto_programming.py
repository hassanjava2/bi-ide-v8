"""
خط الإنتاج البرمجي — Auto-Programming Pipeline
أمر → المجلس يناقش → الكشافة تبحث → الخبراء يصممون → الكود يتولّد

🏭 "سوولي ERP" → النظام يبني أفضل ERP بالكون
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class ProjectBlueprint:
    """مخطط المشروع — ناتج مرحلة التحليل"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.modules: List[Dict] = []
        self.database_tables: List[Dict] = []
        self.api_endpoints: List[Dict] = []
        self.ui_pages: List[Dict] = []
        self.created_at = datetime.now().isoformat()


class AutoProgrammingPipeline:
    """
    خط الإنتاج البرمجي — يحوّل أمر بشري لمشروع كامل
    
    المراحل:
    1. التحليل (Council) — فهم المطلوب
    2. البحث (Scouts) — مراجعة أفضل الممارسات
    3. التصميم (Experts) — بنية المشروع
    4. التوليد (Codegen) — كتابة الكود عبر AI
    5. الفحص (QA) — اختبار الكود
    6. التسليم — المشروع الجاهز
    """
    
    def __init__(self, hierarchy=None):
        self.hierarchy = hierarchy
        self.projects: List[Dict] = []
        self.ollama_url = "http://localhost:11434"
    
    async def execute(self, command: str, output_dir: str = "/tmp/gen_projects") -> Dict[str, Any]:
        """
        تنفيذ أمر برمجي كامل
        مثال: "سوولي نظام ERP كامل"
        """
        project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_dir = os.path.join(output_dir, project_id)
        os.makedirs(project_dir, exist_ok=True)
        
        result = {
            "project_id": project_id,
            "command": command,
            "output_dir": project_dir,
            "phases": {},
            "started_at": datetime.now().isoformat(),
        }
        
        # المرحلة 1: التحليل
        print(f"🏭 [1/5] تحليل الأمر: {command}")
        analysis = await self._phase_analyze(command)
        result["phases"]["analysis"] = analysis
        
        # المرحلة 2: التصميم
        print(f"📐 [2/5] تصميم البنية...")
        design = await self._phase_design(command, analysis)
        result["phases"]["design"] = design
        
        # المرحلة 3: توليد الكود
        print(f"⚡ [3/5] توليد الكود...")
        generated = await self._phase_generate(design, project_dir)
        result["phases"]["generation"] = generated
        
        # المرحلة 4: الفحص
        print(f"🔍 [4/5] فحص الجودة...")
        qa_result = await self._phase_qa(project_dir)
        result["phases"]["qa"] = qa_result
        
        # المرحلة 5: التسليم
        print(f"📦 [5/5] تجهيز التسليم...")
        delivery = self._phase_deliver(result, project_dir)
        result["phases"]["delivery"] = delivery
        
        result["completed_at"] = datetime.now().isoformat()
        result["status"] = "completed"
        
        self.projects.append(result)
        print(f"✅ المشروع جاهز: {project_dir}")
        
        return result
    
    async def _phase_analyze(self, command: str) -> Dict:
        """المرحلة 1: تحليل الأمر عبر AI"""
        prompt = f"""حلل هذا الطلب البرمجي وحدد:
1. نوع المشروع (web app, API, desktop, mobile)
2. الوحدات المطلوبة (modules)
3. جداول قاعدة البيانات
4. API endpoints
5. صفحات الواجهة

الطلب: {command}

أجب بـ JSON فقط."""
        
        ai_response = await self._ask_ai(prompt)
        
        # Try to parse JSON from response
        try:
            # Extract JSON from response
            if "```json" in ai_response:
                json_str = ai_response.split("```json")[1].split("```")[0]
            elif "{" in ai_response:
                start = ai_response.index("{")
                end = ai_response.rindex("}") + 1
                json_str = ai_response[start:end]
            else:
                json_str = "{}"
            
            parsed = json.loads(json_str)
            return {"status": "analyzed", "analysis": parsed, "raw": ai_response[:500]}
        except (json.JSONDecodeError, ValueError):
            return {"status": "analyzed", "analysis": {"type": "web_app", "modules": ["core"]}, "raw": ai_response[:500]}
    
    async def _phase_design(self, command: str, analysis: Dict) -> Dict:
        """المرحلة 2: تصميم البنية عبر AI"""
        prompt = f"""صمم بنية مشروع Python/FastAPI بناءً على:
الأمر: {command}
التحليل: {json.dumps(analysis.get('analysis', {}), ensure_ascii=False)}

أعطني:
1. هيكل المجلدات
2. الملفات المطلوبة مع وصف كل ملف
3. الـ models (Pydantic/SQLAlchemy)
4. الـ API routes

أجب بـ JSON فقط."""
        
        ai_response = await self._ask_ai(prompt)
        
        try:
            if "```json" in ai_response:
                json_str = ai_response.split("```json")[1].split("```")[0]
            elif "{" in ai_response:
                start = ai_response.index("{")
                end = ai_response.rindex("}") + 1
                json_str = ai_response[start:end]
            else:
                json_str = "{}"
            parsed = json.loads(json_str)
            return {"status": "designed", "design": parsed}
        except (json.JSONDecodeError, ValueError):
            # Default design
            return {
                "status": "designed",
                "design": {
                    "files": [
                        {"path": "main.py", "description": "Entry point"},
                        {"path": "models.py", "description": "Database models"},
                        {"path": "routes.py", "description": "API routes"},
                        {"path": "schemas.py", "description": "Pydantic schemas"},
                    ]
                }
            }
    
    async def _phase_generate(self, design: Dict, project_dir: str) -> Dict:
        """المرحلة 3: توليد الكود عبر AI"""
        files_created = []
        design_data = design.get("design", {})
        files = design_data.get("files", [])
        
        for file_info in files:
            filepath = file_info.get("path", "")
            description = file_info.get("description", "")
            
            if not filepath:
                continue
            
            prompt = f"""اكتب كود Python احترافي لملف '{filepath}':
الوصف: {description}
- استخدم FastAPI للـ routes
- استخدم SQLAlchemy/Pydantic للـ models
- أضف type hints
- أضف docstrings بالعربية
- اكتب كود جاهز للتشغيل

اكتب الكود فقط بدون شرح."""
            
            code = await self._ask_ai(prompt)
            
            # Clean code (remove markdown code blocks)
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            
            # Save file
            full_path = os.path.join(project_dir, filepath)
            os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else project_dir, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(code.strip() + "\n")
            
            files_created.append({
                "path": filepath,
                "size": len(code),
                "lines": code.count('\n') + 1,
            })
        
        return {
            "status": "generated",
            "files_created": len(files_created),
            "files": files_created,
        }
    
    async def _phase_qa(self, project_dir: str) -> Dict:
        """المرحلة 4: فحص الجودة"""
        errors = []
        checked = 0
        
        for root, _, files in os.walk(project_dir):
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                filepath = os.path.join(root, fname)
                checked += 1
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        compile(f.read(), filepath, 'exec')
                except SyntaxError as e:
                    errors.append({"file": fname, "error": str(e)})
        
        return {
            "status": "checked",
            "files_checked": checked,
            "syntax_errors": len(errors),
            "passed": len(errors) == 0,
            "errors": errors[:5],
        }
    
    def _phase_deliver(self, result: Dict, project_dir: str) -> Dict:
        """المرحلة 5: التسليم"""
        # Count total files and lines
        total_files = 0
        total_lines = 0
        for root, _, files in os.walk(project_dir):
            for fname in files:
                total_files += 1
                filepath = os.path.join(root, fname)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        total_lines += sum(1 for _ in f)
                except Exception:
                    pass
        
        # Save project manifest
        manifest = {
            "project_id": result["project_id"],
            "command": result["command"],
            "output_dir": project_dir,
            "total_files": total_files,
            "total_lines": total_lines,
            "phases": {k: v.get("status", "unknown") for k, v in result.get("phases", {}).items()},
            "created_at": result["started_at"],
        }
        
        manifest_path = os.path.join(project_dir, "PROJECT_MANIFEST.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "delivered",
            "total_files": total_files,
            "total_lines": total_lines,
            "manifest": manifest_path,
        }
    
    async def _ask_ai(self, prompt: str) -> str:
        """سؤال AI عبر Ollama — بدون أجوبة وهمية"""
        models = ["qwen2.5:1.5b", "llama3.2:latest", "codellama:7b"]
        
        for model in models:
            try:
                import requests
                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=60,
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "")
            except Exception:
                continue
        
        return "// AI غير متاح حالياً"
    
    def get_status(self) -> Dict:
        return {
            "name": "خط الإنتاج البرمجي",
            "active": True,
            "total_projects": len(self.projects),
            "last_project": self.projects[-1]["project_id"] if self.projects else None,
        }


# Singleton
auto_programming = AutoProgrammingPipeline()
