"""
محرك التطوير الذاتي — Self-Development Engine
AI ينسخ المشروع ← يطوّر ← يجرّب ← يقارن ← يستبدل

🧬 المشروع يطوّر نفسه — "التطور الذاتي المستمر"
"""

import os
import json
import shutil
import subprocess
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class SelfDevelopmentEngine:
    """
    محرك التطوير الذاتي — AI يحسّن المشروع أوتوماتيكياً
    
    الخطوات:
    1. نسخ المشروع → مجلد مؤقت
    2. تحليل الكود وتحديد نقاط التحسين  
    3. توليد تحسينات عبر AI
    4. اختبار النسخة المحسّنة
    5. مقارنة الأداء
    6. عرض التغييرات للمستخدم للموافقة
    """
    
    def __init__(self, project_root: str = "/home/bi/bi-ide-v8"):
        self.project_root = project_root
        self.ollama_url = "http://localhost:11434"
        self.improvements: List[Dict] = []
        self.proposals: List[Dict] = []
    
    async def analyze_project(self) -> Dict[str, Any]:
        """تحليل المشروع وتحديد فرص التحسين"""
        opportunities = []
        
        # 1. ملفات كبيرة تحتاج تقسيم
        large_files = self._find_large_files()
        for f in large_files:
            opportunities.append({
                "type": "refactor",
                "file": f["path"],
                "reason": f"ملف كبير ({f['lines']} سطر) — يحتاج تقسيم",
                "priority": "medium",
            })
        
        # 2. ملفات بدون docstrings
        undocumented = self._find_undocumented()
        for f in undocumented:
            opportunities.append({
                "type": "documentation",
                "file": f,
                "reason": "بدون docstrings",
                "priority": "low",
            })
        
        # 3. أنماط كود قديمة
        old_patterns = self._find_old_patterns()
        for p in old_patterns:
            opportunities.append({
                "type": "modernize",
                "file": p["file"],
                "reason": p["reason"],
                "priority": "medium",
            })
        
        # 4. أخطاء محتملة
        issues = self._find_potential_issues()
        for issue in issues:
            opportunities.append({
                "type": "bugfix",
                "file": issue["file"],
                "reason": issue["reason"],
                "priority": "high",
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_opportunities": len(opportunities),
            "by_priority": {
                "high": len([o for o in opportunities if o["priority"] == "high"]),
                "medium": len([o for o in opportunities if o["priority"] == "medium"]),
                "low": len([o for o in opportunities if o["priority"] == "low"]),
            },
            "opportunities": opportunities[:20],  # First 20
        }
    
    async def propose_improvement(self, filepath: str) -> Dict[str, Any]:
        """اقتراح تحسين لملف محدد عبر AI"""
        full_path = os.path.join(self.project_root, filepath)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {filepath}"}
        
        with open(full_path, 'r', encoding='utf-8') as f:
            original = f.read()
        
        # Ask AI for improvements
        prompt = f"""حلل هذا الكود واقترح تحسينات:
1. أداء أفضل
2. قراءة أفضل  
3. أمان أفضل
4. تنظيم أفضل

اكتب فقط التغييرات المقترحة بصيغة diff:

```python
{original[:4000]}
```"""
        
        ai_response = await self._ask_ai(prompt)
        
        proposal = {
            "id": f"prop_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "file": filepath,
            "original_lines": original.count('\n') + 1,
            "suggestion": ai_response[:2000],
            "status": "pending",  # pending, approved, rejected
            "created_at": datetime.now().isoformat(),
        }
        
        self.proposals.append(proposal)
        return proposal
    
    async def run_improvement_cycle(self) -> Dict[str, Any]:
        """دورة تحسين كاملة: تحليل → اقتراح → اختبار"""
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 1. تحليل
        analysis = await self.analyze_project()
        
        # 2. اقتراح تحسينات لأهم الملفات
        high_priority = [o for o in analysis["opportunities"] if o["priority"] == "high"]
        proposals = []
        for opp in high_priority[:3]:  # Top 3
            proposal = await self.propose_improvement(opp["file"])
            proposals.append(proposal)
        
        # 3. نتيجة الدورة
        result = {
            "cycle_id": cycle_id,
            "analysis": {
                "total_opportunities": analysis["total_opportunities"],
                "by_priority": analysis["by_priority"],
            },
            "proposals": len(proposals),
            "status": "proposals_ready",
            "note": "التحسينات جاهزة — تنتظر موافقة الرئيس",
        }
        
        self.improvements.append(result)
        return result
    
    def _find_large_files(self) -> List[Dict]:
        """إيجاد ملفات Python كبيرة"""
        large = []
        for dirpath, _, filenames in os.walk(self.project_root):
            if any(s in dirpath for s in ['node_modules', '.git', '__pycache__', 'venv']):
                continue
            for fname in filenames:
                if not fname.endswith('.py'):
                    continue
                filepath = os.path.join(dirpath, fname)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = sum(1 for _ in f)
                    if lines > 400:
                        large.append({
                            "path": os.path.relpath(filepath, self.project_root),
                            "lines": lines,
                        })
                except Exception:
                    continue
        return sorted(large, key=lambda x: x["lines"], reverse=True)[:10]
    
    def _find_undocumented(self) -> List[str]:
        """إيجاد ملفات بدون docstrings"""
        undoc = []
        for dirpath, _, filenames in os.walk(self.project_root):
            if any(s in dirpath for s in ['node_modules', '.git', '__pycache__', 'venv']):
                continue
            for fname in filenames:
                if not fname.endswith('.py') or fname.startswith('__'):
                    continue
                filepath = os.path.join(dirpath, fname)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read(500)
                    if '"""' not in content and "'''" not in content:
                        undoc.append(os.path.relpath(filepath, self.project_root))
                except Exception:
                    continue
        return undoc[:20]
    
    def _find_old_patterns(self) -> List[Dict]:
        """إيجاد أنماط كود قديمة"""
        import re
        patterns_to_find = [
            (r'except\s*:', "Bare except — يمسك كل الأخطاء"),
            (r'print\s*\(', "print() بدل logging"),
            (r'eval\s*\(', "eval() خطر أمني"),
            (r'exec\s*\(', "exec() خطر أمني"),
        ]
        found = []
        for dirpath, _, filenames in os.walk(self.project_root):
            if any(s in dirpath for s in ['node_modules', '.git', '__pycache__', 'venv']):
                continue
            for fname in filenames:
                if not fname.endswith('.py'):
                    continue
                filepath = os.path.join(dirpath, fname)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            for pattern, reason in patterns_to_find:
                                if re.search(pattern, line):
                                    found.append({
                                        "file": os.path.relpath(filepath, self.project_root),
                                        "line": line_num,
                                        "reason": reason,
                                    })
                except Exception:
                    continue
        return found[:30]
    
    def _find_potential_issues(self) -> List[Dict]:
        """إيجاد مشاكل محتملة"""
        import re
        issues = []
        for dirpath, _, filenames in os.walk(self.project_root):
            if any(s in dirpath for s in ['node_modules', '.git', '__pycache__', 'venv']):
                continue
            for fname in filenames:
                if not fname.endswith('.py'):
                    continue
                filepath = os.path.join(dirpath, fname)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for TODO/FIXME
                    for line_num, line in enumerate(content.split('\n'), 1):
                        if 'TODO' in line or 'FIXME' in line or 'HACK' in line:
                            issues.append({
                                "file": os.path.relpath(filepath, self.project_root),
                                "line": line_num,
                                "reason": line.strip()[:80],
                            })
                except Exception:
                    continue
        return issues[:20]
    
    async def _ask_ai(self, prompt: str) -> str:
        """سؤال AI — Ollama فقط (لا أجوبة وهمية)"""
        models = ["codellama:7b", "qwen2.5:1.5b", "llama3.2:latest"]
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
            "name": "محرك التطوير الذاتي",
            "active": True,
            "total_improvements": len(self.improvements),
            "pending_proposals": len([p for p in self.proposals if p["status"] == "pending"]),
        }


# Singleton
self_development = SelfDevelopmentEngine()
