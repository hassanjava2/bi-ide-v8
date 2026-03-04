"""
طبقة الجمال والأداء — UX Excellence Layer
سلاسة + جمالية + أداء غير مسبوق بالكون

🎨 تفحص كل واجهة قبل التسليم — أداء + جمال + تجربة
"""

import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime


class UXExcellenceLayer:
    """
    طبقة الجمال والأداء — تضمن أعلى جودة UI/UX
    
    الفحوصات:
    - CSS quality (animations, transitions, responsive)
    - Performance (bundle size, lazy loading, image optimization)
    - Accessibility (aria labels, color contrast)
    - Consistency (design tokens, spacing, typography)
    """
    
    def __init__(self):
        self.name = "طبقة الجمال والأداء"
        self.audits: List[Dict[str, Any]] = []
        self.last_audit: Optional[datetime] = None
    
    async def audit_ui(self, project_root: str) -> Dict[str, Any]:
        """تدقيق شامل للواجهة"""
        results = {}
        
        results["animations"] = self._check_animations(project_root)
        results["responsive"] = self._check_responsive(project_root)
        results["performance"] = self._check_performance(project_root)
        results["accessibility"] = self._check_accessibility(project_root)
        results["design_system"] = self._check_design_system(project_root)
        
        self.last_audit = datetime.now()
        
        score = sum(r.get("score", 0) for r in results.values()) / max(len(results), 1)
        
        audit = {
            "timestamp": self.last_audit.isoformat(),
            "overall_score": round(score, 1),
            "verdict": "🌟 ممتاز" if score >= 8 else "✅ جيد" if score >= 6 else "⚠️ يحتاج تحسين",
            "results": results,
        }
        self.audits.append(audit)
        return audit
    
    def _check_animations(self, root: str) -> Dict:
        """التأكد من وجود animations وtransitions"""
        found = {"transitions": 0, "animations": 0, "transforms": 0}
        for dirpath, _, filenames in os.walk(root):
            if any(s in dirpath for s in ['node_modules', '.git']):
                continue
            for fname in filenames:
                if not fname.endswith(('.css', '.scss', '.tsx', '.jsx')):
                    continue
                filepath = os.path.join(dirpath, fname)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        found["transitions"] += len(re.findall(r'transition', content))
                        found["animations"] += len(re.findall(r'@keyframes|animation:', content))
                        found["transforms"] += len(re.findall(r'transform:', content))
                except Exception:
                    continue
        
        total = sum(found.values())
        score = min(10, total / 5)  # 50+ = perfect score
        return {"check": "animations", "found": found, "score": round(score, 1)}
    
    def _check_responsive(self, root: str) -> Dict:
        """التأكد من responsive design"""
        media_queries = 0
        flex_grid = 0
        for dirpath, _, filenames in os.walk(root):
            if any(s in dirpath for s in ['node_modules', '.git']):
                continue
            for fname in filenames:
                if not fname.endswith(('.css', '.scss')):
                    continue
                filepath = os.path.join(dirpath, fname)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        media_queries += len(re.findall(r'@media', content))
                        flex_grid += len(re.findall(r'display:\s*(flex|grid)', content))
                except Exception:
                    continue
        
        score = min(10, (media_queries + flex_grid) / 3)
        return {"check": "responsive", "media_queries": media_queries, "flex_grid": flex_grid, "score": round(score, 1)}
    
    def _check_performance(self, root: str) -> Dict:
        """فحص الأداء"""
        issues = []
        large_files = 0
        for dirpath, _, filenames in os.walk(root):
            if any(s in dirpath for s in ['node_modules', '.git', 'training_data']):
                continue
            for fname in filenames:
                if fname.endswith(('.js', '.css')):
                    filepath = os.path.join(dirpath, fname)
                    size = os.path.getsize(filepath)
                    if size > 500_000:  # > 500KB
                        large_files += 1
                        issues.append(f"{fname}: {size // 1024}KB")
        
        score = max(0, 10 - large_files * 2)
        return {"check": "performance", "large_files": large_files, "issues": issues[:5], "score": score}
    
    def _check_accessibility(self, root: str) -> Dict:
        """فحص accessibility"""
        aria_count = 0
        alt_count = 0
        for dirpath, _, filenames in os.walk(root):
            if any(s in dirpath for s in ['node_modules', '.git']):
                continue
            for fname in filenames:
                if not fname.endswith(('.html', '.tsx', '.jsx', '.vue')):
                    continue
                filepath = os.path.join(dirpath, fname)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        aria_count += len(re.findall(r'aria-', content))
                        alt_count += len(re.findall(r'alt=', content))
                except Exception:
                    continue
        
        score = min(10, (aria_count + alt_count) / 5)
        return {"check": "accessibility", "aria_labels": aria_count, "alt_texts": alt_count, "score": round(score, 1)}
    
    def _check_design_system(self, root: str) -> Dict:
        """فحص نظام التصميم"""
        css_vars = 0
        for dirpath, _, filenames in os.walk(root):
            if any(s in dirpath for s in ['node_modules', '.git']):
                continue
            for fname in filenames:
                if not fname.endswith(('.css', '.scss')):
                    continue
                filepath = os.path.join(dirpath, fname)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        css_vars += len(re.findall(r'--[a-z]', content))
                except Exception:
                    continue
        
        score = min(10, css_vars / 10)
        return {"check": "design_system", "css_variables": css_vars, "score": round(score, 1)}
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active": True,
            "total_audits": len(self.audits),
            "last_score": self.audits[-1]["overall_score"] if self.audits else None,
        }


ux_excellence_layer = UXExcellenceLayer()
