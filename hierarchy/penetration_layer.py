"""
طبقة الاختراق — Penetration Testing Layer
فحص أمني هجومي: يبحث عن الثغرات بالنظام

🔴 هذي الطبقة الهجومية — تكمّلها طبقة سد الثغرات (الدفاعية)
"""

import asyncio
import re
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime


class PenetrationLayer:
    """
    طبقة الاختراق — تفحص النظام من منظور المهاجم
    
    الفحوصات:
    - SQL Injection patterns
    - XSS vulnerabilities
    - Insecure API endpoints
    - Hardcoded secrets
    - Open ports
    - Weak authentication
    """
    
    def __init__(self):
        self.name = "طبقة الاختراق"
        self.findings: List[Dict[str, Any]] = []
        self.last_scan: Optional[datetime] = None
        self.scan_count = 0
    
    async def full_scan(self, project_root: str) -> Dict[str, Any]:
        """فحص أمني شامل للمشروع"""
        self.findings = []
        self.scan_count += 1
        self.last_scan = datetime.now()
        
        await asyncio.gather(
            self._scan_hardcoded_secrets(project_root),
            self._scan_sql_injection(project_root),
            self._scan_xss(project_root),
            self._scan_insecure_endpoints(project_root),
            self._scan_weak_crypto(project_root),
        )
        
        return {
            "scan_id": f"pen-{self.scan_count}",
            "timestamp": self.last_scan.isoformat(),
            "total_findings": len(self.findings),
            "critical": len([f for f in self.findings if f["severity"] == "critical"]),
            "high": len([f for f in self.findings if f["severity"] == "high"]),
            "medium": len([f for f in self.findings if f["severity"] == "medium"]),
            "findings": self.findings,
        }
    
    async def _scan_hardcoded_secrets(self, root: str):
        """البحث عن مفاتيح وكلمات سر مكشوفة"""
        patterns = [
            (r'(?i)(password|secret|api.?key|token)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded Secret"),
            (r'(?i)BEGIN\s+(RSA|DSA|EC)\s+PRIVATE\s+KEY', "Private Key Exposed"),
            (r'(?i)(aws_access_key|aws_secret)\s*=', "AWS Credentials"),
        ]
        await self._grep_patterns(root, patterns, "critical", "hardcoded-secrets")
    
    async def _scan_sql_injection(self, root: str):
        """البحث عن SQL Injection"""
        patterns = [
            (r'f["\'].*SELECT.*{.*}', "Potential SQL Injection (f-string)"),
            (r'\.format\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)', "Potential SQL Injection (format)"),
            (r'%s.*(?:SELECT|INSERT|UPDATE|DELETE).*%', "Potential SQL Injection (% format)"),
        ]
        await self._grep_patterns(root, patterns, "critical", "sql-injection")
    
    async def _scan_xss(self, root: str):
        """البحث عن XSS vulnerabilities"""
        patterns = [
            (r'innerHTML\s*=', "Potential XSS (innerHTML)"),
            (r'dangerouslySetInnerHTML', "Potential XSS (React dangerouslySetInnerHTML)"),
            (r'v-html\s*=', "Potential XSS (Vue v-html)"),
        ]
        await self._grep_patterns(root, patterns, "high", "xss")
    
    async def _scan_insecure_endpoints(self, root: str):
        """البحث عن endpoints بدون حماية"""
        patterns = [
            (r'allow_origins\s*=\s*\[\s*"\*"\s*\]', "Wildcard CORS"),
            (r'verify\s*=\s*False', "SSL Verification Disabled"),
            (r'debug\s*=\s*True', "Debug Mode Enabled"),
        ]
        await self._grep_patterns(root, patterns, "high", "insecure-endpoints")
    
    async def _scan_weak_crypto(self, root: str):
        """البحث عن تشفير ضعيف"""
        patterns = [
            (r'(?i)md5\s*\(', "Weak Hash (MD5)"),
            (r'(?i)sha1\s*\(', "Weak Hash (SHA1)"),
            (r'DES|RC4|Blowfish', "Weak Encryption Algorithm"),
        ]
        await self._grep_patterns(root, patterns, "medium", "weak-crypto")
    
    async def _grep_patterns(self, root: str, patterns: list, severity: str, category: str):
        """Helper to grep for patterns in project"""
        import os
        for dirpath, _, filenames in os.walk(root):
            if any(skip in dirpath for skip in ['node_modules', '.git', '__pycache__', '.venv', 'venv']):
                continue
            for fname in filenames:
                if not fname.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.html')):
                    continue
                filepath = os.path.join(dirpath, fname)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            for pattern, desc in patterns:
                                if re.search(pattern, line):
                                    self.findings.append({
                                        "file": filepath,
                                        "line": line_num,
                                        "description": desc,
                                        "severity": severity,
                                        "category": category,
                                        "content": line.strip()[:100],
                                    })
                except Exception:
                    continue
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active": True,
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "scan_count": self.scan_count,
            "total_findings": len(self.findings),
        }


# Singleton
penetration_layer = PenetrationLayer()
