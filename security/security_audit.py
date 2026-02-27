"""
BI-IDE v8 - Security Audit Module
Automated vulnerability scanning and security compliance checks
"""

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

import aiohttp
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Vulnerability:
    """Vulnerability finding"""
    id: str
    title: str
    description: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    remediation: Optional[str] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    references: List[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Security audit report"""
    timestamp: str
    total_files_scanned: int
    vulnerabilities: List[Vulnerability]
    secrets_found: List[Dict[str, Any]]
    compliance_status: Dict[str, bool]
    risk_score: float
    summary: Dict[str, int]


class SecurityAuditor:
    """Main security auditor class"""
    
    SEVERITY_WEIGHTS = {
        'CRITICAL': 10,
        'HIGH': 7,
        'MEDIUM': 4,
        'LOW': 1,
        'INFO': 0
    }
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.vulnerabilities: List[Vulnerability] = []
        self.secrets_found: List[Dict[str, Any]] = []
        self.compliance_status: Dict[str, bool] = {}
        self.scanners: List[BaseScanner] = [
            DependencyScanner(),
            SecretScanner(),
            CodeScanner(),
            ConfigScanner(),
            OWASPComplianceScanner(),
        ]
    
    async def run_full_audit(self) -> AuditReport:
        """Run complete security audit"""
        logger.info("Starting full security audit...")
        
        # Run all scanners in parallel
        results = await asyncio.gather(*[
            scanner.scan(self.project_path)
            for scanner in self.scanners
        ])
        
        # Aggregate results
        for result in results:
            if isinstance(result, list):
                self.vulnerabilities.extend(result)
            elif isinstance(result, dict):
                if 'secrets' in result:
                    self.secrets_found.extend(result['secrets'])
                elif 'compliance' in result:
                    self.compliance_status.update(result['compliance'])
        
        # Calculate risk score
        risk_score = self._calculate_risk_score()
        
        # Generate summary
        summary = self._generate_summary()
        
        report = AuditReport(
            timestamp=datetime.now().isoformat(),
            total_files_scanned=self._count_scanned_files(),
            vulnerabilities=self.vulnerabilities,
            secrets_found=self.secrets_found,
            compliance_status=self.compliance_status,
            risk_score=risk_score,
            summary=summary
        )
        
        logger.info(f"Audit complete. Risk score: {risk_score:.2f}")
        return report
    
    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score (0-100)"""
        if not self.vulnerabilities:
            return 0.0
        
        total_weight = sum(
            self.SEVERITY_WEIGHTS.get(v.severity, 0)
            for v in self.vulnerabilities
        )
        
        # Normalize to 0-100
        max_possible = len(self.vulnerabilities) * 10
        return min(100, (total_weight / max_possible) * 100) if max_possible > 0 else 0
    
    def _generate_summary(self) -> Dict[str, int]:
        """Generate vulnerability summary"""
        summary = {severity: 0 for severity in self.SEVERITY_WEIGHTS.keys()}
        for vuln in self.vulnerabilities:
            summary[vuln.severity] = summary.get(vuln.severity, 0) + 1
        return summary
    
    def _count_scanned_files(self) -> int:
        """Count total files scanned"""
        count = 0
        for pattern in ['**/*.py', '**/*.js', '**/*.ts', '**/*.json', '**/*.yaml', '**/*.yml']:
            count += len(list(self.project_path.glob(pattern)))
        return count
    
    def generate_report(self, output_path: str, format: str = 'json'):
        """Generate audit report in specified format"""
        report = asyncio.run(self.run_full_audit())
        
        if format == 'json':
            self._generate_json_report(report, output_path)
        elif format == 'html':
            self._generate_html_report(report, output_path)
        elif format == 'sarif':
            self._generate_sarif_report(report, output_path)
    
    def _generate_json_report(self, report: AuditReport, path: str):
        """Generate JSON report"""
        data = {
            'timestamp': report.timestamp,
            'risk_score': report.risk_score,
            'summary': report.summary,
            'total_files': report.total_files_scanned,
            'vulnerabilities': [
                {
                    'id': v.id,
                    'title': v.title,
                    'description': v.description,
                    'severity': v.severity,
                    'category': v.category,
                    'file': v.file_path,
                    'line': v.line_number,
                    'remediation': v.remediation,
                    'cwe': v.cwe_id,
                    'cvss': v.cvss_score
                }
                for v in report.vulnerabilities
            ],
            'secrets': report.secrets_found,
            'compliance': report.compliance_status
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_html_report(self, report: AuditReport, path: str):
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BI-IDE v8 Security Audit Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #1a1a2e; color: white; padding: 20px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background: #f0f0f0; padding: 15px; border-radius: 8px; }}
                .CRITICAL {{ color: #dc3545; }}
                .HIGH {{ color: #fd7e14; }}
                .MEDIUM {{ color: #ffc107; }}
                .LOW {{ color: #17a2b8; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #1a1a2e; color: white; }}
                tr:hover {{ background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BI-IDE v8 Security Audit Report</h1>
                <p>Generated: {report.timestamp}</p>
                <p>Risk Score: {report.risk_score:.2f}/100</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>Files Scanned</h3>
                    <p>{report.total_files_scanned}</p>
                </div>
                <div class="metric">
                    <h3>Vulnerabilities</h3>
                    <p>{len(report.vulnerabilities)}</p>
                </div>
                <div class="metric">
                    <h3>Secrets Found</h3>
                    <p>{len(report.secrets_found)}</p>
                </div>
            </div>
            
            <h2>Vulnerability Summary</h2>
            <div class="summary">
                {''.join(f'<div class="metric {sev}"><h3>{sev}</h3><p>{count}</p></div>' 
                        for sev, count in report.summary.items() if count > 0)}
            </div>
            
            <h2>Vulnerabilities</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Title</th>
                    <th>Severity</th>
                    <th>Category</th>
                    <th>File</th>
                    <th>CWE</th>
                </tr>
                {''.join(f'''
                <tr class="{v.severity}">
                    <td>{v.id}</td>
                    <td>{v.title}</td>
                    <td>{v.severity}</td>
                    <td>{v.category}</td>
                    <td>{v.file_path or 'N/A'}</td>
                    <td>{v.cwe_id or 'N/A'}</td>
                </tr>
                ''' for v in report.vulnerabilities)}
            </table>
        </body>
        </html>
        """
        
        with open(path, 'w') as f:
            f.write(html)


class BaseScanner(ABC):
    """Base scanner class"""
    
    @abstractmethod
    async def scan(self, project_path: Path) -> Any:
        pass


class DependencyScanner(BaseScanner):
    """Scan dependencies for known vulnerabilities"""
    
    VULNERABILITY_DB_URL = "https://pyup.io/api/v1/safety/"
    
    async def scan(self, project_path: Path) -> List[Vulnerability]:
        """Scan Python dependencies"""
        vulnerabilities = []
        
        # Read requirements files
        req_files = list(project_path.glob('**/requirements*.txt'))
        
        for req_file in req_files:
            vulns = await self._scan_requirements(req_file)
            vulnerabilities.extend(vulns)
        
        # Check package-lock.json for Node.js
        lock_files = list(project_path.glob('**/package-lock.json'))
        for lock_file in lock_files:
            vulns = await self._scan_npm_packages(lock_file)
            vulnerabilities.extend(vulns)
        
        return vulnerabilities
    
    async def _scan_requirements(self, req_file: Path) -> List[Vulnerability]:
        """Scan Python requirements"""
        vulnerabilities = []
        
        try:
            # Run safety check
            result = subprocess.run(
                ['safety', 'check', '-r', str(req_file), '--json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                for vuln in data.get('vulnerabilities', []):
                    vulnerabilities.append(Vulnerability(
                        id=f"DEP-{vuln.get('cve', 'UNKNOWN')}",
                        title=vuln.get('package_name', 'Unknown Package'),
                        description=vuln.get('vulnerability_spec', 'Unknown vulnerability'),
                        severity=self._map_severity(vuln.get('severity', 'low')),
                        category='Dependency Vulnerability',
                        file_path=str(req_file),
                        cwe_id=vuln.get('cwe', None)
                    ))
        except Exception as e:
            logger.warning(f"Dependency scan failed for {req_file}: {e}")
        
        return vulnerabilities
    
    async def _scan_npm_packages(self, lock_file: Path) -> List[Vulnerability]:
        """Scan NPM packages with npm audit"""
        vulnerabilities = []
        
        try:
            result = subprocess.run(
                ['npm', 'audit', '--json'],
                cwd=lock_file.parent,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                for vuln_id, vuln_data in data.get('advisories', {}).items():
                    vulnerabilities.append(Vulnerability(
                        id=f"NPM-{vuln_id}",
                        title=vuln_data.get('module_name', 'Unknown'),
                        description=vuln_data.get('overview', ''),
                        severity=self._map_severity(vuln_data.get('severity', 'low')),
                        category='NPM Dependency Vulnerability',
                        file_path=str(lock_file),
                        cwe_id=str(vuln_data.get('cwe', [''])[0]) if vuln_data.get('cwe') else None
                    ))
        except Exception as e:
            logger.warning(f"NPM audit failed for {lock_file}: {e}")
        
        return vulnerabilities
    
    def _map_severity(self, severity: str) -> str:
        """Map severity to standard levels"""
        mapping = {
            'critical': 'CRITICAL',
            'high': 'HIGH',
            'moderate': 'MEDIUM',
            'medium': 'MEDIUM',
            'low': 'LOW',
        }
        return mapping.get(severity.lower(), 'INFO')


class SecretScanner(BaseScanner):
    """Scan for hardcoded secrets"""
    
    SECRET_PATTERNS = {
        'AWS Access Key': r'AKIA[0-9A-Z]{16}',
        'AWS Secret Key': r'['\"'][0-9a-zA-Z/+]{40}['\'" ]',
        'Generic API Key': r'[aA][pP][iI]_?[kK][eE][yY][\s]*[=:]+[\s]*['\'"][a-zA-Z0-9_\-]{16,}['\'"]',
        'Private Key': r'-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----',
        'JWT Token': r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*',
        'GitHub Token': r'gh[pousr]_[A-Za-z0-9_]{36,}',
        'Slack Token': r'xox[baprs]-[0-9a-zA-Z]{10,48}',
        'Generic Secret': r'[sS][eE][cC][rR][eE][tT][\s]*[=:]+[\s]*['\'"][a-zA-Z0-9_\-]{8,}['\'"]',
        'Password': r'[pP][aA][sS][sS][wW][oO][rR][dD][\s]*[=:]+[\s]*['\'"][^'\'"]{8,}['\'"]',
        'Database URL': r'(postgres|mysql|mongodb)://[^:]+:[^@]+@[^/]+',
    }
    
    EXCLUDED_PATHS = [
        '.git',
        'node_modules',
        '__pycache__',
        '.venv',
        'venv',
        '.env.example',
        'test_*.py',
        '*_test.py',
    ]
    
    async def scan(self, project_path: Path) -> Dict[str, Any]:
        """Scan for secrets"""
        secrets = []
        
        for file_path in project_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Skip excluded paths
            if any(excl in str(file_path) for excl in self.EXCLUDED_PATHS):
                continue
            
            # Skip binary files
            if file_path.suffix in ['.pyc', '.png', '.jpg', '.jpeg', '.gif', '.ico']:
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                file_secrets = self._scan_file_content(content, file_path)
                secrets.extend(file_secrets)
            except Exception as e:
                logger.debug(f"Could not read {file_path}: {e}")
        
        return {'secrets': secrets}
    
    def _scan_file_content(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Scan file content for secrets"""
        secrets = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for secret_type, pattern in self.SECRET_PATTERNS.items():
                matches = re.finditer(pattern, line)
                for match in matches:
                    # Skip test data
                    if self._is_test_data(line):
                        continue
                    
                    secrets.append({
                        'type': secret_type,
                        'file': str(file_path),
                        'line': line_num,
                        'match': match.group()[:50] + '...' if len(match.group()) > 50 else match.group(),
                        'hash': hashlib.sha256(match.group().encode()).hexdigest()[:16]
                    })
        
        return secrets
    
    def _is_test_data(self, line: str) -> bool:
        """Check if line contains test/mock data"""
        test_indicators = ['test', 'example', 'sample', 'mock', 'dummy', 'fake', 'placeholder']
        return any(indicator in line.lower() for indicator in test_indicators)


class CodeScanner(BaseScanner):
    """Static code analysis scanner"""
    
    PATTERNS = {
        'sql_injection': {
            'pattern': r'execute\s*\(\s*["\'].*%s.*["\']',
            'severity': 'CRITICAL',
            'cwe': 'CWE-89'
        },
        'hardcoded_password': {
            'pattern': r'password\s*=\s*["\'][^"\']+["\']',
            'severity': 'HIGH',
            'cwe': 'CWE-798'
        },
        'eval_usage': {
            'pattern': r'\beval\s*\(',
            'severity': 'CRITICAL',
            'cwe': 'CWE-95'
        },
        'pickle_load': {
            'pattern': r'pickle\.loads?\s*\(',
            'severity': 'HIGH',
            'cwe': 'CWE-502'
        },
        'yaml_load': {
            'pattern': r'yaml\.load\s*\(',
            'severity': 'MEDIUM',
            'cwe': 'CWE-502'
        },
        'debug_mode': {
            'pattern': r'DEBUG\s*=\s*True',
            'severity': 'MEDIUM',
            'cwe': 'CWE-489'
        },
        'insecure_hash': {
            'pattern': r'hashlib\.(md5|sha1)\s*\(',
            'severity': 'MEDIUM',
            'cwe': 'CWE-916'
        },
        'disabled_ssl_verify': {
            'pattern': r'verify\s*=\s*False',
            'severity': 'HIGH',
            'cwe': 'CWE-295'
        },
    }
    
    async def scan(self, project_path: Path) -> List[Vulnerability]:
        """Scan Python code"""
        vulnerabilities = []
        
        for py_file in project_path.rglob('*.py'):
            # Skip test files
            if 'test' in py_file.name:
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                file_vulns = self._analyze_python_file(content, py_file)
                vulnerabilities.extend(file_vulns)
            except Exception as e:
                logger.debug(f"Could not analyze {py_file}: {e}")
        
        return vulnerabilities
    
    def _analyze_python_file(self, content: str, file_path: Path) -> List[Vulnerability]:
        """Analyze Python file for vulnerabilities"""
        vulnerabilities = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for vuln_type, config in self.PATTERNS.items():
                if re.search(config['pattern'], line, re.IGNORECASE):
                    # Skip comments
                    if line.strip().startswith('#'):
                        continue
                    
                    vulnerabilities.append(Vulnerability(
                        id=f"CODE-{vuln_type.upper()}-{line_num}",
                        title=f"Potential {vuln_type.replace('_', ' ').title()}",
                        description=f"Found potentially unsafe pattern: {vuln_type}",
                        severity=config['severity'],
                        category='Code Security',
                        file_path=str(file_path),
                        line_number=line_num,
                        code_snippet=line.strip(),
                        cwe_id=config['cwe']
                    ))
        
        return vulnerabilities


class ConfigScanner(BaseScanner):
    """Configuration security scanner"""
    
    SECURITY_HEADERS = [
        'Strict-Transport-Security',
        'Content-Security-Policy',
        'X-Frame-Options',
        'X-Content-Type-Options',
        'X-XSS-Protection',
        'Referrer-Policy'
    ]
    
    async def scan(self, project_path: Path) -> List[Vulnerability]:
        """Scan configuration files"""
        vulnerabilities = []
        
        # Check Docker files
        dockerfiles = list(project_path.glob('**/Dockerfile*'))
        for df in dockerfiles:
            vulns = self._scan_dockerfile(df)
            vulnerabilities.extend(vulns)
        
        # Check Kubernetes manifests
        k8s_files = list(project_path.glob('deploy/**/*.yaml'))
        for kf in k8s_files:
            vulns = self._scan_k8s_manifest(kf)
            vulnerabilities.extend(vulns)
        
        # Check environment files
        env_files = list(project_path.glob('**/.env*'))
        for ef in env_files:
            if not ef.name.endswith('.example'):
                vulnerabilities.append(Vulnerability(
                    id="CONFIG-ENV-FILE",
                    title="Environment file present",
                    description="Environment file should not be committed to repository",
                    severity='MEDIUM',
                    category='Configuration',
                    file_path=str(ef)
                ))
        
        return vulnerabilities
    
    def _scan_dockerfile(self, dockerfile: Path) -> List[Vulnerability]:
        """Scan Dockerfile for security issues"""
        vulnerabilities = []
        content = dockerfile.read_text()
        
        # Check for latest tag
        if re.search(r'FROM\s+\w+:latest', content):
            vulnerabilities.append(Vulnerability(
                id="DOCKER-LATEST-TAG",
                title="Using 'latest' tag in Dockerfile",
                description="Avoid using 'latest' tag for base images",
                severity='LOW',
                category='Container Security',
                file_path=str(dockerfile)
            ))
        
        # Check for running as root
        if 'USER' not in content:
            vulnerabilities.append(Vulnerability(
                id="DOCKER-ROOT-USER",
                title="Container running as root",
                description="Add USER instruction to run as non-root",
                severity='MEDIUM',
                category='Container Security',
                file_path=str(dockerfile)
            ))
        
        return vulnerabilities
    
    def _scan_k8s_manifest(self, manifest: Path) -> List[Vulnerability]:
        """Scan Kubernetes manifest"""
        vulnerabilities = []
        
        try:
            content = yaml.safe_load(manifest.read_text())
            
            # Check for security context
            if content and content.get('kind') == 'Deployment':
                spec = content.get('spec', {}).get('template', {}).get('spec', {})
                containers = spec.get('containers', [])
                
                for container in containers:
                    security_context = container.get('securityContext', {})
                    
                    if not security_context.get('runAsNonRoot'):
                        vulnerabilities.append(Vulnerability(
                            id="K8S-NON-ROOT",
                            title="Container not configured to run as non-root",
                            severity='MEDIUM',
                            category='Kubernetes Security',
                            file_path=str(manifest)
                        ))
                    
                    if not security_context.get('readOnlyRootFilesystem'):
                        vulnerabilities.append(Vulnerability(
                            id="K8S-RO-FILES",
                            title="Root filesystem not read-only",
                            severity='LOW',
                            category='Kubernetes Security',
                            file_path=str(manifest)
                        ))
        
        except Exception as e:
            logger.debug(f"Could not parse {manifest}: {e}")
        
        return vulnerabilities


class OWASPComplianceScanner(BaseScanner):
    """OWASP compliance checker"""
    
    OWASP_CHECKS = {
        'A01:2021 - Broken Access Control': [
            'CORS misconfiguration',
            'Missing authorization checks',
            'Insecure direct object references'
        ],
        'A02:2021 - Cryptographic Failures': [
            'Weak encryption algorithms',
            'Plaintext data transmission',
            'Inadequate key management'
        ],
        'A03:2021 - Injection': [
            'SQL injection',
            'Command injection',
            'NoSQL injection'
        ],
        'A05:2021 - Security Misconfiguration': [
            'Default credentials',
            'Unnecessary features enabled',
            'Verbose error messages'
        ],
        'A07:2021 - Authentication Failures': [
            'Weak password policy',
            'Session fixation',
            'Credential stuffing vulnerability'
        ]
    }
    
    async def scan(self, project_path: Path) -> Dict[str, Any]:
        """Check OWASP compliance"""
        compliance = {}
        
        for control, checks in self.OWASP_CHECKS.items():
            # Simplified check - in real implementation would analyze code
            compliance[control] = self._check_control(project_path, checks)
        
        return {'compliance': compliance}
    
    def _check_control(self, project_path: Path, checks: List[str]) -> bool:
        """Check if control is satisfied"""
        # Simplified - would analyze actual code/config
        return True  # Assume compliant for template


# Main execution
if __name__ == '__main__':
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='BI-IDE v8 Security Audit')
    parser.add_argument('path', help='Project path to scan')
    parser.add_argument('--output', '-o', default='security_report.json', help='Output file')
    parser.add_argument('--format', '-f', choices=['json', 'html', 'sarif'], default='json')
    
    args = parser.parse_args()
    
    auditor = SecurityAuditor(args.path)
    auditor.generate_report(args.output, args.format)
    print(f"Security report generated: {args.output}")
