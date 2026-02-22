"""
طبقة الامتثال القانوني - Legal Compliance Council (طبقة 3.5)
بين المجلس العالي وفرق الظل/النور
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import asyncio


class ComplianceType(Enum):
    """أنواع الامتثال"""
    GDPR = "GDPR"  # قوانين الاتحاد الأوروبي
    CCPA = "CCPA"  # قوانين كاليفورنيا
    HIPAA = "HIPAA"  # الصحة (لو طبقناها على ERP صحي)
    SOX = "SOX"  # Sarbanes-Oxley للمحاسبة
    LOCAL = "Local Law"  # قوانين البلد المحلي
    INDUSTRY = "Industry Standard"  # معايير الصناعة


class ComplianceStatus(Enum):
    """حالة الامتثال"""
    COMPLIANT = "متوافق"
    NON_COMPLIANT = "غير متوافق"
    REQUIRES_REVIEW = "يحتاج مراجعة"
    EXEMPT = "معفى"


@dataclass
class LegalReview:
    """مراجعة قانونية"""
    review_id: str
    decision_id: str
    reviewed_by: str
    compliance_types: List[ComplianceType]
    status: ComplianceStatus
    findings: List[str]
    recommendations: List[str]
    risk_level: str  # low, medium, high, critical
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataPrivacyAssessment:
    """تقييم خصوصية البيانات"""
    assessment_id: str
    data_type: str
    personal_data: bool
    sensitive_data: bool
    consent_required: bool
    retention_period: int  # days
    encryption_required: bool
    cross_border_transfer: bool


class GDPRCompliance:
    """
    الامتثال لقانون GDPR (الأوروبي)
    """
    
    def __init__(self):
        self.principles = [
            "lawfulness_fairness_transparency",
            "purpose_limitation",
            "data_minimization",
            "accuracy",
            "storage_limitation",
            "integrity_confidentiality",
            "accountability"
        ]
        print("⚖️ GDPR Compliance module initialized")
    
    def assess_data_processing(self, purpose: str, data_types: List[str]) -> Dict:
        """تقييم معالجة البيانات"""
        assessments = []
        
        for data_type in data_types:
            is_personal = self._is_personal_data(data_type)
            assessment = DataPrivacyAssessment(
                assessment_id=f"GDPR-{datetime.now().timestamp()}",
                data_type=data_type,
                personal_data=is_personal,
                sensitive_data=self._is_sensitive_data(data_type),
                consent_required=is_personal,
                retention_period=self._get_retention_period(data_type),
                encryption_required=is_personal,
                cross_border_transfer=False
            )
            assessments.append(assessment)
        
        return {
            "compliant": all(not a.personal_data or a.consent_required for a in assessments),
            "assessments": assessments,
            "required_actions": self._get_required_actions(assessments)
        }
    
    def _is_personal_data(self, data_type: str) -> bool:
        """هل البيانات شخصية؟"""
        personal_types = ['name', 'email', 'phone', 'address', 'id', 'biometric']
        return any(pt in data_type.lower() for pt in personal_types)
    
    def _is_sensitive_data(self, data_type: str) -> bool:
        """هل البيانات حساسة؟"""
        sensitive = ['health', 'financial', 'biometric', 'ethnic', 'political']
        return any(st in data_type.lower() for st in sensitive)
    
    def _get_retention_period(self, data_type: str) -> int:
        """فترة الاحتفاظ بالبيانات"""
        periods = {
            'financial': 2555,  # 7 سنوات
            'employee': 3650,   # 10 سنوات
            'customer': 1825,   # 5 سنوات
            'default': 365
        }
        for key, period in periods.items():
            if key in data_type.lower():
                return period
        return periods['default']
    
    def _get_required_actions(self, assessments: List[DataPrivacyAssessment]) -> List[str]:
        """الإجراءات المطلوبة"""
        actions = []
        for a in assessments:
            if a.personal_data and not a.consent_required:
                actions.append(f"Obtain consent for {a.data_type}")
            if a.sensitive_data:
                actions.append(f"Implement extra protection for {a.data_type}")
        return actions


class FinancialCompliance:
    """
    الامتثال المالي (SOX للمحاسبة)
    """
    
    def __init__(self):
        self.controls = {
            "access_control": True,
            "change_management": True,
            "audit_trail": True,
            "data_backup": True,
            "segregation_of_duties": True
        }
        print("💰 Financial Compliance module initialized")
    
    def review_financial_decision(self, decision: Dict) -> Dict:
        """مراجعة قرار مالي"""
        risks = []
        
        # التحقق من وجود موافقات كافية
        if decision.get('amount', 0) > 1000000:  # مليون
            risks.append("Amount exceeds approval limit - requires board approval")
        
        # التحقق من فصل المهام
        if decision.get('requested_by') == decision.get('approved_by'):
            risks.append("Segregation of duties violation")
        
        return {
            "compliant": len(risks) == 0,
            "risks": risks,
            "requires_board_approval": decision.get('amount', 0) > 1000000
        }


class IndustryStandards:
    """
    معايير الصناعة
    """
    
    def __init__(self):
        self.standards = {
            "ISO27001": "Information Security",
            "ISO9001": "Quality Management",
            "SOC2": "Service Organization Control"
        }
    
    def check_compliance(self, standard: str, implementation: Dict) -> Dict:
        """التحقق من الامتثال لمعيار"""
        # محاكاة فحص
        checks = {
            "policies_defined": True,
            "controls_implemented": True,
            "documentation_complete": True,
            "audits_passed": True
        }
        
        return {
            "standard": standard,
            "compliant": all(checks.values()),
            "checks": checks,
            "certification_status": "certified" if all(checks.values()) else "pending"
        }


class LegalComplianceCouncil:
    """
    ⚖️ مجلس الامتثال القانوني (طبقة 3.5)
    
    يقع بين مجلس الحكماء وفرق الظل/النور
    يضم 7 قانونيين متخصصين
    """
    
    def __init__(self):
        # الفرق القانونية
        self.legal_teams = {
            'gdpr_experts': [
                {'id': 'GDPR-001', 'name': 'خبير خصوصية البيانات', 'level': 'expert'},
                {'id': 'GDPR-002', 'name': 'مستشار GDPR', 'level': 'senior'}
            ],
            'financial_compliance': [
                {'id': 'FIN-001', 'name': 'مراجع مالي', 'level': 'expert'},
                {'id': 'FIN-002', 'name': 'محاسب قانوني', 'level': 'senior'}
            ],
            'corporate_law': [
                {'id': 'CORP-001', 'name': 'محامي شركات', 'level': 'expert'},
                {'id': 'CORP-002', 'name': 'مستشار قانوني', 'level': 'senior'}
            ],
            'industry_standards': [
                {'id': 'STD-001', 'name': 'مدير الامتثال', 'level': 'expert'}
            ]
        }
        
        # الوحدات المتخصصة
        self.gdpr = GDPRCompliance()
        self.financial = FinancialCompliance()
        self.standards = IndustryStandards()
        
        # سجل المراجعات
        self.reviews: List[LegalReview] = []
        
        print("\n" + "="*60)
        print("⚖️ LEGAL COMPLIANCE COUNCIL INITIALIZED")
        print("="*60)
        print("Legal Teams:")
        for team, members in self.legal_teams.items():
            print(f"  • {team}: {len(members)} experts")
        print("="*60 + "\n")
    
    async def review_decision(self, decision: Dict, decision_type: str = "general") -> LegalReview:
        """
        مراجعة قرار قانونياً قبل تنفيذه
        """
        print(f"⚖️ Legal review for decision: {decision.get('id', 'unknown')}")
        
        findings = []
        recommendations = []
        risk_level = "low"
        
        # 1. مراجعة GDPR (إذا كان القرار يتضمن بيانات)
        if 'data' in decision or 'customer' in decision:
            gdpr_result = self.gdpr.assess_data_processing(
                decision.get('purpose', 'general'),
                decision.get('data_types', [])
            )
            
            if not gdpr_result['compliant']:
                findings.append("GDPR compliance issues detected")
                recommendations.extend(gdpr_result['required_actions'])
                risk_level = "high"
        
        # 2. مراجعة مالية (إذا كان القرار مالياً)
        if decision_type == "financial" or 'amount' in decision:
            fin_result = self.financial.review_financial_decision(decision)
            
            if not fin_result['compliant']:
                findings.extend(fin_result['risks'])
                risk_level = "high" if fin_result['requires_board_approval'] else "medium"
            
            if fin_result['requires_board_approval']:
                recommendations.append("Requires board of directors approval")
        
        # 3. معايير الصناعة
        for standard in ['ISO27001', 'ISO9001']:
            std_result = self.standards.check_compliance(standard, decision)
            if not std_result['compliant']:
                findings.append(f"Non-compliant with {standard}")
        
        # تحديد الحالة النهائية
        if len(findings) == 0:
            status = ComplianceStatus.COMPLIANT
        elif risk_level == "critical":
            status = ComplianceStatus.NON_COMPLIANT
        else:
            status = ComplianceStatus.REQUIRES_REVIEW
        
        review = LegalReview(
            review_id=f"LEGAL-{datetime.now().timestamp()}",
            decision_id=decision.get('id', 'unknown'),
            reviewed_by=self.legal_teams['corporate_law'][0]['name'],
            compliance_types=[ComplianceType.GDPR, ComplianceType.SOX],
            status=status,
            findings=findings,
            recommendations=recommendations,
            risk_level=risk_level
        )
        
        self.reviews.append(review)
        
        print(f"✅ Legal review completed: {status.value}")
        
        return review
    
    def get_compliance_report(self) -> Dict:
        """تقرير الامتثال الشامل"""
        total = len(self.reviews)
        compliant = len([r for r in self.reviews if r.status == ComplianceStatus.COMPLIANT])
        non_compliant = len([r for r in self.reviews if r.status == ComplianceStatus.NON_COMPLIANT])
        
        return {
            "total_decisions_reviewed": total,
            "compliant": compliant,
            "non_compliant": non_compliant,
            "compliance_rate": (compliant / total * 100) if total > 0 else 0,
            "pending_review": len([r for r in self.reviews if r.status == ComplianceStatus.REQUIRES_REVIEW]),
            "high_risk_findings": len([r for r in self.reviews if r.risk_level == "high"])
        }
    
    async def generate_privacy_policy(self, system_description: str) -> str:
        """توليد سياسة الخصوصية"""
        policy = f"""
        سياسة الخصوصية - BI IDE System
        
        1. جمع البيانات:
           {system_description}
        
        2. استخدام البيانات:
           - تحسين النظام
           - تقديم الخدمات
           - الامتثال القانوني
        
        3. حماية البيانات:
           - تشفير كمي
           - وصول مقيد
           - تدقيق مستمر
        
        4. حقوق المستخدم:
           - الحق في الوصول
           - الحق في التصحيح
           - الحق في الحذف
        
        5. الاحتفاظ:
           - فترة محددة حسب نوع البيانات
           - حذف آمن بعد الانتهاء
        
        تاريخ التحديث: {datetime.now().strftime('%Y-%m-%d')}
        """
        return policy


# Singleton
legal_compliance_council = LegalComplianceCouncil()
