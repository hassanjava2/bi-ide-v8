"""
فرق الظل والنور - Shadow & Light Teams
التوازن بين التشاؤم والتفاؤل

⚫ فريق الظل (4 متشائمون):
- يرون الكوارث قبل حدوثها
- يحسبون أسوأ السيناريوهات
- يحمون من المخاطر

⚪ فريق النور (4 متفائلون):
- يرون الفرص في الأزمات
- يحلمون بأفضل المستقبلات
- يحفزون على النمو
"""
import sys; sys.path.insert(0, '.'); import encoding_fix; encoding_fix.safe_print("")

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio


@dataclass
class RiskAssessment:
    """تقييم مخاطر"""
    scenario: str
    probability: float  # 0-1
    impact: float       # 0-1
    risk_score: float   # probability × impact
    mitigation: str
    early_warnings: List[str]


@dataclass
class Opportunity:
    """فرصة"""
    description: str
    potential_gain: float
    probability: float
    required_investment: float
    roi: float
    timeline: str


class ShadowTeam:
    """
    ⚫ فريق الظل (4 متشائمون)
    
    مهمتهم: حماية النظام من الكوارث
    """
    
    def __init__(self):
        self.members = {
            'disaster_barker': 'نبّاح الكوارث',
            'vulnerability_hunter': 'صيّاد الثغرات',
            'failure_simulator': 'محاكي الفشل',
            'boundary_guard': 'حارس الحدود'
        }
        self.risk_database: List[RiskAssessment] = []
        print("⚫ Shadow Team initialized (4 pessimists)")
    
    async def analyze_proposal(self, proposal: Dict) -> Dict:
        """
        تحليل مقترح من منظور الكوارث
        
        Returns:
            تقييم المخاطر + تحذيرات
        """
        risks = []
        
        # 1. نبّاح الكوارث: أسوأ السيناريوهات
        worst_case = self._imagine_worst_case(proposal)
        risks.append(RiskAssessment(
            scenario=worst_case,
            probability=0.3,
            impact=0.9,
            risk_score=0.27,
            mitigation="تأمين احتياطي",
            early_warnings=["انخفاض المؤشرات", "شكاوى العملاء"]
        ))
        
        # 2. صيّاد الثغرات: نقاط الضعف
        vulnerabilities = self._find_vulnerabilities(proposal)
        for vuln in vulnerabilities:
            risks.append(RiskAssessment(
                scenario=vuln,
                probability=0.5,
                impact=0.6,
                risk_score=0.3,
                mitigation="تدقيق أمني",
                early_warnings=["محاولات اختراق"]
            ))
        
        # 3. محاكي الفشل: كيف نفشل؟
        failure_modes = self._simulate_failures(proposal)
        
        # 4. حارس الحدود: خطوط حمراء
        boundaries = self._check_boundaries(proposal)
        
        return {
            'team': 'Shadow',
            'proposal': proposal.get('name', 'Unknown'),
            'risks': risks,
            'failure_modes': failure_modes,
            'boundary_violations': boundaries,
            'recommendation': 'موافق مشروطة' if risks else 'رفض',
            'confidence': 0.85
        }
    
    def _imagine_worst_case(self, proposal: Dict) -> str:
        """تخيل أسوأ سيناريو"""
        scenarios = {
            'expansion': 'افتتاح 10 فروع → خسارة 5 منها → إفلاس',
            'hiring': 'تعيين 100 موظف → 50% استقالة → فوضى',
            'investment': 'استثمار مليون → خسارة 80% → أفلاس',
            'default': 'تنفيذ الخطة → فشل كامل → خسارة كل شيء'
        }
        return scenarios.get(proposal.get('type'), scenarios['default'])
    
    def _find_vulnerabilities(self, proposal: Dict) -> List[str]:
        """البحث عن ثغرات"""
        return [
            'تبعية على مورد واحد',
            'نقص في المواهب',
            'منافسة شرسة',
            'تغيرات تنظيمية'
        ]
    
    def _simulate_failures(self, proposal: Dict) -> List[str]:
        """محاكاة حالات الفشل"""
        return [
            'نفاد المال قبل الربحية',
            'مغادرة المدير التنفيذي',
            'اختراق أمني كبير',
            'مقاطعة المنتج'
        ]
    
    def _check_boundaries(self, proposal: Dict) -> List[str]:
        """التحقق من الخطوط الحمراء"""
        violations = []
        budget = proposal.get('budget', 0)
        if budget > 1000000:  # مليون
            violations.append('تجاوز حد الميزانية المسموح')
        return violations
    
    async def continuous_monitoring(self):
        """مراقبة مستمرة للمخاطر"""
        while True:
            # فحص دوري
            alerts = self._scan_for_risks()
            if alerts:
                print(f"🚨 Shadow Alert: {len(alerts)} risks detected")
                # رفع للحكماء
            await asyncio.sleep(3600)  # كل ساعة
    
    def _scan_for_risks(self) -> List[str]:
        """فحص المخاطر"""
        return []


class LightTeam:
    """
    ⚪ فريق النور (4 متفائلون)
    
    مهمتهم: رؤية الفرص والنمو
    """
    
    def __init__(self):
        self.members = {
            'opportunity_catcher': 'صائد الفرص',
            'future_builder': 'باني المستقبل',
            'luck_maximizer': 'مُحفز الحظ',
            'boundary_expander': 'موسع الحدود'
        }
        self.opportunities: List[Opportunity] = []
        print("⚪ Light Team initialized (4 optimists)")
    
    async def analyze_proposal(self, proposal: Dict) -> Dict:
        """
        تحليل مقترح من منظور الفرص
        
        Returns:
            الفرص المحتملة + توصيات
        """
        opportunities = []
        
        # 1. صائد الفرص: في كل أزمة
        crisis_opps = self._find_opportunities_in_crisis(proposal)
        opportunities.extend(crisis_opps)
        
        # 2. باني المستقبل: أفضل نسخة
        best_case = self._imagine_best_case(proposal)
        opportunities.append(Opportunity(
            description=best_case,
            potential_gain=1000000,
            probability=0.4,
            required_investment=proposal.get('budget', 100000),
            roi=10.0,
            timeline="1-2 years"
        ))
        
        # 3. مُحفز الحظ: استغلال المواقف
        luck_opps = self._maximize_luck(proposal)
        
        # 4. موسع الحدود: أكبر بـ 10 أضعاف
        expanded = self._expand_boundaries(proposal)
        
        return {
            'team': 'Light',
            'proposal': proposal.get('name', 'Unknown'),
            'opportunities': opportunities,
            'best_case_scenario': best_case,
            'expanded_vision': expanded,
            'recommendation': 'موافقة حماسية',
            'enthusiasm': 0.95
        }
    
    def _find_opportunities_in_crisis(self, proposal: Dict) -> List[Opportunity]:
        """البحث عن فرص في الأزمات"""
        return [
            Opportunity(
                description="انخفاض أسعار السوق = فرصة شراء",
                potential_gain=500000,
                probability=0.6,
                required_investment=200000,
                roi=2.5,
                timeline="6 months"
            )
        ]
    
    def _imagine_best_case(self, proposal: Dict) -> str:
        """تخيل أفضل سيناريو"""
        scenarios = {
            'expansion': 'افتتاح 10 فروع → نجاح 15 → امتلاك السوق',
            'hiring': 'تعيين 100 موظف → إبداع غير مسبوق → قيادة الصناعة',
            'investment': 'استثمار مليون → عائد 10 ملايين → ثروة',
            'default': 'تنفيذ الخطة → نجاح ساحق → تحول نوعي'
        }
        return scenarios.get(proposal.get('type'), scenarios['default'])
    
    def _maximize_luck(self, proposal: Dict) -> List[str]:
        """تعظيم الحظ"""
        return [
            'التوقيت مثالي - المنافسون ضعفاء',
            'السوق جاهز للمنتج',
            'قصة نجاح مشابهة حدثت',
            'دعم إعلامي غير متوقع'
        ]
    
    def _expand_boundaries(self, proposal: Dict) -> Dict:
        """توسيع الحدود (10x thinking)"""
        original_budget = proposal.get('budget', 100000)
        return {
            'original': proposal,
            '10x_version': {
                'budget': original_budget * 10,
                'scale': 'global',
                'impact': 'industry-transforming',
                'timeline': '5 years'
            },
            'recommendation': 'فكر أكبر!'
        }
    
    async def generate_moonshot(self) -> Dict:
        """توليد فكرة " moonshot" مجنونة"""
        moonshots = [
            'بناء ERP يعمل بالأحلام',
            'ذكاء اصطناعي يتنبأ بالمستقبل 100 سنة',
            'شركة تتجاوز قيمتها تريليون دولار',
            'تحويل العراق لمركز تكنولوجي عالمي'
        ]
        return {
            'moonshot': moonshots[len(self.opportunities) % len(moonshots)],
            'probability': 0.01,
            'potential_impact': 'game-changing'
        }


class BalanceCouncil:
    """
    مجلس التوازن
    
    يجمع رأي الظل والنور ويصلح بينهما
    """
    
    def __init__(self):
        self.shadow = ShadowTeam()
        self.light = LightTeam()
        print("⚖️ Balance Council initialized")
    
    async def evaluate_proposal(self, proposal: Dict) -> Dict:
        """
        تقييم مقترح من الطرفين
        
        Returns:
            قرار متوازن
        """
        # جلب التحليلات
        shadow_report = await self.shadow.analyze_proposal(proposal)
        light_report = await self.light.analyze_proposal(proposal)
        
        # الموازنة
        risks = len(shadow_report['risks'])
        opportunities = len(light_report['opportunities'])
        
        if risks > opportunities * 2:
            decision = 'رفض'
            reasoning = 'المخاطر تفوق الفرص بكثير'
        elif opportunities > risks * 2:
            decision = 'موافقة قوية'
            reasoning = 'الفرص واضحة والمخاطر محدودة'
        else:
            decision = 'موافقة مشروطة'
            reasoning = 'تنفيذ مع احتياطات Shadow Team'
        
        return {
            'decision': decision,
            'reasoning': reasoning,
            'shadow_report': shadow_report,
            'light_report': light_report,
            'balance_score': (opportunities - risks) / max(opportunities + risks, 1),
            'final_recommendation': self._generate_recommendation(
                shadow_report, light_report, decision
            )
        }
    
    def _generate_recommendation(self, shadow: Dict, light: Dict, 
                                 decision: str) -> str:
        """توليد توصية نهائية"""
        if decision == 'موافقة قوية':
            return f"{light['best_case_scenario']} - مع مراقبة {shadow['risks'][0].scenario if shadow['risks'] else 'المخاطر'}"
        elif decision == 'رفض':
            return f"{shadow['risks'][0].scenario} - انتظر ظروف أفضل"
        else:
            return f"نفذ {light['proposal']} بحذر وانتبه لـ {shadow['risks'][0].early_warnings if shadow['risks'] else 'التنبيهات'}"


# Singletons
shadow_team = ShadowTeam()
light_team = LightTeam()
balance_council = BalanceCouncil()
