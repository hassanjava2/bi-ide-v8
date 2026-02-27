"""
الطبقة الثامنة: الرئيس الأعلى (أنت)
President Interface - Human Control Layer

الصلاحيات:
- دخول 24/7 للمجلس
- أمر فوري ينفذ بدون نقاش
- فيتو على التدمير الذاتي
- مشاهدة حية للمجلس
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable
import asyncio


class CommandType(Enum):
    """أنواع الأوامر"""
    EXECUTE = "execute"           # نفذ فوراً
    STOP = "stop"                 # قف فوراً
    VETO = "veto"                 # فيتو
    WAIT = "wait"                 # انتظر
    DESTROY = "destroy"           # تدمير (يتطلب تأكيد)


class AlertLevel(Enum):
    """مستويات الإنذار"""
    GREEN = "green"      # معلومة فقط
    YELLOW = "yellow"    # قرار عادي (5 دقائق)
    ORANGE = "orange"    # قرار مهم (1 ساعة)
    RED = "red"          # قرار حيوي (15 دقيقة)
    BLACK = "black"      # تدمير/خطر وجودي (يتوقف لحين قرارك)


@dataclass
class PresidentialCommand:
    """أمر رئاسي"""
    command_type: CommandType
    target_layer: int
    description: str
    timestamp: datetime
    requires_confirmation: bool = False


@dataclass
class CouncilMeeting:
    """اجتماع المجلس"""
    meeting_id: str
    start_time: datetime
    topic: str
    participating_sages: list
    status: str = "ongoing"


class PresidentInterface:
    """
    واجهة الرئيس الأعلى
    
    يمكنك الدخول متى شئت، إعطاء الأوامر، والمشاهدة المباشرة
    """
    
    def __init__(self):
        self.is_present: bool = False
        self.current_meeting: Optional[CouncilMeeting] = None
        self.command_history: list = []
        self.alert_handlers: dict = {}
        self.veto_power_active: bool = True
        
    async def enter_council(self) -> CouncilMeeting:
        """
        دخول الرئيس للمجلس
        
        Returns:
            CouncilMeeting: معلومات الاجتماع الحالي
        """
        self.is_present = True
        print("👑 الرئيس دخل المجلس")
        
        # إشعار للحكماء
        await self._notify_sages("الرئيس حاضر")
        
        return self.current_meeting
    
    async def issue_command(self, command: PresidentialCommand) -> bool:
        """
        إصدار أمر رئاسي
        
        Args:
            command: الأمر المراد تنفيذه
            
        Returns:
            bool: هل تم التنفيذ
        """
        # التحقق من الأوامر الخطرة
        if command.command_type == CommandType.DESTROY:
            return await self._handle_destruction_command(command)
        
        # الأمر العادي ينفذ فوراً
        print(f"👑 أمر رئاسي: {command.command_type.value} - {command.description}")
        self.command_history.append(command)
        
        # إرسال للطبقة المستهدفة
        await self._dispatch_to_layer(command)
        return True
    
    async def _handle_destruction_command(self, command: PresidentialCommand) -> bool:
        """
        التعامل مع أمر التدمير - يتطلب تأكيد شخصي
        """
        print("🚨 أمر تدمير الذاتي Detected!")
        print("🚨 هذا الأمر يتطلب تأكيدك الشخصي")
        print("🚨 هل أنت متأكد؟ (yes/no)")
        
        # في الإنتاج، هذا ينتظر إدخالك الفعلي
        confirmation = await self._wait_for_confirmation(timeout=300)  # 5 دقائق
        
        if confirmation:
            print("👑 التدمير مؤكد من الرئيس")
            return True
        else:
            print("👑 التدمير مرفوض")
            return False
    
    async def watch_live(self) -> None:
        """
        مشاهدة حية لمجريات المجلس
        """
        while self.is_present:
            # جلب آخر المستجدات
            updates = await self._get_council_updates()
            for update in updates:
                print(f"📺 {update}")
            await asyncio.sleep(1)  # تحديث كل ثانية
    
    async def veto(self, decision_id: str) -> bool:
        """
        فيتو على قرار
        
        Args:
            decision_id: معرف القرار
            
        Returns:
            bool: هل تم الفيتو
        """
        print(f"👑 فيتو على القرار: {decision_id}")
        await self._cancel_decision(decision_id)
        return True
    
    def receive_alert(self, alert_level: AlertLevel, message: str):
        """
        استلام إنذار من الطبقات السفلى
        
        Args:
            alert_level: مستوى الإنذار
            message: نص الإنذار
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if alert_level == AlertLevel.BLACK:
            print(f"🚨 [{timestamp}] إنذار أسود: {message}")
            print(f"🚨 النظام متوقف في انتظار قرارك")
        elif alert_level == AlertLevel.RED:
            print(f"🔴 [{timestamp}] إنذار أحمر: {message}")
        elif alert_level == AlertLevel.ORANGE:
            print(f"🟠 [{timestamp}] إنذار برتقالي: {message}")
        else:
            print(f"📢 [{timestamp}] {alert_level.value}: {message}")
    
    async def _notify_sages(self, message: str):
        """إشعار الحكماء"""
        # يرسل إشعار للطبقة السادسة
        pass
    
    async def _dispatch_to_layer(self, command: PresidentialCommand):
        """إرسال الأمر للطبقة المستهدفة"""
        # يرسل للطبقة المناسبة
        pass
    
    async def _wait_for_confirmation(self, timeout: int) -> bool:
        """انتظار تأكيد المستخدم"""
        # في الإنتاج، ينتظر إدخال حقيقي
        # الآن نحاكي بالموافقة
        return True
    
    async def _get_council_updates(self) -> list:
        """جلب آخر المستجدات"""
        return ["الحكيم 1: نقترح...", "الحكيم 3: أوافق..."]
    
    async def _cancel_decision(self, decision_id: str):
        """إلغاء قرار"""
        print(f"✅ تم إلغاء القرار {decision_id}")


# Singleton instance
president = PresidentInterface()
