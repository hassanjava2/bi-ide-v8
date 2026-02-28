"""
Firewall Manager - إدارة الجدار الناري
Configure and manage firewall rules
"""

import asyncio
import ipaddress
import logging
import platform
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable, Union

import aiofiles

logger = logging.getLogger(__name__)


class FirewallAction(str, Enum):
    """إجراءات الجدار الناري - Firewall actions"""
    ALLOW = "allow"
    DENY = "deny"
    DROP = "drop"
    REJECT = "reject"
    LOG = "log"


class FirewallProtocol(str, Enum):
    """بروتوكولات الجدار الناري - Firewall protocols"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ALL = "all"


@dataclass
class FirewallRule:
    """قاعدة جدار ناري - Firewall rule"""
    id: str
    name: str
    action: FirewallAction
    protocol: FirewallProtocol
    port: Optional[int]
    source_ip: Optional[str]
    destination_ip: Optional[str]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    rate_limit: Optional[int] = None  # requests per minute
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل إلى قاموس - Convert to dict"""
        return {
            'id': self.id,
            'name': self.name,
            'action': self.action.value,
            'protocol': self.protocol.value,
            'port': self.port,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'enabled': self.enabled,
            'created_at': self.created_at.isoformat(),
            'description': self.description,
            'rate_limit': self.rate_limit
        }


class FirewallManager:
    """
    مدير الجدار الناري
    Firewall Manager
    
    يدير قواعد الجدار الناري لـ BI-IDE
    Manages firewall rules for BI-IDE
    
    يدعم:
    - iptables (Linux)
    - Windows Firewall (Windows)
    - pf (macOS - محدود)
    """
    
    # منافذ BI-IDE الافتراضية - Default BI-IDE ports
    DEFAULT_PORTS = [8000, 8001, 6379, 5432, 8080]
    
    # IPs مشبوهة افتراضية - Default suspicious IPs
    SUSPICIOUS_IPS: List[str] = []
    
    def __init__(
        self,
        default_policy: FirewallAction = FirewallAction.DENY,
        config_file: Optional[str] = None,
        enable_rate_limiting: bool = True
    ):
        """
        تهيئة مدير الجدار الناري
        Initialize firewall manager
        
        Args:
            default_policy: السياسة الافتراضية
            config_file: ملف التكوين
            enable_rate_limiting: تفعيل تقييد المعدل
        """
        self.default_policy = default_policy
        self.enable_rate_limiting = enable_rate_limiting
        self._rules: Dict[str, FirewallRule] = {}
        self._blocked_ips: Set[str] = set()
        self._rate_limits: Dict[str, Dict[str, Any]] = {}
        self._callbacks: List[Callable[[str, FirewallRule], None]] = []
        
        # تحديد النظام
        self.system = platform.system().lower()
        self._is_admin = self._check_admin_privileges()
        
        # تحميل التكوين
        if config_file:
            self.config_file = Path(config_file)
        else:
            self.config_file = Path(__file__).parent / 'firewall_config.json'
        
        self._load_config()
        
        # إضافة قواعد افتراضية
        self._add_default_rules()
        
        logger.info(f"FirewallManager initialized ({self.system}, admin: {self._is_admin})")
    
    def _check_admin_privileges(self) -> bool:
        """التحقق من صلاحيات المسؤول - Check admin privileges"""
        try:
            if self.system == 'windows':
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin()
            else:
                return os.geteuid() == 0
        except:
            return False
    
    def _add_default_rules(self) -> None:
        """إضافة قواعد افتراضية - Add default rules"""
        # السماح بمنافذ BI-IDE
        for port in self.DEFAULT_PORTS:
            self.add_rule(
                name=f"bi_ide_port_{port}",
                action=FirewallAction.ALLOW,
                protocol=FirewallProtocol.TCP,
                port=port,
                description=f"Allow BI-IDE port {port}"
            )
        
        # السماح بالاتصال المحلي
        self.add_rule(
            name="localhost",
            action=FirewallAction.ALLOW,
            protocol=FirewallProtocol.ALL,
            source_ip="127.0.0.1",
            description="Allow localhost connections"
        )
        
        # حظر IPs مشبوهة
        for ip in self.SUSPICIOUS_IPS:
            self.block_ip(ip)
    
    def _load_config(self) -> None:
        """تحميل التكوين - Load configuration"""
        try:
            if self.config_file.exists():
                import json
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # تحميل القواعد
                for rule_data in data.get('rules', []):
                    rule = FirewallRule(
                        id=rule_data['id'],
                        name=rule_data['name'],
                        action=FirewallAction(rule_data['action']),
                        protocol=FirewallProtocol(rule_data['protocol']),
                        port=rule_data.get('port'),
                        source_ip=rule_data.get('source_ip'),
                        destination_ip=rule_data.get('destination_ip'),
                        enabled=rule_data.get('enabled', True),
                        description=rule_data.get('description', ''),
                        rate_limit=rule_data.get('rate_limit')
                    )
                    self._rules[rule.id] = rule
                
                # تحميل IPs المحظورة
                self._blocked_ips = set(data.get('blocked_ips', []))
                
                logger.info(f"Loaded {len(self._rules)} rules and {len(self._blocked_ips)} blocked IPs")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    async def _save_config(self) -> None:
        """حفظ التكوين - Save configuration"""
        try:
            data = {
                'rules': [rule.to_dict() for rule in self._rules.values()],
                'blocked_ips': list(self._blocked_ips),
                'default_policy': self.default_policy.value,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            async with aiofiles.open(self.config_file, 'w') as f:
                await f.write(__import__('json').dumps(data, indent=2))
            
            logger.debug("Config saved")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def add_rule(
        self,
        name: str,
        action: FirewallAction,
        protocol: FirewallProtocol = FirewallProtocol.TCP,
        port: Optional[int] = None,
        source_ip: Optional[str] = None,
        destination_ip: Optional[str] = None,
        description: str = "",
        rate_limit: Optional[int] = None,
        apply: bool = True
    ) -> FirewallRule:
        """
        إضافة قاعدة جديدة
        Add new rule
        
        Args:
            name: اسم القاعدة
            action: الإجراء
            protocol: البروتوكول
            port: المنفذ
            source_ip: IP المصدر
            destination_ip: IP الوجهة
            description: الوصف
            rate_limit: تقييد المعدل (طلب/دقيقة)
            apply: تطبيق القاعدة فوراً
            
        Returns:
            FirewallRule: القاعدة المنشأة
        """
        import uuid
        
        rule = FirewallRule(
            id=str(uuid.uuid4())[:8],
            name=name,
            action=action,
            protocol=protocol,
            port=port,
            source_ip=source_ip,
            destination_ip=destination_ip,
            description=description,
            rate_limit=rate_limit
        )
        
        self._rules[rule.id] = rule
        
        if apply and self._is_admin:
            asyncio.create_task(self._apply_rule(rule))
        
        # استدعاء الدوال المسجلة
        for callback in self._callbacks:
            try:
                callback('added', rule)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
        
        logger.info(f"Added rule: {name} ({action.value})")
        return rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        إزالة قاعدة
        Remove rule
        
        Args:
            rule_id: معرف القاعدة
            
        Returns:
            bool: نجاح العملية
        """
        if rule_id not in self._rules:
            return False
        
        rule = self._rules[rule_id]
        
        if self._is_admin:
            asyncio.create_task(self._remove_rule_from_system(rule))
        
        del self._rules[rule_id]
        
        # استدعاء الدوال المسجلة
        for callback in self._callbacks:
            try:
                callback('removed', rule)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
        
        logger.info(f"Removed rule: {rule.name}")
        return True
    
    def list_rules(
        self,
        enabled_only: bool = False,
        action: Optional[FirewallAction] = None
    ) -> List[FirewallRule]:
        """
        قائمة القواعد
        List rules
        
        Args:
            enabled_only: القواعد المفعلة فقط
            action: تصفية حسب الإجراء
            
        Returns:
            List[FirewallRule]: قائمة القواعد
        """
        rules = list(self._rules.values())
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        if action:
            rules = [r for r in rules if r.action == action]
        
        return rules
    
    def get_rule(self, rule_id: str) -> Optional[FirewallRule]:
        """
        الحصول على قاعدة محددة
        Get specific rule
        
        Args:
            rule_id: معرف القاعدة
            
        Returns:
            Optional[FirewallRule]: القاعدة أو None
        """
        return self._rules.get(rule_id)
    
    def enable_rule(self, rule_id: str) -> bool:
        """
        تفعيل قاعدة
        Enable rule
        
        Args:
            rule_id: معرف القاعدة
            
        Returns:
            bool: نجاح العملية
        """
        if rule_id not in self._rules:
            return False
        
        rule = self._rules[rule_id]
        rule.enabled = True
        
        if self._is_admin:
            asyncio.create_task(self._apply_rule(rule))
        
        logger.info(f"Enabled rule: {rule.name}")
        return True
    
    def disable_rule(self, rule_id: str) -> bool:
        """
        تعطيل قاعدة
        Disable rule
        
        Args:
            rule_id: معرف القاعدة
            
        Returns:
            bool: نجاح العملية
        """
        if rule_id not in self._rules:
            return False
        
        rule = self._rules[rule_id]
        rule.enabled = False
        
        if self._is_admin:
            asyncio.create_task(self._remove_rule_from_system(rule))
        
        logger.info(f"Disabled rule: {rule.name}")
        return True
    
    def enable(self) -> bool:
        """
        تفعيل الجدار الناري
        Enable firewall
        
        Returns:
            bool: نجاح العملية
        """
        if not self._is_admin:
            logger.warning("Admin privileges required to enable firewall")
            return False
        
        asyncio.create_task(self._enable_firewall())
        logger.info("Firewall enabled")
        return True
    
    def disable(self) -> bool:
        """
        تعطيل الجدار الناري
        Disable firewall
        
        Returns:
            bool: نجاح العملية
        """
        if not self._is_admin:
            logger.warning("Admin privileges required to disable firewall")
            return False
        
        asyncio.create_task(self._disable_firewall())
        logger.info("Firewall disabled")
        return True
    
    def block_ip(self, ip: str, reason: str = "") -> bool:
        """
        حظر IP
        Block IP
        
        Args:
            ip: عنوان IP
            reason: السبب
            
        Returns:
            bool: نجاح العملية
        """
        try:
            # التحقق من صحة IP
            ipaddress.ip_address(ip)
        except ValueError:
            logger.error(f"Invalid IP address: {ip}")
            return False
        
        self._blocked_ips.add(ip)
        
        if self._is_admin:
            asyncio.create_task(self._apply_ip_block(ip))
        
        logger.info(f"Blocked IP: {ip} ({reason})")
        return True
    
    def unblock_ip(self, ip: str) -> bool:
        """
        إلغاء حظر IP
        Unblock IP
        
        Args:
            ip: عنوان IP
            
        Returns:
            bool: نجاح العملية
        """
        if ip not in self._blocked_ips:
            return False
        
        self._blocked_ips.discard(ip)
        
        if self._is_admin:
            asyncio.create_task(self._remove_ip_block(ip))
        
        logger.info(f"Unblocked IP: {ip}")
        return True
    
    def is_ip_blocked(self, ip: str) -> bool:
        """
        التحقق مما إذا كان IP محظور
        Check if IP is blocked
        
        Args:
            ip: عنوان IP
            
        Returns:
            bool: محظور أم لا
        """
        return ip in self._blocked_ips
    
    def check_rate_limit(self, ip: str) -> bool:
        """
        التحقق من تقييد المعدل
        Check rate limit
        
        Args:
            ip: عنوان IP
            
        Returns:
            bool: مسموح أم لا
        """
        if not self.enable_rate_limiting:
            return True
        
        now = datetime.utcnow()
        
        if ip not in self._rate_limits:
            self._rate_limits[ip] = {
                'count': 1,
                'window_start': now
            }
            return True
        
        limit_data = self._rate_limits[ip]
        window_start = limit_data['window_start']
        
        # إعادة تعيين النافذة بعد دقيقة
        if (now - window_start).total_seconds() > 60:
            limit_data['count'] = 1
            limit_data['window_start'] = now
            return True
        
        # الحصول على الحد الأقصى للمعدل
        max_rate = 100  # افتراضي
        for rule in self._rules.values():
            if rule.rate_limit and rule.enabled:
                max_rate = min(max_rate, rule.rate_limit)
        
        if limit_data['count'] >= max_rate:
            logger.warning(f"Rate limit exceeded for {ip}")
            return False
        
        limit_data['count'] += 1
        return True
    
    async def _apply_rule(self, rule: FirewallRule) -> bool:
        """تطبيق قاعدة على النظام - Apply rule to system"""
        if self.system == 'linux':
            return await self._apply_iptables_rule(rule)
        elif self.system == 'windows':
            return await self._apply_windows_rule(rule)
        elif self.system == 'darwin':
            return await self._apply_pf_rule(rule)
        else:
            logger.warning(f"Unsupported system: {self.system}")
            return False
    
    async def _apply_iptables_rule(self, rule: FirewallRule) -> bool:
        """تطبيق قاعدة iptables - Apply iptables rule"""
        try:
            cmd = ['iptables', '-A', 'INPUT']
            
            # البروتوكول
            if rule.protocol != FirewallProtocol.ALL:
                cmd.extend(['-p', rule.protocol.value])
            
            # المنفذ
            if rule.port:
                cmd.extend(['--dport', str(rule.port)])
            
            # IP المصدر
            if rule.source_ip:
                cmd.extend(['-s', rule.source_ip])
            
            # IP الوجهة
            if rule.destination_ip:
                cmd.extend(['-d', rule.destination_ip])
            
            # الإجراء
            action_map = {
                FirewallAction.ALLOW: 'ACCEPT',
                FirewallAction.DENY: 'DROP',
                FirewallAction.REJECT: 'REJECT',
                FirewallAction.LOG: 'LOG'
            }
            cmd.extend(['-j', action_map.get(rule.action, 'DROP')])
            
            # تنفيذ الأمر
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            return process.returncode == 0
            
        except Exception as e:
            logger.error(f"Error applying iptables rule: {e}")
            return False
    
    async def _apply_windows_rule(self, rule: FirewallRule) -> bool:
        """تطبيق قاعدة Windows Firewall - Apply Windows Firewall rule"""
        try:
            cmd = [
                'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                f'name={rule.name}',
                f'dir=in',
                f'action={rule.action.value}'
            ]
            
            if rule.protocol != FirewallProtocol.ALL:
                cmd.append(f'protocol={rule.protocol.value}')
            
            if rule.port:
                cmd.append(f'localport={rule.port}')
            
            if rule.source_ip:
                cmd.append(f'remoteip={rule.source_ip}')
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            return process.returncode == 0
            
        except Exception as e:
            logger.error(f"Error applying Windows firewall rule: {e}")
            return False
    
    async def _apply_pf_rule(self, rule: FirewallRule) -> bool:
        """تطبيق قاعدة PF (macOS) - Apply PF rule (macOS)"""
        logger.warning("PF rules not fully implemented")
        return False
    
    async def _remove_rule_from_system(self, rule: FirewallRule) -> bool:
        """إزالة قاعدة من النظام - Remove rule from system"""
        if self.system == 'linux':
            return await self._remove_iptables_rule(rule)
        elif self.system == 'windows':
            return await self._remove_windows_rule(rule)
        return False
    
    async def _remove_iptables_rule(self, rule: FirewallRule) -> bool:
        """إزالة قاعدة iptables - Remove iptables rule"""
        try:
            # استخدام -D لحذف القاعدة
            cmd = ['iptables', '-D', 'INPUT']
            
            if rule.protocol != FirewallProtocol.ALL:
                cmd.extend(['-p', rule.protocol.value])
            
            if rule.port:
                cmd.extend(['--dport', str(rule.port)])
            
            if rule.source_ip:
                cmd.extend(['-s', rule.source_ip])
            
            action_map = {
                FirewallAction.ALLOW: 'ACCEPT',
                FirewallAction.DENY: 'DROP',
                FirewallAction.REJECT: 'REJECT'
            }
            cmd.extend(['-j', action_map.get(rule.action, 'DROP')])
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            return process.returncode == 0
            
        except Exception as e:
            logger.error(f"Error removing iptables rule: {e}")
            return False
    
    async def _remove_windows_rule(self, rule: FirewallRule) -> bool:
        """إزالة قاعدة Windows - Remove Windows rule"""
        try:
            cmd = [
                'netsh', 'advfirewall', 'firewall', 'delete', 'rule',
                f'name={rule.name}'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            return process.returncode == 0
            
        except Exception as e:
            logger.error(f"Error removing Windows firewall rule: {e}")
            return False
    
    async def _apply_ip_block(self, ip: str) -> bool:
        """تطبيق حظر IP - Apply IP block"""
        if self.system == 'linux':
            cmd = ['iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP']
        elif self.system == 'windows':
            cmd = ['netsh', 'advfirewall', 'firewall', 'add', 'rule',
                   f'name=Block_{ip}', 'dir=in', 'action=block', f'remoteip={ip}']
        else:
            return False
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception as e:
            logger.error(f"Error blocking IP: {e}")
            return False
    
    async def _remove_ip_block(self, ip: str) -> bool:
        """إزالة حظر IP - Remove IP block"""
        if self.system == 'linux':
            cmd = ['iptables', '-D', 'INPUT', '-s', ip, '-j', 'DROP']
        elif self.system == 'windows':
            cmd = ['netsh', 'advfirewall', 'firewall', 'delete', 'rule',
                   f'name=Block_{ip}']
        else:
            return False
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception as e:
            logger.error(f"Error unblocking IP: {e}")
            return False
    
    async def _enable_firewall(self) -> bool:
        """تفعيل الجدار الناري - Enable firewall"""
        if self.system == 'linux':
            # تطبيق جميع القواعد
            for rule in self._rules.values():
                if rule.enabled:
                    await self._apply_rule(rule)
            return True
        elif self.system == 'windows':
            cmd = ['netsh', 'advfirewall', 'set', 'allprofiles', 'state', 'on']
            try:
                process = await asyncio.create_subprocess_exec(*cmd)
                await process.communicate()
                return process.returncode == 0
            except Exception as e:
                logger.error(f"Error enabling firewall: {e}")
                return False
        return False
    
    async def _disable_firewall(self) -> bool:
        """تعطيل الجدار الناري - Disable firewall"""
        if self.system == 'linux':
            cmd = ['iptables', '-F']  # مسح جميع القواعد
        elif self.system == 'windows':
            cmd = ['netsh', 'advfirewall', 'set', 'allprofiles', 'state', 'off']
        else:
            return False
        
        try:
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.communicate()
            return process.returncode == 0
        except Exception as e:
            logger.error(f"Error disabling firewall: {e}")
            return False
    
    def add_callback(self, callback: Callable[[str, FirewallRule], None]) -> None:
        """إضافة دالة استدعاء - Add callback"""
        self._callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """
        الحصول على حالة الجدار الناري
        Get firewall status
        
        Returns:
            Dict[str, Any]: حالة الجدار الناري
        """
        return {
            'system': self.system,
            'is_admin': self._is_admin,
            'default_policy': self.default_policy.value,
            'total_rules': len(self._rules),
            'enabled_rules': sum(1 for r in self._rules.values() if r.enabled),
            'blocked_ips': len(self._blocked_ips),
            'rate_limiting_enabled': self.enable_rate_limiting
        }


# استيراد os لاستخدامه في _check_admin_privileges
import os


async def main():
    """الدالة الرئيسية - Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # إنشاء مدير الجدار الناري
    fw = FirewallManager()
    
    # طباعة الحالة
    print("\nFirewall Status:")
    print(__import__('json').dumps(fw.get_status(), indent=2))
    
    # طباعة القواعد
    print("\nFirewall Rules:")
    for rule in fw.list_rules():
        status = "✓" if rule.enabled else "✗"
        print(f"  {status} {rule.name}: {rule.action.value} {rule.protocol.value}", end="")
        if rule.port:
            print(f" port {rule.port}", end="")
        print()
    
    # حظر IP تجريبي
    fw.block_ip("192.168.1.100", "Test block")
    print(f"\nBlocked IPs: {fw._blocked_ips}")


if __name__ == "__main__":
    asyncio.run(main())
