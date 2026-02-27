"""
Firewall Configuration - BI-IDE v8
تكوين جدار الحماية لـ Windows و Ubuntu
"""

import os
import sys
import subprocess
import platform
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from core.logging_config import logger


# المنافذ المطلوبة للتطبيق
PORTS_REQUIRED = [8000, 8080, 9090, 6379, 5432]

# وصف المنافذ
PORT_DESCRIPTIONS = {
    8000: "BI-IDE API Server (Windows)",
    8080: "RTX 4090 Inference Server (Ubuntu)",
    9090: "Prometheus Metrics",
    6379: "Redis Cache",
    5432: "PostgreSQL Database",
}


class FirewallStatus(Enum):
    """حالة جدار الحماية"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


@dataclass
class FirewallRule:
    """قاعدة جدار حماية"""
    port: int
    protocol: str = "tcp"
    action: str = "allow"
    direction: str = "in"
    description: str = ""
    
    def __post_init__(self):
        if not self.description:
            self.description = PORT_DESCRIPTIONS.get(self.port, f"Port {self.port}")


def setup_ufw_rules(
    ports: Optional[List[int]] = None,
    enable_ufw: bool = True,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    إعداد قواعد UFW على Ubuntu
    
    Args:
        ports: قائمة المنافذ (افتراضي: PORTS_REQUIRED)
        enable_ufw: تفعيل UFW بعد الإعداد
        dry_run: عرض الأوامر فقط بدون تنفيذ
        
    Returns:
        Dict: نتيجة الإعداد
    """
    if platform.system() != "Linux":
        return {
            "success": False,
            "error": "UFW is only available on Linux",
            "platform": platform.system(),
        }
    
    ports = ports or PORTS_REQUIRED
    results = {
        "success": True,
        "platform": "linux",
        "firewall": "ufw",
        "commands_executed": [],
        "rules_added": [],
        "errors": [],
    }
    
    def run_command(cmd: List[str], description: str) -> bool:
        """تشغيل أمر shell"""
        results["commands_executed"].append({
            "command": " ".join(cmd),
            "description": description,
        })
        
        if dry_run:
            logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return True
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"✅ {description}")
                return True
            else:
                error_msg = result.stderr or f"Exit code: {result.returncode}"
                logger.error(f"❌ {description}: {error_msg}")
                results["errors"].append(f"{description}: {error_msg}")
                return False
        except Exception as e:
            logger.error(f"❌ {description}: {e}")
            results["errors"].append(f"{description}: {e}")
            return False
    
    # Check if ufw is installed
    if not run_command(["which", "ufw"], "Check UFW installation"):
        results["success"] = False
        results["error"] = "UFW is not installed. Install with: sudo apt install ufw"
        return results
    
    # Reset UFW (optional - be careful in production)
    # run_command(["sudo", "ufw", "--force", "reset"], "Reset UFW")
    
    # Default policies
    run_command(["sudo", "ufw", "default", "deny", "incoming"], "Deny incoming by default")
    run_command(["sudo", "ufw", "default", "allow", "outgoing"], "Allow outgoing by default")
    
    # Allow SSH first (important!)
    run_command(["sudo", "ufw", "allow", "22/tcp"], "Allow SSH (port 22)")
    
    # Add rules for required ports
    for port in ports:
        rule = FirewallRule(port=port)
        cmd = ["sudo", "ufw", "allow", str(port), rule.protocol]
        desc = f"Allow {rule.description} ({port}/{rule.protocol})"
        
        if run_command(cmd, desc):
            results["rules_added"].append({
                "port": port,
                "protocol": rule.protocol,
                "description": rule.description,
            })
    
    # Allow specific IP ranges (optional)
    # run_command(["sudo", "ufw", "allow", "from", "192.168.68.0/24"], "Allow local network")
    
    # Enable UFW
    if enable_ufw:
        # Check status first
        if not dry_run:
            status_result = subprocess.run(
                ["sudo", "ufw", "status"],
                capture_output=True,
                text=True
            )
            if "Status: active" not in status_result.stdout:
                run_command(["sudo", "ufw", "--force", "enable"], "Enable UFW")
            else:
                logger.info("UFW is already active")
                run_command(["sudo", "ufw", "reload"], "Reload UFW rules")
    
    # Check final status
    if not dry_run:
        status_result = subprocess.run(
            ["sudo", "ufw", "status", "verbose"],
            capture_output=True,
            text=True
        )
        results["ufw_status"] = status_result.stdout
    
    results["success"] = len(results["errors"]) == 0
    return results


def setup_windows_firewall(
    ports: Optional[List[int]] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    إعداد Windows Firewall
    
    Args:
        ports: قائمة المنافذ (افتراضي: PORTS_REQUIRED)
        dry_run: عرض الأوامر فقط بدون تنفيذ
        
    Returns:
        Dict: نتيجة الإعداد
    """
    if platform.system() != "Windows":
        return {
            "success": False,
            "error": "Windows Firewall is only available on Windows",
            "platform": platform.system(),
        }
    
    ports = ports or PORTS_REQUIRED
    results = {
        "success": True,
        "platform": "windows",
        "firewall": "windows",
        "commands_executed": [],
        "rules_added": [],
        "errors": [],
    }
    
    def run_powershell(command: str, description: str) -> bool:
        """تشغيل أمر PowerShell"""
        results["commands_executed"].append({
            "command": command,
            "description": description,
        })
        
        if dry_run:
            logger.info(f"[DRY RUN] Would execute: {command}")
            return True
        
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info(f"✅ {description}")
                return True
            else:
                error_msg = result.stderr or f"Exit code: {result.returncode}"
                logger.error(f"❌ {description}: {error_msg}")
                results["errors"].append(f"{description}: {error_msg}")
                return False
        except Exception as e:
            logger.error(f"❌ {description}: {e}")
            results["errors"].append(f"{description}: {e}")
            return False
    
    # Add inbound rules for each port
    for port in ports:
        rule = FirewallRule(port=port)
        rule_name = f"BI-IDE-v8-{port}"
        
        # Remove existing rule if exists
        remove_cmd = f'Remove-NetFirewallRule -DisplayName "{rule_name}" -ErrorAction SilentlyContinue'
        run_powershell(remove_cmd, f"Remove existing rule for port {port}")
        
        # Add new rule
        add_cmd = (
            f'New-NetFirewallRule -DisplayName "{rule_name}" '
            f'-Direction Inbound -Protocol {rule.protocol.upper()} '
            f'-LocalPort {port} -Action Allow '
            f'-Profile Any '
            f'-Description "{rule.description}"'
        )
        
        if run_powershell(add_cmd, f"Add rule for port {port}"):
            results["rules_added"].append({
                "port": port,
                "protocol": rule.protocol,
                "name": rule_name,
                "description": rule.description,
            })
    
    # Check current firewall status
    if not dry_run:
        status_cmd = 'Get-NetFirewallProfile | Select-Object Name, Enabled | Format-Table -AutoSize'
        status_result = subprocess.run(
            ["powershell", "-Command", status_cmd],
            capture_output=True,
            text=True
        )
        results["firewall_status"] = status_result.stdout
    
    results["success"] = len(results["errors"]) == 0
    return results


def get_firewall_status() -> Dict[str, Any]:
    """الحصول على حالة جدار الحماية الحالية"""
    system = platform.system()
    
    if system == "Linux":
        try:
            result = subprocess.run(
                ["sudo", "ufw", "status", "numbered"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return {
                "platform": "linux",
                "firewall": "ufw",
                "status": "active" if "Status: active" in result.stdout else "inactive",
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
            }
        except Exception as e:
            return {
                "platform": "linux",
                "firewall": "ufw",
                "status": "unknown",
                "error": str(e),
            }
    
    elif system == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command", 'Get-NetFirewallProfile | Select-Object Name, Enabled'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                "platform": "windows",
                "firewall": "windows",
                "status": "configured",
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
            }
        except Exception as e:
            return {
                "platform": "windows",
                "firewall": "windows",
                "status": "unknown",
                "error": str(e),
            }
    
    return {
        "platform": system,
        "firewall": "unknown",
        "status": "unsupported",
    }


def verify_port_access(port: int, host: str = "localhost") -> bool:
    """التحقق من إمكانية الوصول لمنفذ معين"""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def verify_all_ports() -> Dict[str, Any]:
    """التحقق من جميع المنافذ المطلوبة"""
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "ports": {},
        "all_accessible": True,
    }
    
    for port in PORTS_REQUIRED:
        is_open = verify_port_access(port)
        results["ports"][port] = {
            "port": port,
            "accessible": is_open,
            "description": PORT_DESCRIPTIONS.get(port, "Unknown"),
        }
        if not is_open:
            results["all_accessible"] = False
    
    return results


if __name__ == "__main__":
    # Test firewall configuration
    print("=" * 60)
    print("BI-IDE v8 Firewall Configuration")
    print("=" * 60)
    
    # Show current status
    status = get_firewall_status()
    print(f"\nPlatform: {status['platform']}")
    print(f"Firewall: {status['firewall']}")
    print(f"Status: {status['status']}")
    
    # Verify ports
    print("\n--- Port Verification ---")
    port_results = verify_all_ports()
    for port_info in port_results["ports"].values():
        status_icon = "✅" if port_info["accessible"] else "❌"
        print(f"{status_icon} Port {port_info['port']}: {port_info['description']}")
    
    print("\n--- Configuration ---")
    if status["platform"] == "Linux":
        result = setup_ufw_rules(dry_run=True)
        print(f"UFW Rules (dry-run): {len(result['rules_added'])} rules")
    elif status["platform"] == "Windows":
        result = setup_windows_firewall(dry_run=True)
        print(f"Windows Firewall Rules (dry-run): {len(result['rules_added'])} rules")
