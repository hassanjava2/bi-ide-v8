"""
تحكم بالأجهزة — Device Control Module
تصوير شاشة + تحليل + تشغيل برامج + تفاعل مع أي تطبيق

🖥️ تحكم كامل بالأجهزة عن بعد وعن قرب
"""

import os
import subprocess
import platform
import asyncio
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime


class DeviceController:
    """
    تحكم بالأجهزة — يقدر:
    - يصوّر الشاشة ويحللها
    - يشغّل برامج ويتحكم بيها
    - يتفاعل مع أي تطبيق أو نظام تشغيل
    - يرسل أوامر SSH لأجهزة أخرى
    """
    
    def __init__(self):
        self.name = "تحكم الأجهزة"
        self.devices: Dict[str, Dict] = {}
        self.screenshots: List[Dict] = []
        self.commands_history: List[Dict] = []
    
    def register_device(self, name: str, host: str, user: str = "bi", 
                       ssh_key: Optional[str] = None) -> Dict:
        """تسجيل جهاز جديد"""
        device = {
            "name": name,
            "host": host,
            "user": user,
            "ssh_key": ssh_key,
            "registered_at": datetime.now().isoformat(),
            "status": "registered",
        }
        self.devices[name] = device
        return device
    
    async def capture_screen(self, device_name: str = "local") -> Dict[str, Any]:
        """تصوير الشاشة"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/tmp/screenshot_{timestamp}.png"
        
        try:
            if device_name == "local":
                system = platform.system()
                if system == "Darwin":  # macOS
                    subprocess.run(["screencapture", "-x", output_path], 
                                 capture_output=True, timeout=10)
                elif system == "Linux":
                    # Try multiple screenshot tools
                    for cmd in [
                        ["gnome-screenshot", "-f", output_path],
                        ["scrot", output_path],
                        ["import", "-window", "root", output_path],
                    ]:
                        try:
                            subprocess.run(cmd, capture_output=True, timeout=10)
                            if os.path.exists(output_path):
                                break
                        except FileNotFoundError:
                            continue
            else:
                # Remote device via SSH
                device = self.devices.get(device_name)
                if not device:
                    return {"error": f"Device {device_name} not registered"}
                
                ssh_cmd = f"ssh {device['user']}@{device['host']} 'DISPLAY=:0 import -window root /tmp/screen.png && cat /tmp/screen.png' > {output_path}"
                subprocess.run(ssh_cmd, shell=True, timeout=15)
            
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                result = {
                    "status": "captured",
                    "path": output_path,
                    "size_kb": round(size / 1024, 1),
                    "device": device_name,
                    "timestamp": timestamp,
                }
                self.screenshots.append(result)
                return result
            else:
                return {"status": "failed", "error": "Screenshot file not created"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def run_program(self, command: str, device_name: str = "local",
                         timeout: int = 30) -> Dict[str, Any]:
        """تشغيل برنامج"""
        entry = {
            "command": command,
            "device": device_name,
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            if device_name == "local":
                result = subprocess.run(
                    command, shell=True,
                    capture_output=True, text=True, timeout=timeout,
                )
            else:
                device = self.devices.get(device_name)
                if not device:
                    return {"error": f"Device {device_name} not registered"}
                
                ssh_prefix = f"ssh {device['user']}@{device['host']}"
                result = subprocess.run(
                    f'{ssh_prefix} "{command}"',
                    shell=True, capture_output=True, text=True, timeout=timeout,
                )
            
            entry["stdout"] = result.stdout[-2000:] if result.stdout else ""
            entry["stderr"] = result.stderr[-500:] if result.stderr else ""
            entry["returncode"] = result.returncode
            entry["status"] = "success" if result.returncode == 0 else "failed"
            
        except subprocess.TimeoutExpired:
            entry["status"] = "timeout"
            entry["error"] = f"Command timed out after {timeout}s"
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)
        
        self.commands_history.append(entry)
        return entry
    
    async def get_system_info(self, device_name: str = "local") -> Dict[str, Any]:
        """الحصول على معلومات النظام"""
        cmd = "echo '=== OS ===' && uname -a && echo '=== CPU ===' && nproc && echo '=== MEM ===' && free -h 2>/dev/null || vm_stat 2>/dev/null && echo '=== DISK ===' && df -h / && echo '=== GPU ===' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU'"
        return await self.run_program(cmd, device_name)
    
    def list_devices(self) -> List[Dict]:
        """عرض الأجهزة المسجّلة"""
        return list(self.devices.values())
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active": True,
            "registered_devices": len(self.devices),
            "screenshots_taken": len(self.screenshots),
            "commands_executed": len(self.commands_history),
        }


# Singleton with pre-registered devices
device_controller = DeviceController()

# Pre-register known devices
device_controller.register_device("rtx5090", "192.168.1.164", "bi")
