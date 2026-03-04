"""
عميل إرسال الحالة — Stats Reporter Agent
يشتغل على كل جهاز ← يرسل heartbeat + إحصائيات للسيرفر كل 10 ثواني

🌐 إنترنت: يرسل لـ VPS (bi-iq.com)
🏠 بدون إنترنت: الأجهزة تستخدم LAN مباشرة

يتم تشغيله كـ systemd service أو Task Scheduler
"""

import os
import time
import json
import socket
import platform
import subprocess
import urllib.request
import urllib.error
from datetime import datetime

# ──── إعدادات ────
VPS_URL = "https://bi-iq.com/api/v1/machines/heartbeat"
REPORT_INTERVAL = 10  # seconds
MACHINE_ID = socket.gethostname()


def get_gpu_stats() -> dict:
    """إحصائيات GPU عبر nvidia-smi"""
    try:
        r = subprocess.run(
            "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,name "
            "--format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True, timeout=5
        )
        parts = [p.strip() for p in r.stdout.strip().split(",")]
        if len(parts) >= 4:
            return {
                "gpu_temp_c": int(parts[0]) if parts[0].isdigit() else 0,
                "gpu_util": int(parts[1]) if parts[1].isdigit() else 0,
                "gpu_mem_used_mb": int(parts[2]) if parts[2].isdigit() else 0,
                "gpu_mem_total_mb": int(parts[3]) if parts[3].isdigit() else 0,
                "gpu_name": parts[4] if len(parts) >= 5 else "Unknown",
            }
    except Exception:
        pass
    return {}


def get_system_stats() -> dict:
    """إحصائيات النظام — بدون psutil"""
    stats = {"machine_id": MACHINE_ID, "os": platform.system().lower()}

    if platform.system() == "Linux":
        # CPU
        try:
            r = subprocess.run(
                "top -bn1 | grep 'Cpu' | awk '{print 100-$8}'",
                shell=True, capture_output=True, text=True, timeout=5
            )
            stats["cpu_percent"] = round(float(r.stdout.strip()), 1)
        except Exception:
            stats["cpu_percent"] = 0

        # CPU Temp
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                stats["cpu_temp_c"] = round(int(f.read().strip()) / 1000, 1)
        except Exception:
            pass

        # RAM
        try:
            r = subprocess.run("free -b | awk '/Mem/{print $3\"/\"$2}'",
                               shell=True, capture_output=True, text=True, timeout=5)
            parts = r.stdout.strip().split("/")
            if len(parts) == 2:
                u, t = float(parts[0]), float(parts[1])
                stats["ram_used_gb"] = round(u / 1e9, 1)
                stats["ram_total_gb"] = round(t / 1e9, 1)
                stats["ram_percent"] = round(u / t * 100, 1) if t else 0
        except Exception:
            pass

        # Disk
        try:
            r = subprocess.run("df -B1 / | awk 'NR==2{print $3\"/\"$2}'",
                               shell=True, capture_output=True, text=True, timeout=5)
            parts = r.stdout.strip().split("/")
            if len(parts) == 2:
                u, t = float(parts[0]), float(parts[1])
                stats["disk_used_gb"] = round(u / 1e9, 1)
                stats["disk_total_gb"] = round(t / 1e9, 1)
                stats["disk_percent"] = round(u / t * 100, 1) if t else 0
        except Exception:
            pass

        # Training
        try:
            r = subprocess.run("systemctl is-active bi-training-daemon",
                               shell=True, capture_output=True, text=True, timeout=3)
            stats["training_active"] = r.stdout.strip() == "active"
        except Exception:
            stats["training_active"] = False

        # Load
        try:
            with open("/proc/loadavg") as f:
                stats["load_1m"] = float(f.read().split()[0])
        except Exception:
            pass

    elif platform.system() == "Windows":
        # CPU
        try:
            r = subprocess.run(
                'powershell -Command "(Get-CimInstance Win32_Processor).LoadPercentage"',
                shell=True, capture_output=True, text=True, timeout=10
            )
            stats["cpu_percent"] = float(r.stdout.strip())
        except Exception:
            stats["cpu_percent"] = 0

        # RAM
        try:
            r = subprocess.run(
                'powershell -Command "$o=Get-CimInstance Win32_OperatingSystem;'
                '$t=$o.TotalVisibleMemorySize*1024;$f=$o.FreePhysicalMemory*1024;'
                'Write-Output \\"$($t-$f)/$t\\""',
                shell=True, capture_output=True, text=True, timeout=10
            )
            parts = r.stdout.strip().split("/")
            if len(parts) == 2:
                u, t = float(parts[0]), float(parts[1])
                stats["ram_used_gb"] = round(u / 1e9, 1)
                stats["ram_total_gb"] = round(t / 1e9, 1)
                stats["ram_percent"] = round(u / t * 100, 1) if t else 0
        except Exception:
            pass

        # Disk
        try:
            r = subprocess.run(
                'powershell -Command "$d=Get-CimInstance Win32_LogicalDisk -Filter \'DriveType=3\' | Select -First 1;'
                'Write-Output \\"$($d.Size-$d.FreeSpace)/$($d.Size)\\""',
                shell=True, capture_output=True, text=True, timeout=10
            )
            parts = r.stdout.strip().split("/")
            if len(parts) == 2:
                u, t = float(parts[0]), float(parts[1])
                stats["disk_used_gb"] = round(u / 1e9, 1)
                stats["disk_total_gb"] = round(t / 1e9, 1)
                stats["disk_percent"] = round(u / t * 100, 1) if t else 0
        except Exception:
            pass

    # GPU
    gpu = get_gpu_stats()
    stats.update(gpu)

    # Network info
    stats["local_ip"] = get_local_ip()
    stats["timestamp"] = datetime.now().isoformat()

    return stats


def get_local_ip() -> str:
    """الحصول على IP المحلي"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def send_heartbeat(stats: dict) -> bool:
    """إرسال heartbeat للسيرفر"""
    try:
        data = json.dumps(stats).encode("utf-8")
        req = urllib.request.Request(
            VPS_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status == 200
    except Exception:
        return False


def main():
    """الحلقة الرئيسية — يرسل heartbeat كل 10 ثواني"""
    print(f"📡 Stats Reporter started — {MACHINE_ID}")
    print(f"   VPS: {VPS_URL}")
    print(f"   Interval: {REPORT_INTERVAL}s")

    consecutive_fails = 0

    while True:
        try:
            stats = get_system_stats()
            success = send_heartbeat(stats)

            if success:
                consecutive_fails = 0
                print(f"✅ [{stats['timestamp'][:19]}] Sent: CPU={stats.get('cpu_percent',0)}% "
                      f"GPU={stats.get('gpu_temp_c','—')}°C RAM={stats.get('ram_used_gb',0)}GB")
            else:
                consecutive_fails += 1
                print(f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] VPS unreachable "
                      f"(fail #{consecutive_fails})")

        except Exception as e:
            consecutive_fails += 1
            print(f"❌ Error: {e}")

        time.sleep(REPORT_INTERVAL)


if __name__ == "__main__":
    main()
