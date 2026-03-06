#!/usr/bin/env python3
"""
mobile_scout.py — كشافة الموبايل 📱

تتصل بجهاز أندرويد عبر ADB وتستخرج بيانات تدريب من:
  1. بنية النظام (/system)
  2. التطبيقات المنصبة
  3. إعدادات النظام
  4. لوغات النظام
  5. خدمات النظام (dumpsys)

الاستخدام:
  python3 mobile_scout.py                    # كشف تلقائي
  python3 mobile_scout.py --device 192.168.1.5:5555  # WiFi ADB
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CAPSULES_DIR = PROJECT_ROOT / "brain" / "capsules"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [mobile] %(message)s",
)
logger = logging.getLogger("mobile")


def run_adb(cmd: str, device: str = None, timeout: int = 30) -> str:
    """تنفيذ أمر ADB وإرجاع النتيجة"""
    full_cmd = ["adb"]
    if device:
        full_cmd += ["-s", device]
    full_cmd += cmd.split()
    try:
        result = subprocess.run(
            full_cmd, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="ignore"
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"ADB error: {e}")
        return ""


def check_adb_connection(device: str = None) -> bool:
    """فحص اتصال ADB"""
    output = run_adb("devices")
    if not output:
        logger.error("❌ ADB not found. Install: brew install android-platform-tools")
        return False
    lines = output.strip().split("\n")
    connected = [l for l in lines[1:] if "device" in l and "offline" not in l]
    if not connected:
        logger.error("❌ No Android device connected. Connect via USB or WiFi ADB.")
        return False
    logger.info(f"📱 Found {len(connected)} device(s)")
    return True


def get_device_info(device: str = None) -> dict:
    """معلومات الجهاز"""
    info = {
        "model": run_adb("shell getprop ro.product.model", device),
        "android_version": run_adb("shell getprop ro.build.version.release", device),
        "sdk": run_adb("shell getprop ro.build.version.sdk", device),
        "brand": run_adb("shell getprop ro.product.brand", device),
        "cpu": run_adb("shell getprop ro.hardware", device),
        "kernel": run_adb("shell uname -r", device),
    }
    logger.info(f"📱 {info['brand']} {info['model']} — Android {info['android_version']}")
    return info


def scan_system_structure(device: str = None) -> list:
    """فحص بنية نظام أندرويد"""
    samples = []

    # مجلدات النظام الرئيسية
    system_dirs = [
        "/system/framework",
        "/system/app",
        "/system/priv-app",
        "/system/etc",
        "/system/lib64",
        "/system/bin",
        "/vendor/etc",
    ]

    for sdir in system_dirs:
        output = run_adb(f"shell ls -la {sdir}", device, timeout=10)
        if output and len(output) > 50:
            samples.append({
                "input_text": f"What files and components are in the Android system directory {sdir}? Explain the purpose of this directory.",
                "output_text": f"The Android system directory `{sdir}` contains:\n\n```\n{output[:3000]}\n```\n\nThis directory is part of the Android OS structure. "
                + _explain_dir(sdir),
            })

    logger.info(f"📂 System structure: {len(samples)} samples")
    return samples


def scan_installed_apps(device: str = None) -> list:
    """فحص التطبيقات المنصبة"""
    samples = []

    # قائمة التطبيقات
    output = run_adb("shell pm list packages -f", device, timeout=15)
    if not output:
        return samples

    packages = output.strip().split("\n")

    # تفاصيل كل تطبيق (أول 50)
    for pkg_line in packages[:50]:
        try:
            # package:/data/app/com.example-xxx/base.apk=com.example
            pkg_name = pkg_line.split("=")[-1].strip()
            if not pkg_name:
                continue

            # معلومات التطبيق
            dump = run_adb(f"shell dumpsys package {pkg_name}", device, timeout=5)
            if dump and len(dump) > 100:
                # استخراج الأذونات
                perms = [l.strip() for l in dump.split("\n") if "permission" in l.lower()][:10]

                samples.append({
                    "input_text": f"What is the Android package `{pkg_name}`? What permissions does it use and what is its structure?",
                    "output_text": f"Package: `{pkg_name}`\n\nPermissions:\n"
                    + "\n".join(f"- {p}" for p in perms[:10])
                    + f"\n\nPackage details (excerpt):\n```\n{dump[:2000]}\n```",
                })
        except Exception:
            continue

    logger.info(f"📦 Apps: {len(samples)} samples from {len(packages)} packages")
    return samples


def scan_system_services(device: str = None) -> list:
    """فحص خدمات النظام"""
    samples = []

    # خدمات مهمة
    services = [
        "activity", "window", "connectivity", "wifi",
        "battery", "display", "audio", "input",
        "notification", "alarm", "power",
    ]

    for svc in services:
        output = run_adb(f"shell dumpsys {svc}", device, timeout=10)
        if output and len(output) > 200:
            samples.append({
                "input_text": f"Explain the Android `{svc}` system service. What does it manage and how does it work?",
                "output_text": f"The Android `{svc}` service manages:\n\n```\n{output[:3000]}\n```\n\n"
                f"This service is part of Android's system server framework.",
            })

    logger.info(f"⚙️ Services: {len(samples)} samples")
    return samples


def scan_system_config(device: str = None) -> list:
    """فحص إعدادات النظام"""
    samples = []

    # ملفات إعدادات مهمة
    config_files = [
        "/system/etc/permissions/platform.xml",
        "/system/etc/security/cacerts.bks",
        "/system/build.prop",
        "/vendor/build.prop",
        "/system/etc/hosts",
        "/system/etc/init/",
    ]

    for cf in config_files:
        output = run_adb(f"shell cat {cf}", device, timeout=5)
        if output and len(output) > 50:
            samples.append({
                "input_text": f"What is the Android configuration file `{cf}`? Explain its contents and purpose.",
                "output_text": f"Configuration file: `{cf}`\n\n```\n{output[:3000]}\n```",
            })

    # إعدادات النظام (Settings)
    for namespace in ["system", "secure", "global"]:
        output = run_adb(f"shell settings list {namespace}", device, timeout=5)
        if output:
            samples.append({
                "input_text": f"What are the Android `{namespace}` settings? List and explain the important ones.",
                "output_text": f"Android {namespace} settings:\n\n```\n{output[:3000]}\n```",
            })

    logger.info(f"🔧 Config: {len(samples)} samples")
    return samples


def scan_logs(device: str = None) -> list:
    """فحص لوغات النظام"""
    samples = []

    # آخر 100 سطر من logcat
    output = run_adb("shell logcat -d -t 100", device, timeout=10)
    if output and len(output) > 100:
        samples.append({
            "input_text": "Analyze these Android system logs. What services are running and are there any errors?",
            "output_text": f"Android logcat output:\n\n```\n{output[:4000]}\n```",
        })

    # Kernel logs
    output = run_adb("shell dmesg", device, timeout=5)
    if output and len(output) > 100:
        samples.append({
            "input_text": "Analyze these Android kernel (dmesg) logs. What hardware is detected and how is the system initialized?",
            "output_text": f"Android kernel logs:\n\n```\n{output[:4000]}\n```",
        })

    logger.info(f"📋 Logs: {len(samples)} samples")
    return samples


def _explain_dir(path: str) -> str:
    """شرح مجلد نظام"""
    explanations = {
        "/system/framework": "Contains Java framework JARs (framework.jar, services.jar) that power Android's core APIs.",
        "/system/app": "Contains pre-installed system applications that can be uninstalled by the user.",
        "/system/priv-app": "Contains privileged system apps with special permissions (Settings, SystemUI, etc.).",
        "/system/etc": "Contains system configuration files, permissions, and security policies.",
        "/system/lib64": "Contains native shared libraries (.so files) used by the system and apps.",
        "/system/bin": "Contains system binaries and daemons (init, servicemanager, surfaceflinger, etc.).",
        "/vendor/etc": "Contains vendor-specific configuration (hardware abstraction, codec configs, etc.).",
    }
    return explanations.get(path, "")


def save_samples(samples: list, capsule_id: str):
    """حفظ العينات في كبسولة"""
    if not samples:
        return 0

    capsule_dir = CAPSULES_DIR / capsule_id / "data"
    capsule_dir.mkdir(parents=True, exist_ok=True)

    out_file = capsule_dir / f"mobile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"💾 Saved {len(samples)} samples → {capsule_id}")
    return len(samples)


def run_scan(device: str = None):
    """تنفيذ فحص كامل"""
    logger.info("═" * 50)
    logger.info("📱 BI-IDE Mobile Scanner")
    logger.info("═" * 50)

    if not check_adb_connection(device):
        return

    info = get_device_info(device)
    total = 0

    # === 1. بنية النظام → كبسولة devops ===
    samples = scan_system_structure(device)
    total += save_samples(samples, "devops")

    # === 2. التطبيقات → security ===
    samples = scan_installed_apps(device)
    total += save_samples(samples, "security")

    # === 3. الخدمات → كبسولة devops ===
    samples = scan_system_services(device)
    total += save_samples(samples, "devops")

    # === 4. الإعدادات → security ===
    samples = scan_system_config(device)
    total += save_samples(samples, "security")

    # === 5. اللوغات → devops ===
    samples = scan_logs(device)
    total += save_samples(samples, "devops")

    # === حفظ معلومات الجهاز ===
    info["scanned_at"] = datetime.now().isoformat()
    info["total_samples"] = total
    info_file = CAPSULES_DIR / ".mobile_scan_info.json"
    info_file.write_text(json.dumps(info, indent=2, ensure_ascii=False))

    logger.info(f"\n{'═' * 50}")
    logger.info(f"✅ Scan complete: {total} training samples generated")
    logger.info(f"📱 Device: {info['brand']} {info['model']}")
    logger.info(f"{'═' * 50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BI-IDE Mobile Scanner")
    parser.add_argument("--device", default=None, help="ADB device ID or IP:port")
    parser.add_argument("--loop", action="store_true", help="Continuous scanning")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between scans (default: 1hr)")
    args = parser.parse_args()

    if args.loop:
        logger.info("♾️ Continuous mode — scanning every {args.interval}s")
        while True:
            try:
                run_scan(args.device)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(60)
    else:
        run_scan(args.device)
