#!/usr/bin/env python3
"""
bi_os_kernel.py — BI-OS: نظام تشغيل للذكاء الاصطناعي 🖥️🧠

نظام تشغيل مصمم من الصفر خصيصاً للـ AI:
  - Kernel: إدارة العمليات والموارد
  - FileSystem: نظام ملفات ذكي
  - ProcessManager: إدارة العمليات
  - MemoryManager: إدارة الذاكرة
  - DeviceManager: إدارة الأجهزة
  - NetworkStack: شبكات محلية
  - AILayer: طبقة الذكاء المدمجة بالنواة
  - SecurityLayer: أمان من النواة

الهدف:
  - نظام أسرع 10x من Linux للـ AI workloads
  - يشتغل offline بالكامل
  - الـ AI مدمج بالنواة (مو تطبيق فوقها)
  - ذاكرة أبدية على مستوى النظام
"""

import json
import logging
import time
import os
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("bi_os")

PROJECT_ROOT = Path(__file__).parent.parent

try:
    from brain.memory_system import memory
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory


# ═══════════════════════════════════════════════════════════
# نواة النظام — Kernel
# ═══════════════════════════════════════════════════════════

class ProcessState(Enum):
    NEW = "new"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    TERMINATED = "terminated"


@dataclass
class Process:
    """عملية بالنظام"""
    pid: int
    name: str
    state: ProcessState = ProcessState.NEW
    priority: int = 5          # 1-10 (10 = أعلى)
    memory_mb: float = 0
    cpu_percent: float = 0
    parent_pid: int = 0
    ai_capsule: str = ""       # كبسولة AI مرتبطة
    created_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class FileNode:
    """ملف/مجلد بنظام الملفات"""
    name: str
    path: str
    is_dir: bool = False
    size_bytes: int = 0
    content: str = ""
    permissions: str = "rwxr-xr-x"
    owner: str = "root"
    ai_indexed: bool = False    # هل الـ AI شافه؟
    ai_summary: str = ""        # ملخص ذكي
    hash: str = ""
    children: List = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)


class ProcessManager:
    """إدارة العمليات — الجدولة والتنفيذ"""

    def __init__(self):
        self.processes: Dict[int, Process] = {}
        self.next_pid = 1
        self._lock = threading.Lock()

        # عمليات النظام الأساسية
        self._create_system_processes()

    def _create_system_processes(self):
        """إنشاء عمليات النظام"""
        system_procs = [
            ("init", 10, "system"),
            ("ai_daemon", 10, "router"),
            ("memory_daemon", 9, "memory"),
            ("scout_daemon", 7, "scout"),
            ("training_daemon", 8, "training"),
            ("council_daemon", 6, "council"),
            ("dream_daemon", 3, "dreams"),
            ("sixth_sense", 5, "monitoring"),
        ]
        for name, priority, capsule in system_procs:
            self.create(name, priority=priority, ai_capsule=capsule)

    def create(self, name: str, priority: int = 5,
               ai_capsule: str = "") -> Process:
        """إنشاء عملية جديدة"""
        with self._lock:
            proc = Process(
                pid=self.next_pid,
                name=name,
                priority=priority,
                ai_capsule=ai_capsule,
                state=ProcessState.READY,
            )
            self.processes[self.next_pid] = proc
            self.next_pid += 1
            return proc

    def kill(self, pid: int) -> bool:
        """إنهاء عملية"""
        with self._lock:
            if pid in self.processes and pid > 1:  # لا تقتل init
                self.processes[pid].state = ProcessState.TERMINATED
                return True
            return False

    def schedule(self) -> Optional[Process]:
        """جدولة — اختيار العملية التالية (Priority Scheduling)"""
        ready = [p for p in self.processes.values() if p.state == ProcessState.READY]
        if not ready:
            return None
        # الأولوية الأعلى أولاً
        ready.sort(key=lambda p: p.priority, reverse=True)
        chosen = ready[0]
        chosen.state = ProcessState.RUNNING
        return chosen

    def list_processes(self) -> List[Dict]:
        """قائمة العمليات"""
        return [
            {
                "pid": p.pid,
                "name": p.name,
                "state": p.state.value,
                "priority": p.priority,
                "memory_mb": p.memory_mb,
                "ai_capsule": p.ai_capsule,
            }
            for p in self.processes.values()
            if p.state != ProcessState.TERMINATED
        ]


class FileSystem:
    """نظام ملفات ذكي — الـ AI يفهرس كل ملف"""

    def __init__(self):
        self.root = FileNode(name="/", path="/", is_dir=True)
        self._files: Dict[str, FileNode] = {"/": self.root}
        self._init_dirs()

    def _init_dirs(self):
        """إنشاء المجلدات الأساسية"""
        dirs = [
            "/bin", "/etc", "/home", "/tmp", "/var",
            "/brain", "/brain/capsules", "/brain/memory",
            "/brain/models", "/brain/training",
            "/data", "/data/knowledge", "/data/research",
            "/ai", "/ai/council", "/ai/dreams", "/ai/imagination",
        ]
        for d in dirs:
            self.mkdir(d)

    def mkdir(self, path: str) -> FileNode:
        """إنشاء مجلد"""
        parts = path.strip("/").split("/")
        current = self.root
        current_path = ""

        for part in parts:
            current_path += f"/{part}"
            if current_path not in self._files:
                node = FileNode(name=part, path=current_path, is_dir=True)
                self._files[current_path] = node
                current.children.append(node)
            current = self._files[current_path]

        return current

    def write(self, path: str, content: str, ai_index: bool = True) -> FileNode:
        """كتابة ملف"""
        parent_path = "/".join(path.rsplit("/", 1)[:-1]) or "/"
        if parent_path not in self._files:
            self.mkdir(parent_path)

        name = path.rsplit("/", 1)[-1]
        file_hash = hashlib.md5(content.encode()).hexdigest()

        node = FileNode(
            name=name, path=path, content=content,
            size_bytes=len(content.encode()),
            hash=file_hash,
        )

        # AI يفهرس الملف أوتوماتيكياً
        if ai_index:
            node.ai_indexed = True
            node.ai_summary = f"File: {name}, {len(content)} chars, hash: {file_hash[:8]}"

        self._files[path] = node
        parent = self._files.get(parent_path)
        if parent:
            parent.children = [c for c in parent.children if c.path != path]
            parent.children.append(node)

        return node

    def read(self, path: str) -> Optional[str]:
        """قراءة ملف"""
        node = self._files.get(path)
        if node and not node.is_dir:
            return node.content
        return None

    def ls(self, path: str = "/") -> List[Dict]:
        """عرض محتويات مجلد"""
        node = self._files.get(path)
        if not node or not node.is_dir:
            return []
        return [
            {
                "name": c.name,
                "type": "dir" if c.is_dir else "file",
                "size": c.size_bytes,
                "ai_indexed": c.ai_indexed,
            }
            for c in node.children
        ]

    def find(self, pattern: str) -> List[str]:
        """بحث بأسماء الملفات"""
        return [p for p in self._files if pattern.lower() in p.lower()]

    def stats(self) -> Dict:
        """إحصائيات نظام الملفات"""
        files = [f for f in self._files.values() if not f.is_dir]
        dirs = [f for f in self._files.values() if f.is_dir]
        total_size = sum(f.size_bytes for f in files)
        indexed = sum(1 for f in files if f.ai_indexed)
        return {
            "total_files": len(files),
            "total_dirs": len(dirs),
            "total_size_bytes": total_size,
            "ai_indexed_files": indexed,
        }


class MemoryManager:
    """إدارة الذاكرة — Virtual Memory + AI Cache"""

    def __init__(self, total_mb: int = 32768):
        self.total_mb = total_mb
        self.used_mb = 0
        self.allocations: Dict[int, float] = {}  # pid: mb
        self.ai_cache_mb = total_mb * 0.3  # 30% للـ AI

    def allocate(self, pid: int, size_mb: float) -> bool:
        """تخصيص ذاكرة"""
        if self.used_mb + size_mb <= self.total_mb:
            self.allocations[pid] = self.allocations.get(pid, 0) + size_mb
            self.used_mb += size_mb
            return True
        return False

    def free(self, pid: int) -> float:
        """تحرير ذاكرة"""
        freed = self.allocations.pop(pid, 0)
        self.used_mb -= freed
        return freed

    def stats(self) -> Dict:
        return {
            "total_mb": self.total_mb,
            "used_mb": round(self.used_mb, 1),
            "free_mb": round(self.total_mb - self.used_mb, 1),
            "ai_cache_mb": round(self.ai_cache_mb, 1),
            "usage_percent": round(self.used_mb / self.total_mb * 100, 1),
            "processes": len(self.allocations),
        }


class DeviceManager:
    """إدارة الأجهزة — GPU, كاميرات, شبكة"""

    def __init__(self):
        self.devices = self._detect_devices()

    def _detect_devices(self) -> List[Dict]:
        """اكتشاف الأجهزة المتصلة"""
        devices = [
            {"name": "CPU", "type": "processor", "status": "active"},
            {"name": "RAM", "type": "memory", "status": "active"},
            {"name": "Disk", "type": "storage", "status": "active"},
        ]

        # GPU
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for gpu in result.stdout.strip().split("\n"):
                    devices.append({"name": gpu.strip(), "type": "gpu", "status": "active"})
        except Exception:
            pass

        # كاميرا
        try:
            import subprocess
            result = subprocess.run(["system_profiler", "SPCameraDataType"],
                                   capture_output=True, text=True, timeout=5)
            if "Camera" in result.stdout:
                devices.append({"name": "Camera", "type": "camera", "status": "active"})
        except Exception:
            pass

        return devices

    def list_devices(self) -> List[Dict]:
        return self.devices


class SecurityLayer:
    """طبقة الأمان — مدمجة بالنواة"""

    def __init__(self):
        self.permissions = {
            "root": {"read": True, "write": True, "execute": True, "ai_admin": True},
            "user": {"read": True, "write": True, "execute": True, "ai_admin": False},
            "guest": {"read": True, "write": False, "execute": False, "ai_admin": False},
        }
        self.audit_log: List[Dict] = []

    def check_permission(self, user: str, action: str) -> bool:
        """فحص الصلاحيات"""
        perms = self.permissions.get(user, self.permissions["guest"])
        allowed = perms.get(action, False)
        self.audit_log.append({
            "user": user, "action": action, "allowed": allowed,
            "time": datetime.now().isoformat(),
        })
        return allowed

    def add_user(self, username: str, role: str = "user"):
        """إضافة مستخدم"""
        if role in self.permissions:
            self.permissions[username] = self.permissions[role].copy()


class AILayer:
    """طبقة الذكاء — مدمجة بالنواة"""

    def __init__(self, kernel):
        self.kernel = kernel
        self.capsules_loaded = []

    def query(self, question: str, mode: str = "fast") -> Dict:
        """سؤال الذكاء المدمج بالنواة"""
        try:
            from brain.capsule_router import ask
            decision = ask(question, mode=mode)
            return {
                "response": f"BI-OS AI: Routed to {decision.selected_capsules}",
                "capsules": decision.selected_capsules,
                "confidence": decision.confidence,
                "mode": mode,
            }
        except Exception as e:
            return {"response": f"AI Layer: {question}", "error": str(e)}

    def auto_index_file(self, path: str, content: str) -> Dict:
        """فهرسة ملف أوتوماتيكياً"""
        try:
            from brain.ide_file_learner import learn
            return learn(path, content)
        except Exception:
            return {"indexed": False}


# ═══════════════════════════════════════════════════════════
# النواة — Kernel
# ═══════════════════════════════════════════════════════════

class BIOSKernel:
    """
    BI-OS Kernel — نواة نظام التشغيل

    مكونات النواة:
      - ProcessManager: إدارة العمليات
      - FileSystem: نظام ملفات ذكي
      - MemoryManager: إدارة الذاكرة
      - DeviceManager: إدارة الأجهزة
      - SecurityLayer: أمان من النواة
      - AILayer: ذكاء مدمج بالنواة
    """

    VERSION = "0.1.0-alpha"
    CODENAME = "Genesis"

    def __init__(self):
        self.boot_time = time.time()

        # مكونات النواة
        self.processes = ProcessManager()
        self.filesystem = FileSystem()
        self.memory_mgr = MemoryManager()
        self.devices = DeviceManager()
        self.security = SecurityLayer()
        self.ai = AILayer(self)

        # حالة النظام
        self.uptime_start = time.time()
        self.syscall_count = 0

        logger.info(f"🖥️ BI-OS Kernel v{self.VERSION} ({self.CODENAME}) — Booted")

    def boot(self) -> Dict:
        """تشغيل النظام"""
        boot_log = [
            f"BI-OS v{self.VERSION} ({self.CODENAME})",
            f"Boot time: {datetime.now().isoformat()}",
            f"Devices: {len(self.devices.list_devices())}",
            f"Processes: {len(self.processes.list_processes())}",
            f"Filesystem: {self.filesystem.stats()['total_dirs']} dirs",
            f"Memory: {self.memory_mgr.total_mb} MB",
            f"AI Layer: Active",
            f"Security: Active",
            "═══ BI-OS Ready ═══",
        ]

        # حفظ بالذاكرة الأبدية
        memory.save_knowledge(
            topic="BI-OS Boot",
            content=f"Kernel v{self.VERSION} booted at {datetime.now().isoformat()}",
            source="kernel",
        )

        return {"status": "booted", "log": boot_log}

    def syscall(self, call: str, args: Dict = None) -> Dict:
        """System Call — واجهة النواة"""
        self.syscall_count += 1
        args = args or {}

        handlers = {
            "ps": lambda: {"processes": self.processes.list_processes()},
            "ls": lambda: {"files": self.filesystem.ls(args.get("path", "/"))},
            "cat": lambda: {"content": self.filesystem.read(args.get("path", ""))},
            "write": lambda: {"file": self.filesystem.write(
                args.get("path", "/tmp/unnamed"),
                args.get("content", ""),
            ).path},
            "mkdir": lambda: {"dir": self.filesystem.mkdir(args.get("path", "/tmp")).path},
            "find": lambda: {"results": self.filesystem.find(args.get("pattern", ""))},
            "kill": lambda: {"killed": self.processes.kill(args.get("pid", 0))},
            "exec": lambda: {"process": self.processes.create(
                args.get("name", "unnamed"),
                priority=args.get("priority", 5),
                ai_capsule=args.get("capsule", ""),
            ).pid},
            "mem": lambda: self.memory_mgr.stats(),
            "devices": lambda: {"devices": self.devices.list_devices()},
            "ai": lambda: self.ai.query(args.get("question", ""), args.get("mode", "fast")),
            "status": lambda: self.get_status(),
        }

        handler = handlers.get(call)
        if handler:
            return handler()
        return {"error": f"Unknown syscall: {call}"}

    def get_status(self) -> Dict:
        """حالة النظام"""
        uptime = time.time() - self.uptime_start
        return {
            "version": self.VERSION,
            "codename": self.CODENAME,
            "uptime_seconds": round(uptime, 1),
            "processes": len(self.processes.list_processes()),
            "filesystem": self.filesystem.stats(),
            "memory": self.memory_mgr.stats(),
            "devices": len(self.devices.list_devices()),
            "syscalls": self.syscall_count,
            "ai_active": True,
            "security_active": True,
        }

    def shell(self, command: str) -> str:
        """BI-OS Shell — سطر الأوامر"""
        parts = command.strip().split()
        if not parts:
            return ""

        cmd = parts[0]
        args = {}

        if cmd == "help":
            return (
                "BI-OS Shell Commands:\n"
                "  ps          — List processes\n"
                "  ls [path]   — List directory\n"
                "  cat <path>  — Read file\n"
                "  mkdir <path> — Create directory\n"
                "  find <pattern> — Search files\n"
                "  mem         — Memory status\n"
                "  devices     — List devices\n"
                "  ai <question> — Ask AI\n"
                "  status      — System status\n"
                "  help        — This help\n"
            )
        elif cmd == "ls":
            args["path"] = parts[1] if len(parts) > 1 else "/"
        elif cmd == "cat":
            args["path"] = parts[1] if len(parts) > 1 else ""
        elif cmd == "mkdir":
            args["path"] = parts[1] if len(parts) > 1 else "/tmp/new"
        elif cmd == "find":
            args["pattern"] = parts[1] if len(parts) > 1 else ""
        elif cmd == "ai":
            args["question"] = " ".join(parts[1:])
        elif cmd == "kill":
            args["pid"] = int(parts[1]) if len(parts) > 1 else 0

        result = self.syscall(cmd, args)
        return json.dumps(result, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════

kernel = BIOSKernel()


# ═══════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🖥️ BI-OS — Boot Sequence\n")

    # Boot
    boot = kernel.boot()
    for line in boot["log"]:
        print(f"  {line}")

    print(f"\n{'═' * 50}")

    # Shell test
    commands = [
        "status",
        "ps",
        "ls /",
        "ls /brain",
        "mem",
        "devices",
        "ai شنو BI-OS؟",
    ]

    for cmd in commands:
        print(f"\n$ {cmd}")
        print(kernel.shell(cmd))
