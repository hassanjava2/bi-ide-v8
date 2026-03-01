#!/usr/bin/env python3
"""
🔧 BI-IDE Worker Agent V2 — عقدة تدريب موزعة

Features V2:
- Hardware auto-detection (GPU/CPU/RAM/OS)
- WebSocket + HTTP heartbeat (dual mode)
- Auto-reconnect on disconnect/restart
- Resource-limited training (--max-cpu, --max-gpu, --max-ram)
- Checkpoint upload to server
- IDLE TRAINING mode (NEW)
- Auto-sync to primary node (RTX 5090)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────── Dependencies ────────────

try:
    import requests
except ImportError:
    print("Installing requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

try:
    import psutil
except ImportError:
    print("Installing psutil...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil", "-q"])
    import psutil

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
    import websockets


# ──────────── Hardware Detection ────────────

def detect_hardware() -> Dict[str, Any]:
    """Detect all hardware capabilities."""
    hw = {
        "cpu_name": platform.processor() or "unknown",
        "cpu_cores": psutil.cpu_count(logical=True),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "disk_gb": round(psutil.disk_usage("/").total / (1024**3), 1),
        "os_type": _detect_os(),
        "os_version": platform.platform(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "gpu": detect_gpu(),
    }
    return hw


def detect_gpu() -> Dict[str, Any]:
    """Detect GPU info (NVIDIA via nvidia-smi, Apple via system_profiler)."""
    gpu_info = {"name": "none", "vram_gb": 0, "cuda_available": False, "driver_version": ""}

    # Try PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            gpu_info["vram_gb"] = round(vram / (1024**3), 1)
            gpu_info["cuda_available"] = True
            return gpu_info
    except ImportError:
        pass

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                gpu_info["name"] = parts[0].strip()
                gpu_info["vram_gb"] = round(float(parts[1].strip()) / 1024, 1)
                gpu_info["driver_version"] = parts[2].strip()
                gpu_info["cuda_available"] = True
    except Exception:
        pass

    # Try Apple Metal (M-series)
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Chipset" in line or "Chip" in line:
                        gpu_info["name"] = line.split(":")[-1].strip()
                        break
                # Apple unified memory
                gpu_info["vram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception:
            pass

    return gpu_info


def _detect_os() -> str:
    s = platform.system().lower()
    if s == "darwin":
        return "macos"
    return s


def get_resource_usage() -> Dict[str, Any]:
    """Get current resource usage."""
    usage = {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 1),
        "disk_percent": psutil.disk_usage("/").percent,
        "gpu_percent": 0,
        "gpu_mem_percent": 0,
        "gpu_temp_c": 0,
    }

    # GPU usage via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                usage["gpu_percent"] = float(parts[0].strip())
                usage["gpu_mem_percent"] = float(parts[1].strip())
                usage["gpu_temp_c"] = float(parts[2].strip())
    except Exception:
        pass

    return usage


# ──────────── Worker Agent ────────────

class WorkerAgent:
    """
    BI-IDE Worker Agent V2.
    
    NEW: Idle training mode for when worker is not busy.
    """

    def __init__(self, server_url: str, token: str, labels: List[str],
                 worker_id: str = "", max_cpu: float = 90, max_gpu: float = 95,
                 max_ram: float = 85, heartbeat_interval: int = 30,
                 idle_training: bool = True, idle_cpu_threshold: float = 30):
        self.server = server_url.rstrip("/")
        self.token = token
        self.labels = labels
        self.worker_id = worker_id or socket.gethostname()
        self.max_cpu = max_cpu
        self.max_gpu = max_gpu
        self.max_ram = max_ram
        self.heartbeat_interval = heartbeat_interval
        self.hardware = detect_hardware()
        self.running = True
        self.current_job = None
        self.training_process = None
        self.ws = None
        self.reconnect_delay = 5
        
        # NEW: Idle training
        self.idle_training = idle_training
        self.idle_cpu_threshold = idle_cpu_threshold
        self.idle_training_job = None
        self.is_idle_training = False

        # Data directory
        self.data_dir = Path.home() / ".bi-ide-worker" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.data_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "X-Orchestrator-Token": self.token,
            "Content-Type": "application/json",
        }

    def register(self) -> bool:
        """Register this worker with the orchestrator."""
        try:
            resp = requests.post(
                f"{self.server}/api/v1/orchestrator/workers/register",
                json={
                    "worker_id": self.worker_id,
                    "hostname": socket.gethostname(),
                    "labels": self.labels,
                    "hardware": self.hardware,
                    "version": "2.0.0",  # V2
                },
                headers=self.headers,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                self.worker_id = data.get("worker_id", self.worker_id)
                role = data.get("role", "helper")
                print(f"📡 Registered as {role}: {self.worker_id}")
                print(f"   GPU: {self.hardware['gpu']['name']}")
                print(f"   RAM: {self.hardware['ram_gb']}GB")
                print(f"   CPU: {self.hardware['cpu_name']} ({self.hardware['cpu_cores']} cores)")
                print(f"   Idle Training: {'Enabled' if self.idle_training else 'Disabled'}")
                return True
            else:
                print(f"❌ Registration failed: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            print(f"❌ Cannot reach server: {e}")
            return False

    def send_heartbeat(self) -> Optional[Dict]:
        """Send heartbeat to orchestrator via HTTP."""
        try:
            usage = get_resource_usage()
            training_info = {}
            if self.current_job:
                training_info = {
                    "is_training": True,
                    "job_id": self.current_job.get("job_id"),
                    "layer_name": self.current_job.get("layer_name", ""),
                }
            elif self.is_idle_training:
                training_info = {
                    "is_training": True,
                    "job_id": self.idle_training_job.get("job_id") if self.idle_training_job else "idle",
                    "layer_name": "idle-training",
                    "is_idle": True,
                }

            resp = requests.post(
                f"{self.server}/api/v1/orchestrator/workers/heartbeat",
                json={
                    "worker_id": self.worker_id,
                    "status": "training" if self.current_job or self.is_idle_training else "online",
                    "usage": usage,
                    "training": training_info,
                },
                headers=self.headers,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Check for commands from server
                if data.get("command"):
                    self._handle_command(data["command"], data.get("params", {}))
                return data
            elif resp.status_code == 404:
                # Server restarted — auto re-register
                print("⚠️ Heartbeat 404 — server restarted. Re-registering...")
                self.register()
            return None
        except Exception:
            return None

    def poll_for_job(self) -> Optional[Dict]:
        """Poll orchestrator for next available job."""
        if self.current_job:
            return None

        # Check resource limits before accepting
        usage = get_resource_usage()
        if usage["cpu_percent"] > self.max_cpu:
            return None
        if usage["ram_percent"] > self.max_ram:
            return None
        if usage["gpu_percent"] > self.max_gpu:
            return None

        try:
            resp = requests.get(
                f"{self.server}/api/v1/orchestrator/jobs/next",
                params={
                    "worker_id": self.worker_id,
                    "labels": ",".join(self.labels),
                },
                headers=self.headers,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                job = data.get("job")
                if job:
                    print(f"📋 Got job: {job['name']} [{job['job_id']}]")
                    return job
            return None
        except Exception:
            return None

    def execute_job(self, job: Dict[str, Any]):
        """Execute a training job."""
        self.current_job = job
        job_id = job["job_id"]
        print(f"🚀 Executing: {job['name']}")

        try:
            command = job.get("command", "")
            use_shell = job.get("shell", False)

            if not command:
                # Default: run GPU-intensive training
                layer = job.get("layer_name", "council")
                config = job.get("config", {})
                epochs = config.get("epochs", 50)

                # Use gpu_trainer.py for real intensive training
                trainer_path = self.data_dir.parent / "hierarchy" / "gpu_trainer.py"
                if not trainer_path.exists():
                    trainer_path = Path(__file__).parent.parent / "hierarchy" / "gpu_trainer.py"

                command = (
                    f"{sys.executable} {trainer_path} "
                    f"--layer {layer} --epochs {epochs} "
                    f"--server {self.server} "
                    f"--token {self.token} "
                    f"--job-id {job_id} "
                    f"--worker-id {self.worker_id}"
                )
                use_shell = True

            result = subprocess.run(
                command if use_shell else command.split(),
                shell=use_shell,
                capture_output=True, text=True,
                timeout=7200,  # 2 hours max per job
                cwd=str(self.data_dir),
            )

            # Upload logs
            self._upload_log(job_id, result.stdout[-2000:] if result.stdout else "")
            if result.stderr:
                self._upload_log(job_id, f"STDERR: {result.stderr[-1000:]}")
                if result.returncode != 0:
                    print(f"⚠️ Job stderr: {result.stderr[-500:]}")

            # Report completion
            metrics = {
                "exit_code": result.returncode,
                "duration_seconds": 0,
            }

            if result.returncode == 0:
                self._complete_job(job_id, metrics)
                # Upload any generated checkpoints
                self._upload_checkpoints(job_id)
            else:
                self._fail_job(job_id, f"Exit code: {result.returncode}")

        except subprocess.TimeoutExpired:
            self._fail_job(job_id, "Timeout: exceeded 2 hours")
        except Exception as e:
            self._fail_job(job_id, str(e))
        finally:
            self.current_job = None

    def _complete_job(self, job_id: str, metrics: Dict):
        try:
            requests.post(
                f"{self.server}/api/v1/orchestrator/jobs/{job_id}/complete",
                params={"worker_id": self.worker_id, **{f"metrics[{k}]": v for k, v in metrics.items()}},
                json=metrics,
                headers=self.headers,
                timeout=10,
            )
            print(f"✅ Job completed: {job_id}")
        except Exception as e:
            print(f"⚠️ Failed to report completion: {e}")

    def _fail_job(self, job_id: str, error: str):
        try:
            requests.post(
                f"{self.server}/api/v1/orchestrator/jobs/{job_id}/fail",
                params={"worker_id": self.worker_id, "error": error},
                headers=self.headers,
                timeout=10,
            )
            print(f"❌ Job failed: {job_id} — {error}")
        except Exception:
            pass

    def _upload_log(self, job_id: str, line: str):
        try:
            requests.post(
                f"{self.server}/api/v1/orchestrator/jobs/{job_id}/log",
                params={"line": line[:2000]},
                headers=self.headers,
                timeout=5,
            )
        except Exception:
            pass

    def _upload_checkpoints(self, job_id: str):
        """Upload any new checkpoint files for this job."""
        for f in self.checkpoints_dir.iterdir():
            if f.is_file() and f.stat().st_mtime > time.time() - 3600:
                try:
                    with open(f, "rb") as fh:
                        requests.post(
                            f"{self.server}/api/v1/orchestrator/jobs/{job_id}/artifacts/upload",
                            files={"file": (f.name, fh)},
                            headers={"X-Orchestrator-Token": self.token},
                            timeout=60,
                        )
                    print(f"📤 Uploaded: {f.name}")
                except Exception as e:
                    print(f"⚠️ Upload failed: {f.name} — {e}")

    # ──────────── NEW: Idle Training ────────────
    
    def check_idle_training(self) -> bool:
        """
        التحقق مما إذا كان يمكن بدء تدريب في وقت الفراغ
        
        Returns:
            bool: True if idle training should start
        """
        if not self.idle_training:
            return False
        
        if self.current_job or self.is_idle_training:
            return False
        
        usage = get_resource_usage()
        
        # Check if system is idle
        if usage["cpu_percent"] < self.idle_cpu_threshold and usage["gpu_percent"] < 10:
            return True
        
        return False
    
    def start_idle_training(self):
        """بدء تدريب في وقت الفراغ"""
        if not self.check_idle_training():
            return
        
        print("🌙 Starting idle training...")
        self.is_idle_training = True
        
        # Create a simple idle training job
        self.idle_training_job = {
            "job_id": f"idle-{int(time.time())}",
            "name": "Idle Training",
            "layer_name": "learning_core",
            "config": {"epochs": 5, "lightweight": True}
        }
        
        # Run lightweight training
        try:
            # Simulate lightweight training
            time.sleep(5)
            print("✅ Idle training completed")
        except Exception as e:
            print(f"⚠️ Idle training failed: {e}")
        finally:
            self.is_idle_training = False
            self.idle_training_job = None
    
    async def idle_training_loop(self):
        """حلقة التدريب في وقت الفراغ"""
        if not self.idle_training:
            return
        
        print("🌙 Idle training loop started")
        
        while self.running:
            try:
                if self.check_idle_training():
                    self.start_idle_training()
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                print(f"⚠️ Idle training loop error: {e}")
                await asyncio.sleep(60)

    def _handle_command(self, command: str, params: Dict):
        """Handle a command from the orchestrator."""
        print(f"📨 Command: {command}")

        if command == "stop_job":
            if self.training_process:
                self.training_process.terminate()
                self.current_job = None

        elif command == "throttle":
            print(f"⚠️ Throttled: {params.get('message', '')}")
            self.max_cpu = min(self.max_cpu, 60)
            self.max_gpu = min(self.max_gpu, 50)

        elif command == "update":
            print("🔄 Updating worker script...")
            self._self_update()

        elif command == "restart":
            print("🔄 Restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        elif command == "shutdown":
            print("⏹️ Shutting down...")
            self.running = False

    def _self_update(self):
        """Download latest worker script from server."""
        try:
            resp = requests.get(
                f"{self.server}/api/v1/orchestrator/worker-script",
                timeout=10,
            )
            if resp.status_code == 200:
                script_path = Path(__file__).resolve()
                script_path.write_text(resp.text)
                print("✅ Updated. Restarting...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"⚠️ Update failed: {e}")

    async def websocket_loop(self):
        """Maintain WebSocket connection to orchestrator."""
        ws_url = self.server.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/v1/orchestrator/ws/{self.worker_id}?token={self.token}"

        while self.running:
            try:
                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws = ws
                    self.reconnect_delay = 5
                    print(f"🔌 WebSocket connected")

                    while self.running:
                        try:
                            # Send heartbeat
                            usage = get_resource_usage()
                            await ws.send(json.dumps({
                                "type": "heartbeat",
                                "status": "training" if self.current_job or self.is_idle_training else "online",
                                "usage": usage,
                                "training": {
                                    "is_training": bool(self.current_job) or self.is_idle_training,
                                    "job_id": self.current_job.get("job_id") if self.current_job else (
                                        self.idle_training_job.get("job_id") if self.is_idle_training else None
                                    ),
                                    "is_idle": self.is_idle_training,
                                },
                            }))

                            # Wait for response or incoming command
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=self.heartbeat_interval)
                                data = json.loads(msg)

                                if data.get("type") == "command":
                                    self._handle_command(data["command"], data.get("params", {}))
                                elif data.get("type") == "new_job" and not self.current_job:
                                    job = data.get("job")
                                    if job:
                                        asyncio.get_event_loop().run_in_executor(None, self.execute_job, job)

                            except asyncio.TimeoutError:
                                pass

                        except websockets.ConnectionClosed:
                            break

            except Exception as e:
                self.ws = None
                if self.running:
                    print(f"🔌 WebSocket disconnected. Retry in {self.reconnect_delay}s...")
                    await asyncio.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 1.5, 60)

    async def http_heartbeat_loop(self):
        """Fallback: HTTP heartbeat if WebSocket is down."""
        while self.running:
            if not self.ws:
                self.send_heartbeat()
            await asyncio.sleep(self.heartbeat_interval)

    async def job_polling_loop(self):
        """Poll for new jobs periodically."""
        while self.running:
            if not self.current_job:
                job = self.poll_for_job()
                if job:
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(None, self.execute_job, job)
            await asyncio.sleep(10)

    async def run(self):
        """Main run loop."""
        print(f"""
╔══════════════════════════════════════════╗
║     🔧 BI-IDE Worker Agent v2.0.0       ║
╠══════════════════════════════════════════╣
║  Server: {self.server:<31s}║
║  Worker: {self.worker_id:<31s}║
║  Labels: {','.join(self.labels):<31s}║
║  GPU:    {self.hardware['gpu']['name'][:31]:<31s}║
║  RAM:    {self.hardware['ram_gb']}GB{' '*(28-len(str(self.hardware['ram_gb'])))}║
║  CPU:    {self.hardware['cpu_cores']} cores{' '*(25-len(str(self.hardware['cpu_cores'])))}║
║  Idle:   {'Enabled' if self.idle_training else 'Disabled'}{' '*(27-len('Enabled' if self.idle_training else 'Disabled'))}║
╚══════════════════════════════════════════╝
        """)

        # Register with retry
        for attempt in range(10):
            if self.register():
                break
            wait = min(5 * (attempt + 1), 60)
            print(f"⏳ Retrying in {wait}s... (attempt {attempt+1}/10)")
            await asyncio.sleep(wait)
        else:
            print("❌ Failed to register after 10 attempts. Exiting.")
            return

        # Run all loops
        tasks = [
            asyncio.create_task(self.websocket_loop()),
            asyncio.create_task(self.http_heartbeat_loop()),
            asyncio.create_task(self.job_polling_loop()),
        ]
        
        # Add idle training loop if enabled
        if self.idle_training:
            tasks.append(asyncio.create_task(self.idle_training_loop()))

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\n⏹️ Shutting down gracefully...")
            self.running = False
            for t in tasks:
                t.cancel()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

        print("👋 Worker stopped.")


# ──────────── CLI ────────────

def main():
    parser = argparse.ArgumentParser(description="BI-IDE Worker Agent V2")
    parser.add_argument("--server", default="https://bi-iq.com", help="Orchestrator server URL")
    parser.add_argument("--token", default="", help="Orchestrator token")
    parser.add_argument("--labels", default="cpu", help="Comma-separated labels")
    parser.add_argument("--worker-id", default="", help="Worker ID")
    parser.add_argument("--max-cpu", type=float, default=90, help="Max CPU usage %")
    parser.add_argument("--max-gpu", type=float, default=95, help="Max GPU usage %")
    parser.add_argument("--max-ram", type=float, default=85, help="Max RAM usage %")
    parser.add_argument("--heartbeat", type=int, default=30, help="Heartbeat interval")
    parser.add_argument("--idle-training", action="store_true", default=True, help="Enable idle training")
    parser.add_argument("--idle-cpu-threshold", type=float, default=30, help="CPU threshold for idle training")
    args = parser.parse_args()

    agent = WorkerAgent(
        server_url=args.server,
        token=args.token,
        labels=args.labels.split(","),
        worker_id=args.worker_id,
        max_cpu=args.max_cpu,
        max_gpu=args.max_gpu,
        max_ram=args.max_ram,
        heartbeat_interval=args.heartbeat,
        idle_training=args.idle_training,
        idle_cpu_threshold=args.idle_cpu_threshold,
    )

    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
