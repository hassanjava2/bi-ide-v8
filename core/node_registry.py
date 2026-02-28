"""
🖧 Node Registry — إدارة العقد الموزعة
Tracks all connected training nodes (RTX 5090, Windows, Mac, Hostinger, etc.)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import WebSocket


# ──────────── Enums ────────────

class NodeRole(str, Enum):
    PRIMARY = "primary"        # RTX 5090 — العقدة الرئيسية
    HELPER = "helper"          # عقدة مساعدة (Windows, Mac, etc.)
    ORCHESTRATOR = "orchestrator"  # السيرفر (Hostinger)


class NodeStatus(str, Enum):
    ONLINE = "online"
    TRAINING = "training"
    SYNCING = "syncing"
    IDLE = "idle"
    THROTTLED = "throttled"    # Hostinger CPU limit
    OFFLINE = "offline"


class NodeOS(str, Enum):
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"


# ──────────── Data Classes ────────────

@dataclass
class GPUInfo:
    name: str = "none"
    vram_gb: float = 0.0
    cuda_cores: int = 0
    driver_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vram_gb": self.vram_gb,
            "cuda_cores": self.cuda_cores,
            "driver_version": self.driver_version,
        }


@dataclass
class HardwareInfo:
    cpu_name: str = "unknown"
    cpu_cores: int = 0
    ram_gb: float = 0.0
    gpu: GPUInfo = field(default_factory=GPUInfo)
    disk_gb: float = 0.0
    os_type: NodeOS = NodeOS.LINUX
    os_version: str = ""
    hostname: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_name": self.cpu_name,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "gpu": self.gpu.to_dict(),
            "disk_gb": self.disk_gb,
            "os_type": self.os_type.value,
            "os_version": self.os_version,
            "hostname": self.hostname,
        }


@dataclass
class ResourceUsage:
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_mem_percent: float = 0.0
    gpu_temp_c: float = 0.0
    disk_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "gpu_percent": self.gpu_percent,
            "gpu_mem_percent": self.gpu_mem_percent,
            "gpu_temp_c": self.gpu_temp_c,
            "disk_percent": self.disk_percent,
        }


@dataclass
class TrainingProgress:
    is_training: bool = False
    layer_name: str = ""
    epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    samples_processed: int = 0
    started_at: Optional[str] = None
    eta_seconds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_training": self.is_training,
            "layer_name": self.layer_name,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "samples_processed": self.samples_processed,
            "started_at": self.started_at,
            "eta_seconds": self.eta_seconds,
        }


@dataclass
class TrainingConfig:
    max_cpu_percent: float = 90.0
    max_gpu_percent: float = 95.0
    max_ram_percent: float = 85.0
    batch_size: int = 32
    learning_rate: float = 0.001
    auto_start: bool = True
    layers_to_train: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_cpu_percent": self.max_cpu_percent,
            "max_gpu_percent": self.max_gpu_percent,
            "max_ram_percent": self.max_ram_percent,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "auto_start": self.auto_start,
            "layers_to_train": self.layers_to_train,
        }


# ──────────── Node ────────────

@dataclass
class Node:
    id: str
    name: str
    role: NodeRole
    status: NodeStatus = NodeStatus.OFFLINE
    hardware: HardwareInfo = field(default_factory=HardwareInfo)
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    training: TrainingProgress = field(default_factory=TrainingProgress)
    config: TrainingConfig = field(default_factory=TrainingConfig)
    ip_address: str = ""
    registered_at: str = ""
    last_heartbeat: float = 0.0
    version: str = "1.0.0"
    websocket: Optional[WebSocket] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "hardware": self.hardware.to_dict(),
            "usage": self.usage.to_dict(),
            "training": self.training.to_dict(),
            "config": self.config.to_dict(),
            "ip_address": self.ip_address,
            "registered_at": self.registered_at,
            "last_heartbeat_ago": f"{time.time() - self.last_heartbeat:.0f}s" if self.last_heartbeat else "never",
            "version": self.version,
            "connected": self.websocket is not None,
        }


# ──────────── Registry ────────────

class NodeRegistry:
    """
    Central registry for all training nodes.
    Thread-safe, supports WebSocket connections.
    """

    HEARTBEAT_TIMEOUT = 60  # seconds before marking node offline

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self._lock = asyncio.Lock()
        self._hostinger_cpu_history: List[float] = []  # CPU readings for throttle logic

    async def register(
        self,
        name: str,
        role: NodeRole,
        hardware: Dict[str, Any],
        ip_address: str = "",
        node_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Node:
        """Register a new node or re-register an existing one."""
        async with self._lock:
            nid = node_id or str(uuid.uuid4())[:12]

            # Check for existing node by name (re-registration after restart)
            for existing in self.nodes.values():
                if existing.name == name and existing.ip_address == ip_address:
                    nid = existing.id
                    break

            gpu_data = hardware.get("gpu", {})
            gpu = GPUInfo(
                name=gpu_data.get("name", "none"),
                vram_gb=gpu_data.get("vram_gb", 0),
                cuda_cores=gpu_data.get("cuda_cores", 0),
                driver_version=gpu_data.get("driver_version", ""),
            )

            hw = HardwareInfo(
                cpu_name=hardware.get("cpu_name", "unknown"),
                cpu_cores=hardware.get("cpu_cores", 0),
                ram_gb=hardware.get("ram_gb", 0),
                gpu=gpu,
                disk_gb=hardware.get("disk_gb", 0),
                os_type=NodeOS(hardware.get("os_type", "linux")),
                os_version=hardware.get("os_version", ""),
                hostname=hardware.get("hostname", name),
            )

            tc = TrainingConfig()
            if config:
                tc.max_cpu_percent = config.get("max_cpu_percent", tc.max_cpu_percent)
                tc.max_gpu_percent = config.get("max_gpu_percent", tc.max_gpu_percent)
                tc.max_ram_percent = config.get("max_ram_percent", tc.max_ram_percent)
                tc.batch_size = config.get("batch_size", tc.batch_size)
                tc.auto_start = config.get("auto_start", tc.auto_start)

            # Hostinger special config: throttle CPU
            if role == NodeRole.ORCHESTRATOR:
                tc.max_cpu_percent = min(tc.max_cpu_percent, 75.0)

            node = Node(
                id=nid,
                name=name,
                role=role,
                status=NodeStatus.ONLINE,
                hardware=hw,
                config=tc,
                ip_address=ip_address,
                registered_at=datetime.now(timezone.utc).isoformat(),
                last_heartbeat=time.time(),
            )

            self.nodes[nid] = node
            print(f"📡 Node registered: {name} ({role.value}) [{nid}] — "
                  f"{hw.gpu.name} / {hw.ram_gb}GB RAM")

            return node

    async def heartbeat(self, node_id: str, usage: Optional[Dict[str, Any]] = None,
                        training: Optional[Dict[str, Any]] = None) -> bool:
        """Update node heartbeat and optionally update usage/training status."""
        async with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                return False

            node.last_heartbeat = time.time()

            if node.status == NodeStatus.OFFLINE:
                node.status = NodeStatus.ONLINE

            if usage:
                node.usage = ResourceUsage(
                    cpu_percent=usage.get("cpu_percent", 0),
                    ram_percent=usage.get("ram_percent", 0),
                    gpu_percent=usage.get("gpu_percent", 0),
                    gpu_mem_percent=usage.get("gpu_mem_percent", 0),
                    gpu_temp_c=usage.get("gpu_temp_c", 0),
                    disk_percent=usage.get("disk_percent", 0),
                )

                # Hostinger throttle check
                if node.role == NodeRole.ORCHESTRATOR:
                    self._check_hostinger_throttle(node)

            if training:
                node.training = TrainingProgress(
                    is_training=training.get("is_training", False),
                    layer_name=training.get("layer_name", ""),
                    epoch=training.get("epoch", 0),
                    total_epochs=training.get("total_epochs", 0),
                    loss=training.get("loss", 0),
                    accuracy=training.get("accuracy", 0),
                    samples_processed=training.get("samples_processed", 0),
                    started_at=training.get("started_at"),
                    eta_seconds=training.get("eta_seconds", 0),
                )
                if node.training.is_training:
                    node.status = NodeStatus.TRAINING

            return True

    def _check_hostinger_throttle(self, node: Node):
        """Hostinger: if CPU > 80% for 3 hours, throttle training."""
        self._hostinger_cpu_history.append(node.usage.cpu_percent)
        # Keep last 360 readings (1 per 30s = 3 hours)
        if len(self._hostinger_cpu_history) > 360:
            self._hostinger_cpu_history = self._hostinger_cpu_history[-360:]

        if len(self._hostinger_cpu_history) >= 360:
            avg = sum(self._hostinger_cpu_history) / len(self._hostinger_cpu_history)
            if avg > 80.0:
                node.status = NodeStatus.THROTTLED
                print(f"⚠️ Hostinger CPU throttled: avg {avg:.1f}% over 3hrs")

    async def set_websocket(self, node_id: str, ws: WebSocket) -> bool:
        """Attach a WebSocket connection to a node."""
        async with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                return False
            node.websocket = ws
            node.status = NodeStatus.ONLINE
            node.last_heartbeat = time.time()
            return True

    async def remove_websocket(self, node_id: str):
        """Remove WebSocket connection from a node."""
        async with self._lock:
            node = self.nodes.get(node_id)
            if node:
                node.websocket = None

    async def send_command(self, node_id: str, command: str,
                           params: Optional[Dict[str, Any]] = None) -> bool:
        """Send a command to a node via WebSocket."""
        node = self.nodes.get(node_id)
        if not node or not node.websocket:
            return False

        try:
            await node.websocket.send_json({
                "type": "command",
                "command": command,
                "params": params or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return True
        except Exception as e:
            print(f"❌ Failed to send command to {node.name}: {e}")
            node.websocket = None
            return False

    async def broadcast_command(self, command: str,
                                params: Optional[Dict[str, Any]] = None,
                                role_filter: Optional[NodeRole] = None) -> int:
        """Broadcast a command to all connected nodes."""
        sent = 0
        for node in self.nodes.values():
            if role_filter and node.role != role_filter:
                continue
            if await self.send_command(node.id, command, params):
                sent += 1
        return sent

    async def unregister(self, node_id: str) -> bool:
        """Remove a node from the registry."""
        async with self._lock:
            if node_id in self.nodes:
                node = self.nodes.pop(node_id)
                if node.websocket:
                    try:
                        await node.websocket.close()
                    except Exception:
                        pass
                print(f"📡 Node unregistered: {node.name} [{node_id}]")
                return True
            return False

    async def check_timeouts(self):
        """Mark nodes as offline if no heartbeat received."""
        now = time.time()
        async with self._lock:
            for node in self.nodes.values():
                if (node.status != NodeStatus.OFFLINE
                        and node.last_heartbeat
                        and now - node.last_heartbeat > self.HEARTBEAT_TIMEOUT):
                    node.status = NodeStatus.OFFLINE
                    print(f"💀 Node timeout: {node.name} [{node.id}]")

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all nodes as dicts."""
        return [n.to_dict() for n in self.nodes.values()]

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node as dict."""
        node = self.nodes.get(node_id)
        return node.to_dict() if node else None

    def get_primary(self) -> Optional[Node]:
        """Get the primary (RTX 5090) node."""
        for node in self.nodes.values():
            if node.role == NodeRole.PRIMARY:
                return node
        return None

    def get_online_nodes(self) -> List[Node]:
        """Get all online nodes (not offline)."""
        return [n for n in self.nodes.values()
                if n.status not in (NodeStatus.OFFLINE, NodeStatus.THROTTLED)]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all nodes."""
        nodes_list = list(self.nodes.values())
        return {
            "total_nodes": len(nodes_list),
            "online": sum(1 for n in nodes_list if n.status != NodeStatus.OFFLINE),
            "training": sum(1 for n in nodes_list if n.status == NodeStatus.TRAINING),
            "primary_connected": self.get_primary() is not None and self.get_primary().status != NodeStatus.OFFLINE,
            "total_gpu_vram_gb": sum(n.hardware.gpu.vram_gb for n in nodes_list),
            "total_ram_gb": sum(n.hardware.ram_gb for n in nodes_list),
            "nodes": self.get_all(),
        }


# ──────────── Singleton ────────────

node_registry = NodeRegistry()


async def _heartbeat_checker():
    """Background task to check node heartbeats."""
    while True:
        await node_registry.check_timeouts()
        await asyncio.sleep(15)
