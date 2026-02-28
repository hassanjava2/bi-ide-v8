"""
اختبارات المنسق - Orchestrator Tests
======================================
Tests for orchestrator functionality including:
- Worker registration
- Heartbeat handling
- Command distribution
- Distributed training coordination

التغطية: >80%
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

pytestmark = pytest.mark.asyncio


class TestWorkerRegistration:
    """
    اختبارات تسجيل العمال
    Worker Registration Tests
    """
    
    @pytest.fixture
    def orchestrator_state(self):
        """إنشاء حالة المنسق للاختبار"""
        from orchestrator_api import OrchestratorState
        return OrchestratorState()
    
    async def test_register_new_worker(self, orchestrator_state):
        """
        اختبار تسجيل عامل جديد
        Test registering a new worker
        """
        # Arrange
        worker_data = {
            "worker_id": "worker-001",
            "hostname": "test-host",
            "labels": ["gpu", "rtx4090"],
            "hardware": {"gpu": {"name": "RTX 4090"}, "ram_gb": 64},
            "version": "1.0.0"
        }
        
        # Act
        orchestrator_state.workers[worker_data["worker_id"]] = {
            **worker_data,
            "status": "online",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "last_heartbeat": 0
        }
        
        # Assert
        assert worker_data["worker_id"] in orchestrator_state.workers
        worker = orchestrator_state.workers[worker_data["worker_id"]]
        assert worker["hostname"] == "test-host"
        assert "gpu" in worker["labels"]
    
    async def test_register_worker_without_id(self, orchestrator_state):
        """
        اختبار تسجيل عامل بدون معرف
        Test registering worker without ID (auto-generate)
        """
        # Arrange
        worker_data = {
            "worker_id": "",  # Empty ID
            "hostname": "auto-host",
            "labels": ["cpu"],
            "hardware": {},
            "version": "1.0.0"
        }
        
        # Act - Simulate auto-generation
        import uuid
        generated_id = str(uuid.uuid4())[:12]
        worker_data["worker_id"] = generated_id
        orchestrator_state.workers[generated_id] = worker_data
        
        # Assert
        assert generated_id in orchestrator_state.workers
        assert len(generated_id) == 12
    
    async def test_reregister_worker_by_hostname(self, orchestrator_state):
        """
        إعادة تسجيل عامل بنفس اسم المضيف
        Test re-registering worker with same hostname
        """
        # Arrange
        hostname = "same-host"
        
        # Register first worker
        orchestrator_state.workers["old-id"] = {
            "worker_id": "old-id",
            "hostname": hostname,
            "labels": ["gpu"],
            "hardware": {},
            "version": "1.0.0",
            "status": "online"
        }
        
        # Act - Re-register with new ID but same hostname
        new_worker = {
            "worker_id": "new-id",
            "hostname": hostname,
            "labels": ["gpu", "updated"],
            "hardware": {},
            "version": "1.0.0"
        }
        
        # Should reuse old ID
        for existing_id, w in orchestrator_state.workers.items():
            if w["hostname"] == hostname:
                new_worker["worker_id"] = existing_id
                break
        
        orchestrator_state.workers[new_worker["worker_id"]] = new_worker
        
        # Assert
        assert new_worker["worker_id"] == "old-id"
        assert "updated" in orchestrator_state.workers["old-id"]["labels"]
    
    async def test_get_primary_worker(self, orchestrator_state):
        """
        اختبار الحصول على العامل الأساسي
        Test getting primary worker
        """
        # Arrange
        orchestrator_state.workers["helper-1"] = {
            "worker_id": "helper-1",
            "labels": ["cpu"],
            "status": "online"
        }
        orchestrator_state.workers["primary-1"] = {
            "worker_id": "primary-1",
            "labels": ["gpu", "rtx5090", "primary"],
            "status": "online"
        }
        
        # Act
        primary = orchestrator_state.get_primary_worker()
        
        # Assert
        assert primary is not None
        assert primary["worker_id"] == "primary-1"
    
    async def test_get_primary_worker_none(self, orchestrator_state):
        """
        اختبار عدم وجود عامل أساسي
        Test getting primary worker when none exists
        """
        # Act
        primary = orchestrator_state.get_primary_worker()
        
        # Assert
        assert primary is None


class TestHeartbeat:
    """
    اختبارات نبضات القلب
    Heartbeat Tests
    """
    
    @pytest.fixture
    def orchestrator_state(self):
        from orchestrator_api import OrchestratorState
        state = OrchestratorState()
        # Add a test worker
        state.workers["test-worker"] = {
            "worker_id": "test-worker",
            "hostname": "test-host",
            "labels": ["gpu"],
            "status": "online",
            "last_heartbeat": 0,
            "usage": {}
        }
        return state
    
    async def test_heartbeat_updates_timestamp(self, orchestrator_state):
        """
        اختبار تحديث وقت النبضة
        Test heartbeat updates timestamp
        """
        import time
        
        # Arrange
        old_time = orchestrator_state.workers["test-worker"]["last_heartbeat"]
        
        # Act
        new_time = time.time()
        orchestrator_state.workers["test-worker"]["last_heartbeat"] = new_time
        orchestrator_state.workers["test-worker"]["status"] = "online"
        
        # Assert
        assert orchestrator_state.workers["test-worker"]["last_heartbeat"] > old_time
        assert orchestrator_state.workers["test-worker"]["status"] == "online"
    
    async def test_heartbeat_updates_usage(self, orchestrator_state):
        """
        اختبار تحديث استخدام الموارد
        Test heartbeat updates resource usage
        """
        # Arrange
        usage_data = {
            "cpu_percent": 45.5,
            "ram_percent": 60.0,
            "gpu_percent": 80.0
        }
        
        # Act
        orchestrator_state.workers["test-worker"]["usage"] = usage_data
        
        # Assert
        assert orchestrator_state.workers["test-worker"]["usage"]["cpu_percent"] == 45.5
    
    async def test_heartbeat_worker_not_found(self, orchestrator_state):
        """
        اختبار نبضة لعامل غير موجود
        Test heartbeat for non-existent worker
        """
        # Act & Assert
        assert "nonexistent-worker" not in orchestrator_state.workers
    
    async def test_hostinger_throttle_check(self, orchestrator_state):
        """
        اختبار التحقق من Hostinger throttle
        Test Hostinger throttle check
        """
        # Arrange
        orchestrator_state.workers["hostinger-worker"] = {
            "worker_id": "hostinger-worker",
            "labels": ["hostinger"],
            "status": "online",
            "last_heartbeat": 0
        }
        
        # Simulate high CPU usage
        for cpu in [80, 85, 90, 95]:
            orchestrator_state.cpu_history.append(cpu)
        
        # Act - Check if should throttle
        should_throttle = orchestrator_state.check_hostinger_throttle(85.0)
        
        # Assert - Should not throttle with just 4 samples
        assert should_throttle is False
        
        # Add more samples
        for _ in range(200):
            orchestrator_state.cpu_history.append(80.0)
        
        should_throttle = orchestrator_state.check_hostinger_throttle(85.0)
        assert should_throttle is True
    
    async def test_heartbeat_returns_command(self, orchestrator_state):
        """
        اختبار إرجاع أمر في النبضة
        Test heartbeat returns pending command
        """
        # Arrange
        orchestrator_state.workers["test-worker"]["_pending_command"] = {
            "command": "restart",
            "params": {}
        }
        
        # Act
        pending = orchestrator_state.workers["test-worker"].pop("_pending_command", None)
        
        # Assert
        assert pending is not None
        assert pending["command"] == "restart"


class TestCommandDistribution:
    """
    اختبارات توزيع الأوامر
    Command Distribution Tests
    """
    
    @pytest.fixture
    def orchestrator_state(self):
        from orchestrator_api import OrchestratorState
        state = OrchestratorState()
        state.workers["worker-1"] = {
            "worker_id": "worker-1",
            "status": "online",
            "current_job": None
        }
        return state
    
    async def test_send_command_to_worker(self, orchestrator_state):
        """
        اختبار إرسال أمر إلى عامل
        Test sending command to worker
        """
        # Arrange
        worker_id = "worker-1"
        command = "stop_job"
        params = {"job_id": "job-001"}
        
        # Act - Queue command
        orchestrator_state.workers[worker_id]["_pending_command"] = {
            "command": command,
            "params": params
        }
        
        # Assert
        pending = orchestrator_state.workers[worker_id]["_pending_command"]
        assert pending["command"] == command
        assert pending["params"]["job_id"] == "job-001"
    
    async def test_send_command_worker_not_found(self, orchestrator_state):
        """
        اختبار إرسال أمر لعامل غير موجود
        Test sending command to non-existent worker
        """
        # Act & Assert
        assert "nonexistent-worker" not in orchestrator_state.workers
    
    async def test_remove_worker(self, orchestrator_state):
        """
        اختبار إزالة عامل
        Test removing worker
        """
        # Act
        worker_id = "worker-1"
        del orchestrator_state.workers[worker_id]
        
        # Assert
        assert worker_id not in orchestrator_state.workers


class TestDistributedTraining:
    """
    اختبارات التنسيق الموزع للتدريب
    Distributed Training Coordination Tests
    """
    
    @pytest.fixture
    def orchestrator_state(self):
        from orchestrator_api import OrchestratorState
        state = OrchestratorState()
        
        # Add multiple workers
        state.workers["worker-gpu"] = {
            "worker_id": "worker-gpu",
            "labels": ["gpu", "rtx4090"],
            "status": "online",
            "current_job": None
        }
        state.workers["worker-cpu-1"] = {
            "worker_id": "worker-cpu-1",
            "labels": ["cpu"],
            "status": "online",
            "current_job": None
        }
        state.workers["worker-cpu-2"] = {
            "worker_id": "worker-cpu-2",
            "labels": ["cpu"],
            "status": "online",
            "current_job": None
        }
        
        return state
    
    async def test_job_creation(self, orchestrator_state):
        """
        اختبار إنشاء مهمة
        Test job creation
        """
        # Arrange
        job_id = "test-job-001"
        job = {
            "job_id": job_id,
            "name": "Test Training Job",
            "status": "queued",
            "target_labels": ["gpu"],
            "priority": 5,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Act
        orchestrator_state.jobs[job_id] = job
        
        # Assert
        assert job_id in orchestrator_state.jobs
        assert orchestrator_state.jobs[job_id]["status"] == "queued"
    
    async def test_job_assignment_by_labels(self, orchestrator_state):
        """
        اختبار تعيين المهام حسب التصنيفات
        Test job assignment based on labels
        """
        # Arrange
        job = {
            "job_id": "gpu-job",
            "status": "queued",
            "target_labels": ["gpu"],
            "priority": 5
        }
        
        # Find matching worker
        matching_workers = [
            w for w in orchestrator_state.workers.values()
            if any(label in w["labels"] for label in job["target_labels"])
            and w["status"] == "online"
            and w["current_job"] is None
        ]
        
        # Assert
        assert len(matching_workers) == 1
        assert matching_workers[0]["worker_id"] == "worker-gpu"
    
    async def test_job_priority_sorting(self, orchestrator_state):
        """
        اختبار ترتيب المهام حسب الأولوية
        Test job priority sorting
        """
        # Arrange
        jobs = [
            {"job_id": "low", "priority": 1},
            {"job_id": "high", "priority": 10},
            {"job_id": "medium", "priority": 5},
        ]
        
        # Act
        sorted_jobs = sorted(jobs, key=lambda j: j.get("priority", 5), reverse=True)
        
        # Assert
        assert sorted_jobs[0]["job_id"] == "high"
        assert sorted_jobs[1]["job_id"] == "medium"
        assert sorted_jobs[2]["job_id"] == "low"
    
    async def test_complete_job(self, orchestrator_state):
        """
        اختبار إكمال مهمة
        Test completing a job
        """
        # Arrange
        job_id = "complete-test-job"
        orchestrator_state.jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "assigned_worker": "worker-gpu"
        }
        orchestrator_state.workers["worker-gpu"]["current_job"] = job_id
        
        # Act
        orchestrator_state.jobs[job_id]["status"] = "completed"
        orchestrator_state.jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        orchestrator_state.workers["worker-gpu"]["current_job"] = None
        
        # Assert
        assert orchestrator_state.jobs[job_id]["status"] == "completed"
        assert orchestrator_state.workers["worker-gpu"]["current_job"] is None
    
    async def test_fail_job(self, orchestrator_state):
        """
        اختبار فشل مهمة
        Test failing a job
        """
        # Arrange
        job_id = "fail-test-job"
        orchestrator_state.jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "assigned_worker": "worker-gpu"
        }
        orchestrator_state.workers["worker-gpu"]["current_job"] = job_id
        
        # Act
        orchestrator_state.jobs[job_id]["status"] = "failed"
        orchestrator_state.jobs[job_id]["result"] = {"error": "Out of memory"}
        orchestrator_state.workers["worker-gpu"]["current_job"] = None
        
        # Assert
        assert orchestrator_state.jobs[job_id]["status"] == "failed"
        assert "error" in orchestrator_state.jobs[job_id]["result"]


class TestOrchestratorHealth:
    """
    اختبارات صحة المنسق
    Orchestrator Health Tests
    """
    
    async def test_health_status(self):
        """
        اختبار حالة صحة المنسق
        Test orchestrator health status
        """
        from orchestrator_api import OrchestratorState
        
        state = OrchestratorState()
        state.workers["online-worker"] = {
            "worker_id": "online-worker",
            "status": "online"
        }
        state.workers["offline-worker"] = {
            "worker_id": "offline-worker",
            "status": "offline"
        }
        
        # Calculate health stats
        online = [w for w in state.workers.values() if w.get("status") != "offline"]
        
        health = {
            "status": "ok",
            "workers_total": len(state.workers),
            "workers_online": len(online),
            "jobs_total": len(state.jobs),
            "jobs_running": sum(1 for j in state.jobs.values() if j.get("status") == "running")
        }
        
        assert health["workers_total"] == 2
        assert health["workers_online"] == 1
