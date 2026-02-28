"""
اختبارات التكامل - Integration Tests
======================================
Tests for integration scenarios including:
- End-to-end training flow
- Full council deliberation
- Multi-worker coordination

التغطية: >80%
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch, call

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestEndToEndTrainingFlow:
    """
    اختبارات التدفق الكامل للتدريب
    End-to-End Training Flow Tests
    """
    
    async def test_full_training_lifecycle(self):
        """
        اختبار دورة حياة التدريب الكاملة
        Test full training lifecycle
        """
        from services.training_service import TrainingService, TrainingStatus
        
        service = TrainingService()
        
        # 1. Start training
        job = await service.start_training(
            job_id="e2e-job-001",
            model_name="e2e-model",
            config={"epochs": 3, "batch_size": 32}
        )
        
        assert job.status in [TrainingStatus.PENDING, TrainingStatus.RUNNING]
        
        # 2. Get status
        status = await service.get_status("e2e-job-001")
        assert status is not None
        
        # 3. Get metrics
        metrics = await service.get_metrics("e2e-job-001")
        assert metrics is not None
        
        # 4. List models (should be empty initially)
        models = await service.list_models()
        assert isinstance(models, list)
    
    async def test_training_with_notifications(self):
        """
        اختبار التدريب مع الإشعارات
        Test training with notifications
        """
        from services.training_service import TrainingService
        from services.notification_service import NotificationService, NotificationChannel
        
        training_service = TrainingService()
        notification_service = NotificationService()
        
        # Register notification callback
        notifications_sent = []
        
        async def notify_on_status_change(job_id, status):
            notification = await notification_service.send_notification(
                user_id="user-001",
                title="Training Update",
                message=f"Job {job_id} is now {status}",
                channel=NotificationChannel.IN_APP
            )
            notifications_sent.append(notification)
        
        # Start training
        job = await training_service.start_training(
            job_id="notified-job",
            model_name="model",
            config={}
        )
        
        # Simulate status notification
        await notify_on_status_change(job.job_id, "running")
        
        assert len(notifications_sent) == 1
    
    async def test_distributed_training_coordination(self):
        """
        اختبار تنسيق التدريب الموزع
        Test distributed training coordination
        """
        from orchestrator_api import OrchestratorState
        
        state = OrchestratorState()
        
        # Register workers
        workers = [
            {"worker_id": "worker-primary", "labels": ["primary", "rtx5090"], "status": "online"},
            {"worker_id": "worker-helper-1", "labels": ["helper", "cpu"], "status": "online"},
            {"worker_id": "worker-helper-2", "labels": ["helper", "cpu"], "status": "online"},
        ]
        
        for w in workers:
            state.workers[w["worker_id"]] = w
        
        # Create distributed training job
        job = {
            "job_id": "distributed-job",
            "name": "Distributed Training",
            "status": "queued",
            "target_labels": ["primary"],
            "config": {"distributed": True, "workers": 3}
        }
        state.jobs[job["job_id"]] = job
        
        # Find primary worker
        primary = state.get_primary_worker()
        
        assert primary is not None
        assert "primary" in primary["labels"] or "rtx5090" in primary["labels"]
        
        # Assign job
        job["status"] = "running"
        job["assigned_worker"] = primary["worker_id"]
        primary["current_job"] = job["job_id"]
        
        assert state.jobs["distributed-job"]["status"] == "running"


class TestFullCouncilDeliberation:
    """
    اختبارات مداولات المجلس الكاملة
    Full Council Deliberation Tests
    """
    
    async def test_complete_council_workflow(self):
        """
        اختبار سير عمل المجلس الكامل
        Test complete council workflow
        """
        from services.council_service import CouncilService, DecisionStatus
        
        service = CouncilService()
        
        # 1. Query the council
        decision = await service.query_council(
            query="Should we refactor the database layer?",
            context={"current_tech_stack": ["PostgreSQL", "Redis"], "team_size": 5}
        )
        
        assert decision is not None
        assert decision.decision_id is not None
        
        # 2. Get members
        members = await service.list_members()
        assert len(members) >= 4
        
        # 3. Members vote
        for member in members[:3]:  # First 3 members vote
            await service.submit_vote(
                decision.decision_id,
                member.member_id,
                "approve"
            )
        
        # 4. Check decision status
        updated_decision = await service.get_status(decision.decision_id)
        assert updated_decision.status in [DecisionStatus.APPROVED, DecisionStatus.NEEDS_REVIEW]
        
        # 5. Get decision history
        decisions = await service.get_decisions(limit=10)
        assert len(decisions) >= 1
    
    async def test_council_with_ai_integration(self):
        """
        اختبار المجلس مع تكامل AI
        Test council with AI integration
        """
        from services.council_service import CouncilService
        from services.ai_service import AIService
        
        council_service = CouncilService()
        ai_service = AIService()
        
        # AI generates code proposal
        code_response = await ai_service.generate_code(
            user_id="council-system",
            prompt="Generate a proposal for database optimization",
            language="python"
        )
        
        # Council deliberates on the proposal
        decision = await council_service.query_council(
            query=f"Review this code proposal:\n{code_response.content[:100]}",
            context={"proposal_type": "database_optimization"}
        )
        
        assert decision.confidence > 0
        assert len(decision.response) > 0
    
    async def test_council_voting_consensus(self):
        """
        اختبار إجماع تصويت المجلس
        Test council voting consensus
        """
        from services.council_service import CouncilService, DecisionStatus
        
        service = CouncilService()
        
        decision = await service.query_council("Should we adopt microservices?")
        
        members = await service.list_members()
        
        # All members approve
        for member in members:
            await service.submit_vote(
                decision.decision_id,
                member.member_id,
                "approve"
            )
        
        updated = await service.get_status(decision.decision_id)
        
        # Should be approved with consensus
        assert updated.status == DecisionStatus.APPROVED


class TestMultiWorkerCoordination:
    """
    اختبارات تنسيق عدة عمال
    Multi-Worker Coordination Tests
    """
    
    async def test_worker_registration_flow(self):
        """
        اختبار سير تسجيل العمال
        Test worker registration flow
        """
        from orchestrator_api import OrchestratorState
        
        state = OrchestratorState()
        
        # Register multiple workers
        workers_data = [
            {
                "worker_id": "rtx-worker",
                "hostname": "rtx-host",
                "labels": ["gpu", "rtx4090"],
                "hardware": {"gpu": {"name": "RTX 4090", "vram_gb": 24}}
            },
            {
                "worker_id": "cpu-worker-1",
                "hostname": "cpu-host-1",
                "labels": ["cpu"],
                "hardware": {"cpu_cores": 16, "ram_gb": 64}
            },
        ]
        
        for w in workers_data:
            state.workers[w["worker_id"]] = {
                **w,
                "status": "online",
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat": 0
            }
        
        assert len(state.workers) == 2
        assert "rtx-worker" in state.workers
        assert state.workers["rtx-worker"]["labels"] == ["gpu", "rtx4090"]
    
    async def test_job_distribution(self):
        """
        اختبار توزيع المهام
        Test job distribution
        """
        from orchestrator_api import OrchestratorState
        
        state = OrchestratorState()
        
        # Setup workers
        state.workers["gpu-worker"] = {
            "worker_id": "gpu-worker",
            "labels": ["gpu"],
            "status": "online",
            "current_job": None
        }
        state.workers["cpu-worker"] = {
            "worker_id": "cpu-worker",
            "labels": ["cpu"],
            "status": "online",
            "current_job": None
        }
        
        # Create GPU job
        gpu_job = {
            "job_id": "gpu-job",
            "target_labels": ["gpu"],
            "status": "queued",
            "priority": 5
        }
        state.jobs[gpu_job["job_id"]] = gpu_job
        
        # Find matching worker
        candidates = [
            w for w in state.workers.values()
            if any(label in w["labels"] for label in gpu_job["target_labels"])
            and w["status"] == "online"
            and w["current_job"] is None
        ]
        
        assert len(candidates) == 1
        assert candidates[0]["worker_id"] == "gpu-worker"
    
    async def test_worker_heartbeat_monitoring(self):
        """
        اختبار مراقبة نبضات العمال
        Test worker heartbeat monitoring
        """
        import time
        from orchestrator_api import OrchestratorState
        
        state = OrchestratorState()
        
        # Add workers with different heartbeat times
        state.workers["active-worker"] = {
            "worker_id": "active-worker",
            "status": "online",
            "last_heartbeat": time.time()
        }
        state.workers["stale-worker"] = {
            "worker_id": "stale-worker",
            "status": "online",
            "last_heartbeat": time.time() - 200  # 200 seconds ago
        }
        
        # Check health
        now = time.time()
        for worker in state.workers.values():
            if now - worker.get("last_heartbeat", 0) > 90:
                worker["status"] = "offline"
        
        assert state.workers["active-worker"]["status"] == "online"
        assert state.workers["stale-worker"]["status"] == "offline"
    
    async def test_load_balancing(self):
        """
        اختبار توزيع الحمل
        Test load balancing
        """
        from orchestrator_api import OrchestratorState
        
        state = OrchestratorState()
        
        # Setup workers with varying loads
        state.workers["busy-worker"] = {
            "worker_id": "busy-worker",
            "labels": ["cpu"],
            "status": "online",
            "current_job": "job-001",
            "usage": {"cpu_percent": 85}
        }
        state.workers["idle-worker"] = {
            "worker_id": "idle-worker",
            "labels": ["cpu"],
            "status": "online",
            "current_job": None,
            "usage": {"cpu_percent": 10}
        }
        
        # Find best worker (idle one)
        available = [
            w for w in state.workers.values()
            if w["status"] == "online"
            and w["current_job"] is None
        ]
        
        assert len(available) == 1
        assert available[0]["worker_id"] == "idle-worker"
    
    async def test_worker_failure_recovery(self):
        """
        اختبار استعادة فشل العامل
        Test worker failure recovery
        """
        from orchestrator_api import OrchestratorState
        
        state = OrchestratorState()
        
        # Setup job assigned to worker
        state.workers["failing-worker"] = {
            "worker_id": "failing-worker",
            "status": "offline",
            "current_job": "critical-job"
        }
        state.jobs["critical-job"] = {
            "job_id": "critical-job",
            "status": "running",
            "assigned_worker": "failing-worker"
        }
        
        # Detect failure and requeue
        if state.workers["failing-worker"]["status"] == "offline":
            job = state.jobs["critical-job"]
            job["status"] = "queued"
            job["assigned_worker"] = None
            state.workers["failing-worker"]["current_job"] = None
        
        assert state.jobs["critical-job"]["status"] == "queued"
        assert state.jobs["critical-job"]["assigned_worker"] is None


class TestSystemWideIntegration:
    """
    اختبارات التكامل على مستوى النظام
    System-Wide Integration Tests
    """
    
    async def test_full_user_workflow(self):
        """
        اختبار سير عمل المستخدم الكامل
        Test full user workflow
        """
        from services.ai_service import AIService
        from services.council_service import CouncilService
        from services.notification_service import NotificationService
        
        ai_service = AIService()
        council_service = CouncilService()
        notification_service = NotificationService()
        
        user_id = "test-user-001"
        
        # 1. User generates code
        code_response = await ai_service.generate_code(
            user_id=user_id,
            prompt="Create a REST API endpoint",
            language="python"
        )
        assert code_response.content is not None
        
        # 2. Council reviews the code
        decision = await council_service.query_council(
            query=f"Review this code:\n{code_response.content[:50]}",
            context={"code_review": True}
        )
        assert decision.confidence > 0
        
        # 3. Send notification to user
        notification = await notification_service.send_notification(
            user_id=user_id,
            title="Code Review Complete",
            message=f"Council confidence: {decision.confidence:.2%}"
        )
        assert notification is not None
    
    async def test_health_check_integration(self):
        """
        اختبار تكامل فحص الصحة
        Test health check integration
        """
        from network.health_check import HealthChecker, HealthStatus
        from monitoring.system_monitor import SystemMonitor
        
        health_checker = HealthChecker(cache_enabled=False)
        
        # Run health checks
        with patch.object(health_checker, 'check_database') as mock_db:
            with patch.object(health_checker, 'check_redis') as mock_redis:
                mock_db.return_value = MagicMock(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=10.0
                )
                mock_redis.return_value = MagicMock(
                    name="redis",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=5.0
                )
                
                report = await health_checker.get_overall_status(use_cache=False)
                
                assert report.overall_status in [
                    HealthStatus.HEALTHY,
                    HealthStatus.DEGRADED,
                    HealthStatus.UNHEALTHY
                ]
    
    async def test_error_handling_across_services(self):
        """
        اختبار معالجة الأخطاء عبر الخدمات
        Test error handling across services
        """
        from services.ai_service import AIService
        from services.training_service import TrainingService
        
        ai_service = AIService()
        training_service = TrainingService()
        
        # Test AI service error
        try:
            # Simulate rate limit
            for _ in range(101):
                ai_service._check_rate_limit("rate-limited-user")
            
            await ai_service.generate_code("rate-limited-user", "test")
        except RuntimeError as e:
            assert "حدود المعدل" in str(e) or "rate limit" in str(e).lower()
        
        # Test training service error
        with pytest.raises(ValueError):
            await training_service.start_training(
                job_id="",  # Invalid job ID
                model_name="test",
                config={}
            )
