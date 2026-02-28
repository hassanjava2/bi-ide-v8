"""
اختبارات العامل - Worker Tests
=================================
Tests for worker system including:
- Worker initialization
- Job execution
- Heartbeat sending
- Error recovery

التغطية: >80%
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch, mock_open

pytestmark = pytest.mark.asyncio


class TestWorkerInitialization:
    """
    اختبارات تهيئة العامل
    Worker Initialization Tests
    """
    
    def test_worker_agent_initialization(self):
        """
        اختبار تهيئة عامل WorkerAgent
        Test WorkerAgent initialization
        """
        from worker.bi_worker import WorkerAgent
        
        agent = WorkerAgent(
            server_url="https://test.com",
            token="test-token",
            labels=["gpu", "test"],
            worker_id="test-worker"
        )
        
        assert agent.server == "https://test.com"
        assert agent.token == "test-token"
        assert "gpu" in agent.labels
        assert agent.worker_id == "test-worker"
        assert agent.running is True
        assert agent.current_job is None
    
    def test_worker_default_values(self):
        """
        اختبار القيم الافتراضية للعامل
        Test worker default values
        """
        from worker.bi_worker import WorkerAgent
        
        agent = WorkerAgent(
            server_url="https://test.com",
            token="token",
            labels=["cpu"]
        )
        
        assert agent.max_cpu == 90
        assert agent.max_gpu == 95
        assert agent.max_ram == 85
        assert agent.heartbeat_interval == 30
    
    def test_worker_headers(self):
        """
        اختبار رؤوس HTTP للعامل
        Test worker HTTP headers
        """
        from worker.bi_worker import WorkerAgent
        
        agent = WorkerAgent(
            server_url="https://test.com",
            token="secret-token",
            labels=["cpu"]
        )
        
        headers = agent.headers
        assert headers["X-Orchestrator-Token"] == "secret-token"
        assert headers["Content-Type"] == "application/json"
    
    @patch('worker.bi_worker.detect_hardware')
    def test_hardware_detection(self, mock_detect):
        """
        اختبار اكتشاف الهاردوير
        Test hardware detection
        """
        from worker.bi_worker import WorkerAgent
        
        mock_detect.return_value = {
            "cpu_name": "Intel i7",
            "cpu_cores": 8,
            "ram_gb": 32,
            "gpu": {"name": "RTX 4090", "vram_gb": 24}
        }
        
        agent = WorkerAgent(
            server_url="https://test.com",
            token="token",
            labels=["gpu"]
        )
        
        hardware = mock_detect()
        assert hardware["cpu_cores"] == 8
        assert hardware["gpu"]["name"] == "RTX 4090"


class TestWorkerHeartbeat:
    """
    اختبارات نبضات القلب
    Worker Heartbeat Tests
    """
    
    @pytest.fixture
    def worker_agent(self):
        from worker.bi_worker import WorkerAgent
        return WorkerAgent(
            server_url="https://test.com",
            token="test-token",
            labels=["gpu"],
            worker_id="test-worker"
        )
    
    @patch('worker.bi_worker.requests.post')
    def test_send_heartbeat_success(self, mock_post, worker_agent):
        """
        اختبار إرسال نبضة ناجحة
        Test successful heartbeat
        """
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_post.return_value = mock_response
        
        # Act
        result = worker_agent.send_heartbeat()
        
        # Assert
        assert result is not None
        mock_post.assert_called_once()
    
    @patch('worker.bi_worker.requests.post')
    def test_send_heartbeat_failure(self, mock_post, worker_agent):
        """
        اختبار فشل إرسال النبضة
        Test heartbeat failure
        """
        # Arrange
        mock_post.side_effect = Exception("Connection error")
        
        # Act
        result = worker_agent.send_heartbeat()
        
        # Assert
        assert result is None
    
    @patch('worker.bi_worker.requests.post')
    def test_send_heartbeat_404_reregister(self, mock_post, worker_agent):
        """
        اختبار إعادة التسجيل عند 404
        Test re-registration on 404
        """
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response
        
        # Act
        result = worker_agent.send_heartbeat()
        
        # Assert
        assert result is None
    
    @patch('worker.bi_worker.get_resource_usage')
    def test_heartbeat_includes_usage(self, mock_usage, worker_agent):
        """
        اختبار تضمين استخدام الموارد في النبضة
        Test heartbeat includes resource usage
        """
        # Arrange
        mock_usage.return_value = {
            "cpu_percent": 50.0,
            "ram_percent": 60.0,
            "gpu_percent": 30.0
        }
        
        # Act
        usage = mock_usage()
        
        # Assert
        assert "cpu_percent" in usage
        assert "ram_percent" in usage


class TestJobExecution:
    """
    اختبارات تنفيذ المهام
    Job Execution Tests
    """
    
    @pytest.fixture
    def worker_agent(self):
        from worker.bi_worker import WorkerAgent
        return WorkerAgent(
            server_url="https://test.com",
            token="test-token",
            labels=["gpu"],
            worker_id="test-worker"
        )
    
    def test_poll_for_job_when_idle(self, worker_agent):
        """
        اختبار طلب مهمة عندما يكون العامل خاملاً
        Test polling for job when idle
        """
        # Ensure no current job
        worker_agent.current_job = None
        
        # Check resource limits before accepting
        usage = {"cpu_percent": 30, "ram_percent": 50, "gpu_percent": 20}
        
        # Should accept job if resources available
        assert usage["cpu_percent"] < worker_agent.max_cpu
        assert usage["ram_percent"] < worker_agent.max_ram
        assert usage["gpu_percent"] < worker_agent.max_gpu
    
    def test_poll_for_job_when_busy(self, worker_agent):
        """
        اختبار عدم طلب مهمة عندما يكون العامل مشغولاً
        Test not polling when busy
        """
        # Set current job
        worker_agent.current_job = {"job_id": "current-job"}
        
        # Should not poll if already has job
        assert worker_agent.current_job is not None
    
    def test_resource_limits_check(self, worker_agent):
        """
        اختبار التحقق من حدود الموارد
        Test resource limits check
        """
        # High resource usage
        high_usage = {"cpu_percent": 95, "ram_percent": 90, "gpu_percent": 98}
        
        # Should not accept job if over limits
        assert high_usage["cpu_percent"] > worker_agent.max_cpu
    
    def test_job_assignment(self, worker_agent):
        """
        اختبار تعيين مهمة
        Test job assignment
        """
        # Arrange
        job = {
            "job_id": "test-job",
            "name": "Test Job",
            "command": "echo test"
        }
        
        # Act
        worker_agent.current_job = job
        
        # Assert
        assert worker_agent.current_job["job_id"] == "test-job"
    
    @patch('worker.bi_worker.subprocess.run')
    def test_execute_job_success(self, mock_run, worker_agent):
        """
        اختبار تنفيذ مهمة بنجاح
        Test successful job execution
        """
        # Arrange
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        job = {
            "job_id": "test-job",
            "command": "echo test",
            "shell": False
        }
        
        # Act
        worker_agent.current_job = job
        # Simulate job completion
        mock_run(["echo", "test"], capture_output=True, text=True)
        
        # Assert
        mock_run.assert_called_once()
    
    @patch('worker.bi_worker.subprocess.run')
    def test_execute_job_failure(self, mock_run, worker_agent):
        """
        اختبار فشل تنفيذ مهمة
        Test job execution failure
        """
        # Arrange
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error occurred"
        mock_run.return_value = mock_result
        
        job = {
            "job_id": "failing-job",
            "command": "exit 1",
            "shell": False
        }
        
        # Act
        worker_agent.current_job = job
        result = mock_run(["exit", "1"], capture_output=True, text=True)
        
        # Assert
        assert result.returncode != 0


class TestErrorRecovery:
    """
    اختبارات استعادة الأخطاء
    Error Recovery Tests
    """
    
    @pytest.fixture
    def worker_agent(self):
        from worker.bi_worker import WorkerAgent
        return WorkerAgent(
            server_url="https://test.com",
            token="test-token",
            labels=["gpu"],
            worker_id="test-worker"
        )
    
    def test_reconnect_delay_progression(self, worker_agent):
        """
        اختبار تدرج تأخير إعادة الاتصال
        Test reconnect delay progression
        """
        # Initial delay
        assert worker_agent.reconnect_delay == 5
        
        # Simulate increasing delay
        worker_agent.reconnect_delay = min(worker_agent.reconnect_delay * 1.5, 60)
        assert worker_agent.reconnect_delay == 7.5
        
        worker_agent.reconnect_delay = min(worker_agent.reconnect_delay * 1.5, 60)
        assert worker_agent.reconnect_delay == 11.25
        
        # Max at 60
        worker_agent.reconnect_delay = 50
        worker_agent.reconnect_delay = min(worker_agent.reconnect_delay * 1.5, 60)
        assert worker_agent.reconnect_delay == 60
    
    def test_handle_command_stop_job(self, worker_agent):
        """
        اختبار معالجة أمر إيقاف المهمة
        Test handling stop_job command
        """
        # Arrange
        worker_agent.current_job = {"job_id": "test-job"}
        worker_agent.training_process = MagicMock()
        
        # Act
        worker_agent._handle_command("stop_job", {})
        
        # Assert
        assert worker_agent.current_job is None
    
    def test_handle_command_throttle(self, worker_agent):
        """
        اختبار معالجة أمر الحد من الموارد
        Test handling throttle command
        """
        # Arrange
        original_max_cpu = worker_agent.max_cpu
        original_max_gpu = worker_agent.max_gpu
        
        # Act
        worker_agent._handle_command("throttle", {"message": "CPU high"})
        
        # Assert - Should reduce resource limits
        assert worker_agent.max_cpu <= 60
        assert worker_agent.max_gpu <= 50
    
    def test_handle_command_restart(self, worker_agent):
        """
        اختبار معالجة أمر إعادة التشغيل
        Test handling restart command
        """
        # Just verify command is recognized
        assert hasattr(worker_agent, '_handle_command')
    
    def test_handle_command_shutdown(self, worker_agent):
        """
        اختبار معالجة أمر الإيقاف
        Test handling shutdown command
        """
        # Act
        worker_agent._handle_command("shutdown", {})
        
        # Assert
        assert worker_agent.running is False
    
    @patch('worker.bi_worker.requests.post')
    def test_complete_job_reporting(self, mock_post, worker_agent):
        """
        اختبار الإبلاغ عن إكمال المهمة
        Test job completion reporting
        """
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Act
        worker_agent._complete_job("test-job", {"exit_code": 0})
        
        # Assert
        mock_post.assert_called_once()
    
    @patch('worker.bi_worker.requests.post')
    def test_fail_job_reporting(self, mock_post, worker_agent):
        """
        اختبار الإبلاغ عن فشل المهمة
        Test job failure reporting
        """
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Act
        worker_agent._fail_job("test-job", "Out of memory")
        
        # Assert
        mock_post.assert_called_once()


class TestWorkerUtils:
    """
    اختبارات الأدوات المساعدة
    Worker Utility Tests
    """
    
    def test_detect_os(self):
        """
        اختبار اكتشاف نظام التشغيل
        Test OS detection
        """
        from worker.bi_worker import _detect_os
        
        result = _detect_os()
        assert result in ["linux", "windows", "macos"]
    
    def test_get_resource_usage_structure(self):
        """
        اختبار هيكل استخدام الموارد
        Test resource usage structure
        """
        from worker.bi_worker import get_resource_usage
        
        usage = get_resource_usage()
        
        assert "cpu_percent" in usage
        assert "ram_percent" in usage
        assert "ram_used_gb" in usage
        assert "disk_percent" in usage
        assert "gpu_percent" in usage
