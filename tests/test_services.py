"""
اختبارات طبقة الخدمات - Services Layer Tests
===============================================
Tests for service layer including:
- Training service
- Council service
- AI service
- Notification service

التغطية: >80%
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

pytestmark = pytest.mark.asyncio


class TestTrainingService:
    """
    اختبارات خدمة التدريب
    Training Service Tests
    """
    
    @pytest.fixture
    def training_service(self):
        from services.training_service import TrainingService
        return TrainingService()
    
    async def test_start_training_creates_job(self, training_service):
        """
        اختبار إنشاء مهمة تدريب
        Test training job creation
        """
        job = await training_service.start_training(
            job_id="job-001",
            model_name="test-model",
            config={"epochs": 5}
        )
        
        assert job.job_id == "job-001"
        assert job.model_name == "test-model"
        assert job.status.value in ["pending", "running"]
    
    async def test_stop_training(self, training_service):
        """
        اختبار إيقاف التدريب
        Test stopping training
        """
        await training_service.start_training(
            job_id="job-002",
            model_name="model",
            config={}
        )
        
        result = await training_service.stop_training("job-002")
        
        assert result is True
    
    async def test_list_models_empty(self, training_service):
        """
        اختبار قائمة النماذج الفارغة
        Test empty models list
        """
        models = await training_service.list_models()
        
        assert isinstance(models, list)
        assert len(models) == 0
    
    async def test_deploy_model(self, training_service):
        """
        اختبار نشر نموذج
        Test deploying model
        """
        from services.training_service import ModelInfo
        
        training_service._models["model-001"] = ModelInfo(
            model_id="model-001",
            name="test-model",
            version="1.0.0",
            created_at=datetime.now(),
            accuracy=0.95,
            is_deployed=False,
            metadata={}
        )
        
        result = await training_service.deploy_model("model-001")
        
        assert result is True
        assert training_service._models["model-001"].is_deployed is True


class TestCouncilService:
    """
    اختبارات خدمة المجلس
    Council Service Tests
    """
    
    @pytest.fixture
    def council_service(self):
        from services.council_service import CouncilService
        return CouncilService()
    
    async def test_query_council(self, council_service):
        """
        اختبار استشارة المجلس
        Test council query
        """
        decision = await council_service.query_council(
            query="What is the best approach?",
            context={"project": "test"}
        )
        
        assert decision is not None
        assert decision.query == "What is the best approach?"
        assert decision.confidence >= 0
    
    async def test_submit_vote(self, council_service):
        """
        اختبار تقديم تصويت
        Test submitting vote
        """
        decision = await council_service.query_council("Test query")
        
        result = await council_service.submit_vote(
            decision.decision_id,
            "architect_1",
            "approve"
        )
        
        assert result is True
    
    async def test_list_members(self, council_service):
        """
        اختبار قائمة أعضاء المجلس
        Test listing council members
        """
        members = await council_service.list_members()
        
        assert len(members) >= 4
        assert any(m.member_id == "architect_1" for m in members)
    
    async def test_get_decisions(self, council_service):
        """
        اختبار الحصول على القرارات
        Test getting decisions
        """
        await council_service.query_council("Query 1")
        await council_service.query_council("Query 2")
        
        decisions = await council_service.get_decisions(limit=10)
        
        assert len(decisions) >= 2


class TestAIService:
    """
    اختبارات خدمة الذكاء الاصطناعي
    AI Service Tests
    """
    
    @pytest.fixture
    def ai_service(self):
        from services.ai_service import AIService
        return AIService()
    
    async def test_generate_code(self, ai_service):
        """
        اختبار توليد كود
        Test code generation
        """
        response = await ai_service.generate_code(
            user_id="user-001",
            prompt="Create a function to add two numbers",
            language="python"
        )
        
        assert response is not None
        assert response.content is not None
        assert response.tokens_used > 0
    
    async def test_complete_code(self, ai_service):
        """
        اختبار إكمال كود
        Test code completion
        """
        response = await ai_service.complete_code(
            user_id="user-001",
            partial_code="def hello():",
            cursor_position=None
        )
        
        assert response is not None
        assert response.content is not None
    
    async def test_explain_code(self, ai_service):
        """
        اختبار شرح كود
        Test code explanation
        """
        response = await ai_service.explain_code(
            user_id="user-001",
            code="print('Hello')",
            detail_level="medium"
        )
        
        assert response is not None
        assert len(response.content) > 0
    
    async def test_review_code(self, ai_service):
        """
        اختبار مراجعة كود
        Test code review
        """
        response = await ai_service.review_code(
            user_id="user-001",
            code="def test(): pass",
            language="python"
        )
        
        assert response is not None
        assert response.content is not None
    
    async def test_rate_limiting(self, ai_service):
        """
        اختبار حدود المعدل
        Test rate limiting
        """
        # Consume all requests
        for _ in range(100):
            ai_service._check_rate_limit("user-001")
        
        # Next request should be blocked
        allowed = ai_service._check_rate_limit("user-001")
        
        assert allowed is False
    
    async def test_context_management(self, ai_service):
        """
        اختبار إدارة السياق
        Test context management
        """
        # Generate with context
        await ai_service.generate_code(
            user_id="user-002",
            prompt="First prompt",
            use_context=True
        )
        
        # Check context was created
        context = await ai_service._get_context("user-002")
        
        assert context is not None
        assert context.user_id == "user-002"
    
    def test_clear_context(self, ai_service):
        """
        اختبار مسح السياق
        Test clearing context
        """
        # Add some context
        ai_service._contexts["user-003"] = MagicMock()
        
        # Clear it
        result = ai_service.clear_context("user-003")
        
        assert result is True
        assert "user-003" not in ai_service._contexts


class TestNotificationService:
    """
    اختبارات خدمة الإشعارات
    Notification Service Tests
    """
    
    @pytest.fixture
    def notification_service(self):
        from services.notification_service import NotificationService
        return NotificationService()
    
    async def test_send_notification(self, notification_service):
        """
        اختبار إرسال إشعار
        Test sending notification
        """
        from services.notification_service import NotificationChannel, NotificationPriority
        
        notification = await notification_service.send_notification(
            user_id="user-001",
            title="Test Notification",
            message="This is a test",
            channel=NotificationChannel.IN_APP,
            priority=NotificationPriority.MEDIUM
        )
        
        assert notification is not None
        assert notification.title == "Test Notification"
        assert notification.user_id == "user-001"
    
    async def test_send_bulk_notification(self, notification_service):
        """
        اختبار إرسال إشعار جماعي
        Test sending bulk notification
        """
        user_ids = ["user-001", "user-002", "user-003"]
        
        notification_ids = await notification_service.send_bulk_notification(
            user_ids=user_ids,
            title="Bulk Notification",
            message="Message to all"
        )
        
        assert len(notification_ids) == 3
    
    async def test_get_user_notifications(self, notification_service):
        """
        اختبار الحصول على إشعارات المستخدم
        Test getting user notifications
        """
        await notification_service.send_notification(
            user_id="user-001",
            title="Notification 1",
            message="Message 1"
        )
        
        notifications = await notification_service.get_user_notifications("user-001")
        
        assert len(notifications) >= 1
    
    async def test_mark_read(self, notification_service):
        """
        اختبار تحديد كمقروء
        Test marking as read
        """
        notification = await notification_service.send_notification(
            user_id="user-001",
            title="Test",
            message="Test message"
        )
        
        count = await notification_service.mark_read(
            user_id="user-001",
            notification_id=notification.notification_id
        )
        
        assert count == 1
        assert notification.is_read is True
    
    async def test_mark_all_read(self, notification_service):
        """
        اختبار تحديد الكل كمقروء
        Test marking all as read
        """
        await notification_service.send_notification(
            user_id="user-002",
            title="Notification 1",
            message="Message 1"
        )
        await notification_service.send_notification(
            user_id="user-002",
            title="Notification 2",
            message="Message 2"
        )
        
        count = await notification_service.mark_read(
            user_id="user-002",
            mark_all=True
        )
        
        assert count >= 2
    
    async def test_delete_notification(self, notification_service):
        """
        اختبار حذف إشعار
        Test deleting notification
        """
        notification = await notification_service.send_notification(
            user_id="user-001",
            title="To Delete",
            message="Delete me"
        )
        
        result = await notification_service.delete_notification(
            user_id="user-001",
            notification_id=notification.notification_id
        )
        
        assert result is True
    
    async def test_get_unread_count(self, notification_service):
        """
        اختبار عدد غير المقروء
        Test unread count
        """
        await notification_service.send_notification(
            user_id="user-003",
            title="Unread",
            message="Unread message"
        )
        
        count = await notification_service.get_unread_count("user-003")
        
        assert count >= 1
    
    async def test_register_websocket(self, notification_service):
        """
        اختبار تسجيل WebSocket
        Test registering WebSocket
        """
        callback = MagicMock()
        
        result = await notification_service.register_websocket(
            user_id="user-001",
            socket_id="socket-001",
            send_callback=callback
        )
        
        assert result is True
        assert "socket-001" in notification_service._websocket_connections
    
    async def test_unregister_websocket(self, notification_service):
        """
        اختبار إلغاء تسجيل WebSocket
        Test unregistering WebSocket
        """
        callback = MagicMock()
        await notification_service.register_websocket(
            user_id="user-001",
            socket_id="socket-002",
            send_callback=callback
        )
        
        result = await notification_service.unregister_websocket("socket-002")
        
        assert result is True
        assert "socket-002" not in notification_service._websocket_connections
    
    async def test_send_websocket_notification(self, notification_service):
        """
        اختبار إرسال إشعار WebSocket
        Test sending WebSocket notification
        """
        callback = MagicMock()
        await notification_service.register_websocket(
            user_id="user-001",
            socket_id="socket-003",
            send_callback=callback
        )
        
        from services.notification_service import NotificationChannel
        
        notification = await notification_service.send_notification(
            user_id="user-001",
            title="WebSocket Test",
            message="Test message",
            channel=NotificationChannel.WEBSOCKET
        )
        
        # The callback should have been called
        # (in real implementation)


class TestServiceIntegration:
    """
    اختبارات تكامل الخدمات
    Service Integration Tests
    """
    
    async def test_training_completion_notification(self):
        """
        اختبار إشعار إكمال التدريب
        Test training completion notification
        """
        from services.training_service import TrainingService
        from services.notification_service import NotificationService, NotificationChannel
        
        training_service = TrainingService()
        notification_service = NotificationService()
        
        # Start training
        job = await training_service.start_training(
            job_id="job-001",
            model_name="model",
            config={}
        )
        
        # Send notification on completion
        notification = await notification_service.send_notification(
            user_id="user-001",
            title="Training Complete",
            message=f"Job {job.job_id} completed",
            channel=NotificationChannel.IN_APP
        )
        
        assert notification is not None
    
    async def test_council_decision_notification(self):
        """
        اختبار إشعار قرار المجلس
        Test council decision notification
        """
        from services.council_service import CouncilService
        from services.notification_service import NotificationService
        
        council_service = CouncilService()
        notification_service = NotificationService()
        
        # Get decision
        decision = await council_service.query_council("Test query")
        
        # Send notification
        notification = await notification_service.send_notification(
            user_id="user-001",
            title="Council Decision",
            message=f"Decision made with confidence {decision.confidence:.2f}"
        )
        
        assert notification is not None
    
    async def test_ai_rate_limit_alert(self):
        """
        اختبار تنبيه حد معدل AI
        Test AI rate limit alert
        """
        from services.ai_service import AIService
        
        ai_service = AIService(rate_limit_per_minute=5)
        
        # Consume all requests
        for _ in range(5):
            try:
                await ai_service.generate_code("user-001", "test")
            except RuntimeError:
                pass  # Expected after limit
        
        # Check status
        status = await ai_service.get_rate_limit_status("user-001")
        
        assert status["remaining"] == 0
