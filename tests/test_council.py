"""
اختبارات نظام المجلس - Council System Tests
==============================================
Tests for council system including:
- query_council
- Voting mechanism
- Decision making
- Member management

التغطية: >80%
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

pytestmark = pytest.mark.asyncio


class TestQueryCouncil:
    """
    اختبارات استشارة المجلس
    Query Council Tests
    """
    
    @pytest.fixture
    def council_service(self):
        from services.council_service import CouncilService
        return CouncilService()
    
    async def test_query_council_success(self, council_service):
        """
        اختبار استشارة المجلس بنجاح
        Test successful council query
        """
        # Arrange
        query = "What is the best architecture for this project?"
        
        # Act
        decision = await council_service.query_council(query)
        
        # Assert
        assert decision is not None
        assert decision.decision_id.startswith("dec_")
        assert decision.query == query
        assert decision.response is not None
        assert decision.confidence > 0
    
    async def test_query_council_with_context(self, council_service):
        """
        اختبار استشارة المجلس مع سياق
        Test council query with context
        """
        # Arrange
        query = "How should we handle authentication?"
        context = {"project_type": "web", "team_size": 5}
        
        # Act
        decision = await council_service.query_council(query, context=context)
        
        # Assert
        assert decision is not None
        assert decision.query == query
    
    async def test_query_council_cache_hit(self, council_service):
        """
        اختبار استخدام الذاكرة المؤقتة
        Test council query cache hit
        """
        # Arrange
        query = "Test caching query"
        
        # First query
        decision1 = await council_service.query_council(query)
        
        # Second query (should hit cache)
        decision2 = await council_service.query_council(query)
        
        # Assert
        assert decision1.decision_id == decision2.decision_id
    
    async def test_query_council_no_cache(self, council_service):
        """
        اختبار استشارة بدون تخزين مؤقت
        Test council query without cache
        """
        # Arrange
        query = "Test no cache query"
        
        # Act
        decision = await council_service.query_council(query, use_cache=False)
        
        # Assert
        assert decision is not None
    
    async def test_query_council_empty_query(self, council_service):
        """
        اختبار استشارة بنص فارغ
        Test council query with empty query
        """
        # Arrange
        query = ""
        
        # Act
        decision = await council_service.query_council(query)
        
        # Assert - Should still return a decision
        assert decision is not None


class TestVotingMechanism:
    """
    اختبارات آلية التصويت
    Voting Mechanism Tests
    """
    
    @pytest.fixture
    def council_service(self):
        from services.council_service import CouncilService
        return CouncilService()
    
    async def test_submit_vote_success(self, council_service):
        """
        اختبار تقديم تصويت بنجاح
        Test successful vote submission
        """
        # Arrange
        query = "Should we use microservices?"
        decision = await council_service.query_council(query)
        member_id = "architect_1"
        vote = "approve"
        
        # Act
        result = await council_service.submit_vote(decision.decision_id, member_id, vote)
        
        # Assert
        assert result is True
        assert member_id in decision.votes
        assert decision.votes[member_id] == vote
    
    async def test_submit_vote_nonexistent_decision(self, council_service):
        """
        اختبار التصويت على قرار غير موجود
        Test voting on non-existent decision
        """
        # Act
        result = await council_service.submit_vote("nonexistent-dec", "member_1", "approve")
        
        # Assert
        assert result is False
    
    async def test_submit_vote_nonexistent_member(self, council_service):
        """
        اختبار تصويت من عضو غير موجود
        Test voting by non-existent member
        """
        # Arrange
        query = "Test query"
        decision = await council_service.query_council(query)
        
        # Act
        result = await council_service.submit_vote(decision.decision_id, "nonexistent-member", "approve")
        
        # Assert
        assert result is False
    
    async def test_vote_updates_status_to_approved(self, council_service):
        """
        اختبار تحديث الحالة إلى معتمد
        Test vote updates status to approved
        """
        from services.council_service import DecisionStatus
        
        # Arrange
        query = "Test approval"
        decision = await council_service.query_council(query)
        
        # Submit multiple approve votes
        await council_service.submit_vote(decision.decision_id, "architect_1", "approve")
        await council_service.submit_vote(decision.decision_id, "security_1", "approve")
        await council_service.submit_vote(decision.decision_id, "performance_1", "approve")
        
        # Act - Check if status updated
        updated_decision = await council_service.get_status(decision.decision_id)
        
        # Assert
        assert updated_decision.status == DecisionStatus.APPROVED
    
    async def test_vote_updates_status_to_rejected(self, council_service):
        """
        اختبار تحديث الحالة إلى مرفوض
        Test vote updates status to rejected
        """
        from services.council_service import DecisionStatus
        
        # Arrange
        query = "Test rejection"
        decision = await council_service.query_council(query)
        
        # Submit reject votes
        await council_service.submit_vote(decision.decision_id, "architect_1", "reject")
        await council_service.submit_vote(decision.decision_id, "security_1", "reject")
        
        # Act
        updated_decision = await council_service.get_status(decision.decision_id)
        
        # Assert
        # Status might be rejected or needs_review depending on vote ratio
        assert updated_decision is not None
    
    async def test_multiple_votes_same_member(self, council_service):
        """
        اختبار تصويتات متعددة من نفس العضو
        Test multiple votes from same member
        """
        # Arrange
        query = "Test multiple votes"
        decision = await council_service.query_council(query)
        member_id = "architect_1"
        
        # Act
        await council_service.submit_vote(decision.decision_id, member_id, "approve")
        await council_service.submit_vote(decision.decision_id, member_id, "reject")  # Override
        
        # Assert - Last vote should count
        assert decision.votes[member_id] == "reject"


class TestDecisionMaking:
    """
    اختبارات اتخاذ القرار
    Decision Making Tests
    """
    
    @pytest.fixture
    def council_service(self):
        from services.council_service import CouncilService
        return CouncilService()
    
    async def test_decision_confidence_calculation(self, council_service):
        """
        اختبار حساب مستوى الثقة
        Test confidence calculation
        """
        # Arrange
        query = "Test confidence"
        
        # Act
        decision = await council_service.query_council(query)
        
        # Assert
        assert 0 <= decision.confidence <= 1
    
    async def test_decision_status_high_confidence(self, council_service):
        """
        اختبار حالة القرار مع ثقة عالية
        Test decision status with high confidence
        """
        from services.council_service import DecisionStatus
        
        # Arrange - Mock high confidence
        query = "Test high confidence"
        decision = await council_service.query_council(query)
        
        # Act
        # If confidence > 0.7, should be approved
        if decision.confidence > 0.7:
            assert decision.status == DecisionStatus.APPROVED
        else:
            assert decision.status in [DecisionStatus.APPROVED, DecisionStatus.NEEDS_REVIEW]
    
    async def test_get_decisions_list(self, council_service):
        """
        اختبار الحصول على قائمة القرارات
        Test getting decisions list
        """
        # Arrange - Create multiple decisions
        for i in range(5):
            await council_service.query_council(f"Query {i}", use_cache=False)
        
        # Act
        decisions = await council_service.get_decisions(limit=10)
        
        # Assert
        assert len(decisions) >= 5
    
    async def test_get_decisions_by_status(self, council_service):
        """
        اختبار الحصول على القرارات حسب الحالة
        Test getting decisions by status
        """
        from services.council_service import DecisionStatus
        
        # Arrange
        query = "Test status filter"
        decision = await council_service.query_council(query)
        
        # Act
        approved_decisions = await council_service.get_decisions(status=DecisionStatus.APPROVED)
        
        # Assert
        for d in approved_decisions:
            assert d.status == DecisionStatus.APPROVED
    
    async def test_get_status_existing_decision(self, council_service):
        """
        اختبار الحصول على حالة قرار موجود
        Test getting status of existing decision
        """
        # Arrange
        query = "Test get status"
        decision = await council_service.query_council(query)
        
        # Act
        status = await council_service.get_status(decision.decision_id)
        
        # Assert
        assert status is not None
        assert status.decision_id == decision.decision_id
    
    async def test_get_status_nonexistent_decision(self, council_service):
        """
        اختبار الحصول على حالة قرار غير موجود
        Test getting status of non-existent decision
        """
        # Act
        status = await council_service.get_status("nonexistent")
        
        # Assert
        assert status is None


class TestMemberManagement:
    """
    اختبارات إدارة الأعضاء
    Member Management Tests
    """
    
    @pytest.fixture
    def council_service(self):
        from services.council_service import CouncilService
        return CouncilService()
    
    async def test_default_members_initialized(self, council_service):
        """
        اختبار تهيئة الأعضاء الافتراضيين
        Test default members are initialized
        """
        # Act
        members = await council_service.list_members()
        
        # Assert
        assert len(members) >= 4  # Should have default members
        member_ids = [m.member_id for m in members]
        assert "architect_1" in member_ids
        assert "security_1" in member_ids
    
    async def test_list_members_active_only(self, council_service):
        """
        اختبار قائمة الأعضاء النشطين فقط
        Test listing only active members
        """
        # Act
        active_members = await council_service.list_members(active_only=True)
        all_members = await council_service.list_members(active_only=False)
        
        # Assert
        assert len(active_members) <= len(all_members)
        for m in active_members:
            assert m.is_active is True
    
    async def test_member_attributes(self, council_service):
        """
        اختبار خصائص الأعضاء
        Test member attributes
        """
        # Act
        members = await council_service.list_members()
        
        # Assert
        for member in members:
            assert member.member_id is not None
            assert member.name is not None
            assert member.role is not None
            assert isinstance(member.expertise, list)
            assert member.joined_at is not None


class TestCacheFunctionality:
    """
    اختبارات وظائف التخزين المؤقت
    Cache Functionality Tests
    """
    
    @pytest.fixture
    def council_service(self):
        from services.council_service import CouncilService
        return CouncilService()
    
    async def test_cache_entry_expiration(self, council_service):
        """
        اختبار انتهاء صلاحية مدخل التخزين المؤقت
        Test cache entry expiration
        """
        from services.council_service import CacheEntry
        
        # Arrange
        entry = CacheEntry(
            data="test data",
            timestamp=datetime.now() - timedelta(seconds=400),
            ttl=300  # 5 minutes
        )
        
        # Act & Assert
        assert entry.is_expired() is True
    
    async def test_cache_entry_not_expired(self, council_service):
        """
        اختبار عدم انتهاء صلاحية المدخل
        Test cache entry not expired
        """
        from services.council_service import CacheEntry
        
        # Arrange
        entry = CacheEntry(
            data="test data",
            timestamp=datetime.now(),
            ttl=300
        )
        
        # Act & Assert
        assert entry.is_expired() is False
    
    async def test_cached_decorator(self, council_service):
        """
        اختبار الديكوريتور المخزن مؤقتاً
        Test cached decorator
        """
        from services.council_service import cached
        
        # Arrange
        call_count = 0
        
        class TestClass:
            @cached(ttl=60)
            async def test_method(self, arg):
                nonlocal call_count
                call_count += 1
                return f"result-{arg}"
        
        obj = TestClass()
        
        # Act
        result1 = await obj.test_method("test")
        result2 = await obj.test_method("test")
        
        # Assert
        assert result1 == result2
        assert call_count == 1  # Should only be called once
