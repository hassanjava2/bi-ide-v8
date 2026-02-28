"""
اختبارات ذاكرة الذكاء الاصطناعي - AI Memory Tests
==================================================
Tests for AI memory system including:
- Vector DB operations
- Context awareness
- Conversation history
- User preferences

التغطية: >80%
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

pytestmark = pytest.mark.asyncio


class TestVectorDBOperations:
    """
    اختبارات عمليات قاعدة بيانات المتجهات
    Vector DB Operations Tests
    """
    
    @pytest.fixture
    def vector_store(self):
        from ai.memory.vector_db import VectorStore
        return VectorStore(backend='faiss', embedding_dim=128)
    
    def test_store_document(self, vector_store):
        """
        اختبار تخزين مستند
        Test storing a document
        """
        # Arrange
        text = "This is a test document about Python programming"
        metadata = {"category": "programming", "language": "en"}
        
        # Act
        doc_id = vector_store.store(text, metadata=metadata)
        
        # Assert
        assert doc_id is not None
        assert isinstance(doc_id, str)
    
    def test_search_similar_documents(self, vector_store):
        """
        اختبار البحث عن مستندات مشابهة
        Test searching similar documents
        """
        # Arrange
        docs = [
            "Python is a great programming language",
            "JavaScript is used for web development",
            "Machine learning uses Python extensively",
            "Docker is a containerization platform",
        ]
        
        for doc in docs:
            vector_store.store(doc, metadata={"type": "test"})
        
        # Act
        query = "programming with Python"
        results = vector_store.search(query, k=2)
        
        # Assert
        assert len(results) <= 2
        assert all("similarity" in r for r in results)
    
    def test_search_with_filter(self, vector_store):
        """
        اختبار البحث مع تصفية
        Test search with metadata filter
        """
        # Arrange
        vector_store.store("Python doc", metadata={"category": "python"})
        vector_store.store("JavaScript doc", metadata={"category": "js"})
        vector_store.store("Another Python doc", metadata={"category": "python"})
        
        # Act
        results = vector_store.search("programming", filter_metadata={"category": "python"})
        
        # Assert
        assert len(results) > 0
        for r in results:
            assert r['metadata']['category'] == 'python'
    
    def test_delete_document(self, vector_store):
        """
        اختبار حذف مستند
        Test deleting a document
        """
        # Arrange
        doc_id = vector_store.store("Document to delete")
        
        # Act
        result = vector_store.delete(doc_id)
        
        # Assert
        assert result is True
    
    def test_delete_nonexistent_document(self, vector_store):
        """
        اختبار حذف مستند غير موجود
        Test deleting non-existent document
        """
        # Act
        result = vector_store.delete("nonexistent-id")
        
        # Assert
        assert result is False
    
    def test_get_relevant_context(self, vector_store):
        """
        اختبار الحصول على سياق ذي صلة
        Test getting relevant context
        """
        # Arrange
        vector_store.store("Python was created by Guido van Rossum")
        vector_store.store("Python is widely used in data science")
        vector_store.store("Java is a different programming language")
        
        # Act
        context = vector_store.get_relevant_context("Python history", k=2)
        
        # Assert
        assert isinstance(context, str)
        assert len(context) > 0
    
    def test_cosine_similarity(self, vector_store):
        """
        اختبار حساب تشابه جيب التمام
        Test cosine similarity calculation
        """
        # Arrange
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        
        # Act
        similarity = vector_store._backend._cosine_similarity(a, b)
        
        # Assert
        assert abs(similarity - 1.0) < 0.001  # Should be nearly identical
    
    def test_embed_text(self, vector_store):
        """
        اختبار تضمين النص
        Test text embedding
        """
        # Act
        embedding = vector_store.embed_text("Test text")
        
        # Assert
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == vector_store.embedding_dim


class TestContextAwareness:
    """
    اختبارات الوعي بالسياق
    Context Awareness Tests
    """
    
    @pytest.fixture
    def context_manager(self):
        from ai.memory.context_awareness import ContextManager
        return ContextManager(max_context_tokens=1000)
    
    def test_add_message(self, context_manager):
        """
        اختبار إضافة رسالة
        Test adding a message
        """
        # Act
        result = context_manager.add_message("user", "Hello, how are you?")
        
        # Assert
        assert result['message_added'] is True
        assert result['active_messages'] == 1
    
    def test_get_context(self, context_manager):
        """
        اختبار الحصول على السياق
        Test getting context
        """
        # Arrange
        context_manager.add_message("user", "What is Python?")
        context_manager.add_message("assistant", "Python is a programming language")
        context_manager.add_message("user", "Tell me more")
        
        # Act
        context = context_manager.get_context()
        
        # Assert
        assert len(context) > 0
        assert all('role' in msg for msg in context)
    
    def test_get_context_with_max_tokens(self, context_manager):
        """
        اختبار الحصول على السياق مع حد للرموز
        Test getting context with max tokens limit
        """
        # Arrange - Add many messages
        for i in range(20):
            context_manager.add_message("user", f"Message {i} with some content")
        
        # Act
        context = context_manager.get_context(max_tokens=500)
        
        # Assert
        assert len(context) <= 20
    
    def test_search_context(self, context_manager):
        """
        اختبار البحث في السياق
        Test searching context
        """
        # Arrange
        context_manager.add_message("user", "I want to learn about Python")
        context_manager.add_message("assistant", "Python is great for beginners")
        context_manager.add_message("user", "What about JavaScript?")
        
        # Act
        results = context_manager.search_context("Python")
        
        # Assert
        assert len(results) > 0
        assert any("Python" in msg['content'] for msg in results)
    
    def test_clear_context(self, context_manager):
        """
        اختبار مسح السياق
        Test clearing context
        """
        # Arrange
        context_manager.add_message("user", "Test message")
        
        # Act
        context_manager.clear_context()
        
        # Assert
        stats = context_manager.get_context_stats()
        assert stats['active_messages'] == 0
    
    def test_get_context_stats(self, context_manager):
        """
        اختبار الحصول على إحصائيات السياق
        Test getting context statistics
        """
        # Arrange
        context_manager.add_message("user", "Message 1")
        context_manager.add_message("assistant", "Response 1")
        
        # Act
        stats = context_manager.get_context_stats()
        
        # Assert
        assert 'active_messages' in stats
        assert 'estimated_tokens' in stats
        assert stats['active_messages'] == 2
    
    def test_context_summarization(self, context_manager):
        """
        اختبار تلخيص السياق
        Test context summarization
        """
        # Arrange - Add many messages to trigger summarization
        for i in range(50):
            context_manager.add_message("user" if i % 2 == 0 else "assistant", 
                                        f"Message {i} with content about Python programming")
        
        # Act
        stats = context_manager.get_context_stats()
        
        # Assert
        # Should have created summaries
        assert stats['summarized_segments'] > 0


class TestConversationHistory:
    """
    اختبارات سجل المحادثات
    Conversation History Tests
    """
    
    @pytest.fixture
    def conversation_store(self, tmp_path):
        from ai.memory.conversation_history import ConversationStore
        db_path = str(tmp_path / "test_conversations.db")
        return ConversationStore(db_type='sqlite', db_path=db_path)
    
    def test_create_conversation(self, conversation_store):
        """
        اختبار إنشاء محادثة
        Test creating a conversation
        """
        # Act
        conv_id = conversation_store.create_conversation(
            user_id="user_001",
            session_id="session_001",
            topic="Python Programming"
        )
        
        # Assert
        assert conv_id is not None
        assert isinstance(conv_id, int)
    
    def test_get_conversation(self, conversation_store):
        """
        اختبار الحصول على محادثة
        Test getting a conversation
        """
        # Arrange
        conv_id = conversation_store.create_conversation(
            user_id="user_001",
            session_id="session_001",
            topic="Test Topic"
        )
        
        # Act
        conversation = conversation_store.get_conversation(conv_id)
        
        # Assert
        assert conversation is not None
        assert conversation.id == conv_id
        assert conversation.topic == "Test Topic"
    
    def test_add_message(self, conversation_store):
        """
        اختبار إضافة رسالة
        Test adding a message
        """
        # Arrange
        conv_id = conversation_store.create_conversation(
            user_id="user_001",
            session_id="session_001"
        )
        
        # Act
        result = conversation_store.add_message(
            conv_id, "user", "Hello, can you help me?"
        )
        
        # Assert
        assert result is True
        
        # Verify message was added
        conversation = conversation_store.get_conversation(conv_id)
        assert len(conversation.messages) == 1
    
    def test_search_by_user(self, conversation_store):
        """
        اختبار البحث حسب المستخدم
        Test searching by user
        """
        # Arrange
        conversation_store.create_conversation(
            user_id="user_001", session_id="s1", topic="Topic 1"
        )
        conversation_store.create_conversation(
            user_id="user_001", session_id="s2", topic="Topic 2"
        )
        conversation_store.create_conversation(
            user_id="user_002", session_id="s3", topic="Topic 3"
        )
        
        # Act
        results = conversation_store.search_by_user("user_001")
        
        # Assert
        assert len(results) == 2
    
    def test_search_by_topic(self, conversation_store):
        """
        اختبار البحث حسب الموضوع
        Test searching by topic
        """
        # Arrange
        conversation_store.create_conversation(
            user_id="user_001", session_id="s1", topic="Python Programming"
        )
        conversation_store.create_conversation(
            user_id="user_001", session_id="s2", topic="JavaScript Development"
        )
        
        # Act
        results = conversation_store.search_by_topic("Python")
        
        # Assert
        assert len(results) == 1
        assert "Python" in results[0].topic
    
    def test_delete_conversation(self, conversation_store):
        """
        اختبار حذف محادثة
        Test deleting a conversation
        """
        # Arrange
        conv_id = conversation_store.create_conversation(
            user_id="user_001", session_id="s1"
        )
        
        # Act
        result = conversation_store.delete_conversation(conv_id)
        
        # Assert
        assert result is True
        assert conversation_store.get_conversation(conv_id) is None
    
    def test_export_conversations(self, conversation_store, tmp_path):
        """
        اختبار تصدير المحادثات
        Test exporting conversations
        """
        # Arrange
        conversation_store.create_conversation(
            user_id="user_001", session_id="s1", topic="Export Test"
        )
        export_path = str(tmp_path / "export.json")
        
        # Act
        result = conversation_store.export_conversations(
            user_id="user_001", filepath=export_path
        )
        
        # Assert
        assert result == export_path
        import os
        assert os.path.exists(export_path)


class TestUserPreferences:
    """
    اختبارات تفضيلات المستخدم
    User Preferences Tests
    """
    
    @pytest.fixture
    def preference_store(self, tmp_path):
        from ai.memory.user_preferences import PreferenceStore
        db_path = str(tmp_path / "test_preferences.db")
        return PreferenceStore(db_path=db_path)
    
    def test_create_profile(self, preference_store):
        """
        اختبار إنشاء ملف شخصي
        Test creating a profile
        """
        # Act
        profile = preference_store.create_profile("user_001")
        
        # Assert
        assert profile.user_id == "user_001"
        assert profile.preferred_language == "en"
    
    def test_get_profile(self, preference_store):
        """
        اختبار الحصول على ملف شخصي
        Test getting a profile
        """
        # Arrange
        preference_store.create_profile("user_001", preferred_language="ar")
        
        # Act
        profile = preference_store.get_profile("user_001")
        
        # Assert
        assert profile is not None
        assert profile.preferred_language == "ar"
    
    def test_update_preference(self, preference_store):
        """
        اختبار تحديث تفضيل
        Test updating a preference
        """
        # Arrange
        preference_store.create_profile("user_001")
        
        # Act
        result = preference_store.update_preference(
            "user_001", "response_style", "detailed"
        )
        
        # Assert
        assert result is True
        profile = preference_store.get_profile("user_001")
        assert profile.response_style == "detailed"
    
    def test_delete_profile(self, preference_store):
        """
        اختبار حذف ملف شخصي
        Test deleting a profile
        """
        # Arrange
        preference_store.create_profile("user_001")
        
        # Act
        result = preference_store.delete_profile("user_001")
        
        # Assert
        assert result is True
        assert preference_store.get_profile("user_001") is None
    
    def test_preference_learner_detect_language(self, preference_store):
        """
        اختبار تعلم تفضيل اللغة
        Test learning language preference
        """
        from ai.memory.user_preferences import PreferenceLearner
        
        learner = PreferenceLearner(preference_store)
        
        # Act
        learned = learner.learn_from_message("user_001", "مرحبا كيف حالك؟")
        
        # Assert
        assert 'language' in learned
        assert learned['language'] == 'ar'
    
    def test_preference_learner_detect_code(self, preference_store):
        """
        اختبار تعلم تفضيل الكود
        Test learning code preference
        """
        from ai.memory.user_preferences import PreferenceLearner
        
        learner = PreferenceLearner(preference_store)
        
        # Act
        learned = learner.learn_from_message(
            "user_001", 
            "def hello():\n    print('Hello')"
        )
        
        # Assert
        assert 'code_language' in learned
        assert learned['code_language'] == 'python'
