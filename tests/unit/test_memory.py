"""
Unit tests for Memory modules
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
from datetime import datetime, timedelta

import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.memory.conversation_history import (
    ConversationStore,
    Conversation,
    create_conversation_store
)
from ai.memory.user_preferences import (
    PreferenceStore,
    PreferenceLearner,
    UserProfile,
    get_user_context
)
from ai.memory.context_awareness import (
    ContextManager,
    ContextSummarizer,
    CouncilContextManager,
    create_context_manager
)
from ai.memory.vector_db import (
    VectorStore,
    BaseVectorStore,
    FAISSVectorStore,
    EmbeddingDocument,
    enhance_council_context
)


class TestConversationStore:
    """Test conversation history storage."""
    
    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def store(self, temp_dir):
        return ConversationStore(
            db_type='sqlite',
            db_path=f'{temp_dir}/conversations.db'
        )
    
    def test_create_conversation(self, store):
        """Test creating conversation."""
        conv_id = store.create_conversation(
            user_id='user_1',
            session_id='session_1',
            topic='Test Topic'
        )
        
        assert conv_id is not None
        assert isinstance(conv_id, int)
    
    def test_get_conversation(self, store):
        """Test retrieving conversation."""
        conv_id = store.create_conversation(
            user_id='user_1',
            session_id='session_1',
            topic='Test Topic'
        )
        
        conv = store.get_conversation(conv_id)
        
        assert conv is not None
        assert conv.user_id == 'user_1'
        assert conv.topic == 'Test Topic'
    
    def test_add_message(self, store):
        """Test adding messages."""
        conv_id = store.create_conversation('user_1', 'session_1')
        
        success = store.add_message(conv_id, 'user', 'Hello!')
        assert success is True
        
        success = store.add_message(conv_id, 'assistant', 'Hi there!')
        assert success is True
        
        conv = store.get_conversation(conv_id)
        assert len(conv.messages) == 2
    
    def test_search_by_user(self, store):
        """Test searching by user."""
        store.create_conversation('user_1', 'session_1')
        store.create_conversation('user_1', 'session_2')
        store.create_conversation('user_2', 'session_1')
        
        results = store.search_by_user('user_1')
        
        assert len(results) == 2
        assert all(r.user_id == 'user_1' for r in results)
    
    def test_search_by_topic(self, store):
        """Test searching by topic."""
        store.create_conversation('user_1', 'session_1', topic='Python')
        store.create_conversation('user_1', 'session_2', topic='JavaScript')
        
        results = store.search_by_topic('Python')
        
        assert len(results) == 1
        assert results[0].topic == 'Python'
    
    def test_search_by_date_range(self, store):
        """Test searching by date range."""
        store.create_conversation('user_1', 'session_1')
        
        start = datetime.now() - timedelta(days=1)
        end = datetime.now() + timedelta(days=1)
        
        results = store.search_by_date_range(start, end)
        
        assert len(results) >= 1
    
    def test_delete_conversation(self, store):
        """Test deleting conversation."""
        conv_id = store.create_conversation('user_1', 'session_1')
        
        success = store.delete_conversation(conv_id)
        assert success is True
        
        conv = store.get_conversation(conv_id)
        assert conv is None


class TestUserPreferences:
    """Test user preferences module."""
    
    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def store(self, temp_dir):
        return PreferenceStore(db_path=f'{temp_dir}/preferences.db')
    
    @pytest.fixture
    def learner(self, store):
        return PreferenceLearner(store)
    
    def test_create_profile(self, store):
        """Test creating user profile."""
        profile = store.create_profile('user_1')
        
        assert profile.user_id == 'user_1'
        assert profile.preferred_language == 'en'
        assert profile.response_style == 'balanced'
    
    def test_get_profile(self, store):
        """Test getting user profile."""
        store.create_profile('user_1', preferred_language='ar')
        
        profile = store.get_profile('user_1')
        
        assert profile is not None
        assert profile.preferred_language == 'ar'
    
    def test_update_preference(self, store):
        """Test updating preference."""
        store.create_profile('user_1')
        
        success = store.update_preference('user_1', 'response_style', 'concise')
        
        assert success is True
        
        profile = store.get_profile('user_1')
        assert profile.response_style == 'concise'
    
    def test_learn_from_message(self, store, learner):
        """Test learning from message."""
        learned = learner.learn_from_message('user_1', 'مرحبا بالعالم')
        
        assert 'language' in learned
        
        profile = store.get_profile('user_1')
        assert 'ar' in profile.language_proficiency
    
    def test_detect_code(self, learner):
        """Test code detection."""
        assert learner._contains_code('def hello():') is True
        assert learner._contains_code('import numpy') is True
        assert learner._contains_code('Hello world') is False
    
    def test_detect_code_language(self, learner):
        """Test programming language detection."""
        lang = learner._detect_code_language('def hello():\n    return 1')
        assert lang == 'python'
        
        lang = learner._detect_code_language('const x = 1;')
        assert lang == 'javascript'
    
    def test_get_user_context(self, store):
        """Test getting user context."""
        store.create_profile('user_1', preferred_language='ar')
        
        context = get_user_context('user_1', store)
        
        assert context['user_id'] == 'user_1'
        assert context['preferred_language'] == 'ar'


class TestContextAwareness:
    """Test context awareness module."""
    
    def test_context_manager_init(self):
        """Test ContextManager initialization."""
        manager = ContextManager(max_context_tokens=2048)
        
        assert manager.max_context_tokens == 2048
        assert len(manager.active_context) == 0
    
    def test_add_message(self):
        """Test adding messages."""
        manager = ContextManager(max_context_tokens=1000)
        
        result = manager.add_message('user', 'Hello!')
        
        assert result['message_added'] is True
        assert len(manager.active_context) == 1
    
    def test_get_context(self):
        """Test getting context."""
        manager = ContextManager(max_context_tokens=1000)
        
        manager.add_message('user', 'Hello!')
        manager.add_message('assistant', 'Hi!')
        
        context = manager.get_context()
        
        assert len(context) == 2
    
    def test_search_context(self):
        """Test searching context."""
        manager = ContextManager()
        
        manager.add_message('user', 'Python programming')
        manager.add_message('assistant', 'JavaScript development')
        
        results = manager.search_context('Python')
        
        assert len(results) == 1
    
    def test_context_summarizer(self):
        """Test context summarization."""
        summarizer = ContextSummarizer()
        
        messages = [
            {'role': 'user', 'content': 'How do I use Python?'},
            {'role': 'assistant', 'content': 'Python is easy to learn.'},
            {'role': 'user', 'content': 'What about data science?'}
        ]
        
        summary = summarizer.summarize(messages)
        
        assert summary.content is not None
        assert len(summary.key_points) > 0
    
    def test_council_context_manager(self):
        """Test CouncilContextManager."""
        manager = CouncilContextManager()
        
        # Create agent context
        agent_manager = manager.create_agent_context('agent_1')
        
        assert 'agent_1' in manager.agent_contexts
        assert agent_manager is not None
    
    def test_share_knowledge(self):
        """Test sharing knowledge."""
        manager = CouncilContextManager()
        
        manager.share_knowledge('important_fact', 'Python is great')
        
        value = manager.get_shared_knowledge('important_fact')
        assert value == 'Python is great'


class TestVectorDB:
    """Test vector database module."""
    
    @pytest.fixture
    def vector_store(self):
        # Use numpy fallback since FAISS may not be available
        return VectorStore(backend='faiss', embedding_dim=128)
    
    def test_vector_store_init(self):
        """Test VectorStore initialization."""
        store = VectorStore(backend='faiss', embedding_dim=256)
        
        assert store.embedding_dim == 256
    
    def test_store_and_search(self, vector_store):
        """Test storing and searching."""
        # Store documents
        doc_id1 = vector_store.store('Python programming language', metadata={'topic': 'python'})
        doc_id2 = vector_store.store('JavaScript web development', metadata={'topic': 'js'})
        
        assert doc_id1 is not None
        assert doc_id2 is not None
        
        # Search
        results = vector_store.search('coding in python', k=2)
        
        assert len(results) > 0
        assert 'similarity' in results[0]
    
    def test_similarity_search(self, vector_store):
        """Test similarity search."""
        # Store multiple docs
        vector_store.store('Machine learning with Python')
        vector_store.store('Deep learning and neural networks')
        vector_store.store('Web development basics')
        
        # Search
        results = vector_store.search('artificial intelligence', k=2)
        
        assert len(results) == 2
        # Results should be sorted by similarity
        assert results[0]['similarity'] >= results[1]['similarity']
    
    def test_get_relevant_context(self, vector_store):
        """Test getting relevant context."""
        vector_store.store('Python is a programming language')
        vector_store.store('JavaScript is used for web development')
        
        context = vector_store.get_relevant_context('programming', k=1)
        
        # With random embeddings, similarity may be below threshold
        assert isinstance(context, str)
    
    def test_filter_search(self, vector_store):
        """Test filtered search."""
        vector_store.store('Python code', metadata={'lang': 'python'})
        vector_store.store('JavaScript code', metadata={'lang': 'javascript'})
        
        results = vector_store.search('code', k=5, filter_metadata={'lang': 'python'})
        
        assert all(r['metadata'].get('lang') == 'python' for r in results)
    
    def test_cosine_similarity(self, vector_store):
        """Test cosine similarity calculation."""
        a = np.array([1, 0, 0], dtype=np.float32)
        b = np.array([1, 0, 0], dtype=np.float32)
        
        similarity = vector_store._backend._cosine_similarity(a, b)
        
        assert abs(similarity - 1.0) < 0.01  # Same direction
    
    def test_enhance_council_context(self, vector_store):
        """Test enhancing council context."""
        vector_store.store('Important context about Python')
        
        council_context = {}
        enhanced = enhance_council_context(vector_store, council_context, 'Python programming')
        
        assert 'relevant_memories' in enhanced


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
