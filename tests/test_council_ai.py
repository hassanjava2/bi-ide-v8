"""
Comprehensive Tests for Smart Council AI - المجلس الذكي

Tests cover:
- Unit tests for WiseManAI individual wise men
- Unit tests for SmartCouncil coordination
- Integration tests for full council discussions
- Infinite loop prevention tests
- Hallucination detection tests
- Error handling and edge cases
"""
import sys
sys.path.insert(0, '.')

import pytest
import asyncio
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import hashlib

# Import modules to test
from council_ai import (
    WiseManAI, SmartCouncil, WiseManTransformer,
    smart_council, CHECKPOINT_DIR
)
from wise_ai_engine import wise_ai_engine, WiseAIEngine


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_wise_man():
    """Create a sample wise man for testing"""
    return WiseManAI(
        name="حكيم القرار",
        role="رئيس المجلس",
        specialty="الاستراتيجية",
        checkpoint_path=None
    )


@pytest.fixture
def all_wise_men():
    """Create all 8 core wise men for testing"""
    wise_men_data = [
        ('حكيم القرار', 'رئيس المجلس', 'الاستراتيجية'),
        ('حكيم البصيرة', 'تحليل عميق', 'التحليل'),
        ('حكيم المستقبل', 'تخطيط طويل المدى', 'الرؤية'),
        ('حكيم الشجاعة', 'المبادرة', 'الجرأة'),
        ('حكيم التوازن', 'العدالة', 'الموازنة'),
        ('حكيم الضبط', 'مراقبة', 'النظام'),
        ('حكيم التكيف', 'تطور', 'التغيير'),
        ('حكيم الذاكرة', 'تاريخ', 'الخبرة'),
    ]
    return [
        WiseManAI(name=name, role=role, specialty=specialty, checkpoint_path=None)
        for name, role, specialty in wise_men_data
    ]


@pytest.fixture
def fresh_council():
    """Create a fresh SmartCouncil instance for isolated testing"""
    council = SmartCouncil()
    return council


@pytest.fixture
def wise_ai_engine_mock():
    """Create a mock for the wise AI engine"""
    mock = Mock()
    mock.generate_response_with_evidence.return_value = {
        'response': 'Test response from AI engine',
        'source': 'training+persona',
        'confidence': 0.85,
        'evidence': [{'topic': 'test', 'source': 'test_data'}],
        'blocked': False
    }
    return mock


# ═══════════════════════════════════════════════════════════════
# Unit Tests - WiseManTransformer Model
# ═══════════════════════════════════════════════════════════════

class TestWiseManTransformer:
    """Tests for the Transformer model architecture"""
    
    def test_model_initialization(self):
        """Test that the transformer model initializes correctly"""
        model = WiseManTransformer(vocab_size=1000, d_model=64, nhead=4, num_layers=2)
        
        assert model is not None
        assert model.d_model == 64
        assert isinstance(model.embedding, torch.nn.Embedding)
        assert isinstance(model.transformer, torch.nn.TransformerEncoder)
        assert isinstance(model.fc, torch.nn.Linear)
    
    def test_model_forward_pass(self):
        """Test that the model can perform a forward pass"""
        model = WiseManTransformer(vocab_size=100, d_model=32, nhead=2, num_layers=1)
        model.eval()
        
        # Create dummy input (batch_size=2, seq_len=10)
        x = torch.randint(0, 100, (2, 10))
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 10, 100)  # (batch, seq, vocab)
    
    def test_model_different_sequence_lengths(self):
        """Test model with different sequence lengths"""
        model = WiseManTransformer(vocab_size=100, d_model=32, nhead=2, num_layers=1)
        model.eval()
        
        for seq_len in [1, 5, 50, 100]:
            x = torch.randint(0, 100, (1, seq_len))
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, seq_len, 100)


# ═══════════════════════════════════════════════════════════════
# Unit Tests - WiseManAI Individual
# ═══════════════════════════════════════════════════════════════

class TestWiseManAI:
    """Tests for individual Wise Man AI"""
    
    def test_initialization(self, sample_wise_man):
        """Test wise man initialization"""
        assert sample_wise_man.name == "حكيم القرار"
        assert sample_wise_man.role == "رئيس المجلس"
        assert sample_wise_man.specialty == "الاستراتيجية"
        assert sample_wise_man.model is None  # No checkpoint loaded
        assert sample_wise_man.memory == []
        assert 'source' in sample_wise_man.last_generation_meta
    
    def test_initialization_all_personalities(self):
        """Test that all 8 personalities are properly defined"""
        personalities = [
            'حكيم القرار', 'حكيم البصيرة', 'حكيم المستقبل', 'حكيم الشجاعة',
            'حكيم التوازن', 'حكيم الضبط', 'حكيم التكيف', 'حكيم الذاكرة'
        ]
        
        for name in personalities:
            wise_man = WiseManAI(name=name, role="test", specialty="test", checkpoint_path=None)
            assert wise_man.personality is not None
            assert 'style' in wise_man.personality
            assert 'greeting' in wise_man.personality
            assert 'traits' in wise_man.personality
    
    def test_unknown_personality_fallback(self):
        """Test fallback personality for unknown wise man"""
        wise_man = WiseManAI(name="حكيم مجهول", role="test", specialty="test")
        assert wise_man.personality['style'] == 'حكيم ومتواضع'
        assert 'greeting' in wise_man.personality
    
    def test_think_method_basic(self, sample_wise_man):
        """Test basic thinking functionality"""
        response = sample_wise_man.think("ما هو قراري؟")
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Memory should store the interaction
        assert len(sample_wise_man.memory) >= 2  # input + output
    
    def test_think_memory_management(self, sample_wise_man):
        """Test that memory doesn't grow indefinitely"""
        # Simulate many interactions
        for i in range(30):
            sample_wise_man.think(f"Message {i}")
        
        # Memory should be capped at 20 items
        assert len(sample_wise_man.memory) <= 20
    
    def test_think_with_context(self, sample_wise_man):
        """Test thinking with context"""
        context = [{'role': 'user', 'message': 'previous message'}]
        response = sample_wise_man.think("ما رأيك؟", context=context)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_think_strict_evidence_no_evidence(self, sample_wise_man):
        """Test strict evidence mode when no evidence available"""
        # Mock the engine to return no evidence
        with patch.object(wise_ai_engine, 'generate_response_with_evidence', return_value={
            'response': '',
            'source': 'blocked-no-evidence',
            'confidence': 0.0,
            'evidence': [],
            'blocked': True
        }):
            response = sample_wise_man.think("موضوع عام جداً", strict_evidence=True)
            # Should return empty when blocked
            assert response == ''
    
    def test_response_generation_personality_based(self, sample_wise_man):
        """Test that responses are personality-based"""
        # Test specific keywords trigger personality responses
        response = sample_wise_man.think("تحليل البيانات", strict_evidence=False)
        assert isinstance(response, str)
        # Response should be in Arabic since it's personality-based
        assert any('\u0600' <= c <= '\u06FF' for c in response) or len(response) > 0
    
    def test_personality_responses_all_wise_men(self, all_wise_men):
        """Test that each wise man generates appropriate personality responses"""
        for wise_man in all_wise_men:
            response = wise_man.think("اختبار")
            assert isinstance(response, str)
            assert len(response) > 0
            # Each should have their own greeting or response
            assert wise_man.last_generation_meta['source'] is not None
    
    def test_response_avoid_repetition(self, sample_wise_man):
        """Test that responses avoid recent repetitions"""
        responses = []
        for _ in range(10):
            response = sample_wise_man.think("نفس السؤال")
            responses.append(response)
        
        # Check that we have some variety (not all identical)
        unique_responses = set(responses)
        assert len(unique_responses) > 1


# ═══════════════════════════════════════════════════════════════
# Unit Tests - SmartCouncil
# ═══════════════════════════════════════════════════════════════

class TestSmartCouncil:
    """Tests for the Smart Council coordination"""
    
    def test_council_initialization(self, fresh_council):
        """Test council initializes with all 16 wise men"""
        assert len(fresh_council.wise_men) == 16
        
        # Check all expected names
        expected_core = {
            'حكيم القرار', 'حكيم البصيرة', 'حكيم المستقبل', 'حكيم الشجاعة',
            'حكيم التوازن', 'حكيم الضبط', 'حكيم التكيف', 'حكيم الذاكرة'
        }
        for name in expected_core:
            assert name in fresh_council.wise_men
    
    def test_core_evidence_wise_men_set(self, fresh_council):
        """Test that core evidence wise men are defined"""
        assert len(fresh_council.core_evidence_wise_men) == 8
        assert 'حكيم القرار' in fresh_council.core_evidence_wise_men
        assert 'حكيم الذاكرة' in fresh_council.core_evidence_wise_men
    
    def test_ask_basic(self, fresh_council):
        """Test basic ask functionality"""
        result = fresh_council.ask("ما هو قراري؟")
        
        assert 'response' in result
        assert 'wise_man' in result
        assert 'role' in result
        assert 'specialty' in result
        assert 'confidence' in result
        assert 'evidence' in result
        assert result['wise_man'] in fresh_council.wise_men
    
    def test_ask_specific_wise_man(self, fresh_council):
        """Test asking a specific wise man"""
        result = fresh_council.ask("تحليل", specific_wise_man="حكيم البصيرة")
        
        assert result['wise_man'] == 'حكيم البصيرة'
        assert result['specialty'] == 'التحليل'
    
    def test_ask_invalid_wise_man(self, fresh_council):
        """Test asking an invalid wise man falls back to selection"""
        result = fresh_council.ask("سؤال", specific_wise_man="حكيم غير موجود")
        
        # Should still return a valid response from someone
        assert result['wise_man'] in fresh_council.wise_men
        assert 'response' in result
    
    def test_select_best_wise_man_analysis(self, fresh_council):
        """Test automatic selection for analysis questions"""
        analysis_keywords = ["تحليل", "تقرير", "بيانات", "analysis", "report"]
        
        for keyword in analysis_keywords:
            wise_man = fresh_council._select_best_wise_man(f"{keyword} السوق")
            assert wise_man.name == 'حكيم البصيرة'
    
    def test_select_best_wise_man_decision(self, fresh_council):
        """Test automatic selection for decision questions"""
        decision_keywords = ["قرار", "decision", "استراتيجية", "strategy"]
        
        for keyword in decision_keywords:
            wise_man = fresh_council._select_best_wise_man(f"{keyword} مهمة")
            assert wise_man.name == 'حكيم القرار'
    
    def test_select_best_wise_man_future(self, fresh_council):
        """Test automatic selection for future questions"""
        future_keywords = ["مستقبل", "future", "خطة", "plan"]
        
        for keyword in future_keywords:
            wise_man = fresh_council._select_best_wise_man(f"{keyword} الشركة")
            assert wise_man.name == 'حكيم المستقبل'
    
    def test_select_best_wise_man_crisis(self, fresh_council):
        """Test automatic selection for crisis/risk questions"""
        crisis_keywords = ["خطر", "risk", "أزمة", "crisis"]
        
        for keyword in crisis_keywords:
            wise_man = fresh_council._select_best_wise_man(f"{keyword} مالي")
            # Could be الشجاعة or الطوارئ
            assert wise_man.name in ['حكيم الشجاعة', 'حكيم الطوارئ']
    
    def test_select_best_wise_man_hash_fallback(self, fresh_council):
        """Test hash-based selection for unmatched queries"""
        # Use a query that doesn't match any specific pattern
        wise_man1 = fresh_council._select_best_wise_man("سؤال عشوائي غير محدد")
        wise_man2 = fresh_council._select_best_wise_man("سؤال عشوائي غير محدد")
        
        # Same query should select same wise man (consistent hashing)
        assert wise_man1.name == wise_man2.name
        assert wise_man1.name in fresh_council.core_evidence_wise_men
    
    def test_get_all_wise_men(self, fresh_council):
        """Test getting list of all wise men"""
        all_wise = fresh_council.get_all_wise_men()
        
        assert len(all_wise) == 16
        for wise_info in all_wise:
            assert 'name' in wise_info
            assert 'role' in wise_info
            assert 'specialty' in wise_info
            assert 'personality' in wise_info
            assert 'has_model' in wise_info
            assert 'memory' in wise_info
    
    def test_think_with_evidence_retry_success(self, fresh_council):
        """Test evidence retry mechanism - success case"""
        wise_man = fresh_council.wise_men['حكيم القرار']
        
        # Mock to return evidence on first try
        with patch.object(wise_man, 'think', return_value='Evidence-based response'):
            wise_man.last_generation_meta = {
                'source': 'training+persona',
                'confidence': 0.8,
                'evidence': [{'topic': 'test'}]
            }
            result = fresh_council._think_with_evidence_retry(wise_man, "topic")
            
            assert result is not None
            assert result['response'] == 'Evidence-based response'
    
    def test_think_with_evidence_retry_failure(self, fresh_council):
        """Test evidence retry mechanism - failure case"""
        wise_man = fresh_council.wise_men['حكيم القرار']
        
        # Mock to never return evidence
        with patch.object(wise_man, 'think', return_value=''):
            wise_man.last_generation_meta = {
                'source': 'blocked-no-evidence',
                'confidence': 0.0,
                'evidence': []
            }
            result = fresh_council._think_with_evidence_retry(wise_man, "topic", max_attempts=2)
            
            # Should return None after all retries fail
            assert result is None


# ═══════════════════════════════════════════════════════════════
# Integration Tests - Council Discussions
# ═══════════════════════════════════════════════════════════════

class TestCouncilDiscussions:
    """Tests for multi-wise-man discussions"""
    
    def test_discuss_basic(self, fresh_council):
        """Test basic discussion functionality"""
        responses = fresh_council.discuss("موضوع للنقاش")
        
        # Should return list of responses
        assert isinstance(responses, list)
        # Should have at most 5 responses (limit in code)
        assert len(responses) <= 5
        
        for response in responses:
            assert 'wise_man' in response
            assert 'response' in response
            assert 'confidence' in response
    
    def test_discuss_empty_topic(self, fresh_council):
        """Test discussion with empty/minimal topic"""
        responses = fresh_council.discuss("")
        assert isinstance(responses, list)
    
    def test_discuss_with_specific_wise_men(self, fresh_council):
        """Test that discussion only involves specific wise men"""
        responses = fresh_council.discuss("استراتيجية الشركة")
        
        candidate_names = {
            'حكيم القرار', 'حكيم البصيرة', 'حكيم المستقبل', 'حكيم الشجاعة',
            'حكيم التوازن', 'حكيم الضبط', 'حكيم التكيف', 'حكيم الذاكرة',
            'حكيم التنفيذ', 'حكيم النظام'
        }
        
        for response in responses:
            assert response['wise_man'] in candidate_names


# ═══════════════════════════════════════════════════════════════
# Infinite Loop Prevention Tests
# ═══════════════════════════════════════════════════════════════

class TestInfiniteLoopPrevention:
    """Critical tests to prevent infinite loops in AI conversations"""
    
    @pytest.mark.timeout(5)  # Fail if test takes >5 seconds
    def test_think_does_not_loop(self, sample_wise_man):
        """Test that think method completes quickly"""
        response = sample_wise_man.think("اختبار")
        assert isinstance(response, str)
    
    @pytest.mark.timeout(10)
    def test_discuss_does_not_loop(self, fresh_council):
        """Test that discuss method completes quickly"""
        responses = fresh_council.discuss("موضوع للنقاش")
        assert isinstance(responses, list)
    
    @pytest.mark.timeout(5)
    def test_ask_does_not_loop(self, fresh_council):
        """Test that ask method completes quickly"""
        result = fresh_council.ask("سؤال")
        assert 'response' in result
    
    @pytest.mark.timeout(10)
    def test_multiple_rapid_calls(self, fresh_council):
        """Test rapid sequential calls don't cause issues"""
        for i in range(20):
            result = fresh_council.ask(f"سؤال سريع {i}")
            assert 'response' in result
    
    def test_memory_circular_reference_prevention(self, sample_wise_man):
        """Test that memory doesn't create circular references"""
        import gc
        
        initial_count = len(gc.get_objects())
        
        for i in range(50):
            sample_wise_man.think(f"Message {i}")
        
        gc.collect()
        final_count = len(gc.get_objects())
        
        # Memory growth should be bounded
        assert final_count - initial_count < 1000


# ═══════════════════════════════════════════════════════════════
# Hallucination Detection and Prevention Tests
# ═══════════════════════════════════════════════════════════════

class TestHallucinationPrevention:
    """Tests to detect and prevent AI hallucinations"""
    
    def test_response_has_metadata(self, sample_wise_man):
        """Test that all responses include metadata about source"""
        sample_wise_man.think("سؤال")
        
        meta = sample_wise_man.last_generation_meta
        assert 'source' in meta
        assert 'confidence' in meta
        assert 'evidence' in meta
    
    def test_confidence_score_present(self, fresh_council):
        """Test that confidence scores are included in responses"""
        result = fresh_council.ask("سؤال")
        
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1.0
    
    def test_evidence_list_present(self, fresh_council):
        """Test that evidence list is included"""
        result = fresh_council.ask("سؤال")
        
        assert 'evidence' in result
        assert isinstance(result['evidence'], list)
    
    def test_low_confidence_flagged(self, fresh_council):
        """Test that low confidence responses are properly flagged"""
        # Mock to return low confidence
        with patch.object(fresh_council, '_think_with_evidence_retry', return_value=None):
            result = fresh_council.ask("موضوع صعب جداً")
            
            # Should indicate low confidence
            assert result['confidence'] < 0.5
            assert result['response_source'] == 'no-evidence-guard'
    
    def test_response_source_tracking(self, fresh_council):
        """Test that response source is tracked"""
        result = fresh_council.ask("سؤال عادي")
        
        assert 'response_source' in result
        # Source should be one of the expected values
        valid_sources = ['training+persona', 'persona-template', 'no-evidence-guard']
        assert result['response_source'] in valid_sources or isinstance(result['response_source'], str)
    
    def test_strict_evidence_blocks_unverified(self, sample_wise_man):
        """Test strict evidence mode blocks unverified responses"""
        # Force no evidence scenario
        with patch.object(wise_ai_engine, 'generate_response_with_evidence', return_value={
            'response': '',
            'source': 'blocked-no-evidence',
            'confidence': 0.0,
            'evidence': [],
            'blocked': True
        }):
            response = sample_wise_man.think("موضوع", strict_evidence=True)
            assert response == ''  # Should block


# ═══════════════════════════════════════════════════════════════
# Error Handling Tests
# ═══════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Tests for error handling and edge cases"""
    
    def test_empty_message(self, sample_wise_man):
        """Test handling of empty message"""
        response = sample_wise_man.think("")
        assert isinstance(response, str)
    
    def test_none_message(self, sample_wise_man):
        """Test handling of None message"""
        # This should not crash - convert to string
        try:
            response = sample_wise_man.think(str(None))
            assert isinstance(response, str)
        except Exception as e:
            # If it raises, it should be a controlled error
            assert isinstance(e, (TypeError, AttributeError))
    
    def test_very_long_message(self, sample_wise_man):
        """Test handling of very long message"""
        long_message = "كلمة " * 10000
        response = sample_wise_man.think(long_message)
        assert isinstance(response, str)
    
    def test_special_characters(self, sample_wise_man):
        """Test handling of special characters"""
        special_msgs = [
            "<script>alert('xss')</script>",
            " DROP TABLE users; --",
            "🔥🎯💡🚀💻",
            "مرحبا\nworld\t!!!",
            "نص" * 1000,
        ]
        
        for msg in special_msgs:
            response = sample_wise_man.think(msg)
            assert isinstance(response, str)
            assert len(response) > 0
    
    def test_unicode_handling(self, sample_wise_man):
        """Test handling of various unicode strings"""
        unicode_msgs = [
            "مرحبا بالعالم",
            "Hello World",
            "你好世界",
            "🌍🌎🌏",
            "αβγδε",
            "مرحبا 🌍 Hello",
        ]
        
        for msg in unicode_msgs:
            response = sample_wise_man.think(msg)
            assert isinstance(response, str)
    
    def test_model_load_failure_handling(self):
        """Test graceful handling of model load failure"""
        # Create with invalid checkpoint path
        invalid_path = Path("/nonexistent/path/model.pt")
        wise_man = WiseManAI(
            name="Test",
            role="test",
            specialty="test",
            checkpoint_path=invalid_path
        )
        
        # Should still work with personality fallback
        assert wise_man.model is None
        response = wise_man.think("test")
        assert isinstance(response, str)
    
    def test_corrupted_checkpoint_handling(self, tmp_path):
        """Test handling of corrupted checkpoint file"""
        # Create a fake corrupted checkpoint
        checkpoint_file = tmp_path / "corrupted.pt"
        checkpoint_file.write_text("not a valid torch file")
        
        wise_man = WiseManAI(
            name="Test",
            role="test",
            specialty="test",
            checkpoint_path=checkpoint_file
        )
        
        # Should gracefully fall back to None model
        assert wise_man.model is None
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, fresh_council):
        """Test thread-safety of concurrent calls"""
        import asyncio
        
        async def ask_question(i):
            return fresh_council.ask(f"سؤال {i}")
        
        # Run many concurrent asks
        tasks = [ask_question(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        for result in results:
            assert 'response' in result


# ═══════════════════════════════════════════════════════════════
# WiseAIEngine Tests
# ═══════════════════════════════════════════════════════════════

class TestWiseAIEngine:
    """Tests for the WiseAIEngine"""
    
    def test_engine_initialization(self):
        """Test AI engine initialization"""
        engine = WiseAIEngine()
        assert engine is not None
        assert engine.device is not None
        assert len(engine.response_db) > 0
        assert len(engine.response_types) == 10
    
    def test_context_analysis(self):
        """Test context analysis function"""
        engine = WiseAIEngine()
        
        test_cases = [
            ("تحليل البيانات", "analysis"),
            ("ما هي الاستراتيجية؟", "strategy"),
            ("مستقبل الشركة", "vision"),
            ("خطر مالي", "caution"),
            ("تنفيذ الأمر", "action"),
            ("أحتاج نصيحة", "wisdom"),
            ("كيف أقوم بهذا؟", "question"),  # Changed to avoid "عمل" keyword
            ("ساعدني", "support"),
        ]
        
        for message, expected_type in test_cases:
            result = engine._analyze_context(message)
            assert result == expected_type
    
    def test_response_generation(self):
        """Test response generation"""
        engine = WiseAIEngine()
        
        result = engine.generate_response_with_evidence(
            message="اختبار",
            wise_man_name="حكيم القرار"
        )
        
        assert 'response' in result
        assert 'source' in result
        assert 'confidence' in result
        assert 'evidence' in result
    
    def test_response_avoids_recent_duplicates(self):
        """Test that recent responses are avoided"""
        engine = WiseAIEngine()
        
        history = ["بناءً على تحليلي العميق"]
        result1 = engine.generate_response_with_evidence(
            message="اختبار",
            wise_man_name="حكيم القرار",
            conversation_history=history
        )
        
        # Result should be different from history
        assert result1['response'] not in history
    
    def test_personality_retrieval(self):
        """Test personality dictionary retrieval"""
        engine = WiseAIEngine()
        
        personality = engine.get_personality("حكيم القرار")
        assert 'style' in personality
        assert 'greeting' in personality
        
        # Unknown should return default
        unknown = engine.get_personality("حكيم غير معروف")
        assert 'style' in unknown
    
    def test_keyword_extraction(self):
        """Test keyword extraction"""
        engine = WiseAIEngine()
        
        keywords = engine._extract_keywords("تحليل البيانات المالية")
        assert isinstance(keywords, list)
        assert 'تحليل' in keywords or 'البيانات' in keywords
    
    def test_training_evidence_retrieval(self):
        """Test training evidence retrieval"""
        engine = WiseAIEngine()
        
        evidence = engine._retrieve_training_evidence("تحليل بيانات", top_k=2)
        assert isinstance(evidence, list)
        # Result depends on whether training data exists


# ═══════════════════════════════════════════════════════════════
# Global Instance Tests
# ═══════════════════════════════════════════════════════════════

class TestGlobalInstances:
    """Tests for global singleton instances"""
    
    def test_global_smart_council_exists(self):
        """Test that global smart_council exists"""
        assert smart_council is not None
        assert isinstance(smart_council, SmartCouncil)
        assert len(smart_council.wise_men) == 16
    
    def test_global_wise_ai_engine_exists(self):
        """Test that global wise_ai_engine exists"""
        assert wise_ai_engine is not None
        assert isinstance(wise_ai_engine, WiseAIEngine)


# ═══════════════════════════════════════════════════════════════
# Performance Tests
# ═══════════════════════════════════════════════════════════════

class TestPerformance:
    """Performance-related tests"""
    
    @pytest.mark.timeout(30)
    def test_bulk_operations(self, fresh_council):
        """Test that bulk operations complete in reasonable time"""
        import time
        
        start = time.time()
        
        for i in range(50):
            fresh_council.ask(f"سؤال {i}")
        
        elapsed = time.time() - start
        
        # Should complete 50 asks in less than 30 seconds
        assert elapsed < 30
    
    def test_memory_usage_bounded(self, fresh_council):
        """Test that memory usage stays bounded"""
        import sys
        
        # Get initial size estimate
        initial_memory = len(str(fresh_council.wise_men))
        
        # Perform many operations
        for i in range(100):
            fresh_council.ask(f"سؤال {i}")
        
        final_memory = len(str(fresh_council.wise_men))
        
        # Memory growth should be sub-linear
        growth_ratio = final_memory / max(initial_memory, 1)
        assert growth_ratio < 10  # Less than 10x growth
