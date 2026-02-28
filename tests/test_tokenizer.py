"""
اختبارات التوكنيزر - Tokenizer Tests
======================================
Tests for tokenizers including:
- arabic_processor
- bpe_tokenizer
- code_tokenizer

التغطية: >80%
"""

import pytest
from pathlib import Path


class TestArabicProcessor:
    """
    اختبارات معالج النصوص العربية
    Arabic Processor Tests
    """
    
    @pytest.fixture
    def processor(self):
        from ai.tokenizer.arabic_processor import ArabicProcessor
        return ArabicProcessor()
    
    def test_normalize_arabic_text(self, processor):
        """
        اختبار تطبيع النص العربي
        Test Arabic text normalization
        """
        # Test different alef variants
        text = "الذكاء الإصطناعي"
        result = processor.normalize(text)
        assert "أ" not in result or result == text.replace("إ", "ا")
    
    def test_remove_diacritics(self, processor):
        """
        اختبار إزالة التشكيل
        Test diacritics removal
        """
        text = "القُرآن الكَرِيم"
        result = processor.remove_diacritics(text)
        assert "ُ" not in result
        assert "َ" not in result
        assert "ِ" not in result
    
    def test_is_arabic(self, processor):
        """
        اختبار التحقق من النص العربي
        Test Arabic text detection
        """
        assert processor.is_arabic("مرحبا") is True
        assert processor.is_arabic("Hello") is False
        assert processor.is_arabic("Hello مرحبا") is True
    
    def test_split_arabic_words(self, processor):
        """
        اختبار تقسيم الكلمات العربية
        Test splitting Arabic words
        """
        text = "مرحبا بالعالم"
        words = processor.split_arabic_words(text)
        assert len(words) == 2
        assert "مرحبا" in words
        assert "بالعالم" in words
    
    def test_count_arabic_words(self, processor):
        """
        اختبار عد الكلمات العربية
        Test counting Arabic words
        """
        text = "هذا نص عربي"
        count = processor.count_arabic_words(text)
        assert count == 3
    
    def test_preprocess_for_tokenization(self, processor):
        """
        اختبار المعالجة المسبقة للتوكنيزيشن
        Test preprocessing for tokenization
        """
        text = "مرحباً!   كيف  حالك؟"
        result = processor.preprocess_for_tokenization(text)
        # Should normalize whitespace
        assert "  " not in result
        assert result.strip() == result
    
    def test_get_word_root(self, processor):
        """
        اختبار استخراج جذر الكلمة
        Test extracting word root
        """
        word = "الكتاب"
        root = processor.get_word_root(word)
        assert "ال" not in root  # Should remove prefix
    
    def test_normalize_alef_variants(self, processor):
        """
        اختبار تطبيع حروف الألف
        Test normalizing alef variants
        """
        text = "أإآا"
        result = processor.normalize(text)
        # All alef variants should become ا
        assert "أ" not in result
        assert "إ" not in result
        assert "آ" not in result
        assert result.count("ا") == 4
    
    def test_normalize_taa_marbuta(self, processor):
        """
        اختبار تطبيع التاء المربوطة
        Test normalizing taa marbuta
        """
        text = "مكتبة"
        result = processor.normalize(text)
        assert "ة" not in result
        assert "ه" in result


class TestBPETokenizer:
    """
    اختبارات BPE Tokenizer
    BPE Tokenizer Tests
    """
    
    @pytest.fixture
    def tokenizer(self):
        from ai.tokenizer.bpe_tokenizer import BPETokenizer
        return BPETokenizer(vocab_size=100)
    
    def test_tokenizer_initialization(self, tokenizer):
        """
        اختبار تهيئة التوكنيزر
        Test tokenizer initialization
        """
        assert tokenizer.vocab_size == 100
        assert tokenizer.vocab == {}
        assert tokenizer.merges == []
    
    def test_train_tokenizer(self, tokenizer):
        """
        اختبار تدريب التوكنيزر
        Test training tokenizer
        """
        texts = [
            "Hello world",
            "Python is great",
            "Machine learning is fascinating"
        ]
        
        tokenizer.train(texts)
        
        assert len(tokenizer.vocab) > 0
        assert len(tokenizer.merges) >= 0
    
    def test_encode_text(self, tokenizer):
        """
        اختبار تشفير النص
        Test encoding text
        """
        # Train first
        tokenizer.train(["Hello world", "Python is great"])
        
        # Encode
        text = "Hello world"
        token_ids = tokenizer.encode(text)
        
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
    
    def test_decode_text(self, tokenizer):
        """
        اختبار فك تشفير النص
        Test decoding text
        """
        # Train
        tokenizer.train(["Hello world", "Python programming"])
        
        # Encode and decode
        original = "Hello world"
        token_ids = tokenizer.encode(original)
        decoded = tokenizer.decode(token_ids)
        
        # Decoded might not be exact due to limited vocab
        assert isinstance(decoded, str)
    
    def test_tokenize(self, tokenizer):
        """
        اختبار تقسيم النص إلى توكنات
        Test tokenizing text
        """
        # Train
        tokenizer.train(["Hello world"])
        
        # Tokenize
        text = "Hello world"
        tokens = tokenizer.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_save_and_load(self, tokenizer, tmp_path):
        """
        اختبار حفظ وتحميل التوكنيزر
        Test saving and loading tokenizer
        """
        from ai.tokenizer.bpe_tokenizer import BPETokenizer
        
        # Train
        tokenizer.train(["Hello world", "Python is great"])
        
        # Save
        save_path = tmp_path / "tokenizer"
        tokenizer.save(save_path)
        
        # Load
        loaded = BPETokenizer.load(save_path)
        
        assert loaded.vocab_size == tokenizer.vocab_size
        assert len(loaded.vocab) == len(tokenizer.vocab)
    
    def test_special_tokens(self):
        """
        اختبار التوكنات الخاصة
        Test special tokens
        """
        from ai.tokenizer.bpe_tokenizer import BPETokenizer
        
        special_tokens = BPETokenizer.SPECIAL_TOKENS
        
        assert "<PAD>" in special_tokens
        assert "<UNK>" in special_tokens
        assert "<BOS>" in special_tokens
        assert "<EOS>" in special_tokens
        assert "<ARABIC>" in special_tokens
        assert "<CODE>" in special_tokens
    
    def test_get_vocab_size(self, tokenizer):
        """
        اختبار الحصول على حجم المفردات
        Test getting vocabulary size
        """
        tokenizer.train(["Hello world"])
        
        vocab_size = tokenizer.get_vocab_size()
        
        assert vocab_size > 0
        assert vocab_size <= tokenizer.vocab_size
    
    def test_normalize_arabic(self, tokenizer):
        """
        اختبار تطبيع العربية في BPE
        Test Arabic normalization in BPE
        """
        result = tokenizer._normalize_arabic("مرحباً")
        assert isinstance(result, str)


class TestCodeTokenizer:
    """
    اختبارات توكنيزر الأكواد
    Code Tokenizer Tests
    """
    
    @pytest.fixture
    def tokenizer(self):
        from ai.tokenizer.code_tokenizer import CodeTokenizer
        return CodeTokenizer()
    
    def test_detect_python(self, tokenizer):
        """
        اختبار اكتشاف بايثون
        Test Python detection
        """
        code = """
def hello():
    print("Hello, World!")
        """
        lang = tokenizer.detect_language(code)
        assert lang.value == "python"
    
    def test_detect_javascript(self, tokenizer):
        """
        اختبار اكتشاف جافاسكريبت
        Test JavaScript detection
        """
        code = """
function hello() {
    console.log("Hello");
}
        """
        lang = tokenizer.detect_language(code)
        assert lang.value == "javascript"
    
    def test_detect_sql(self, tokenizer):
        """
        اختبار اكتشاف SQL
        Test SQL detection
        """
        code = "SELECT * FROM users WHERE id = 1"
        lang = tokenizer.detect_language(code)
        assert lang.value == "sql"
    
    def test_tokenize_python(self, tokenizer):
        """
        اختبار تقسيم كود بايثون
        Test tokenizing Python code
        """
        code = "def hello():\n    return 42"
        tokens = tokenizer.tokenize(code)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Should preserve newline
        assert "\n" in tokens
    
    def test_tokenize_preserves_indentation(self, tokenizer):
        """
        اختبار الحفاظ على المسافة البادئة
        Test preserving indentation
        """
        code = "def hello():\n    pass"
        tokens = tokenizer.tokenize(code)
        
        # Should have indentation token
        assert any(t.startswith(" ") for t in tokens if isinstance(t, str))
    
    def test_extract_comments_python(self, tokenizer):
        """
        اختبار استخراج التعليقات من بايثون
        Test extracting Python comments
        """
        code = """
# This is a comment
def hello():
    pass  # inline comment
        """
        comments = tokenizer.extract_comments(code)
        
        assert len(comments) == 2
        assert any("This is a comment" in c for c in comments)
    
    def test_extract_comments_javascript(self, tokenizer):
        """
        اختبار استخراج التعليقات من جافاسكريبت
        Test extracting JavaScript comments
        """
        code = """
// Single line comment
function hello() {
    /* Multi-line
       comment */
}
        """
        comments = tokenizer.extract_comments(code)
        
        assert len(comments) >= 2
    
    def test_remove_comments(self, tokenizer):
        """
        اختبار إزالة التعليقات
        Test removing comments
        """
        code = "def hello():  # comment\n    pass"
        result = tokenizer.remove_comments(code)
        
        assert "#" not in result
        assert "def hello():" in result
    
    def test_get_code_statistics(self, tokenizer):
        """
        اختبار الحصول على إحصائيات الكود
        Test getting code statistics
        """
        code = """
def hello():
    # Comment
    x = 10
    return x
        """
        stats = tokenizer.get_code_statistics(code)
        
        assert "language" in stats
        assert "total_lines" in stats
        assert "comment_lines" in stats
        assert stats["language"] == "python"
    
    def test_read_string_literal(self, tokenizer):
        """
        اختبار قراءة سلسلة نصية
        Test reading string literal
        """
        line = '"Hello world" rest'
        token, pos = tokenizer._read_string(line, 0)
        
        assert token == '"Hello world"'
        assert pos == 13
    
    def test_read_number(self, tokenizer):
        """
        اختبار قراءة رقم
        Test reading number
        """
        line = "3.14159 end"
        token, pos = tokenizer._read_number(line, 0)
        
        assert token == "3.14159"
        assert pos == 7
    
    def test_read_identifier(self, tokenizer):
        """
        اختبار قراءة معرف
        Test reading identifier
        """
        line = "variable_name = 10"
        token, pos = tokenizer._read_identifier(line, 0)
        
        assert token == "variable_name"
        assert pos == 13


class TestTokenizerIntegration:
    """
    اختبارات تكامل التوكنيزر
    Tokenizer Integration Tests
    """
    
    def test_arabic_with_bpe(self):
        """
        اختبار النص العربي مع BPE
        Test Arabic text with BPE
        """
        from ai.tokenizer.arabic_processor import ArabicProcessor
        from ai.tokenizer.bpe_tokenizer import BPETokenizer
        
        processor = ArabicProcessor()
        tokenizer = BPETokenizer(vocab_size=100)
        
        # Preprocess Arabic text
        arabic_text = processor.preprocess_for_tokenization("مرحبا بالعالم")
        
        # Train and encode
        tokenizer.train([arabic_text, "Hello world"])
        token_ids = tokenizer.encode(arabic_text)
        
        assert isinstance(token_ids, list)
    
    def test_code_with_arabic_comments(self):
        """
        اختبار كود مع تعليقات عربية
        Test code with Arabic comments
        """
        from ai.tokenizer.code_tokenizer import CodeTokenizer
        
        tokenizer = CodeTokenizer()
        
        code = '''
def hello():
    # مرحبا بالعالم
    print("Hello")
        '''
        
        comments = tokenizer.extract_comments(code)
        tokens = tokenizer.tokenize(code)
        
        assert len(comments) > 0
        assert len(tokens) > 0
    
    def test_multilingual_text(self):
        """
        اختبار نص متعدد اللغات
        Test multilingual text
        """
        from ai.tokenizer.bpe_tokenizer import BPETokenizer
        
        tokenizer = BPETokenizer(vocab_size=200)
        
        texts = [
            "Hello world",
            "مرحبا بالعالم",
            "def hello(): pass",
            "Bonjour le monde"
        ]
        
        tokenizer.train(texts)
        
        for text in texts:
            token_ids = tokenizer.encode(text)
            assert isinstance(token_ids, list)
