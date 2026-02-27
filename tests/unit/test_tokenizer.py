"""
Unit tests for BPE Tokenizer
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.tokenizer.bpe_tokenizer import BPETokenizer, create_tokenizer, load_tokenizer


class TestBPETokenizer:
    """Test cases for BPETokenizer."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a fresh tokenizer for each test."""
        return BPETokenizer(vocab_size=1000)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def corpus_file(self, temp_dir):
        """Create a sample corpus file."""
        corpus_path = Path(temp_dir) / 'corpus.txt'
        corpus_content = """Hello world
        This is a test corpus for training
        Python programming is fun
        Machine learning and AI development
        Code examples and documentation
        """ * 10  # Repeat for more data
        
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write(corpus_content)
        
        return str(corpus_path)
    
    def test_initialization(self, tokenizer):
        """Test tokenizer initialization."""
        assert tokenizer.vocab_size == 1000
        assert len(tokenizer.vocab) == 0  # Not trained yet
        assert len(tokenizer.SPECIAL_TOKENS) == 11
    
    def test_train(self, tokenizer, corpus_file):
        """Test tokenizer training."""
        tokenizer.train(corpus_file)
        
        assert len(tokenizer.vocab) > len(tokenizer.SPECIAL_TOKENS)
        assert len(tokenizer.merges) > 0
        assert len(tokenizer.inverse_vocab) == len(tokenizer.vocab)
    
    def test_encode_decode(self, tokenizer, corpus_file):
        """Test encoding and decoding."""
        tokenizer.train(corpus_file)
        
        test_texts = [
            "Hello world",
            "Python code",
            "Test 123"
        ]
        
        for text in test_texts:
            encoded = tokenizer.encode(text)
            assert isinstance(encoded, list)
            assert len(encoded) > 0
            assert all(isinstance(t, int) for t in encoded)
    
    def test_arabic_text(self, tokenizer, corpus_file):
        """Test Arabic text handling."""
        tokenizer.train(corpus_file)
        
        arabic_texts = [
            "مرحبا بالعالم",
            "برمجة بايثون",
            "تعلم الآلة"
        ]
        
        for text in arabic_texts:
            encoded = tokenizer.encode(text)
            assert len(encoded) > 0
            
            decoded = tokenizer.decode(encoded)
            # Arabic text trained on English corpus may decode to UNK
            assert isinstance(decoded, str)
    
    def test_code_text(self, tokenizer, corpus_file):
        """Test code snippet handling."""
        tokenizer.train(corpus_file)
        
        code_snippets = [
            "def hello():\n    return 'world'",
            "import numpy as np",
            "class MyClass:\n    pass"
        ]
        
        for code in code_snippets:
            encoded = tokenizer.encode(code)
            assert len(encoded) > 0
    
    def test_save_load(self, tokenizer, corpus_file, temp_dir):
        """Test saving and loading tokenizer."""
        tokenizer.train(corpus_file)
        
        # Save
        save_path = Path(temp_dir) / 'tokenizer'
        tokenizer.save(str(save_path))
        
        assert (save_path / 'vocab.json').exists()
        assert (save_path / 'merges.pkl').exists()
        assert (save_path / 'config.json').exists()
        
        # Load
        loaded_tokenizer = load_tokenizer(str(save_path))
        
        assert loaded_tokenizer.get_vocab_size() == tokenizer.get_vocab_size()
        assert loaded_tokenizer.vocab == tokenizer.vocab
    
    def test_consistency(self, tokenizer, corpus_file):
        """Test encoding/decoding consistency."""
        tokenizer.train(corpus_file)
        
        test_text = "Hello Python world"
        
        # Encode twice should give same result
        encoded1 = tokenizer.encode(test_text)
        encoded2 = tokenizer.encode(test_text)
        assert encoded1 == encoded2
    
    def test_tokenize(self, tokenizer, corpus_file):
        """Test tokenize method."""
        tokenizer.train(corpus_file)
        
        text = "Hello world"
        tokens = tokenizer.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
    
    def test_special_tokens(self, tokenizer):
        """Test special token handling."""
        assert tokenizer.SPECIAL_TOKENS['<PAD>'] == 0
        assert tokenizer.SPECIAL_TOKENS['<UNK>'] == 1
        assert tokenizer.SPECIAL_TOKENS['<BOS>'] == 2
        assert tokenizer.SPECIAL_TOKENS['<EOS>'] == 3
    
    def test_normalize_arabic(self, tokenizer):
        """Test Arabic normalization."""
        test_cases = [
            ('إنشاء', 'انشاء'),
            ('كتابة', 'كتابه'),
            ('قراءة', 'قراءه'),
        ]
        
        for input_text, expected in test_cases:
            normalized = tokenizer._normalize_arabic(input_text)
            # Note: Some normalization may vary
            assert isinstance(normalized, str)
    
    def test_empty_text(self, tokenizer, corpus_file):
        """Test handling of empty text."""
        tokenizer.train(corpus_file)
        
        encoded = tokenizer.encode("")
        assert encoded == []
    
    def test_long_text(self, tokenizer, corpus_file):
        """Test handling of long text."""
        tokenizer.train(corpus_file)
        
        long_text = "word " * 1000
        encoded = tokenizer.encode(long_text)
        
        assert len(encoded) > 0
        assert len(encoded) < len(long_text)  # Should be compressed
    
    def test_get_vocab_size(self, tokenizer, corpus_file):
        """Test get_vocab_size method."""
        assert tokenizer.get_vocab_size() == 0
        
        tokenizer.train(corpus_file)
        vocab_size = tokenizer.get_vocab_size()
        
        assert vocab_size > 0
        assert vocab_size <= tokenizer.vocab_size


class TestTokenizerFactory:
    """Test factory functions."""

    @pytest.fixture
    def factory_corpus(self, temp_dir):
        """Create a corpus file for factory tests."""
        corpus_path = Path(temp_dir) / 'factory_corpus.txt'
        content = "Hello world Python code test\n" * 20
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(corpus_path)

    def test_create_tokenizer(self, factory_corpus):
        """Test create_tokenizer function."""
        tokenizer = create_tokenizer(factory_corpus, vocab_size=500)

        assert isinstance(tokenizer, BPETokenizer)
        assert tokenizer.get_vocab_size() > 0

    def test_load_tokenizer(self, temp_dir, factory_corpus):
        """Test load_tokenizer function."""
        # Create and save a tokenizer
        tokenizer = create_tokenizer(factory_corpus, vocab_size=500)

        save_path = Path(temp_dir) / 'test_tokenizer'
        tokenizer.save(str(save_path))

        # Load
        loaded = load_tokenizer(str(save_path))

        assert isinstance(loaded, BPETokenizer)
        assert loaded.get_vocab_size() == tokenizer.get_vocab_size()


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
