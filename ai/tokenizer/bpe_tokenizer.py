"""BPE Tokenizer - تعلم Byte Pair Encoding
=====================================
Implements a simple BPE tokenizer.

This module exposes a test-facing API used by the repository test suite:
- BPETokenizer.SPECIAL_TOKENS (dict)
- create_tokenizer / load_tokenizer
- BPETokenizer.save() writing vocab.json + merges.pkl + config.json

It also keeps compatibility with the previous in-file pickle save/load.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union


class BPETokenizer:
    """Byte Pair Encoding Tokenizer."""

    SPECIAL_TOKENS: Dict[str, int] = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "<MASK>": 4,
        "<SEP>": 5,
        "<CLS>": 6,
        "<ARABIC>": 7,
        "<CODE>": 8,
        "<NUM>": 9,
        "<PUNC>": 10,
    }

    def __init__(self, vocab_size: int = 10000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        # Populated after training (or manual assignment during checkpoint conversion)
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[Tuple[str, str], str]] = []

        self._trained = False
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text into words
        Handles Arabic, English, and code
        """
        # Pattern to match words, numbers, and code tokens
        # Arabic: \u0600-\u06FF
        # English: a-zA-Z
        # Numbers: 0-9
        # Code: special chars
        pattern = r'[\u0600-\u06FF]+|[a-zA-Z]+|\d+|[^\s\u0600-\u06FFa-zA-Z\d]'
        
        words = re.findall(pattern, text)
        return words
    
    def _get_word_tokens(self, word: str) -> List[str]:
        """Convert word to initial tokens (characters + </w>)"""
        # End of word symbol
        return list(word) + ["</w>"]
    
    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """Get all adjacent pairs in word"""
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs
    
    def _merge_vocab(
        self,
        pair: Tuple[str, str],
        word_freqs: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """Apply merge operation to all words"""
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        
        new_word_freqs = {}
        for word_tokens, freq in word_freqs.items():
            word_str = " ".join(word_tokens)
            new_word_str = pattern.sub("".join(pair), word_str)
            new_word_tokens = tuple(new_word_str.split())
            new_word_freqs[new_word_tokens] = freq
        
        return new_word_freqs
    
    def train(self, texts: Union[List[str], str, Path]):
        """
        Train BPE tokenizer on corpus
        
        Args:
            texts: List of training texts
        """
        # Accept either a list of strings or a corpus file path
        if isinstance(texts, (str, Path)):
            corpus_path = Path(texts)
            if corpus_path.exists() and corpus_path.is_file():
                with open(corpus_path, "r", encoding="utf-8") as f:
                    texts = [line.strip() for line in f if line.strip()]
            else:
                raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        print(f"Training BPE tokenizer (vocab_size={self.vocab_size})...")

        # Initialize vocab with special tokens at training time
        self.vocab = dict(self.SPECIAL_TOKENS)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Count word frequencies
        word_freqs: Dict[Tuple[str, ...], int] = defaultdict(int)
        
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                word_tokens = tuple(self._get_word_tokens(word))
                word_freqs[word_tokens] += 1
        
        print(f"  Unique words: {len(word_freqs)}")
        
        # Build initial character vocabulary
        all_chars = set()
        for word_tokens in word_freqs:
            for token in word_tokens:
                if token not in self.vocab:
                    all_chars.add(token)
        
        # Add characters to vocabulary
        for char in sorted(all_chars):
            if len(self.vocab) < self.vocab_size:
                self.vocab[char] = len(self.vocab)
            self.inverse_vocab[self.vocab[char]] = char
        
        print(f"  Initial vocab size: {len(self.vocab)}")
        
        # BPE training loop
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            # Count all pairs
            pairs_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
            
            for word_tokens, freq in word_freqs.items():
                pairs = self._get_pairs(list(word_tokens))
                for pair in pairs:
                    pairs_freqs[pair] += freq
            
            if not pairs_freqs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs_freqs, key=pairs_freqs.get)
            
            if pairs_freqs[best_pair] < self.min_frequency:
                break
            
            # Create new token
            new_token = "".join(best_pair)
            
            # Add to vocabulary
            if new_token not in self.vocab and len(self.vocab) < self.vocab_size:
                self.vocab[new_token] = len(self.vocab)
                self.inverse_vocab[self.vocab[new_token]] = new_token
                self.merges.append((best_pair, new_token))
                
                # Apply merge to word frequencies
                word_freqs = self._merge_vocab(best_pair, word_freqs)
            
            if (i + 1) % 100 == 0:
                print(f"  Merges: {i + 1}/{num_merges}, Vocab: {len(self.vocab)}")
        
        self._trained = True
        print(f"Training complete. Final vocab size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        if not text:
            return []

        if not self.vocab:
            # Not trained yet
            return []
        
        words = self._pre_tokenize(text)
        token_ids = []
        
        for word in words:
            word_tokens = self._get_word_tokens(word)
            
            # Apply BPE merges
            for pair, new_token in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        word_tokens = word_tokens[:i] + [new_token] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            # Convert to IDs
            for token in word_tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    token_ids.append(self.SPECIAL_TOKENS["<UNK>"])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens: List[str] = []
        for idx in token_ids:
            token = self.inverse_vocab.get(idx)
            if token is None:
                continue
            if token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)
        
        # Join tokens and remove </w>
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()
    
    def save(self, path: Union[str, Path]):
        """Save tokenizer to disk.

        If `path` is a directory (or has no suffix), writes:
        - vocab.json
        - merges.pkl
        - config.json

        Otherwise falls back to the previous single-file pickle format.
        """
        path = Path(path)

        # Directory format (expected by tests)
        if path.suffix == "" or path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "vocab.json", "w", encoding="utf-8") as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            with open(path / "merges.pkl", "wb") as f:
                pickle.dump(self.merges, f)
            with open(path / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"vocab_size": self.vocab_size, "min_frequency": self.min_frequency},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Tokenizer saved to {path}")
            return

        # Single-file pickle format
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "vocab": self.vocab,
            "merges": self.merges,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        """Load tokenizer from disk (directory or single-file format)."""
        path = Path(path)

        # Directory format
        if path.exists() and path.is_dir() and (path / "config.json").exists():
            with open(path / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            tokenizer = cls(
                vocab_size=config.get("vocab_size", 10000),
                min_frequency=config.get("min_frequency", 2),
            )
            with open(path / "vocab.json", "r", encoding="utf-8") as f:
                tokenizer.vocab = {k: int(v) for k, v in json.load(f).items()}
            with open(path / "merges.pkl", "rb") as f:
                tokenizer.merges = pickle.load(f)
            tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
            tokenizer._trained = True
            print(f"Tokenizer loaded from {path}")
            return tokenizer

        # Single-file pickle format
        with open(path, "rb") as f:
            data = pickle.load(f)
        tokenizer = cls(vocab_size=data["vocab_size"], min_frequency=data["min_frequency"])
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = data.get("merges", [])
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer._trained = True
        print(f"Tokenizer loaded from {path}")
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)

    def _normalize_arabic(self, text: str) -> str:
        # Minimal normalization (tests only require it returns a string)
        replacements = {
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ة": "ه",
            "ى": "ي",
            "ؤ": "و",
            "ئ": "ي",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to tokens (without converting to IDs)"""
        words = self._pre_tokenize(text)
        all_tokens = []
        
        for word in words:
            word_tokens = self._get_word_tokens(word)
            
            # Apply BPE merges
            for pair, new_token in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        word_tokens = word_tokens[:i] + [new_token] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            all_tokens.extend(word_tokens)
        
        return all_tokens


def train_bpe_tokenizer(
    texts: List[str],
    vocab_size: int = 10000,
    output_path: Optional[Path] = None
) -> BPETokenizer:
    """
    Train and save BPE tokenizer
    
    Args:
        texts: Training corpus
        vocab_size: Target vocabulary size
        output_path: Path to save tokenizer
        
    Returns:
        Trained tokenizer
    """
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    
    if output_path:
        tokenizer.save(output_path)
    
    return tokenizer


def create_tokenizer(corpus_path: str, vocab_size: int = 10000) -> BPETokenizer:
    """Factory function used by unit tests."""
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(corpus_path)
    return tokenizer


def load_tokenizer(path: str) -> BPETokenizer:
    """Load tokenizer from directory path (unit-test API)."""
    return BPETokenizer.load(path)


# Example usage
if __name__ == "__main__":
    # Sample training data
    texts = [
        "Hello world! This is a test.",
        "مرحبا بالعالم! هذا اختبار.",
        "def hello_world(): print('Hello')",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a great programming language.",
        "الذكاء الاصطناعي مجال مثير للاهتمام.",
    ]
    
    # Train tokenizer
    tokenizer = train_bpe_tokenizer(texts, vocab_size=500)
    
    # Test encoding
    test_text = "Hello world! مرحبا بالعالم"
    token_ids = tokenizer.encode(test_text)
    tokens = tokenizer.tokenize(test_text)
    decoded = tokenizer.decode(token_ids)
    
    print(f"\nTest text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded}")
