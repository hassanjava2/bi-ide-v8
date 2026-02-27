"""
Data Preprocessing Module
Text cleaning, tokenization, batching, and augmentation
"""

import re
import random
from typing import List, Dict, Optional, Iterator, Callable, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing."""
    # Text cleaning
    normalize_whitespace: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    lowercase: bool = False
    
    # Tokenization
    max_length: int = 512
    padding: str = 'max_length'  # 'max_length', 'longest', 'none'
    truncation: bool = True
    
    # Augmentation
    enable_augmentation: bool = False
    augment_prob: float = 0.1
    
    # Special tokens
    bos_token: str = '<BOS>'
    eos_token: str = '<EOS>'
    pad_token: str = '<PAD>'
    unk_token: str = '<UNK>'


class TextCleaner:
    """Clean and normalize text."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def clean(self, text: str) -> str:
        """Clean text according to configuration."""
        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emails
        if self.config.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = ' '.join(text.split())
        
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text."""
        # Normalize alef variants
        text = re.sub('[ٱإأآا]', 'ا', text)
        # Normalize alef maksura
        text = re.sub('ى', 'ي', text)
        # Normalize teh marbuta
        text = re.sub('ة', 'ه', text)
        # Remove tatweel
        text = re.sub('ـ', '', text)
        # Remove extra spaces
        text = re.sub('\s+', ' ', text)
        return text.strip()
    
    def clean_code(self, code: str) -> str:
        """Clean code snippets."""
        # Remove excessive blank lines
        code = re.sub(r'\n\s*\n\s*\n+', '\n\n', code)
        # Normalize indentation (convert tabs to spaces)
        code = code.replace('\t', '    ')
        # Remove trailing whitespace
        code = '\n'.join(line.rstrip() for line in code.split('\n'))
        return code.strip()


class DataAugmenter:
    """Augment training data."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def augment(self, text: str) -> str:
        """Apply random augmentations."""
        if random.random() > self.config.augment_prob:
            return text
        
        augmentations = [
            self._synonym_replacement,
            self._random_insertion,
            self._random_swap,
            self._random_deletion
        ]
        
        # Apply 1-2 random augmentations
        num_aug = random.randint(1, 2)
        for _ in range(num_aug):
            aug_func = random.choice(augmentations)
            text = aug_func(text)
        
        return text
    
    def _synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace words with synonyms."""
        words = text.split()
        if len(words) < 2:
            return text
        
        # Simple synonym map (expand in production)
        synonyms = {
            'good': ['great', 'excellent', 'fine'],
            'bad': ['poor', 'terrible', 'awful'],
            'big': ['large', 'huge', 'massive'],
            'small': ['tiny', 'little', 'compact'],
            'fast': ['quick', 'rapid', 'swift'],
            'slow': ['gradual', 'sluggish'],
        }
        
        new_words = words.copy()
        random.shuffle(new_words)
        
        num_replaced = 0
        for i, word in enumerate(new_words):
            if word.lower() in synonyms:
                synonym = random.choice(synonyms[word.lower()])
                # Find and replace in original
                for j, orig_word in enumerate(words):
                    if orig_word.lower() == word.lower():
                        words[j] = synonym
                        num_replaced += 1
                        break
            
            if num_replaced >= n:
                break
        
        return ' '.join(words)
    
    def _random_insertion(self, text: str, n: int = 1) -> str:
        """Randomly insert words."""
        words = text.split()
        if len(words) < 3:
            return text
        
        for _ in range(n):
            # Insert a filler word
            fillers = ['very', 'quite', 'really', 'actually', 'indeed']
            idx = random.randint(0, len(words))
            words.insert(idx, random.choice(fillers))
        
        return ' '.join(words)
    
    def _random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap words."""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words."""
        words = text.split()
        if len(words) <= 1:
            return text
        
        new_words = [w for w in words if random.random() > p]
        
        if not new_words:
            return random.choice(words)
        
        return ' '.join(new_words)


class DataPreprocessor:
    """
    Main preprocessing pipeline.
    """
    
    def __init__(
        self,
        tokenizer = None,
        config: Optional[PreprocessingConfig] = None
    ):
        self.config = config or PreprocessingConfig()
        self.tokenizer = tokenizer
        self.cleaner = TextCleaner(self.config)
        self.augmenter = DataAugmenter(self.config)
    
    def preprocess(
        self,
        text: str,
        apply_augmentation: bool = False
    ) -> Dict[str, Any]:
        """
        Preprocess single text sample.
        
        Returns:
            Dictionary with processed data
        """
        # Clean
        text = self.cleaner.clean(text)
        
        # Augment
        if apply_augmentation and self.config.enable_augmentation:
            text = self.augmenter.augment(text)
        
        result = {
            'text': text,
            'length': len(text),
        }
        
        # Tokenize if tokenizer available
        if self.tokenizer:
            tokens = self.tokenize(text)
            result['tokens'] = tokens
            result['token_count'] = len(tokens)
        
        return result
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")
        
        # Use BPE tokenizer if available
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text)
        else:
            # Fallback to simple tokenization
            return text.split()
    
    def create_batches(
        self,
        data: List[Dict[str, Any]],
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Create batches from preprocessed data.
        
        Args:
            data: List of preprocessed samples
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of samples
        """
        if shuffle:
            random.shuffle(data)
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Pad/truncate batch
            if self.tokenizer:
                batch = self._pad_batch(batch)
            
            yield batch
    
    def _pad_batch(self, batch: List[Dict]) -> List[Dict]:
        """Pad batch to same length."""
        if not batch:
            return batch
        
        # Find max length in batch
        max_len = max(len(s.get('tokens', [])) for s in batch)
        
        # Respect max_length config
        if self.config.max_length:
            max_len = min(max_len, self.config.max_length)
        
        pad_id = 0  # Assuming 0 is pad token
        
        for sample in batch:
            tokens = sample.get('tokens', [])
            
            # Truncate
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            
            # Pad
            if len(tokens) < max_len:
                tokens = tokens + [pad_id] * (max_len - len(tokens))
            
            sample['tokens'] = tokens
            sample['attention_mask'] = [1 if t != pad_id else 0 for t in tokens]
        
        return batch
    
    def preprocess_dataset(
        self,
        input_file: str,
        output_file: str,
        text_column: str = 'text',
        apply_augmentation: bool = False
    ) -> Dict[str, int]:
        """
        Preprocess entire dataset file.
        
        Args:
            input_file: Input JSONL file
            output_file: Output JSONL file
            text_column: Column containing text
            apply_augmentation: Whether to augment data
            
        Returns:
            Statistics dict
        """
        stats = {
            'processed': 0,
            'filtered': 0,
            'augmented': 0,
            'total_tokens': 0
        }
        
        input_path = Path(input_file)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_file, 'r', encoding='utf-8') as in_f, \
             open(output_file, 'w', encoding='utf-8') as out_f:
            
            for line in in_f:
                try:
                    data = json.loads(line)
                    text = data.get(text_column, '')
                    
                    # Preprocess
                    processed = self.preprocess(text, apply_augmentation)
                    
                    # Skip empty
                    if not processed['text']:
                        stats['filtered'] += 1
                        continue
                    
                    # Merge with original metadata
                    output_data = {**data, **processed}
                    
                    out_f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    
                    stats['processed'] += 1
                    if 'token_count' in processed:
                        stats['total_tokens'] += processed['token_count']
                    
                    if apply_augmentation and processed.get('augmented'):
                        stats['augmented'] += 1
                
                except Exception as e:
                    print(f"Error processing line: {e}")
                    stats['filtered'] += 1
        
        print(f"Preprocessing complete:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Filtered: {stats['filtered']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        
        return stats
    
    def prepare_training_data(
        self,
        data: List[str],
        batch_size: int = 32,
        validation_split: float = 0.1
    ) -> Tuple[Iterator, Iterator]:
        """
        Prepare data for training.
        
        Returns:
            (train_batches, val_batches)
        """
        # Preprocess all
        processed = [self.preprocess(text) for text in data]
        
        # Split
        split_idx = int(len(processed) * (1 - validation_split))
        train_data = processed[:split_idx]
        val_data = processed[split_idx:]
        
        # Create batch iterators
        train_batches = self.create_batches(train_data, batch_size, shuffle=True)
        val_batches = self.create_batches(val_data, batch_size, shuffle=False)
        
        return train_batches, val_batches


def create_preprocessing_pipeline(
    tokenizer = None,
    max_length: int = 512,
    enable_augmentation: bool = False
) -> DataPreprocessor:
    """Create preprocessing pipeline with default config."""
    config = PreprocessingConfig(
        max_length=max_length,
        enable_augmentation=enable_augmentation
    )
    return DataPreprocessor(tokenizer=tokenizer, config=config)


if __name__ == '__main__':
    print("Preprocessing Module Demo")
    print("="*50)
    
    # Create preprocessor
    config = PreprocessingConfig(
        normalize_whitespace=True,
        enable_augmentation=True
    )
    preprocessor = DataPreprocessor(config=config)
    
    # Sample texts
    texts = [
        "  This   is   a   test   with   extra   spaces.  ",
        "Check out https://example.com for more info.",
        "Contact me at email@example.com please.",
    ]
    
    # Preprocess
    for text in texts:
        processed = preprocessor.preprocess(text, apply_augmentation=True)
        print(f"Input:  {text}")
        print(f"Output: {processed['text']}")
        print()
