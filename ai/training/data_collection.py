"""
Data Collection Module
Collect and process training data from multiple sources
"""

import json
import hashlib
import re
from typing import List, Dict, Optional, Iterator, Set, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
import string


@dataclass
class DataSample:
    """Single training data sample."""
    id: str
    text: str
    source: str
    metadata: Dict
    quality_score: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'source': self.source,
            'metadata': self.metadata,
            'quality_score': self.quality_score,
            'timestamp': self.timestamp.isoformat()
        }


class QualityFilter:
    """Filter data samples based on quality criteria."""
    
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 10000,
        min_word_count: int = 10,
        max_repetition_ratio: float = 0.5,
        require_code_ratio: Optional[float] = None
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_word_count = min_word_count
        self.max_repetition_ratio = max_repetition_ratio
        self.require_code_ratio = require_code_ratio
    
    def filter(self, text: str) -> Tuple[bool, float, str]:
        """
        Filter text and return quality score.
        
        Returns:
            (is_valid, quality_score, reason)
        """
        # Length check
        if len(text) < self.min_length:
            return False, 0.0, f"too_short ({len(text)} < {self.min_length})"
        
        if len(text) > self.max_length:
            return False, 0.0, f"too_long ({len(text)} > {self.max_length})"
        
        # Word count check
        words = text.split()
        if len(words) < self.min_word_count:
            return False, 0.0, f"too_few_words ({len(words)} < {self.min_word_count})"
        
        # Repetition check
        unique_words = set(word.lower() for word in words)
        repetition_ratio = 1 - len(unique_words) / len(words)
        if repetition_ratio > self.max_repetition_ratio:
            return False, 0.0, f"too_repetitive ({repetition_ratio:.2f})"
        
        # Code ratio check (if specified)
        if self.require_code_ratio is not None:
            code_chars = len(re.findall(r'[{}();=+\-*/<>]', text))
            code_ratio = code_chars / len(text)
            if code_ratio < self.require_code_ratio:
                return False, 0.0, f"insufficient_code ({code_ratio:.2f})"
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(text, repetition_ratio)
        
        return True, quality_score, "passed"
    
    def _calculate_quality_score(self, text: str, repetition_ratio: float) -> float:
        """Calculate overall quality score."""
        score = 1.0
        
        # Penalize repetition
        score -= repetition_ratio * 0.5
        
        # Reward code content
        code_indicators = ['def ', 'class ', 'function', 'import ', 'return']
        code_score = sum(1 for ind in code_indicators if ind in text) / len(code_indicators)
        score += code_score * 0.2
        
        # Reward balanced length
        optimal_length = 2000
        length_penalty = abs(len(text) - optimal_length) / optimal_length
        score -= length_penalty * 0.1
        
        return max(0.0, min(1.0, score))


class Deduplicator:
    """Remove duplicate or near-duplicate samples."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
        self.seen_minhashes: List[Set] = []
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        # Exact hash check
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        
        # Near-duplicate check using MinHash-like approach
        text_shingles = self._get_shingles(text)
        
        for seen_shingles in self.seen_minhashes:
            similarity = self._jaccard_similarity(text_shingles, seen_shingles)
            if similarity > self.similarity_threshold:
                return True
        
        # Add to seen
        self.seen_hashes.add(text_hash)
        self.seen_minhashes.append(text_shingles)
        
        return False
    
    def _get_shingles(self, text: str, k: int = 5) -> Set[str]:
        """Get k-gram shingles from text."""
        text = text.lower()
        return set(text[i:i+k] for i in range(len(text) - k + 1))
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity."""
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0


class DataCollector:
    """
    Collect training data from multiple sources.
    """
    
    def __init__(
        self,
        quality_filter: Optional[QualityFilter] = None,
        deduplicator: Optional[Deduplicator] = None
    ):
        self.quality_filter = quality_filter or QualityFilter()
        self.deduplicator = deduplicator or Deduplicator()
        self.collected_samples: List[DataSample] = []
        self.stats = {
            'total_collected': 0,
            'filtered_out': 0,
            'duplicates_removed': 0,
            'by_source': {}
        }
    
    def collect_from_user_interactions(
        self,
        conversation_store,
        min_rating: int = 4
    ) -> List[DataSample]:
        """
        Collect data from user interactions.
        
        Args:
            conversation_store: Conversation store instance
            min_rating: Minimum rating to include
            
        Returns:
            List of data samples
        """
        print("Collecting from user interactions...")
        
        samples = []
        conversations = conversation_store.get_all_profiles() if hasattr(conversation_store, 'get_all_profiles') else []
        
        for conversation in conversations:
            # Process conversation messages
            for msg in conversation.messages:
                if msg.get('role') == 'user':
                    text = msg.get('content', '')
                    
                    # Quality filter
                    is_valid, quality_score, reason = self.quality_filter.filter(text)
                    
                    if not is_valid:
                        self.stats['filtered_out'] += 1
                        continue
                    
                    # Deduplicate
                    if self.deduplicator.is_duplicate(text):
                        self.stats['duplicates_removed'] += 1
                        continue
                    
                    sample = DataSample(
                        id=self._generate_id(text),
                        text=text,
                        source='user_interaction',
                        metadata={
                            'conversation_id': conversation.id,
                            'user_id': conversation.user_id,
                            'timestamp': msg.get('timestamp')
                        },
                        quality_score=quality_score,
                        timestamp=datetime.now()
                    )
                    
                    samples.append(sample)
                    self.collected_samples.append(sample)
                    self.stats['total_collected'] += 1
        
        self.stats['by_source']['user_interaction'] = len(samples)
        print(f"  Collected {len(samples)} samples from user interactions")
        
        return samples
    
    def collect_from_files(
        self,
        directory: str,
        file_extensions: List[str] = None,
        source_name: str = 'local_files'
    ) -> List[DataSample]:
        """
        Collect data from local files.
        
        Args:
            directory: Directory to search
            file_extensions: File extensions to include
            source_name: Name for source tracking
            
        Returns:
            List of data samples
        """
        print(f"Collecting from {directory}...")
        
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.py', '.json', '.jsonl']
        
        samples = []
        directory = Path(directory)
        
        for ext in file_extensions:
            for file_path in directory.rglob(f'*{ext}'):
                try:
                    file_samples = self._process_file(file_path, source_name)
                    samples.extend(file_samples)
                except Exception as e:
                    print(f"  Error processing {file_path}: {e}")
        
        self.stats['by_source'][source_name] = len(samples)
        print(f"  Collected {len(samples)} samples from files")
        
        return samples
    
    def _process_file(self, file_path: Path, source: str) -> List[DataSample]:
        """Process a single file."""
        samples = []
        ext = file_path.suffix
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if ext == '.json':
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text = item.get('text', str(item))
                        sample = self._process_text(text, source, {'file': str(file_path)})
                        if sample:
                            samples.append(sample)
                elif isinstance(data, dict):
                    text = data.get('text', str(data))
                    sample = self._process_text(text, source, {'file': str(file_path)})
                    if sample:
                        samples.append(sample)
            
            elif ext == '.jsonl':
                for line in f:
                    data = json.loads(line)
                    text = data.get('text', str(data))
                    sample = self._process_text(text, source, {'file': str(file_path)})
                    if sample:
                        samples.append(sample)
            
            else:
                # Text file
                content = f.read()
                # Split into chunks if too large
                chunks = self._chunk_text(content, max_length=5000)
                for chunk in chunks:
                    sample = self._process_text(chunk, source, {'file': str(file_path)})
                    if sample:
                        samples.append(sample)
        
        return samples
    
    def _process_text(
        self,
        text: str,
        source: str,
        metadata: Dict
    ) -> Optional[DataSample]:
        """Process and filter text."""
        # Quality filter
        is_valid, quality_score, reason = self.quality_filter.filter(text)
        
        if not is_valid:
            self.stats['filtered_out'] += 1
            return None
        
        # Deduplicate
        if self.deduplicator.is_duplicate(text):
            self.stats['duplicates_removed'] += 1
            return None
        
        sample = DataSample(
            id=self._generate_id(text),
            text=text,
            source=source,
            metadata=metadata,
            quality_score=quality_score,
            timestamp=datetime.now()
        )
        
        self.collected_samples.append(sample)
        self.stats['total_collected'] += 1
        
        return sample
    
    def collect_from_huggingface(
        self,
        dataset_name: str,
        text_column: str = 'text',
        split: str = 'train',
        max_samples: Optional[int] = None,
        streaming: bool = True
    ) -> List[DataSample]:
        """
        Collect data from HuggingFace datasets.
        
        Args:
            dataset_name: HF dataset name
            text_column: Column containing text
            split: Dataset split
            max_samples: Maximum samples to collect
            streaming: Use streaming mode
            
        Returns:
            List of data samples
        """
        print(f"Collecting from HuggingFace dataset: {dataset_name}...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            print("  datasets library not available")
            return []
        
        samples = []
        
        try:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
            
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                
                text = item.get(text_column, '')
                
                # Quality filter
                is_valid, quality_score, reason = self.quality_filter.filter(text)
                
                if not is_valid:
                    self.stats['filtered_out'] += 1
                    continue
                
                # Deduplicate
                if self.deduplicator.is_duplicate(text):
                    self.stats['duplicates_removed'] += 1
                    continue
                
                sample = DataSample(
                    id=self._generate_id(text),
                    text=text,
                    source=f'hf:{dataset_name}',
                    metadata={'dataset': dataset_name, 'split': split},
                    quality_score=quality_score,
                    timestamp=datetime.now()
                )
                
                samples.append(sample)
                self.collected_samples.append(sample)
                self.stats['total_collected'] += 1
                
                if i % 1000 == 0:
                    print(f"  Processed {i} items, collected {len(samples)}")
        
        except Exception as e:
            print(f"  Error loading dataset: {e}")
        
        self.stats['by_source'][f'hf:{dataset_name}'] = len(samples)
        print(f"  Collected {len(samples)} samples from {dataset_name}")
        
        return samples
    
    def generate_synthetic_data(
        self,
        num_samples: int,
        generator_func: Optional[Callable] = None,
        topics: List[str] = None
    ) -> List[DataSample]:
        """
        Generate synthetic training data.
        
        Args:
            num_samples: Number of samples to generate
            generator_func: Custom generator function
            topics: Topics for synthetic generation
            
        Returns:
            List of synthetic data samples
        """
        print(f"Generating {num_samples} synthetic samples...")
        
        samples = []
        
        if generator_func:
            for i in range(num_samples):
                text = generator_func()
                sample = self._process_text(text, 'synthetic', {'generated': True})
                if sample:
                    samples.append(sample)
        else:
            # Simple synthetic generation (for demo)
            templates = [
                "How do I {action} in {language}?",
                "What is the best way to {action}?",
                "Can you explain {concept} in {language}?",
                "I'm having trouble with {problem} in {context}.",
            ]
            
            actions = ['implement', 'optimize', 'debug', 'test', 'deploy']
            languages = ['Python', 'JavaScript', 'Java', 'C++', 'Go']
            concepts = ['recursion', 'inheritance', 'async/await', 'memory management']
            problems = ['performance issues', 'memory leaks', 'compilation errors']
            contexts = ['production', 'development', 'testing']
            
            for i in range(num_samples):
                template = random.choice(templates)
                text = template.format(
                    action=random.choice(actions),
                    language=random.choice(languages),
                    concept=random.choice(concepts),
                    problem=random.choice(problems),
                    context=random.choice(contexts)
                )
                
                sample = self._process_text(text, 'synthetic', {'generated': True})
                if sample:
                    samples.append(sample)
        
        self.stats['by_source']['synthetic'] = len(samples)
        print(f"  Generated {len(samples)} synthetic samples")
        
        return samples
    
    def save_dataset(
        self,
        filepath: str,
        min_quality: float = 0.0
    ) -> str:
        """
        Save collected dataset to file.
        
        Args:
            filepath: Output file path
            min_quality: Minimum quality score to include
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Filter by quality
        filtered_samples = [
            s for s in self.collected_samples
            if s.quality_score >= min_quality
        ]
        
        # Save as JSONL
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in filtered_samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        
        print(f"Saved {len(filtered_samples)} samples to {filepath}")
        
        return str(filepath)
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return self.stats.copy()
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID."""
        return hashlib.md5(f"{text}_{datetime.now().timestamp()}".encode()).hexdigest()[:16]
    
    def _chunk_text(self, text: str, max_length: int = 5000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            
            # Try to break at sentence boundary
            if end < len(text):
                for i in range(end, max(start, end - 200), -1):
                    if text[i] in '.!?' and i + 1 < len(text) and text[i + 1] == ' ':
                        end = i + 1
                        break
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks


if __name__ == '__main__':
    print("Data Collection Module Demo")
    print("="*50)
    
    # Create collector
    collector = DataCollector()
    
    # Generate synthetic data
    synthetic = collector.generate_synthetic_data(100)
    
    # Print stats
    stats = collector.get_stats()
    print(f"\nCollection Stats:")
    print(f"  Total collected: {stats['total_collected']}")
    print(f"  Filtered out: {stats['filtered_out']}")
    print(f"  Duplicates: {stats['duplicates_removed']}")
    print(f"  By source: {stats['by_source']}")
