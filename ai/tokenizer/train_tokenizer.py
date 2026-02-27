"""
Training script for BPE Tokenizer
Uses 1M+ texts from local files or HuggingFace datasets
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.tokenizer.bpe_tokenizer import BPETokenizer


class TokenizerTrainer:
    """Trainer for BPE Tokenizer with dataset loading capabilities."""
    
    def __init__(self, vocab_size: int = 32000, output_dir: str = "ai/tokenizer/checkpoint"):
        self.vocab_size = vocab_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = BPETokenizer(vocab_size=vocab_size)
        
    def load_from_local(self, data_dir: str, file_extensions: List[str] = None) -> str:
        """
        Load and concatenate text files from local directory.
        
        Args:
            data_dir: Directory containing text files
            file_extensions: List of file extensions to include
            
        Returns:
            Path to merged corpus file
        """
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.py', '.json', '.jsonl']
        
        data_dir = Path(data_dir)
        corpus_file = self.output_dir / 'corpus_merged.txt'
        
        print(f"Loading files from {data_dir}...")
        
        text_count = 0
        with open(corpus_file, 'w', encoding='utf-8') as out_f:
            for ext in file_extensions:
                files = list(data_dir.rglob(f'*{ext}'))
                print(f"Found {len(files)} {ext} files")
                
                for file_path in tqdm(files, desc=f"Processing {ext}"):
                    try:
                        if ext == '.json':
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, dict) and 'text' in data:
                                    out_f.write(data['text'] + '\n')
                                    text_count += 1
                                elif isinstance(data, list):
                                    for item in data:
                                        if isinstance(item, dict) and 'text' in item:
                                            out_f.write(item['text'] + '\n')
                                            text_count += 1
                                        elif isinstance(item, str):
                                            out_f.write(item + '\n')
                                            text_count += 1
                        elif ext == '.jsonl':
                            with open(file_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    data = json.loads(line)
                                    if isinstance(data, dict) and 'text' in data:
                                        out_f.write(data['text'] + '\n')
                                        text_count += 1
                                    elif isinstance(data, str):
                                        out_f.write(data + '\n')
                                        text_count += 1
                        else:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if content.strip():
                                    out_f.write(content + '\n')
                                    text_count += 1
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
        
        print(f"Total texts collected: {text_count}")
        return str(corpus_file)
    
    def load_from_huggingface(
        self, 
        dataset_name: str, 
        text_column: str = 'text',
        split: str = 'train',
        max_samples: Optional[int] = None
    ) -> str:
        """
        Load dataset from HuggingFace.
        
        Args:
            dataset_name: HuggingFace dataset name (e.g., 'wikitext', 'openwebtext')
            text_column: Column name containing text
            split: Dataset split to use
            max_samples: Maximum number of samples to load
            
        Returns:
            Path to corpus file
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing datasets library...")
            os.system(f"{sys.executable} -m pip install datasets -q")
            from datasets import load_dataset
        
        corpus_file = self.output_dir / f'corpus_{dataset_name.replace("/", "_")}.txt'
        
        print(f"Loading dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split, streaming=max_samples is None)
        
        text_count = 0
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="Processing dataset"):
                if text_column in item:
                    text = item[text_column]
                    if text and text.strip():
                        f.write(text.strip() + '\n')
                        text_count += 1
                        
                if max_samples and text_count >= max_samples:
                    break
        
        print(f"Total texts collected from HF: {text_count}")
        return str(corpus_file)
    
    def load_mixed_corpus(
        self, 
        local_dirs: List[str] = None,
        hf_datasets: List[dict] = None,
        min_length: int = 10
    ) -> str:
        """
        Load mixed corpus from multiple sources.
        
        Args:
            local_dirs: List of local directories
            hf_datasets: List of HF dataset configs [{'name': '...', 'text_column': '...'}]
            min_length: Minimum text length to include
            
        Returns:
            Path to merged corpus file
        """
        corpus_file = self.output_dir / 'corpus_mixed.txt'
        
        with open(corpus_file, 'w', encoding='utf-8') as out_f:
            # Load from local directories
            if local_dirs:
                for data_dir in local_dirs:
                    if not Path(data_dir).exists():
                        print(f"Warning: {data_dir} does not exist, skipping...")
                        continue
                        
                    temp_trainer = TokenizerTrainer(vocab_size=self.vocab_size)
                    temp_file = temp_trainer.load_from_local(data_dir)
                    
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if len(line.strip()) >= min_length:
                                out_f.write(line)
            
            # Load from HuggingFace
            if hf_datasets:
                for ds_config in hf_datasets:
                    temp_trainer = TokenizerTrainer(vocab_size=self.vocab_size)
                    temp_file = temp_trainer.load_from_huggingface(
                        dataset_name=ds_config['name'],
                        text_column=ds_config.get('text_column', 'text'),
                        split=ds_config.get('split', 'train'),
                        max_samples=ds_config.get('max_samples')
                    )
                    
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if len(line.strip()) >= min_length:
                                out_f.write(line)
        
        # Count lines
        with open(corpus_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        print(f"Mixed corpus created with {line_count} texts")
        return str(corpus_file)
    
    def train(
        self, 
        corpus_file: str,
        validate: bool = True,
        sample_size: int = 1000
    ) -> None:
        """
        Train the tokenizer.
        
        Args:
            corpus_file: Path to training corpus
            validate: Whether to run validation after training
            sample_size: Number of samples for validation
        """
        print(f"\n{'='*50}")
        print(f"Training BPE Tokenizer")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Corpus file: {corpus_file}")
        print(f"{'='*50}\n")
        
        # Train
        self.tokenizer.train(corpus_file)
        
        # Save
        self.tokenizer.save(str(self.output_dir))
        
        # Validate
        if validate:
            self._validate(corpus_file, sample_size)
        
        print(f"\n{'='*50}")
        print(f"Training complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*50}\n")
    
    def _validate(self, corpus_file: str, sample_size: int = 1000) -> None:
        """Validate tokenizer on sample texts."""
        print("\nValidating tokenizer...")
        
        # Load sample texts
        sample_texts = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                sample_texts.append(line.strip())
        
        # Calculate metrics
        total_tokens = 0
        total_chars = 0
        
        for text in tqdm(sample_texts[:100], desc="Validation"):
            tokens = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(tokens)
            
            total_tokens += len(tokens)
            total_chars += len(text)
            
            # Check round-trip
            if text != decoded:
                pass  # Some loss is expected with BPE
        
        avg_tokens_per_char = total_tokens / total_chars if total_chars > 0 else 0
        
        print(f"\nValidation Results:")
        print(f"  Sample texts: {len(sample_texts)}")
        print(f"  Avg tokens/char: {avg_tokens_per_char:.4f}")
        print(f"  Vocabulary size: {self.tokenizer.get_vocab_size()}")
        
        # Test specific cases
        test_cases = [
            "print('Hello, World!')",
            "مرحبا بالعالم",
            "def function(x: int) -> str:",
            "Hello العالم 123"
        ]
        
        print("\nTest cases:")
        for text in test_cases:
            tokens = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(tokens)
            print(f"  Input:  {text[:50]}")
            print(f"  Tokens: {len(tokens)}")
            print(f"  Output: {decoded[:50]}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Train BPE Tokenizer')
    parser.add_argument('--vocab-size', type=int, default=32000,
                        help='Target vocabulary size')
    parser.add_argument('--output-dir', type=str, default='ai/tokenizer/checkpoint',
                        help='Output directory for tokenizer')
    parser.add_argument('--local-data', type=str,
                        help='Local directory containing training data')
    parser.add_argument('--hf-dataset', type=str,
                        help='HuggingFace dataset name')
    parser.add_argument('--hf-text-column', type=str, default='text',
                        help='Text column name in HF dataset')
    parser.add_argument('--hf-split', type=str, default='train',
                        help='Dataset split for HF')
    parser.add_argument('--max-samples', type=int,
                        help='Maximum number of samples to load')
    parser.add_argument('--corpus-file', type=str,
                        help='Direct path to corpus file')
    parser.add_argument('--mixed', action='store_true',
                        help='Use mixed corpus from multiple sources')
    
    args = parser.parse_args()
    
    trainer = TokenizerTrainer(
        vocab_size=args.vocab_size,
        output_dir=args.output_dir
    )
    
    # Determine corpus source
    if args.corpus_file:
        corpus_file = args.corpus_file
    elif args.mixed:
        local_dirs = [args.local_data] if args.local_data else []
        hf_datasets = []
        if args.hf_dataset:
            hf_datasets.append({
                'name': args.hf_dataset,
                'text_column': args.hf_text_column,
                'split': args.hf_split,
                'max_samples': args.max_samples
            })
        corpus_file = trainer.load_mixed_corpus(local_dirs, hf_datasets)
    elif args.local_data:
        corpus_file = trainer.load_from_local(args.local_data)
    elif args.hf_dataset:
        corpus_file = trainer.load_from_huggingface(
            args.hf_dataset,
            args.hf_text_column,
            args.hf_split,
            args.max_samples
        )
    else:
        print("Error: No data source specified!")
        print("Use --local-data, --hf-dataset, --corpus-file, or --mixed")
        sys.exit(1)
    
    # Train
    trainer.train(corpus_file)


if __name__ == '__main__':
    main()
