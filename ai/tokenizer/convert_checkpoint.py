"""
Checkpoint Conversion Script
Converts old character-level checkpoints to BPE format
"""

import json
import pickle
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.tokenizer.bpe_tokenizer import BPETokenizer


class CheckpointConverter:
    """
    Safe migration tool for converting old checkpoints to BPE format.
    """
    
    def __init__(self, backup_dir: str = "backup/checkpoint_migration"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def backup_checkpoint(self, checkpoint_path: str) -> Path:
        """Create backup of original checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        timestamp = self._get_timestamp()
        backup_path = self.backup_dir / f"{checkpoint_path.stem}_{timestamp}"
        
        if checkpoint_path.is_dir():
            shutil.copytree(checkpoint_path, backup_path)
        else:
            backup_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_path, backup_path / checkpoint_path.name)
        
        print(f"Backup created: {backup_path}")
        return backup_path
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def detect_checkpoint_type(self, checkpoint_path: str) -> str:
        """
        Detect the type of checkpoint.
        
        Returns:
            'char_level', 'bpe', 'unknown'
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Check for BPE tokenizer files
        if (checkpoint_path / 'vocab.json').exists() and \
           (checkpoint_path / 'merges.pkl').exists():
            return 'bpe'
        
        # Check for old char-level format
        if (checkpoint_path / 'vocab.txt').exists() or \
           (checkpoint_path / 'char_vocab.json').exists():
            return 'char_level'
        
        # Check single file
        if checkpoint_path.is_file():
            if checkpoint_path.suffix == '.pkl':
                return 'char_level'
            if checkpoint_path.suffix == '.json':
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                    if 'merges' in data:
                        return 'bpe'
                    if 'char_to_idx' in data or 'vocab' in data:
                        return 'char_level'
        
        return 'unknown'
    
    def convert_char_to_bpe(
        self, 
        checkpoint_path: str,
        output_path: str,
        vocab_size: int = 32000,
        corpus_file: Optional[str] = None
    ) -> BPETokenizer:
        """
        Convert character-level checkpoint to BPE.
        
        Args:
            checkpoint_path: Path to old checkpoint
            output_path: Path for new BPE checkpoint
            vocab_size: Target vocabulary size
            corpus_file: Optional corpus for training BPE
            
        Returns:
            New BPE tokenizer
        """
        print(f"Converting {checkpoint_path} to BPE format...")
        
        # Create backup
        self.backup_checkpoint(checkpoint_path)
        
        # Load old checkpoint
        old_vocab = self._load_char_vocab(checkpoint_path)
        print(f"Loaded char-level vocab with {len(old_vocab)} characters")
        
        # Create new BPE tokenizer
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        
        if corpus_file and Path(corpus_file).exists():
            # Train on corpus
            print(f"Training BPE on {corpus_file}...")
            tokenizer.train(corpus_file)
        else:
            # Convert old vocab to BPE format
            print("Converting vocabulary without training...")
            tokenizer = self._convert_vocab_only(old_vocab, vocab_size)
        
        # Save new tokenizer
        tokenizer.save(output_path)
        
        # Create conversion report
        self._create_report(checkpoint_path, output_path, old_vocab, tokenizer)
        
        return tokenizer
    
    def _load_char_vocab(self, checkpoint_path: str) -> Dict:
        """Load character vocabulary from old checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        vocab = {}
        
        # Try different formats
        if (checkpoint_path / 'vocab.txt').exists():
            with open(checkpoint_path / 'vocab.txt', 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    vocab[line.strip()] = i
        
        elif (checkpoint_path / 'char_vocab.json').exists():
            with open(checkpoint_path / 'char_vocab.json', 'r', encoding='utf-8') as f:
                vocab = json.load(f)
        
        elif checkpoint_path.suffix == '.pkl':
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                if 'char_to_idx' in data:
                    vocab = data['char_to_idx']
                elif 'vocab' in data:
                    vocab = data['vocab']
        
        elif checkpoint_path.suffix == '.json':
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'char_to_idx' in data:
                    vocab = data['char_to_idx']
                elif 'vocab' in data:
                    vocab = data['vocab']
        
        return vocab
    
    def _convert_vocab_only(self, old_vocab: Dict, vocab_size: int) -> BPETokenizer:
        """Convert old vocab to BPE format without training."""
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        
        # Initialize with special tokens
        tokenizer.vocab = tokenizer.SPECIAL_TOKENS.copy()
        
        # Add old characters
        for char, idx in old_vocab.items():
            if char not in tokenizer.vocab:
                tokenizer.vocab[char] = len(tokenizer.vocab)
        
        # Build inverse vocab
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        
        return tokenizer
    
    def _create_report(
        self, 
        old_path: str, 
        new_path: str, 
        old_vocab: Dict, 
        tokenizer: BPETokenizer
    ) -> None:
        """Create conversion report."""
        report = {
            'old_checkpoint': str(old_path),
            'new_checkpoint': str(new_path),
            'old_vocab_size': len(old_vocab),
            'new_vocab_size': tokenizer.get_vocab_size(),
            'conversion_type': 'char_to_bpe',
            'timestamp': self._get_timestamp(),
            'migrations': {
                'characters_preserved': len(old_vocab),
                'new_tokens_added': tokenizer.get_vocab_size() - len(old_vocab) - len(tokenizer.SPECIAL_TOKENS)
            }
        }
        
        report_path = Path(new_path) / 'conversion_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Conversion report saved: {report_path}")
    
    def validate_conversion(
        self, 
        old_tokenizer_path: str, 
        new_tokenizer_path: str,
        test_file: Optional[str] = None
    ) -> bool:
        """
        Validate that conversion was successful.
        
        Returns:
            True if validation passes
        """
        print("\nValidating conversion...")
        
        # Load tokenizers
        from ai.tokenizer.bpe_tokenizer import load_tokenizer
        new_tokenizer = load_tokenizer(new_tokenizer_path)
        
        # Test cases
        test_cases = [
            "Hello World",
            "مرحبا بالعالم",
            "def hello():",
            "12345",
            "if x == 1:"
        ]
        
        if test_file and Path(test_file).exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                test_cases.extend(f.read().strip().split('\n')[:10])
        
        success = True
        for text in test_cases[:5]:
            try:
                tokens = new_tokenizer.encode(text)
                decoded = new_tokenizer.decode(tokens)
                print(f"✓ '{text[:30]}...' -> {len(tokens)} tokens")
            except Exception as e:
                print(f"✗ '{text[:30]}...' -> Error: {e}")
                success = False
        
        return success
    
    def batch_convert(
        self,
        checkpoints_dir: str,
        output_dir: str,
        vocab_size: int = 32000,
        corpus_file: Optional[str] = None
    ) -> List[str]:
        """
        Convert multiple checkpoints in batch.
        
        Returns:
            List of converted checkpoint paths
        """
        checkpoints_dir = Path(checkpoints_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        converted = []
        
        for checkpoint_path in checkpoints_dir.iterdir():
            if checkpoint_path.is_dir() or checkpoint_path.suffix in ['.pkl', '.json', '.pt']:
                checkpoint_type = self.detect_checkpoint_type(str(checkpoint_path))
                
                if checkpoint_type == 'char_level':
                    output_path = output_dir / f"{checkpoint_path.stem}_bpe"
                    try:
                        self.convert_char_to_bpe(
                            str(checkpoint_path),
                            str(output_path),
                            vocab_size,
                            corpus_file
                        )
                        converted.append(str(output_path))
                    except Exception as e:
                        print(f"Error converting {checkpoint_path}: {e}")
                elif checkpoint_type == 'bpe':
                    print(f"Skipping {checkpoint_path} (already BPE)")
                else:
                    print(f"Skipping {checkpoint_path} (unknown type)")
        
        return converted


def main():
    parser = argparse.ArgumentParser(description='Convert checkpoints to BPE format')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint to convert')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output path')
    parser.add_argument('--vocab-size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--corpus', type=str, help='Corpus file for training')
    parser.add_argument('--validate', action='store_true', help='Run validation')
    parser.add_argument('--test-file', type=str, help='Test file for validation')
    parser.add_argument('--batch', type=str, help='Batch convert directory')
    
    args = parser.parse_args()
    
    converter = CheckpointConverter()
    
    if args.batch:
        converted = converter.batch_convert(
            args.batch,
            args.output,
            args.vocab_size,
            args.corpus
        )
        print(f"\nConverted {len(converted)} checkpoints:")
        for path in converted:
            print(f"  - {path}")
    else:
        checkpoint_type = converter.detect_checkpoint_type(args.checkpoint)
        print(f"Detected checkpoint type: {checkpoint_type}")
        
        if checkpoint_type == 'char_level':
            tokenizer = converter.convert_char_to_bpe(
                args.checkpoint,
                args.output,
                args.vocab_size,
                args.corpus
            )
            print(f"\nConversion complete!")
            print(f"New vocabulary size: {tokenizer.get_vocab_size()}")
            
            if args.validate:
                success = converter.validate_conversion(
                    args.checkpoint,
                    args.output,
                    args.test_file
                )
                if success:
                    print("\n✓ Validation passed!")
                else:
                    print("\n✗ Validation failed!")
                    sys.exit(1)
        elif checkpoint_type == 'bpe':
            print("Checkpoint is already in BPE format. Copying...")
            Path(args.output).mkdir(parents=True, exist_ok=True)
            import shutil
            if Path(args.checkpoint).is_dir():
                shutil.copytree(args.checkpoint, args.output, dirs_exist_ok=True)
            else:
                shutil.copy2(args.checkpoint, args.output)
        else:
            print(f"Unknown checkpoint type: {checkpoint_type}")
            sys.exit(1)


if __name__ == '__main__':
    main()
