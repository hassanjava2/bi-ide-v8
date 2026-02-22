"""
Inference Engine v2 - Character-Level Tokenization
أكثر استقراراً للتوليد
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json

CHECKPOINT_DIR = Path("learning_data/checkpoints")

# Arabic + English + Numbers + Punctuation
ALL_CHARS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + \
    list('abcdefghijklmnopqrstuvwxyz') + \
    list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + \
    list('0123456789') + \
    list(' .,!?;:()[]{}"\'-') + \
    list('ابتثجحخدذرزسشصضطظعغفقكلمنهوي') + \
    list('ًٌٍَُِّْ') + ['ى', 'ة', 'ء', 'آ', 'أ', 'إ', 'ؤ', 'ئ']


class CharVocabulary:
    """Character-level Vocabulary"""
    def __init__(self):
        self.char2idx = {c: i for i, c in enumerate(ALL_CHARS)}
        self.idx2char = {i: c for i, c in enumerate(ALL_CHARS)}
        self.vocab_size = len(ALL_CHARS)
    
    def encode(self, text: str) -> List[int]:
        """نص إلى أرقام"""
        return [self.char2idx.get(c, self.char2idx['<UNK>']) for c in text]
    
    def decode(self, indices: List[int]) -> str:
        """أرقام إلى نص"""
        chars = []
        for idx in indices:
            if idx in [0, 1, 2]:  # PAD, SOS, EOS
                continue
            chars.append(self.idx2char.get(idx, '<UNK>'))
        return ''.join(chars)


class TransformerModel(nn.Module):
    """Transformer للتوليد"""
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(512, d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) + self.pos_encoder(positions)
        x = self.transformer(x)
        return self.fc(x)


class InferenceEngine:
    """محرك الاستدلال"""
    
    def __init__(self):
        self.models: Dict[str, TransformerModel] = {}
        self.vocab = CharVocabulary()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inference Device: {self.device}")
        print(f"Vocab Size: {self.vocab.vocab_size} (character-level)")
        
        self._load_all_models()
    
    def _load_all_models(self):
        """تحميل النماذج"""
        if not CHECKPOINT_DIR.exists():
            print("No checkpoints directory")
            return
        
        for layer_dir in CHECKPOINT_DIR.iterdir():
            if layer_dir.is_dir():
                layer_name = layer_dir.name
                try:
                    self._load_model(layer_name, layer_dir)
                except Exception as e:
                    print(f"  Failed {layer_name}: {str(e)[:40]}")
        
        print(f"Loaded {len(self.models)} inference models\n")
    
    def _load_model(self, name: str, layer_dir: Path):
        """تحميل نموذج"""
        checkpoints = sorted(layer_dir.glob("*.pt"))
        if not checkpoints:
            return
        
        latest = checkpoints[-1]
        checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
        
        vocab_size = checkpoint.get('vocab_size', self.vocab.vocab_size)
        model = TransformerModel(vocab_size=vocab_size)
        
        state_dict = checkpoint.get('model_state', checkpoint)
        model_dict = model.state_dict()
        
        # Filter matching keys
        filtered = {k: v for k, v in state_dict.items() 
                   if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        
        self.models[name] = model
        print(f"  [OK] {name}: {len(filtered)}/{len(state_dict)} layers")
    
    def generate_response(self, prompt: str, model_name: str = "high_council", 
                         max_length: int = 100, temperature: float = 0.8) -> str:
        """توليد رد"""
        if model_name not in self.models:
            return "[Model not found]"
        
        model = self.models[model_name]
        
        # تحضير الإدخال
        input_ids = self.vocab.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # توليد
        model.eval()
        generated = input_tensor.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(generated)
                next_logits = outputs[:, -1, :] / temperature
                
                # Top-k sampling
                top_k = 20
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = -float('Inf')
                
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                # توقف عند EOS
                if next_token.item() == 2:
                    break
        
        # فك التشفير
        response = self.vocab.decode(generated[0].cpu().tolist())
        
        # إزالة الـ prompt
        response = response.replace(prompt, '').strip()
        
        # تنظيف
        if not response or len(response) < 5:
            response = "أفكر في ما قلت..."
        
        return response
    
    def get_status(self):
        return {
            'device': str(self.device),
            'models': len(self.models),
            'vocab_size': self.vocab.vocab_size
        }


# Global
inference_engine = InferenceEngine()
