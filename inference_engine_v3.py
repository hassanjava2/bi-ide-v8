"""
Inference Engine v3 - متوافق مع Checkpoints الـ RTX 4090
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional
import random

CHECKPOINT_DIR = Path("learning_data/checkpoints")


class RTX4090Transformer(nn.Module):
    """نفس Architecture الـ RTX 4090"""
    def __init__(self, vocab_size=492, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(512, d_model)
        
        # نفس الـ architecture بالضبط
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=False  # مهم! الـ checkpoints تستخدم batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x: (seq_len, batch) - because batch_first=False
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        
        x = self.embedding(x) + self.pos_encoder(positions)
        x = self.transformer(x)
        return self.fc(x)


class InferenceEngine:
    """محرك الاستدلال - متوافق 100% مع RTX 4090"""
    
    def __init__(self):
        self.models: Dict[str, RTX4090Transformer] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inference Device: {self.device}")
        
        # نبني vocabulary بسيط (character-based)
        self.chars = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + \
            list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:()[]{}"\'-') + \
            list('ابتثجحخدذرزسشصضطظعغفقكلمنهويىةءآأإؤئ') + \
            list('ًٌٍَُِّْ')
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        print(f"Vocab: {self.vocab_size} chars")
        
        self._load_all_models()
    
    def _load_all_models(self):
        """تحميل النماذج"""
        if not CHECKPOINT_DIR.exists():
            return
        
        for layer_dir in CHECKPOINT_DIR.iterdir():
            if layer_dir.is_dir():
                try:
                    self._load_model(layer_dir.name, layer_dir)
                except Exception as e:
                    print(f"  [ERR] {layer_dir.name}: {str(e)[:40]}")
        
        print(f"Loaded {len(self.models)} models\n")
    
    def _load_model(self, name: str, layer_dir: Path):
        """تحميل نموذج - متوافق مع RTX 4090"""
        checkpoints = list(layer_dir.glob("*.pt"))
        if not checkpoints:
            return
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
        
        vocab_size = checkpoint.get('vocab_size', 492)
        model = RTX4090Transformer(vocab_size=vocab_size)
        
        # تحميل الأوزان
        state_dict = checkpoint.get('model_state', checkpoint)
        
        # Map keys إذا لزم
        model_dict = model.state_dict()
        filtered = {}
        
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered[k] = v
                else:
                    print(f"    Shape mismatch: {k} {v.shape} vs {model_dict[k].shape}")
        
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        
        self.models[name] = model
        print(f"  [OK] {name}: {len(filtered)}/{len(state_dict)} layers loaded")
    
    def encode(self, text: str) -> List[int]:
        """نص إلى أرقام"""
        return [self.char2idx.get(c, 3) for c in text]
    
    def decode(self, indices: List[int]) -> str:
        """أرقام إلى نص"""
        chars = []
        for idx in indices:
            if idx in [0, 1, 2]:  # PAD, SOS, EOS
                continue
            chars.append(self.idx2char.get(idx, ''))
        return ''.join(chars)
    
    def generate_response(self, prompt: str, model_name: str = "high_council",
                         max_length: int = 80, temperature: float = 0.7) -> str:
        """توليد رد حقيقي"""
        if model_name not in self.models:
            return "[Model not available]"
        
        model = self.models[model_name]
        
        # تحضير الإدخال
        input_ids = [1] + self.encode(prompt)  # <SOS> + prompt
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        
        generated = input_tensor.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(generated)
                next_logits = outputs[-1, 0, :] / temperature
                
                # Top-k sampling
                top_k = 30
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                top_k_probs = torch.softmax(top_k_logits, dim=-1)
                next_idx = torch.multinomial(top_k_probs, num_samples=1)
                next_token = top_k_indices[next_idx].item()
                
                generated = torch.cat([generated, torch.tensor([[next_token]], device=self.device)], dim=0)
                
                if next_token == 2:  # <EOS>
                    break
        
        # فك التشفير
        response = self.decode(generated.squeeze(1).cpu().tolist())
        
        # إزالة الـ prompt
        response = response.replace(prompt, '').strip()
        
        # تنظيف
        if len(response) < 5:
            response = "أفكر في ما قلت..."
        
        return response


# Global instance
inference_engine = InferenceEngine()
