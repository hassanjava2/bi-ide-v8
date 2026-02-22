"""
Inference Engine v4 - متوافق 100% مع Checkpoints RTX 4090
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict
import random

CHECKPOINT_DIR = Path("learning_data/checkpoints")


class RTX4090Transformer(nn.Module):
    """نفس Architecture الـ RTX 4090 تماماً"""
    def __init__(self, vocab_size=10000, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(512, d_model)
        
        # PyTorch Transformer - نفس الإعدادات
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4,
                dropout=dropout,
                batch_first=False  # مهم!
            ),
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x: (seq_len, batch)
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        
        x = self.embedding(x) + self.pos_encoder(positions)
        x = self.transformer(x)
        return self.fc(x)


class SimpleCharTokenizer:
    """Tokenizer بسيط"""
    def __init__(self):
        self.chars = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + \
            list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:()[]') + \
            list("'\"") + \
            list('ابتثجحخدذرزسشصضطظعغفقكلمنهويىةءآأإؤئ') + \
            list('ًٌٍَُِّْ')
        
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        
    def encode(self, text: str) -> List[int]:
        return [self.char2idx.get(c, 3) for c in text]  # 3 = <UNK>
    
    def decode(self, indices: List[int]) -> str:
        return ''.join(self.idx2char.get(i, '') for i in indices if i not in [0, 1, 2])


class InferenceEngine:
    """محرك الاستدلال"""
    
    def __init__(self):
        self.models: Dict[str, RTX4090Transformer] = {}
        self.tokenizer = SimpleCharTokenizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inference Device: {self.device}")
        
        self._load_models()
    
    def _load_models(self):
        """تحميل النماذج"""
        if not CHECKPOINT_DIR.exists():
            return
        
        for layer_dir in CHECKPOINT_DIR.iterdir():
            if layer_dir.is_dir():
                try:
                    self._load_model(layer_dir.name, layer_dir)
                except Exception as e:
                    print(f"  [ERR] {layer_dir.name}: {str(e)[:50]}")
        
        print(f"Loaded {len(self.models)} models\n")
    
    def _load_model(self, name: str, layer_dir: Path):
        """تحميل نموذج واحد"""
        checkpoints = list(layer_dir.glob("*.pt"))
        if not checkpoints:
            return
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
        
        # نستخدم vocab_size=10000 (نفس التدريب)
        model = RTX4090Transformer(vocab_size=10000)
        
        # تحميل الأوزان
        state_dict = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(state_dict, strict=True)  # strict=True لأننا متطابقين 100%
        
        model.to(self.device)
        model.eval()
        
        self.models[name] = model
        print(f"  [OK] {name}: Model loaded")
    
    def generate(self, prompt: str, model_name: str = "high_council",
                max_length: int = 60, temperature: float = 0.8) -> str:
        """توليد رد"""
        if model_name not in self.models:
            return "[Model not available]"
        
        model = self.models[model_name]
        
        # Tokenize
        input_ids = [1] + self.tokenizer.encode(prompt)  # <SOS> = 1
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        
        generated = input_tensor.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(generated)
                logits = outputs[-1, 0, :] / temperature
                
                # Top-k sampling
                top_k = 40
                top_logits, top_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_logits, dim=-1)
                next_token = top_indices[torch.multinomial(probs, 1)].item()
                
                generated = torch.cat([generated, torch.tensor([[next_token]], device=self.device)], dim=0)
                
                if next_token == 2:  # <EOS>
                    break
        
        # Decode
        response = self.tokenizer.decode(generated.squeeze(1).cpu().tolist())
        response = response.replace(prompt, '').strip()
        
        # Post-process
        response = response.replace('<UNK>', '').strip()
        if len(response) < 3:
            response = "أحتاج مزيداً من الوقت للتفكير..."
        
        return response


# Global
inference_engine = InferenceEngine()
