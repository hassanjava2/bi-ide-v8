"""
Inference Engine - محرك الاستدلال الحقيقي
يستخدم Checkpoints من RTX 4090 للتوليد
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json
import pickle

CHECKPOINT_DIR = Path("learning_data/checkpoints")
VOCAB_PATH = Path("learning_data/vocab.pkl")


class TransformerModel(nn.Module):
    """Transformer للتوليد"""
    def __init__(self, vocab_size=10000, d_model=256, nhead=8, num_layers=4, dropout=0.1):
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
    
    def generate(self, input_ids, max_length=100, temperature=0.8, top_k=50):
        """توليد نص جديد"""
        self.eval()
        device = next(self.parameters()).device
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(generated)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Top-k sampling
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Stop if EOS (assuming EOS token is 2)
                if next_token.item() == 2:
                    break
        
        return generated


class Vocabulary:
    """إدارة المفردات"""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
    def build_from_texts(self, texts: List[str]):
        """بناء المفردات من النصوص"""
        idx = 4
        for text in texts:
            for word in text.split():
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1
    
    def encode(self, text: str) -> List[int]:
        """تحويل نص إلى أرقام"""
        return [self.word2idx.get(word, 3) for word in text.split()]
    
    def decode(self, indices: List[int]) -> str:
        """تحويل أرقام إلى نص"""
        words = []
        for idx in indices:
            if idx in [0, 1, 2]:  # PAD, SOS, EOS
                continue
            words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = Vocabulary()
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        return vocab


class InferenceEngine:
    """محرك الاستدلال الرئيسي"""
    
    def __init__(self):
        self.models: Dict[str, TransformerModel] = {}
        self.vocab: Optional[Vocabulary] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inference Device: {self.device}")
        
        self._load_vocabulary()
        self._load_all_models()
    
    def _load_vocabulary(self):
        """تحميل المفردات"""
        if VOCAB_PATH.exists():
            self.vocab = Vocabulary.load(VOCAB_PATH)
            print(f"📚 Vocabulary loaded: {len(self.vocab.word2idx)} words")
        else:
            print("⚠️ No vocabulary found - building from checkpoints...")
            self.vocab = self._build_vocab_from_checkpoints()
    
    def _build_vocab_from_checkpoints(self) -> Vocabulary:
        """بناء المفردات من الـ Checkpoints"""
        vocab = Vocabulary()
        
        # قراءة كل النصوص المتاحة
        all_texts = []
        for layer_dir in CHECKPOINT_DIR.iterdir():
            if layer_dir.is_dir():
                stats_file = layer_dir / "training_stats.json"
                if stats_file.exists():
                    try:
                        with open(stats_file) as f:
                            stats = json.load(f)
                            if 'sample_texts' in stats:
                                all_texts.extend(stats['sample_texts'])
                    except:
                        pass
        
        if all_texts:
            vocab.build_from_texts(all_texts)
            vocab.save(VOCAB_PATH)
            print(f"✅ Vocabulary built: {len(vocab.word2idx)} words")
        else:
            # Fallback: استخدم مفردات افتراضية
            default_texts = [
                "التأسيس المتين يحتاج صبراً",
                "القرار الحكيم يأتي من تحليل البيانات",
                "التنويع استراتيجية حماية",
                "الثقة تُبنى بالنتائج",
                "التكيف مع التغيير هو مفتاح البقاء"
            ]
            vocab.build_from_texts(default_texts)
            vocab.save(VOCAB_PATH)
        
        return vocab
    
    def _load_all_models(self):
        """تحميل كل النماذج"""
        if not CHECKPOINT_DIR.exists():
            print("❌ No checkpoints directory")
            return
        
        for layer_dir in CHECKPOINT_DIR.iterdir():
            if layer_dir.is_dir():
                layer_name = layer_dir.name
                try:
                    self._load_model(layer_name, layer_dir)
                except Exception as e:
                    print(f"  ⚠️ Failed to load {layer_name}: {str(e)[:50]}")
        
        print(f"🧠 Loaded {len(self.models)} inference models\n")
    
    def _load_model(self, name: str, layer_dir: Path):
        """تحميل نموذج واحد"""
        # ابحث عن checkpoint
        checkpoints = sorted(layer_dir.glob("*.pt"))
        if not checkpoints:
            return
        
        latest = checkpoints[-1]
        
        # تحميل checkpoint
        checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
        
        # إنشاء نموذج جديد
        vocab_size = checkpoint.get('vocab_size', 10000)
        model = TransformerModel(vocab_size=vocab_size)
        
        # تحميل الأوزان (مع التعامل مع الاختلافات)
        state_dict = checkpoint.get('model_state', checkpoint)
        
        # Filter to matching keys only
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        
        self.models[name] = model
        print(f"  ✅ {name}: Loaded ({len(filtered_dict)}/{len(state_dict)} layers)")
    
    def generate_response(self, prompt: str, model_name: str = "high_council", max_length: int = 50) -> str:
        """
        توليد رد حقيقي من النموذج
        """
        if model_name not in self.models:
            return f"[Model {model_name} not loaded]"
        
        if not self.vocab:
            return "[No vocabulary]"
        
        model = self.models[model_name]
        
        # تحضير الإدخال
        input_ids = self.vocab.encode(prompt)
        if not input_ids:
            input_ids = [1]  # SOS token
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # التوليد
        try:
            with torch.no_grad():
                output = model.generate(input_tensor, max_length=max_length, temperature=0.8)
            
            # فك التشفير
            response = self.vocab.decode(output[0].cpu().tolist())
            
            # تنظيف الرد
            response = response.replace(prompt, '').strip()
            if not response:
                response = "أنا أفكر في ما قلت..."
            
            return response
            
        except Exception as e:
            return f"[Generation error: {str(e)[:30]}]"
    
    def get_model_status(self) -> Dict:
        """حالة النماذج"""
        return {
            'device': str(self.device),
            'models_loaded': len(self.models),
            'model_names': list(self.models.keys()),
            'vocab_size': len(self.vocab.word2idx) if self.vocab else 0
        }


# Global instance
inference_engine = InferenceEngine()
