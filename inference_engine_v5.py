"""
Inference Engine v5 - Word-Level Tokenizer (متوافق مع RTX 4090)
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional
import json

CHECKPOINT_DIR = Path("learning_data/checkpoints")

# كلمات عربية شائعة للبناء عليها
ARABIC_WORDS = [
    "<PAD>", "<SOS>", "<EOS>", "<UNK>",
    "التأسيس", "المتين", "يحتاج", "صبراً", "رؤية", "طويلة", "المدى",
    "القرار", "الحكيم", "يأتي", "من", "تحليل", "البيانات", "وليس", "العواطف",
    "التنويع", "استراتيجية", "حماية", "للمستقبل",
    "الثقة", "تبنى", "بالنتائج", "لا", "بالوعود",
    "التكيف", "مع", "التغيير", "هو", "مفتاح", "البقاء",
    "الصبر", "مفتاح", "النجاح", "الأسواق", "المتقلبة",
    "أنا", "في", "خدمتك", "سيادة", "الرئيس",
    "نفهم", "طلبك", "وسنعمل", "على", "تحليله",
    "أرى", "ما", "وراء", "الظواهر", "أخبرني", "بما", "تبحث", "عنه",
    "المستقبل", "ليس", "مكتوباً", "نحن", "نكتبه", "بقراراتنا", "اليوم",
    "الشجاعة", "تقتضي", "الاعتراف", "بالمشكلة", "والمواجهة",
    "النظام", "الحالي", "يحتاج", "مراجعة",
    "العدل", "أساس", "الملك",
    "التعلم", "المستمر", "سر", "التفوق",
    "نعم", "لا", "ربما", "أحياناً", "دائماً", "أبداً",
    "جيد", "سيئ", "ممتاز", "ضعيف", "قوي",
    "كبير", "صغير", "كثير", "قليل",
    "أول", "ثان", "ثالث", "رابع", "خامس",
    "عمل", "فعل", "قال", "كان", "صار",
    "يريد", "يحب", "يكره", "يفعل",
    "الآن", "اليوم", "الأمس", "الغد",
    "هنا", "هناك", "كل", "بعض", "جميع",
    "أنت", "أنا", "هو", "هي", "نحن", "هم",
    "التقرير", "التحليل", "البيان", "المعلومة",
    "الخطر", "الأزمة", "المشكلة", "الحل",
    "الربح", "الخسارة", "الفائدة", "الضرر",
    "الموظف", "المدير", "العميل", "المستثمر"
]


class WordTokenizer:
    """Word-level tokenizer"""
    def __init__(self, vocab_size=492):
        self.vocab_size = vocab_size
        # استخدم الكلمات المتاحة + padding بالـ <UNK>
        self.word2idx = {}
        self.idx2word = {}
        
        for i, word in enumerate(ARABIC_WORDS[:vocab_size-1]):
            self.word2idx[word] = i
            self.idx2word[i] = word
        
        # Fill rest with UNK
        for i in range(len(ARABIC_WORDS), vocab_size):
            self.idx2word[i] = "<UNK>"
        
        self.word2idx["<UNK>"] = 3  # Default for unknown
        
    def encode(self, text: str) -> List[int]:
        """نص إلى أرقام"""
        words = text.split()
        return [self.word2idx.get(word, 3) for word in words]
    
    def decode(self, indices: List[int]) -> str:
        """أرقام إلى نص"""
        words = [self.idx2word.get(i, "<UNK>") for i in indices if i not in [0, 1, 2]]
        return " ".join(words)


class RTX4090Transformer(nn.Module):
    """نفس Architecture الـ RTX 4090"""
    def __init__(self, vocab_size=492, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(512, d_model)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4,
                dropout=dropout,
                batch_first=False
            ),
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        x = self.embedding(x) + self.pos_encoder(positions)
        x = self.transformer(x)
        return self.fc(x)


class InferenceEngine:
    """محرك الاستدلال"""
    
    def __init__(self):
        self.models: Dict[str, RTX4090Transformer] = {}
        self.tokenizers: Dict[str, WordTokenizer] = {}
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
        """تحميل نموذج"""
        checkpoints = list(layer_dir.glob("*.pt"))
        if not checkpoints:
            return
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
        
        vocab_size = checkpoint.get('vocab_size', 492)
        
        # إنشاء tokenizer خاص بهذا النموذج
        self.tokenizers[name] = WordTokenizer(vocab_size)
        
        # إنشاء النموذج
        model = RTX4090Transformer(vocab_size=vocab_size)
        
        # تحميل الأوزان
        state_dict = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(state_dict, strict=True)
        
        model.to(self.device)
        model.eval()
        
        self.models[name] = model
        print(f"  [OK] {name}: vocab={vocab_size}")
    
    def generate(self, prompt: str, model_name: str = "high_council",
                max_length: int = 20, temperature: float = 0.8) -> str:
        """توليد رد"""
        if model_name not in self.models:
            return "[Model not available]"
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize
        input_ids = [1] + tokenizer.encode(prompt)  # <SOS> = 1
        if not input_ids:
            input_ids = [1, 3]  # <SOS>, <UNK>
        
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        generated = input_tensor.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(generated)
                logits = outputs[-1, 0, :] / temperature
                
                # Top-k sampling
                top_k = 10
                top_logits, top_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_logits, dim=-1)
                next_token = top_indices[torch.multinomial(probs, 1)].item()
                
                generated = torch.cat([generated, torch.tensor([[next_token]], device=self.device)], dim=0)
                
                if next_token == 2:  # <EOS>
                    break
        
        # Decode
        response = tokenizer.decode(generated.squeeze(1).cpu().tolist())
        
        # Post-process
        response = response.replace(prompt, '').strip()
        response = response.replace('<UNK>', '').strip()
        
        if len(response) < 2:
            # Fallback - استخدم شخصية الحكيم
            responses = {
                "high_council": "القرار يحتاج تفكيراً عميقاً",
                "seventh_dimension": "المستقبل يحمل الفرص",
                "domain_experts": "البيانات تكشف الحقيقة",
                "shadow_light": "الشجاعة تُبنى بالمواقف",
                "guardian": "النظام يحمي من الفوضى",
                "learning_core": "التعلم المستمر هو سر التفوق"
            }
            response = responses.get(model_name, "أحتاج مزيداً من التفكير...")
        
        return response


# Global
inference_engine = InferenceEngine()
