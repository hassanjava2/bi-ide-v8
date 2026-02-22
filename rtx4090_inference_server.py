#!/usr/bin/env python3
"""
RTX 4090 Inference Server
يشتغل على Ubuntu + RTX 4090
يوفر API للـ Windows للحصول على ردود AI حقيقية
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import json
import logging
from datetime import datetime
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Paths
_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "learning_data" / "checkpoints"
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR))).expanduser()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f"🖥️  Device: {DEVICE}")
if DEVICE.type == 'cuda':
    logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


class TransformerModel(nn.Module):
    """Transformer للتوليد"""
    def __init__(self, vocab_size=10000, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        x = self.embedding(x) + self.pos_encoder(positions)
        x = self.transformer(x)
        return self.fc(x)


class InferenceManager:
    """يدير التوليد من Checkpoints"""
    
    def __init__(self):
        self.models: Dict[str, TransformerModel] = {}
        self.tokenizers: Dict[str, 'SimpleTokenizer'] = {}
        self.model_info: Dict[str, Dict] = {}
        
        self._load_all_models()
    
    def _load_all_models(self):
        """تحميل كل الـ Checkpoints"""
        if not CHECKPOINT_DIR.exists():
            logger.error(f"❌ Checkpoints directory not found: {CHECKPOINT_DIR}")
            return
        
        logger.info(f"📂 Loading checkpoints from: {CHECKPOINT_DIR}")
        
        for layer_dir in CHECKPOINT_DIR.iterdir():
            if layer_dir.is_dir():
                try:
                    self._load_model(layer_dir.name, layer_dir)
                except Exception as e:
                    logger.error(f"❌ Failed to load {layer_dir.name}: {e}")
        
        logger.info(f"✅ Loaded {len(self.models)} models")
    
    def _load_model(self, name: str, layer_dir: Path):
        """تحميل نموذج واحد"""
        checkpoints = list(layer_dir.glob("*.pt"))
        if not checkpoints:
            return
        
        # Latest checkpoint
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"📥 Loading {name} from {latest.name}...")
        
        checkpoint = torch.load(latest, map_location=DEVICE, weights_only=False)
        
        vocab_size = checkpoint.get('vocab_size', 492)
        epoch = checkpoint.get('epoch', 0)
        
        # Create model
        model = TransformerModel(vocab_size=vocab_size)
        
        # Load weights
        state_dict = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        model.eval()
        
        # Create tokenizer
        self.tokenizers[name] = SimpleTokenizer(vocab_size)
        
        self.models[name] = model
        self.model_info[name] = {
            'vocab_size': vocab_size,
            'epoch': epoch,
            'checkpoint': str(latest),
            'loaded_at': datetime.now().isoformat()
        }
        
        logger.info(f"  ✅ {name}: epoch={epoch}, vocab={vocab_size}")
    
    def generate(self, prompt: str, model_name: str = "high_council",
                 max_length: int = 100, temperature: float = 0.8) -> str:
        """توليد رد"""
        if model_name not in self.models:
            return f"[Model {model_name} not found. Available: {list(self.models.keys())}]"
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [1]  # <SOS>
        
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(1)
        generated = input_tensor.clone()

        temperature = max(0.5, float(temperature))
        repetition_penalty = 1.15
        top_k = 40
        recent_window = 80
        max_repeat_per_token = 6
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(generated)
                logits = outputs[-1, 0, :] / temperature

                # Repetition penalty: قلل احتمال التوكنات المكررة
                generated_ids = generated.squeeze(1).tolist()
                recent_ids = generated_ids[-recent_window:]
                token_counts = Counter(recent_ids)
                for token_id, count in token_counts.items():
                    if count >= 2:
                        logits[token_id] = logits[token_id] / (repetition_penalty * min(count, 4))
                    if count >= max_repeat_per_token:
                        logits[token_id] = -float('inf')
                
                # Top-k sampling
                top_logits, top_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_logits, dim=-1)
                next_token = top_indices[torch.multinomial(probs, 1)].item()
                
                generated = torch.cat([generated, torch.tensor([[next_token]], device=DEVICE)], dim=0)
                
                if next_token == 2:  # <EOS>
                    break
        
        # Decode
        response = tokenizer.decode(generated.squeeze(1).cpu().tolist())
        
        # Post-process
        response = response.replace(prompt, '').strip()
        response = self._dedupe_repeated_chunks(response)
        
        if len(response) < 5:
            response = "أحتاج مزيداً من الوقت للتفكير..."
        
        return response

    def _dedupe_repeated_chunks(self, text: str) -> str:
        """تقليل التكرار الواضح في النص الناتج"""
        if not text:
            return text

        chunks = [chunk.strip() for chunk in text.split('.') if chunk.strip()]
        if len(chunks) <= 1:
            return text

        deduped = []
        seen = set()
        for chunk in chunks:
            key = chunk.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)

        cleaned = '. '.join(deduped).strip()
        if cleaned and not cleaned.endswith('.'):
            cleaned += '.'
        return cleaned or text
    
    def get_status(self) -> Dict:
        return {
            'device': str(DEVICE),
            'models_loaded': len(self.models),
            'models': list(self.models.keys()),
            'model_info': self.model_info
        }


class SimpleTokenizer:
    """Tokenizer بسيط"""
    def __init__(self, vocab_size=492):
        self.vocab_size = vocab_size
        # Character-level fallback
        self.chars = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + \
            list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:()[]') + \
            list("'\"") + \
            list('ابتثجحخدذرزسشصضطظعغفقكلمنهويىةءآأإؤئ') + \
            list('ًٌٍَُِّْ')
        
        self.char2idx = {c: min(i, vocab_size-1) for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars[:vocab_size])}
    
    def encode(self, text: str) -> List[int]:
        return [self.char2idx.get(c, 3) for c in text]
    
    def decode(self, indices: List[int]) -> str:
        return ''.join(self.idx2char.get(i, '') for i in indices if i not in [0, 1, 2])


# Global manager
manager = InferenceManager()


# API Endpoints

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'device': str(DEVICE),
        'models': len(manager.models),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/status', methods=['GET'])
def status():
    """Detailed status"""
    return jsonify(manager.get_status())


@app.route('/generate', methods=['POST'])
def generate():
    """توليد رد من AI"""
    data = request.json or {}
    
    prompt = data.get('prompt', '')
    model_name = data.get('model', 'high_council')
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.8)
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    logger.info(f"📝 Generate request: model={model_name}, prompt='{prompt[:30]}...'")
    
    try:
        response = manager.generate(
            prompt=prompt,
            model_name=model_name,
            max_length=max_length,
            temperature=temperature
        )
        
        logger.info(f"✅ Generated: {response[:50]}...")
        
        return jsonify({
            'response': response,
            'model': model_name,
            'prompt': prompt,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """قائمة النماذج المتاحة"""
    return jsonify({
        'models': list(manager.models.keys()),
        'info': manager.model_info
    })


@app.route('/council/message', methods=['POST'])
def council_message():
    """API للمجلس (متوافق مع Windows)"""
    data = request.json or {}
    
    message = data.get('message', '')
    user_id = data.get('user_id', 'president')
    
    # اختيار النموذج المناسب حسب السياق
    model_name = select_model_by_context(message)
    
    try:
        response = manager.generate(
            prompt=message,
            model_name=model_name,
            max_length=80,
            temperature=0.7
        )
        
        # Mapping model to wise man name
        model_to_wise_man = {
            'high_council': 'حكيم القرار',
            'seventh_dimension': 'حكيم المستقبل',
            'domain_experts': 'حكيم البصيرة',
            'shadow_light': 'حكيم الشجاعة',
            'guardian': 'حكيم الضبط',
            'learning_core': 'حكيم التكيف',
            'eternity': 'حكيم الذاكرة',
            'execution': 'حكيم النظام',
            'meta_team': 'حكيم التنفيذ',
            'scouts': 'حكيم التقارير',
            'executive_controller': 'حكيم الطوارئ'
        }
        
        return jsonify({
            'response': response,
            'council_member': model_to_wise_man.get(model_name, 'حكيم القرار'),
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def select_model_by_context(message: str) -> str:
    """يختار النموذج المناسب حسب السياق"""
    msg_lower = message.lower()
    
    if any(w in msg_lower for w in ['تحليل', 'تقرير', 'بيانات', 'analysis', 'data']):
        return 'domain_experts'
    elif any(w in msg_lower for w in ['مستقبل', 'خطة', 'رؤية', 'future', 'plan']):
        return 'seventh_dimension'
    elif any(w in msg_lower for w in ['خطر', 'أزمة', 'مشكلة', 'risk', 'crisis']):
        return 'shadow_light'
    elif any(w in msg_lower for w in ['نظام', 'مراقبة', 'أمان', 'system', 'security']):
        return 'guardian'
    elif any(w in msg_lower for w in ['تغيير', 'تطور', 'change', 'evolve']):
        return 'learning_core'
    elif any(w in msg_lower for w in ['تاريخ', 'ماضي', 'history', 'past']):
        return 'eternity'
    elif any(w in msg_lower for w in ['تنفيذ', 'عمل', 'execute', 'work']):
        return 'execution'
    else:
        return 'high_council'


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 RTX 4090 Inference Server Starting...")
    logger.info("=" * 60)
    
    # Run server
    app.run(host='0.0.0.0', port=8080, threaded=True)
