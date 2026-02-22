"""
إعادة بناء Vocabulary من Checkpoints
"""
import torch
import pickle
from pathlib import Path
import json

checkpoint_dir = Path("learning_data/checkpoints")

# جمع كل الكلمات
all_words = set()
all_words.update(['<PAD>', '<SOS>', '<EOS>', '<UNK>'])

# أضف كلمات عربية شائعة
common_words = [
    'التأسيس', 'المتين', 'يحتاج', 'صبراً', 'رؤية', 'طويلة', 'المدى',
    'القرار', 'الحكيم', 'يأتي', 'من', 'تحليل', 'البيانات', 'وليس', 'العواطف',
    'التنويع', 'استراتيجية', 'حماية', 'للمستقبل',
    'الثقة', 'تُبنى', 'بالنتائج', 'لا', 'بالوعود',
    'التكيف', 'مع', 'التغيير', 'هو', 'مفتاح', 'البقاء',
    'الصبر', 'النجاح', 'الأسواق', 'المتقلبة',
    'أنا', 'في', 'خدمتك', 'سيادة', 'الرئيس',
    'نفهم', 'طلبك', 'وسنعمل', 'على', 'تحليله',
    'أرى', 'ما', 'وراء', 'الظواهر', 'أخبرني', 'بما', 'تبحث', 'عنه',
    'المستقبل', 'ليس', 'مكتوباً', 'نحن', 'من', 'نكتبه', 'بقراراتنا', 'اليوم',
    'الشجاعة', 'تقتضي', 'الاعتراف', 'بالمشكلة', 'والمواجهة',
    'النظام', 'الحالي', 'يحتاج', 'مراجعة',
    'العدل', 'أساس', 'الملك',
    'التعلم', 'المستمر', 'سر', 'التفوق'
]
all_words.update(common_words)

# ابحث عن أي ملفات JSON في Checkpoints
for layer_dir in checkpoint_dir.iterdir():
    if layer_dir.is_dir():
        for json_file in layer_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    # استخرج أي نصوص
                    def extract_text(obj):
                        if isinstance(obj, str):
                            return obj
                        elif isinstance(obj, list):
                            return ' '.join(str(x) for x in obj)
                        elif isinstance(obj, dict):
                            return ' '.join(extract_text(v) for v in obj.values())
                        return ''
                    
                    text = extract_text(data)
                    words = text.split()
                    all_words.update(words)
            except:
                pass

print(f"Total unique words: {len(all_words)}")

# بناء Vocabulary
word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}

for idx, word in enumerate(sorted(all_words), start=4):
    word2idx[word] = idx
    idx2word[idx] = word

print(f"Vocabulary size: {len(word2idx)}")

# حفظ
vocab_path = Path("learning_data/vocab.pkl")
with open(vocab_path, 'wb') as f:
    pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)

print(f"Saved to: {vocab_path}")
