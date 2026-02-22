"""
Wise AI Engine - Hybrid System
الـ Neural Network يحدد نوع الرد، والردود من قاعدة بيانات متطورة
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple
import random
import hashlib
import re
import json

CHECKPOINT_DIR = Path("learning_data/checkpoints")
SMART_LEARNED_DATA_FILE = Path("data/knowledge/smart-learned-data.json")


class ResponseSelector(nn.Module):
    """يحدد نوع الرد بناءً على السياق"""
    def __init__(self, vocab_size=10000, d_model=256, num_response_types=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=False),
            num_layers=2
        )
        self.classifier = nn.Linear(d_model, num_response_types)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.classifier(x.mean(dim=0))


class WiseAIEngine:
    """محرك الحكماء الذكي"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Wise AI Engine: {self.device}")
        
        # قاعدة بيانات الردود المتطورة (كل حكيم له شخصية)
        self.response_db = self._build_response_database()
        
        # أنواع الردود
        self.response_types = [
            'analysis',      # تحليل
            'strategy',      # استراتيجية
            'vision',        # رؤية
            'caution',       # حذر
            'action',        # عمل
            'wisdom',        # حكمة
            'question',      # سؤال
            'support',       # دعم
            'warning',       # تحذير
            'encouragement'  # تشجيع
        ]

        self.training_knowledge = self._load_training_knowledge()
        
        print(f"Response database: {sum(len(v) for v in self.response_db.values())} responses")
        print(f"Training knowledge loaded: {len(self.training_knowledge)} entries\n")

    def _load_training_knowledge(self) -> List[Dict[str, str]]:
        """تحميل بيانات تدريب مبسطة للاستشهاد أثناء الرد"""
        if not SMART_LEARNED_DATA_FILE.exists():
            return []

        try:
            raw = json.loads(SMART_LEARNED_DATA_FILE.read_text(encoding='utf-8') or '[]')
            entries: List[Dict[str, str]] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                instruction = str(item.get('instruction', '') or '').strip()
                output = str(item.get('output', '') or '').strip()
                metadata = item.get('metadata', {}) if isinstance(item.get('metadata', {}), dict) else {}
                source = str(metadata.get('source', 'unknown') or 'unknown').strip()
                topic = str(metadata.get('topic', '') or '').strip()
                if not instruction and not output:
                    continue
                entries.append({
                    'instruction': instruction,
                    'output': output,
                    'source': source,
                    'topic': topic
                })
            return entries
        except Exception:
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[\u0600-\u06FFA-Za-z][\u0600-\u06FFA-Za-z0-9_-]{2,}", text.lower())
        stop_words = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'your', 'what',
            'على', 'من', 'الى', 'إلى', 'في', 'عن', 'هذا', 'هذه', 'كيف', 'ما', 'هو', 'هي', 'تم', 'عند'
        }
        return [token for token in tokens if token not in stop_words]

    def _retrieve_training_evidence(self, message: str, top_k: int = 2) -> List[Dict[str, str]]:
        """استرجاع أمثلة تدريب الأقرب للنص"""
        if not self.training_knowledge:
            return []

        query_keywords = set(self._extract_keywords(message))
        if not query_keywords:
            return []

        scored: List[Tuple[int, Dict[str, str]]] = []
        for item in self.training_knowledge:
            corpus = f"{item.get('instruction', '')} {item.get('output', '')} {item.get('topic', '')}".lower()
            item_keywords = set(self._extract_keywords(corpus))
            overlap = len(query_keywords.intersection(item_keywords))
            if overlap <= 0:
                continue
            scored.append((overlap, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best = [item for _, item in scored[:top_k]]
        return best
    
    def _build_response_database(self) -> Dict[str, List[str]]:
        """بناء قاعدة بيانات الردود"""
        return {
            'حكيم القرار': [
                "بناءً على تحليلي العميق، القرار الأمثل هو التنويع. لا تضع كل مواردك في مشروع واحد.",
                "القرار الحكيم يجمع بين البيانات والحدس. ما هي بياناتك؟",
                "أنصحك بالصبر. القرارات الكبرى تحتاج تفكيراً عميقاً وليس استعجالاً.",
                "التأسيس المتين يحتاج رؤية طويلة المدى. هل أنت مستعد للمضي قدماً؟",
                "الثقة تُبنى بالنتائج، لا بالوعود. دع أفعالك تتحدث عنك.",
                "التنويع ليس تشتيتاً، بل حماية للمستقبل.",
                "كل قرار كبير يحمل مخاطر. المهم أن تكون المخاطر محسوبة.",
                "الاستراتيجية الناجحة تتطور مع الوقت. لا تتشبث بالخطة الأصلية إذا تغيرت الظروف."
            ],
            'حكيم البصيرة': [
                "أرى في البيانات نمطاً مقلقاً. التركيز الزائد على جانب واحد يخفي المخاطر الحقيقية.",
                "تحليلي يشير إلى فرصة في السوق لم تلحظها بعد. هل تريد التفاصيل؟",
                "البيانات لا تكذب، لكنها تحتاج من يفهم لغتها. دعني أترجم لك.",
                "أرى ما وراء الظواهر. السطح يخفي عمقاً يجب استكشافه.",
                "الأرقام تخبرني بقصة. السؤال: هل أنت جاهز لسماع الحقيقة؟",
                "التفاصيل الصغيرة تكشف الصورة الكبيرة. لا تهملها.",
                "التحليل العميق يحتاج صبراً. الخلاصات السريعة غالباً ما تكون خاطئة.",
                "كل بيان له سياق. فهم السياق أهم من البيان نفسه."
            ],
            'حكيم المستقبل': [
                "رؤيتي للمستقبل: التكيف التقني سيفصل الناجحين عن الباقين. استعد الآن.",
                "الخطط الخمسية ناجحة فقط إذا كانت مرنة. الجمود يقتل المستقبل.",
                "أرى فرصة كبيرة في السوق خلال 6 أشهر. هل أنت مستعد؟",
                "المستقبل ليس مكتوباً، نحن من نكتبه بقراراتنا اليوم.",
                "ما هي رؤيتك للسنوات القادمة؟ بدون رؤية، كل قرار عشوائي.",
                "التغيير هو القانون الوحيد الثابت. من لا يتكيف يتخلف.",
                "الاستثمار في المستقبل يبدأ اليوم. لا تؤجل لغداً.",
                "الرؤية البعيدة تحمي من الأزمات القريبة."
            ],
            'حكيم الشجاعة': [
                "الشجاعة ليست غياب الخوف، بل التحرر رغم الخوف. تحرك الآن!",
                "الخطر الحقيقي ليس في الفعل، بل في الجمود. لا تنتظر الظروف المثالية.",
                "الشجاعة تقتضي الاعتراف بالمشكلة والمواجهة. أنا معك.",
                "كل أزمة تحمل فرصة. دعنا نستغل هذه الفرصة بجرأة.",
                "أحياناً يجب أن نقفز قبل أن نكون جاهزين تماماً.",
                "ما هو التحدي الذي تخشى مواجهته؟ المواجهة أقل صعوبة من التخيل.",
                "القرار الجريء اليوم يمنحك السيطرة غداً.",
                "لا تخف من الفشل. خف من عدم المحاولة."
            ],
            'حكيم الضبط': [
                "النظام الحالي يحتاج مراجعة. وجدت ثغرات يجب سدها فوراً.",
                "المراقبة الوقائية أفضل من العلاج. أنصح بتدقيق شهري.",
                "الأمان ليس كلفة، بل استثمار. لا تبخل به.",
                "النظام يحمينا من الفوضى. هل النظام لديك سليم؟",
                "ما هي إجراءات الضبط التي تريد مراجعتها؟",
                "الانضباط اليوم يمنح الحرية غداً.",
                "الرقابة الذكية تكشف المشاكل قبل أن تكبر.",
                "لا تثق بالعواطف في الأمور المهمة. ثق بالنظام."
            ],
            'حكيم التوازن': [
                "العدل يقتضي سماع الطرفين. ما هي القصة الكاملة؟",
                "التوازن مهم: لا تفرط في العمل على حساب الصحة.",
                "الحل العادل يخدم الجميع على المدى الطويل.",
                "الاعتدال هو السبيل. تجنب التطرف في كل شيء.",
                "هل هناك موضوع يحتاج لنظرة محايدة؟",
                "العدالة تأخر أحياناً، لكنها لا تغيب.",
                "التوازن بين العقل والقلب هو سر الحكمة.",
                "لا تنحاز لأحد طرفي النزاع قبل أن تسمع الطرفين."
            ],
            'حكيم التكيف': [
                "التغيير فرصة، ليس تهديداً. دعنا نتكيف معه.",
                "من لا يتكيف يتخلف. العالم يتغير بسرعة.",
                "المرونة في الخطة تضمن النجاح.",
                "التكيف هو سر البقاء. هل أنت جاهز للتغيير؟",
                "الثبات في الهدف، المرونة في الطريقة.",
                "كل تغيير يحمل درساً. هل تتعلم منه؟",
                "لا تقاوم التغيير. استخدمه لصالحك.",
                "التكيف السريع يمنحك الأفضلية على المنافسين."
            ],
            'حكيم الذاكرة': [
                "مررنا بمثل هذا في 2019. الحل كان... هل تريد أن أروي لك التجربة؟",
                "التاريخ يعيد نفسه. دعنا نتعلم من أخطاء الماضي.",
                "خبرة 20 سنة تقول: الصبر يُكافأ دائماً.",
                "لدينا أرشيف غني من التجارب. استفد منه.",
                "هل تريد أن أروي لك قصة مشابهة مرت علينا؟",
                "العبرة من الماضي تبني مستقبلاً أفضل.",
                "من لا يتعلم من التاريخ، يرتكب نفس الأخطاء.",
                "الخبرة لا تُدرس في الكتب، بل تُكتسب بالمواقف."
            ],
            'default': [
                "أنا هنا لمساعدتك. ما هو سؤالك؟",
                "أفكر في ما قلت. هل لديك تفاصيل إضافية؟",
                "هذا موضوع مهم. دعني أحلله بعمق.",
                "وجهة نظرك محترمة. لدي رأي آخر...",
                "دعنا نناقش هذا بمزيد من التفصيل."
            ]
        }
    
    def _analyze_context(self, message: str) -> str:
        """تحليل السياق وتحديد نوع الرد"""
        msg_lower = message.lower()
        
        # تحديد نوع الرد بناءً على الكلمات المفتاحية
        if any(w in msg_lower for w in ['تحليل', 'تقرير', 'بيانات', 'data', 'analysis']):
            return 'analysis'
        elif any(w in msg_lower for w in ['استراتيجية', 'قرار', 'plan', 'strategy']):
            return 'strategy'
        elif any(w in msg_lower for w in ['مستقبل', 'رؤية', 'future', 'vision']):
            return 'vision'
        elif any(w in msg_lower for w in ['خطر', 'أزمة', 'مشكلة', 'risk', 'crisis']):
            return 'caution'
        elif any(w in msg_lower for w in ['تنفيذ', 'عمل', 'do', 'action']):
            return 'action'
        elif any(w in msg_lower for w in ['حكمة', 'نصيحة', 'advice', 'wisdom']):
            return 'wisdom'
        elif any(w in msg_lower for w in ['؟', 'question', 'what', 'how', 'why']):
            return 'question'
        elif any(w in msg_lower for w in ['ساعد', 'support', 'help']):
            return 'support'
        elif any(w in msg_lower for w in ['تحذير', 'انتبه', 'warning']):
            return 'warning'
        else:
            return 'encouragement'
    
    def generate_response(self, message: str, wise_man_name: str, 
                         conversation_history: List[str] = None) -> str:
        """واجهة متوافقة: ترجع النص فقط"""
        result = self.generate_response_with_evidence(message, wise_man_name, conversation_history)
        return result['response']

    def generate_response_with_evidence(self, message: str, wise_man_name: str,
                                        conversation_history: List[str] = None,
                                        strict_evidence: bool = False) -> Dict:
        """
        توليد رد ذكي مع ميتاداتا مصدر/استشهاد
        
        Args:
            message: رسالة المستخدم
            wise_man_name: اسم الحكيم
            conversation_history: سجل المحادثة (للتنويع)
        """
        # تحليل السياق
        context_type = self._analyze_context(message)
        
        # اختيار قاعدة الردود المناسبة
        if wise_man_name in self.response_db:
            responses = self.response_db[wise_man_name]
        else:
            responses = self.response_db['default']
        
        # تجنب تكرار آخر عدة ردود (مو بس آخر رد)
        history = conversation_history or []
        recent = history[-4:]
        recent_norm = {self._normalize_text(item) for item in recent if item}

        available_responses = [
            r for r in responses
            if self._normalize_text(r) not in recent_norm
        ]
        if not available_responses:
            available_responses = responses
        
        # اختيار الرد (مع تنويع بناءً على السياق)
        # نستخدم hash للرسالة للحصول على نفس الرد للرسالة المماثلة
        salt = f"{message}|{wise_man_name}|{len(history)}"
        msg_hash = int(hashlib.md5(salt.encode()).hexdigest(), 16)
        index = msg_hash % len(available_responses)
        
        selected_response = available_responses[index]

        # تعديل الاختيار بناءً على السياق
        if context_type in ['analysis', 'strategy', 'vision']:
            # اختر الردود الأكثر تفصيلاً
            long_responses = [r for r in available_responses if len(r) > 50]
            if long_responses:
                selected_response = long_responses[msg_hash % len(long_responses)]

        evidence = self._retrieve_training_evidence(message, top_k=2)
        response_source = 'persona-template'
        confidence = 0.45

        if evidence:
            primary = evidence[0]
            topic = primary.get('topic') or 'موضوع تدريبي'
            source = primary.get('source') or 'dataset'
            selected_response = f"{selected_response}\n\nمرجع تدريبي: {topic} ({source})"
            response_source = 'training+persona'
            confidence = 0.72
        elif strict_evidence:
            return {
                'response': '',
                'source': 'blocked-no-evidence',
                'context_type': context_type,
                'confidence': 0.0,
                'evidence': [],
                'blocked': True
            }

        return {
            'response': selected_response,
            'source': response_source,
            'context_type': context_type,
            'confidence': confidence,
            'blocked': False,
            'evidence': [
                {
                    'topic': item.get('topic', ''),
                    'source': item.get('source', ''),
                    'instruction': item.get('instruction', '')[:160]
                }
                for item in evidence
            ]
        }

    def _normalize_text(self, text: str) -> str:
        """تطبيع نص بسيط للمقارنة وتقليل التكرار"""
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        normalized = normalized.replace("،", ",").replace("؟", "?")
        return normalized
    
    def get_personality(self, wise_man_name: str) -> Dict:
        """الحصول على شخصية الحكيم"""
        personalities = {
            'حكيم القرار': {'style': 'حازم واستراتيجي', 'greeting': 'سيادة الرئيس، ما هو قرارك؟'},
            'حكيم البصيرة': {'style': 'تحليلي وعميق', 'greeting': 'أرى ما لا يراه الآخرون. أخبرني.'},
            'حكيم المستقبل': {'style': 'رؤيوي وبعيد النظر', 'greeting': 'المستقبل يكشف أسراره لمن يصبر.'},
            'حكيم الشجاعة': {'style': 'جريء وقاطع', 'greeting': 'الشجاعة ليست غياب الخوف.'},
            'حكيم الضبط': {'style': 'دقيق ومنظم', 'greeting': 'النظام يحمي من الفوضى.'},
            'حكيم التوازن': {'style': 'عادل ومحايد', 'greeting': 'العدل أساس الملك.'},
            'حكيم التكيف': {'style': 'مرن وذكي', 'greeting': 'التغيير هو القانون الوحيد الثابت.'},
            'حكيم الذاكرة': {'style': 'حكيم وذو خبرة', 'greeting': 'من التاريخ نتعلم.'}
        }
        return personalities.get(wise_man_name, {'style': 'حكيم', 'greeting': 'أنا في خدمتك.'})


# Global instance
wise_ai_engine = WiseAIEngine()
