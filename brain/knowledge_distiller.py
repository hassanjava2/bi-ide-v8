#!/usr/bin/env python3
"""
knowledge_distiller.py — جمع بيانات تدريب من AI APIs 🧠📚

المصادر:
  - Kimi K2.5 API (الأساسي)
  - أي OpenAI-compatible API (GPT-4, Claude, etc.)

الناتج:
  - JSONL files لكل كبسولة (Q&A pairs)
  - جاهز للـ LoRA training
  - يُرفع على Google Drive أوتوماتيكياً

⚠️ السيادة: البيانات تنحفظ محلياً فقط
"""

import json
import os
import time
import hashlib
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

# ═══════════════════════════════════════════════════════════
# Load credentials
# ═══════════════════════════════════════════════════════════
ENV_FILE = Path(__file__).parent / ".env"

def _load_env():
    """Load .env file"""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
                os.environ.setdefault(k.strip(), v.strip())
    return env

_load_env()

KIMI_API_KEY = os.environ.get("KIMI_API_KEY", "")
KIMI_API_BASE = os.environ.get("KIMI_API_BASE", "https://api.moonshot.cn/v1")


# ═══════════════════════════════════════════════════════════
# Data storage
# ═══════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).parent / "distillation_data"
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "distiller_state.json"


# ═══════════════════════════════════════════════════════════
# Capsule Training Topics — كل كبسولة + مواضيعها
# ═══════════════════════════════════════════════════════════
CAPSULE_TOPICS = {
    # === العلوم ===
    "science.physics.mechanics": {
        "name_ar": "ميكانيكا",
        "topics": [
            "قوانين نيوتن الثلاثة مع أمثلة عملية",
            "ميكانيكا الموائع — برنولي وتطبيقاته",
            "قوة الاحتكاك وتطبيقاتها في المصانع",
            "الرافعات والآلات البسيطة — تصميم وحساب",
            "ديناميكا المواد الصلبة — إجهاد وانفعال",
            "مقاومة المواد — حسابات الهياكل",
            "الاهتزازات الميكانيكية — تحليل وتخميد",
            "هيدروليك — مضخات وأسطوانات",
            "حركة المقذوفات — بالستيك",
            "التوازن الاستاتيكي — جسور ومباني",
        ],
    },
    "science.physics.thermodynamics": {
        "name_ar": "ديناميكا حرارية",
        "topics": [
            "القانون الأول — حفظ الطاقة بالمصانع",
            "القانون الثاني — أنتروبيا وكفاءة المحركات",
            "دورة كارنو — المحرك المثالي",
            "انتقال الحرارة — توصيل وحمل وإشعاع",
            "تصميم مبادلات حرارية للمصانع",
            "التبريد الصناعي — أنظمة ومبرّدات",
            "أفران صناعية — تصميم وحسابات",
            "توليد البخار — غلايات ومولدات",
            "خصائص الغاز المثالي وتطبيقاته",
            "نقطة الانصهار والغليان لكل المعادن",
        ],
    },
    "science.physics.quantum": {
        "name_ar": "ميكانيكا الكم",
        "topics": [
            "أشباه الموصلات — كيف تعمل الترانزستورات",
            "تأثير النفق الكمي — استخدامه بالإلكترونيات",
            "فيزياء المواد الصلبة — بنية البلورات",
            "التوصيل الكهربائي — معادن وعوازل",
            "أشعة الليزر — مبدأ العمل والتطبيقات",
            "الطاقة النووية — انشطار واندماج",
            "تقنية النانو — مواد بخصائص كمية",
            "الخلايا الشمسية — تأثير فوتوفولطي",
            "الحوسبة الكمية — مبادئ أساسية",
            "التحليل الطيفي — تحديد المواد",
        ],
    },

    # === الكيمياء ===
    "science.chemistry.organic": {
        "name_ar": "كيمياء عضوية",
        "topics": [
            "البوليمرات — تصنيع البلاستيك من الصفر",
            "المطاط الصناعي — تركيب وإنتاج",
            "الأدوية — تصنيع الأسبرين والمضادات الحيوية",
            "البتروكيماويات — تكرير النفط",
            "الأسمدة العضوية — إنتاج ومواصفات",
            "المبيدات — تركيب آمن",
            "الألياف الصناعية — نايلون وبوليستر",
            "الصابون والمنظفات — تصنيع كامل",
            "المواد اللاصقة — أنواع وتركيب",
            "الأصباغ — صناعة الألوان",
        ],
    },
    "science.chemistry.inorganic": {
        "name_ar": "كيمياء غير عضوية",
        "topics": [
            "استخراج المعادن من الخامات",
            "تنقية السيليكون — نقاء 99.999%",
            "تصنيع الأسمنت — كيمياء التفاعل",
            "تصنيع الزجاج — مواد وحرارة",
            "تصنيع السيراميك — طين وحرق",
            "كيمياء المياه — تنقية وتحلية",
            "تآكل المعادن — أسباب وحلول",
            "البطاريات — كيمياء ليثيوم أيون",
            "الأملاح الصناعية — إنتاج واستخدام",
            "الغازات الصناعية — أكسجين ونيتروجين",
        ],
    },
    "science.chemistry.industrial": {
        "name_ar": "كيمياء صناعية",
        "topics": [
            "عملية هابر — إنتاج الأمونيا",
            "عملية كلور-قلوي — إنتاج الكلور والصودا",
            "تصنيع حمض الكبريتيك — عملية التلامس",
            "تصنيع حمض النيتريك — عملية أوستفالد",
            "معالجة مياه الصرف الصناعي",
            "التحفيز الصناعي — أنواع المحفزات",
            "تقطير النفط — أبراج التقطير",
            "التخمير الصناعي — إيثانول وخميرة",
            "صناعة الورق — من الخشب للمنتج",
            "معالجة النفايات الخطرة",
        ],
    },

    # === الهندسة ===
    "engineering.civil.structures": {
        "name_ar": "هندسة إنشائية",
        "topics": [
            "تصميم أساسات المصانع — حسابات تربة",
            "هياكل حديدية — حساب وتصميم",
            "خرسانة مسلحة — خلطات وتسليح",
            "تصميم مخازن ومستودعات كبيرة",
            "مقاومة الزلازل — تصميم مضاد",
            "جسور — أنواع وتصميم",
            "أنفاق — حفر وتبطين",
            "سدود — أنواع وحسابات",
            "معالجة التربة الضعيفة",
            "صيانة المباني الصناعية",
        ],
    },
    "engineering.civil.roads": {
        "name_ar": "طرق ومواصلات",
        "topics": [
            "تصميم طرق — منحنيات وميول",
            "أسفلت — خلطات وتنفيذ",
            "سكك حديد — تصميم وإنشاء",
            "مطارات — مدارج ومرافق",
            "موانئ — تصميم أرصفة",
        ],
    },
    "engineering.mechanical.engines": {
        "name_ar": "محركات",
        "topics": [
            "محركات ديزل — تصميم وصيانة",
            "محركات بنزين — 4 أشواط",
            "توربينات غازية — مبدأ عمل",
            "توربينات بخارية — محطات طاقة",
            "محركات كهربائية — AC وDC",
            "مضخات — أنواع وتصميم",
            "ضواغط هواء — صناعي",
            "تروس ونقل حركة",
            "أنظمة هيدروليك",
            "أنظمة نيوماتيك (هوائية)",
        ],
    },
    "engineering.mechanical.hvac": {
        "name_ar": "تدفئة وتبريد",
        "topics": [
            "تصميم أنظمة تكييف مركزي",
            "تبريد صناعي — غرف تبريد وتجميد",
            "تهوية مصانع — معالجة هواء",
            "عزل حراري — مواد وتطبيق",
            "أبراج تبريد — تصميم وصيانة",
        ],
    },
    "engineering.electrical.power_systems": {
        "name_ar": "أنظمة قدرة",
        "topics": [
            "تصميم شبكة كهرباء لمصنع",
            "محولات — أنواع وحماية",
            "مولدات ديزل — احتياطي طوارئ",
            "لوحات توزيع كهربائية",
            "حماية كهربائية — قواطع ومرحلات",
            "أنظمة UPS — بدون انقطاع",
            "كابلات كهربائية — اختيار وتمديد",
            "تأريض كهربائي — سلامة",
            "جودة الطاقة — هارمونيات",
            "توفير الطاقة بالمصانع",
        ],
    },
    "engineering.electrical.electronics": {
        "name_ar": "إلكترونيات",
        "topics": [
            "تصميم PCB — من الصفر",
            "ترانزستورات — أنواع واستخدام",
            "متحكمات دقيقة — Arduino/ESP32",
            "FPGA — تصميم رقمي",
            "حساسات صناعية — أنواع ومعايرة",
            "PLC — برمجة وتطبيقات",
            "SCADA — مراقبة صناعية",
            "اتصالات صناعية — Modbus/CAN",
            "أنظمة مدمجة — تصميم",
            "تصنيع رقائق — عملية lithography",
        ],
    },
    "engineering.electrical.renewable": {
        "name_ar": "طاقة متجددة",
        "topics": [
            "ألواح شمسية — تصميم نظام كامل",
            "توربينات رياح — تصميم وتشغيل",
            "بطاريات تخزين — ليثيوم وصوديوم",
            "طاقة مائية — سدود صغيرة",
            "هيدروجين أخضر — إنتاج وتخزين",
        ],
    },
    "engineering.electrical.solar_energy": {
        "name_ar": "طاقة شمسية",
        "topics": [
            "حسابات الألواح الشمسية — عدد ومساحة",
            "انفرتر — أنواع واختيار",
            "بطاريات — سعة وتوصيل",
            "on-grid vs off-grid",
            "صيانة أنظمة شمسية",
        ],
    },

    # === البرمجة ===
    "engineering.software.backend": {
        "name_ar": "برمجة خلفية",
        "topics": [
            "Python — هياكل بيانات متقدمة",
            "FastAPI — بناء API كامل",
            "PostgreSQL — تصميم قاعدة بيانات",
            "Redis — caching وقوائم انتظار",
            "Docker — حاويات وتنسيق",
            "تصميم أنظمة — microservices",
            "أمان API — JWT و OAuth",
            "performance — تحسين أداء",
            "GraphQL — بديل REST",
            "WebSocket — اتصال حي",
        ],
    },
    "engineering.software.web": {
        "name_ar": "تطوير ويب",
        "topics": [
            "React/Next.js — بناء تطبيق كامل",
            "TypeScript — أنماط متقدمة",
            "CSS — تصميم responsive",
            "أداء الويب — Core Web Vitals",
            "PWA — تطبيقات ويب تقدمية",
        ],
    },
    "engineering.software.mobile": {
        "name_ar": "تطوير موبايل",
        "topics": [
            "React Native — تطبيق كامل",
            "Flutter — بديل cross-platform",
            "iOS/Android native — أساسيات",
            "push notifications — تنفيذ",
            "offline-first — بدون إنترنت",
        ],
    },
    "engineering.software.devops": {
        "name_ar": "DevOps",
        "topics": [
            "CI/CD — GitHub Actions",
            "Kubernetes — تنسيق حاويات",
            "مراقبة — Prometheus/Grafana",
            "Infrastructure as Code — Terraform",
            "أمان — حماية الخوادم",
        ],
    },

    # === AI/ML ===
    "computing.ai_ml.nlp": {
        "name_ar": "معالجة لغة طبيعية",
        "topics": [
            "Transformer — بنية النموذج بالتفصيل",
            "LoRA/QLoRA — تدريب فعّال",
            "RAG — بحث واسترجاع",
            "Tokenization — عربي وإنجليزي",
            "Fine-tuning — خطوات كاملة",
            "تحليل مشاعر — عربي",
            "ترجمة آلية — نماذج",
            "تلخيص نصوص — خوارزميات",
            "استخراج معلومات — NER",
            "توليد نصوص — GPT-like",
        ],
    },
    "computing.ai_ml.vision_ai": {
        "name_ar": "رؤية حاسوبية",
        "topics": [
            "CNN — بنية وتدريب",
            "YOLO — كشف أجسام",
            "Segmentation — تقسيم صور",
            "OCR — قراءة نص من صور",
            "Face Recognition — تعرف وجوه",
            "Object Tracking — تتبع حركة",
            "Image Classification — تصنيف",
            "Data Augmentation — تكثير بيانات",
            "Transfer Learning — نقل تعلم",
            "Video Analysis — تحليل فيديو",
        ],
    },

    # === الأمن ===
    "computing.security.crypto": {
        "name_ar": "تشفير",
        "topics": [
            "AES — تشفير متماثل",
            "RSA — تشفير غير متماثل",
            "SHA — دوال تجزئة",
            "PKI — بنية مفاتيح عامة",
            "TLS/SSL — اتصال آمن",
        ],
    },
    "computing.security.pentest": {
        "name_ar": "اختبار اختراق",
        "topics": [
            "OWASP Top 10 — ثغرات ويب",
            "فحص شبكات — nmap وwireshark",
            "هندسة اجتماعية — أساليب وحماية",
            "اختراق تطبيقات — ويب وموبايل",
            "تحليل malware — أساسيات",
        ],
    },

    # === التصنيع ===
    "manufacturing.metals.steelmaking": {
        "name_ar": "صناعة حديد",
        "topics": [
            "فرن عالٍ — من خام لحديد خام",
            "فرن قوس كهربائي — صلب من خردة",
            "صب مستمر — billets وslabs",
            "درفلة — ألواح وقضبان",
            "معالجة حرارية — تقسية وتلدين",
            "سبائك حديد — أنواع ومواصفات",
            "اختبارات جودة — شد وصلادة",
            "لحام — أنواع وتقنيات",
            "طلاء وحماية من التآكل",
            "تصنيع أنابيب فولاذية",
        ],
    },
    "manufacturing.metals.casting": {
        "name_ar": "سباكة معادن",
        "topics": [
            "صب بالرمل — قوالب وصب",
            "صب بالقالب المعدني — die casting",
            "صب بالشمع المفقود — دقة عالية",
            "صب الألمنيوم — خفيف ومقاوم",
            "صب النحاس والبرونز",
            "عيوب الصب — أسباب وحلول",
            "تشطيب وتشغيل — CNC",
            "مراقبة جودة — أشعة سينية",
        ],
    },

    # === نانو ===
    "science.nanotech": {
        "name_ar": "تقنية النانو",
        "topics": [
            "مواد نانوية — أنابيب كربونية",
            "طلاءات نانوية — مقاومة ماء",
            "أغشية نانوية — تنقية مياه",
            "حساسات نانوية — كشف غازات",
            "تصنيع نانوي — bottom-up وtop-down",
        ],
    },
}

# ═══════════════════════════════════════════════════════════
# سلاسل الإمداد — مواضيع إضافية
# ═══════════════════════════════════════════════════════════
SUPPLY_CHAIN_TOPICS = [
    "سلسلة إمداد مصنع رقائق إلكترونية — من الرمل للمعالج",
    "سلسلة إمداد مصنع سيارات — كل المكونات ومصادرها",
    "سلسلة إمداد مصنع أدوية — مواد خام ومعايير",
    "سلسلة إمداد مصنع طائرات — مواد وتقنيات",
    "سلسلة إمداد مصنع صواريخ — وقود ومحركات ومواد",
    "سلسلة إمداد مصنع هواتف — شاشات وبطاريات ورقائق",
    "سلسلة إمداد مصنع ألواح شمسية — سيليكون وتغليف",
    "سلسلة إمداد مصنع بطاريات ليثيوم — تعدين لتجميع",
    "سلسلة إمداد مصنع أسمنت — حجر كلسي وطاقة",
    "سلسلة إمداد مصنع زجاج — رمل وصودا وحرارة",
    "سلسلة إمداد مصنع ورق — خشب ولب وتبييض",
    "سلسلة إمداد مصنع ملابس — قطن ونسيج وصباغة",
    "سلسلة إمداد مصنع إنترنت (كابلات بحرية + أبراج)",
    "بدائل المواد الخام — ماذا لو مادة غير متوفرة؟",
    "إعادة بناء سلسلة إمداد بعد كارثة — خطة كاملة",
    "كيف تحدد أي مصنع تبنيه أولاً بعد الكارثة",
    "ترتيب أولويات المصانع — أيها الأهم للبقاء",
    "إنتاج كهرباء بدون شبكة — مولدات ذاتية",
    "تصنيع أدوات يدوية من الصفر — حدادة وتشكيل",
    "استخراج المياه النظيفة — آبار وتنقية",
]


@dataclass
class DistillerState:
    """حالة المجمّع — يحفظ التقدم"""
    total_generated: int = 0
    capsules_done: Dict[str, int] = field(default_factory=dict)
    supply_chain_done: int = 0
    seen_hashes: List[str] = field(default_factory=list)
    last_run: str = ""
    errors: int = 0


class KnowledgeDistiller:
    """
    يجمع بيانات تدريب من Kimi API + أي OpenAI-compatible API

    الناتج: JSONL files لكل كبسولة
    """

    def __init__(self):
        self.state = self._load_state()
        self.api_key = KIMI_API_KEY
        self.api_base = KIMI_API_BASE

    def _load_state(self) -> DistillerState:
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                return DistillerState(**data)
            except Exception:
                pass
        return DistillerState()

    def _save_state(self):
        STATE_FILE.write_text(json.dumps(asdict(self.state), ensure_ascii=False, indent=2))

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _call_kimi(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> Optional[str]:
        """Call Kimi K2.5 API (OpenAI-compatible)"""
        try:
            import urllib.request
            import ssl

            url = f"{self.api_base}/chat/completions"
            payload = {
                "model": "moonshot-v1-auto",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": 4096,
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )

            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                result = json.loads(resp.read().decode())
                return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"  ❌ Kimi error: {e}")
            self.state.errors += 1
            return None

    def generate_qa_for_capsule(self, capsule_id: str, num_per_topic: int = 10) -> int:
        """
        توليد Q&A لكبسولة واحدة

        Returns: عدد الأزواج المولّدة
        """
        if capsule_id not in CAPSULE_TOPICS:
            print(f"  ⚠️ كبسولة غير معروفة: {capsule_id}")
            return 0

        capsule = CAPSULE_TOPICS[capsule_id]
        output_file = DATA_DIR / f"{capsule_id.replace('.', '_')}.jsonl"
        count = 0

        system_prompt = f"""أنت خبير عالمي في {capsule['name_ar']}.
مهمتك: توليد أسئلة وأجوبة تعليمية دقيقة وعملية.
الهدف: تدريب ذكاء اصطناعي يقدر يبني مصانع ويعيد بناء الحضارة.
القواعد:
1. الأجوبة لازم تكون دقيقة علمياً 100%
2. تشمل أرقام وحسابات حقيقية
3. تشمل خطوات عملية قابلة للتنفيذ
4. باللغة العربية + المصطلحات الإنجليزية التقنية
5. رد بصيغة JSON array من objects بمفاتيح "question" و "answer"
"""

        for topic in capsule["topics"]:
            h = self._hash(f"{capsule_id}:{topic}")
            if h in self.state.seen_hashes:
                print(f"  ⏭️ تخطي (موجود): {topic[:40]}")
                continue

            user_prompt = f"""أنشئ {num_per_topic} أسئلة وأجوبة عن:
"{topic}"

حالات الاستخدام: بناء مصنع من الصfr, إعادة بناء حضارة, تدريب بشر.

رد بصيغة JSON:
[
  {{"question": "...", "answer": "..."}},
  ...
]"""

            print(f"  📚 {capsule_id} | {topic[:50]}...")
            response = self._call_kimi(system_prompt, user_prompt)
            if not response:
                time.sleep(2)
                continue

            # Parse JSON from response
            try:
                # Extract JSON array from response
                start = response.find("[")
                end = response.rfind("]") + 1
                if start >= 0 and end > start:
                    qa_pairs = json.loads(response[start:end])
                else:
                    qa_pairs = json.loads(response)

                with open(output_file, "a", encoding="utf-8") as f:
                    for qa in qa_pairs:
                        if "question" in qa and "answer" in qa:
                            record = {
                                "capsule_id": capsule_id,
                                "topic": topic,
                                "question": qa["question"],
                                "answer": qa["answer"],
                                "source": "kimi_k2.5",
                                "timestamp": datetime.now().isoformat(),
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            count += 1

                self.state.seen_hashes.append(h)
                self.state.total_generated += len(qa_pairs)
                print(f"    ✅ {len(qa_pairs)} Q&A pairs")

            except (json.JSONDecodeError, Exception) as e:
                print(f"    ⚠️ Parse error: {e}")
                # Save raw response as fallback
                raw_file = DATA_DIR / f"{capsule_id.replace('.', '_')}_raw.txt"
                with open(raw_file, "a", encoding="utf-8") as f:
                    f.write(f"\n--- {topic} ---\n{response}\n")

            # Rate limiting — Kimi allows reasonable rates
            time.sleep(1)

        self.state.capsules_done[capsule_id] = count
        self._save_state()
        return count

    def generate_supply_chain_data(self, num_per_topic: int = 5) -> int:
        """توليد بيانات سلاسل الإمداد"""
        output_file = DATA_DIR / "supply_chains.jsonl"
        count = 0

        system_prompt = """أنت خبير عالمي في سلاسل الإمداد الصناعية والتصنيع.
مهمتك: وصف سلسلة إمداد كاملة بتفاصيل دقيقة.
الهدف: بناء مصانع من الصفر بعد كارثة.
القواعد:
1. اذكر كل المواد الخام المطلوبة + مصادرها
2. اذكر كل خطوة تصنيع بالترتيب
3. اذكر المعدات المطلوبة لكل خطوة
4. اذكر الطاقة المطلوبة
5. اذكر البدائل إذا مادة غير متوفرة
6. رد بصيغة JSON array من objects بمفاتيح "question" و "answer"
"""

        for topic in SUPPLY_CHAIN_TOPICS:
            h = self._hash(f"supply:{topic}")
            if h in self.state.seen_hashes:
                continue

            user_prompt = f"""أنشئ {num_per_topic} أسئلة وأجوبة عن:
"{topic}"

رد بصيغة JSON:
[{{"question": "...", "answer": "..."}}]"""

            print(f"  🔗 Supply: {topic[:50]}...")
            response = self._call_kimi(system_prompt, user_prompt)
            if not response:
                time.sleep(2)
                continue

            try:
                start = response.find("[")
                end = response.rfind("]") + 1
                if start >= 0 and end > start:
                    qa_pairs = json.loads(response[start:end])
                else:
                    qa_pairs = json.loads(response)

                with open(output_file, "a", encoding="utf-8") as f:
                    for qa in qa_pairs:
                        if "question" in qa and "answer" in qa:
                            record = {
                                "capsule_id": "supply_chain",
                                "topic": topic,
                                "question": qa["question"],
                                "answer": qa["answer"],
                                "source": "kimi_k2.5",
                                "timestamp": datetime.now().isoformat(),
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            count += 1

                self.state.seen_hashes.append(h)
                self.state.supply_chain_done += count
                self.state.total_generated += len(qa_pairs)
                print(f"    ✅ {len(qa_pairs)} Q&A pairs")
            except Exception as e:
                print(f"    ⚠️ Parse error: {e}")

            time.sleep(1)

        self._save_state()
        return count

    def distill_all(self, num_per_topic: int = 10):
        """تشغيل الكل — كل الكبسولات + سلاسل الإمداد"""
        print("🧠 Knowledge Distiller — بدء جمع بيانات\n")
        print(f"  API: Kimi K2.5 ({self.api_base})")
        print(f"  كبسولات: {len(CAPSULE_TOPICS)}")
        print(f"  مواضيع سلاسل إمداد: {len(SUPPLY_CHAIN_TOPICS)}")
        total_topics = sum(len(c["topics"]) for c in CAPSULE_TOPICS.values())
        print(f"  إجمالي مواضيع: {total_topics + len(SUPPLY_CHAIN_TOPICS)}")
        print(f"  Q&A/موضوع: {num_per_topic}")
        print(f"  الحد الأقصى المتوقع: {(total_topics + len(SUPPLY_CHAIN_TOPICS)) * num_per_topic} Q&A pair")
        print(f"  تم سابقاً: {self.state.total_generated}")
        print()

        total = 0

        # 1. كل كبسولة
        for capsule_id in CAPSULE_TOPICS:
            print(f"\n{'='*60}")
            print(f"  📦 كبسولة: {capsule_id}")
            print(f"{'='*60}")
            count = self.generate_qa_for_capsule(capsule_id, num_per_topic)
            total += count
            print(f"  → {count} Q&A pairs لهذي الكبسولة")

        # 2. سلاسل إمداد
        print(f"\n{'='*60}")
        print(f"  🔗 سلاسل الإمداد")
        print(f"{'='*60}")
        sc_count = self.generate_supply_chain_data(num_per_topic)
        total += sc_count

        self.state.last_run = datetime.now().isoformat()
        self._save_state()

        print(f"\n{'='*60}")
        print(f"✅ انتهى — {total} Q&A pair جديدة")
        print(f"  الإجمالي الكلي: {self.state.total_generated}")
        print(f"  الملفات: {DATA_DIR}")
        print(f"{'='*60}")

        return total

    def get_status(self) -> Dict:
        return {
            "total_generated": self.state.total_generated,
            "capsules_done": len(self.state.capsules_done),
            "capsules_total": len(CAPSULE_TOPICS),
            "supply_chain_done": self.state.supply_chain_done,
            "errors": self.state.errors,
            "last_run": self.state.last_run,
            "data_dir": str(DATA_DIR),
        }


# Singleton
distiller = KnowledgeDistiller()


if __name__ == "__main__":
    print("🧠 BI-IDE Knowledge Distiller\n")

    if not KIMI_API_KEY:
        print("❌ KIMI_API_KEY not set! Check brain/.env")
        exit(1)

    print(f"API Key: {KIMI_API_KEY[:15]}...")
    print(f"Topics: {sum(len(c['topics']) for c in CAPSULE_TOPICS.values())} + {len(SUPPLY_CHAIN_TOPICS)} supply chain")
    print()

    # Start distillation
    distiller.distill_all(num_per_topic=10)
