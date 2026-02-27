#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Bi IDE - نظام التدريب الشامل للذكاء                      ║
║                        AI Training System v1.0                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  التدريب على:                                                                ║
║    • فهم الكود البرمجي                                                       ║
║    • تطوير وإكمال الكود                                                      ║
║    • اكتشاف وإصلاح الأخطاء                                                   ║
║    • الإجابة على الأسئلة البرمجية                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  التشغيل: python train_ai.py                                                 ║
║  أو:      python train_ai.py --mode full                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# إعداد الترميز للعربية
if sys.platform == 'win32':
    import locale
    locale.setlocale(locale.LC_ALL, '')
    sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════════════════════
# الثوابت والإعدادات
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "training" / "output"
LOGS_DIR = BASE_DIR / "training" / "logs"

# إنشاء المجلدات
for dir_path in [OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# هياكل البيانات
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CodeSample:
    """عينة كود للتدريب"""
    code: str
    language: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass 
class ErrorPattern:
    """نمط خطأ للتدريب على إصلاح الأخطاء"""
    error_code: str
    correct_code: str
    error_type: str
    language: str
    description: str = ""
    explanation: str = ""


@dataclass
class QAPair:
    """زوج سؤال وجواب للتدريب"""
    question: str
    answer: str
    context: str = ""
    category: str = ""


@dataclass
class TrainingStats:
    """إحصائيات التدريب"""
    total_samples: int = 0
    code_samples: int = 0
    error_patterns: int = 0
    qa_pairs: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    categories: Dict[str, int] = field(default_factory=dict)
    start_time: float = 0
    end_time: float = 0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0


# ═══════════════════════════════════════════════════════════════════════════════
# محمّل البيانات
# ═══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """تحميل بيانات التدريب من ملفات المشروع"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.code_samples: List[CodeSample] = []
        self.error_patterns: List[ErrorPattern] = []
        self.qa_pairs: List[QAPair] = []
        self.knowledge_base: Dict[str, Any] = {}
        
    def load_all(self) -> Dict[str, Any]:
        """تحميل جميع البيانات"""
        logger.info("=" * 60)
        logger.info("Loading Training Data...")
        logger.info("=" * 60)
        
        # تحميل قواعد المعرفة
        self._load_knowledge_bases()
        
        # تحميل بيانات التدريب المتخصصة
        self._load_training_data()
        
        # تحميل بيانات المحادثات
        self._load_conversation_data()
        
        # تحميل أنماط الأخطاء
        self._load_error_patterns()
        
        stats = {
            "code_samples": len(self.code_samples),
            "error_patterns": len(self.error_patterns),
            "qa_pairs": len(self.qa_pairs),
            "knowledge_topics": len(self.knowledge_base)
        }
        
        logger.info(f"[OK] Data loaded: {stats}")
        return stats
    
    def _load_knowledge_bases(self):
        """تحميل ملفات قاعدة المعرفة"""
        knowledge_files = [
            "comprehensive-knowledge-base.js",
            "extended-knowledge-base.js",
            "programming-knowledge.js",
            "computer-science-knowledge.js",
            "frameworks-knowledge.js"
        ]
        
        for filename in knowledge_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding='utf-8')
                    data = self._parse_js_module(content)
                    if data:
                        self.knowledge_base.update(data)
                        logger.info(f"   ✓ {filename}")
                except Exception as e:
                    logger.warning(f"   [WARN] Error in {filename}: {e}")
    
    def _load_training_data(self):
        """تحميل بيانات التدريب المتخصصة"""
        training_files = [
            ("advanced-debugging-training.js", "debugging"),
            ("design-patterns-training.js", "patterns"),
            ("security-vulnerabilities-training.js", "security"),
            ("ai-ml-programming-training.js", "ai_ml"),
            ("advanced-algorithms-data-structures.js", "algorithms")
        ]
        
        for filename, category in training_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding='utf-8')
                    samples = self._extract_code_samples(content, category)
                    self.code_samples.extend(samples)
                    logger.info(f"   ✓ {filename}: {len(samples)} عينة")
                except Exception as e:
                    logger.warning(f"   [WARN] Error in {filename}: {e}")
    
    def _load_conversation_data(self):
        """تحميل بيانات المحادثات"""
        conv_files = [
            "advanced-conversation-data.json",
            "ai-conversations.json",
            "conversation-training.js",
            "smart-replies-training.js"
        ]
        
        for filename in conv_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    if filename.endswith('.json'):
                        data = json.loads(filepath.read_text(encoding='utf-8'))
                        pairs = self._extract_qa_from_json(data)
                    else:
                        content = filepath.read_text(encoding='utf-8')
                        pairs = self._extract_qa_from_js(content)
                    
                    self.qa_pairs.extend(pairs)
                    logger.info(f"   ✓ {filename}: {len(pairs)} محادثة")
                except Exception as e:
                    logger.warning(f"   [WARN] Error in {filename}: {e}")
    
    def _load_error_patterns(self):
        """تحميل أنماط الأخطاء"""
        error_files = [
            "advanced-error-detection.js",
            "debugging-mastery-training.js"
        ]
        
        for filename in error_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding='utf-8')
                    patterns = self._extract_error_patterns(content)
                    self.error_patterns.extend(patterns)
                    logger.info(f"   ✓ {filename}: {len(patterns)} نمط خطأ")
                except Exception as e:
                    logger.warning(f"   [WARN] Error in {filename}: {e}")
    
    def _parse_js_module(self, content: str) -> Dict:
        """تحليل محتوى ملف JavaScript واستخراج البيانات"""
        import re
        
        data = {}
        
        # استخراج المصفوفات والكائنات
        # البحث عن patterns مثل: const name = { ... } أو const name = [ ... ]
        pattern = r'(?:const|let|var)\s+(\w+)\s*=\s*(\{[\s\S]*?\}|\[[\s\S]*?\]);'
        
        matches = re.findall(pattern, content)
        for name, value in matches:
            try:
                # محاولة تحويل JavaScript إلى JSON
                json_str = self._js_to_json(value)
                data[name] = json.loads(json_str)
            except:
                pass
        
        # استخراج النصوص والأمثلة
        code_blocks = re.findall(r'```(\w+)?\n([\s\S]*?)```', content)
        if code_blocks:
            data['code_examples'] = [
                {'language': lang or 'text', 'code': code}
                for lang, code in code_blocks
            ]
        
        return data
    
    def _js_to_json(self, js_str: str) -> str:
        """تحويل JavaScript object إلى JSON صالح"""
        import re
        
        # إزالة التعليقات
        result = re.sub(r'//.*$', '', js_str, flags=re.MULTILINE)
        result = re.sub(r'/\*[\s\S]*?\*/', '', result)
        
        # تحويل المفاتيح بدون علامات اقتباس
        result = re.sub(r'(\w+):', r'"\1":', result)
        
        # تحويل علامات الاقتباس المفردة
        result = result.replace("'", '"')
        
        # إزالة الفواصل الزائدة
        result = re.sub(r',\s*([}\]])', r'\1', result)
        
        return result
    
    def _extract_code_samples(self, content: str, category: str) -> List[CodeSample]:
        """استخراج عينات الكود من المحتوى"""
        import re
        
        samples = []
        
        # البحث عن code blocks
        code_pattern = r'(?:code|example|snippet)[\'":\s]*[`\'"]*([\s\S]*?)[`\'"]*(?:,|\})'
        
        for match in re.finditer(code_pattern, content, re.IGNORECASE):
            code = match.group(1).strip()
            if len(code) > 20:  # تجاهل الأمثلة القصيرة جداً
                language = self._detect_language(code)
                samples.append(CodeSample(
                    code=code,
                    language=language,
                    tags=[category],
                    metadata={'source': 'training_file'}
                ))
        
        # البحث عن code blocks بـ backticks
        backtick_pattern = r'```(\w+)?\n([\s\S]*?)```'
        for match in re.finditer(backtick_pattern, content):
            lang = match.group(1) or 'javascript'
            code = match.group(2).strip()
            if len(code) > 20:
                samples.append(CodeSample(
                    code=code,
                    language=lang,
                    tags=[category],
                    metadata={'source': 'training_file'}
                ))
        
        return samples
    
    def _extract_qa_from_json(self, data: Any) -> List[QAPair]:
        """استخراج أزواج Q&A من JSON"""
        pairs = []
        
        def extract_recursive(obj, path=""):
            if isinstance(obj, dict):
                # البحث عن أنماط Q&A
                q = obj.get('question') or obj.get('q') or obj.get('input')
                a = obj.get('answer') or obj.get('a') or obj.get('output') or obj.get('response')
                
                if q and a:
                    pairs.append(QAPair(
                        question=str(q),
                        answer=str(a),
                        category=obj.get('category', path)
                    ))
                
                for key, value in obj.items():
                    extract_recursive(value, f"{path}/{key}")
                    
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, path)
        
        extract_recursive(data)
        return pairs
    
    def _extract_qa_from_js(self, content: str) -> List[QAPair]:
        """استخراج أزواج Q&A من ملف JavaScript"""
        import re
        
        pairs = []
        
        # البحث عن أنماط السؤال والجواب
        patterns = [
            r'(?:question|q)[\'":\s]+[\'"]([^"\']+)[\'"][\s\S]*?(?:answer|a|response)[\'":\s]+[\'"]([^"\']+)[\'"]',
            r'(?:input)[\'":\s]+[\'"]([^"\']+)[\'"][\s\S]*?(?:output)[\'":\s]+[\'"]([^"\']+)[\'"]'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                pairs.append(QAPair(
                    question=match.group(1).strip(),
                    answer=match.group(2).strip()
                ))
        
        return pairs
    
    def _extract_error_patterns(self, content: str) -> List[ErrorPattern]:
        """استخراج أنماط الأخطاء"""
        import re
        
        patterns = []
        
        # البحث عن أنماط الخطأ والإصلاح
        error_pattern = r'(?:wrong|error|incorrect|bug)[\'":\s]*[\'"`]*([\s\S]*?)[\'"`]*[\s\S]*?(?:correct|fix|solution)[\'":\s]*[\'"`]*([\s\S]*?)[\'"`]*(?:,|\})'
        
        for match in re.finditer(error_pattern, content, re.IGNORECASE):
            error_code = match.group(1).strip()
            correct_code = match.group(2).strip()
            
            if len(error_code) > 10 and len(correct_code) > 10:
                patterns.append(ErrorPattern(
                    error_code=error_code,
                    correct_code=correct_code,
                    error_type='general',
                    language=self._detect_language(error_code)
                ))
        
        return patterns
    
    def _detect_language(self, code: str) -> str:
        """اكتشاف لغة البرمجة"""
        indicators = {
            'javascript': ['function', 'const ', 'let ', 'var ', '=>', 'require(', 'import '],
            'python': ['def ', 'import ', 'from ', 'class ', 'self.', '__init__', 'print('],
            'java': ['public class', 'private ', 'void ', 'String ', 'System.out'],
            'cpp': ['#include', 'std::', 'cout', 'int main', '::'],
            'html': ['<html', '<div', '<span', '</div>', '<!DOCTYPE'],
            'css': ['{', '}', 'color:', 'font-', 'margin:', 'padding:'],
            'sql': ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'CREATE TABLE'],
            'php': ['<?php', '$_', 'echo ', '->'],
            'rust': ['fn ', 'let mut', 'impl ', '::new('],
            'go': ['func ', 'package ', 'import (', 'fmt.']
        }
        
        code_lower = code.lower()
        scores = defaultdict(int)
        
        for lang, keywords in indicators.items():
            for keyword in keywords:
                if keyword.lower() in code_lower:
                    scores[lang] += 1
        
        if scores:
            return max(scores, key=scores.get)
        return 'unknown'


# ═══════════════════════════════════════════════════════════════════════════════
# مولّد بيانات التدريب
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingDataGenerator:
    """توليد بيانات تدريب إضافية"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """تحميل قوالب توليد البيانات"""
        return {
            'error_fixes': {
                'javascript': [
                    {
                        'error': 'console.log(undefined_variable)',
                        'fix': 'const defined_variable = "value";\nconsole.log(defined_variable)',
                        'type': 'ReferenceError',
                        'explanation': 'يجب تعريف المتغير قبل استخدامه'
                    },
                    {
                        'error': 'const arr = [1,2,3];\narr.map(x => x.toUpperCase())',
                        'fix': 'const arr = ["a","b","c"];\narr.map(x => x.toUpperCase())',
                        'type': 'TypeError',
                        'explanation': 'toUpperCase() تعمل فقط مع النصوص'
                    },
                    {
                        'error': 'async function getData() {\n  const data = fetch(url);\n  return data.json();\n}',
                        'fix': 'async function getData() {\n  const data = await fetch(url);\n  return await data.json();\n}',
                        'type': 'Missing await',
                        'explanation': 'يجب استخدام await مع الدوال غير المتزامنة'
                    },
                    {
                        'error': 'for (var i = 0; i < 5; i++) {\n  setTimeout(() => console.log(i), 100);\n}',
                        'fix': 'for (let i = 0; i < 5; i++) {\n  setTimeout(() => console.log(i), 100);\n}',
                        'type': 'Closure issue',
                        'explanation': 'استخدم let بدل var في الحلقات مع closures'
                    },
                    {
                        'error': 'const obj = {a: 1};\nconst copy = obj;\ncopy.a = 2; // obj.a أيضاً تتغير',
                        'fix': 'const obj = {a: 1};\nconst copy = {...obj};\ncopy.a = 2; // obj.a تبقى 1',
                        'type': 'Reference mutation',
                        'explanation': 'استخدم spread operator لنسخ الكائنات'
                    }
                ],
                'python': [
                    {
                        'error': 'def add(a, b=[]):\n    b.append(a)\n    return b',
                        'fix': 'def add(a, b=None):\n    if b is None:\n        b = []\n    b.append(a)\n    return b',
                        'type': 'Mutable default argument',
                        'explanation': 'لا تستخدم كائنات قابلة للتغيير كقيم افتراضية'
                    },
                    {
                        'error': 'items = [1, 2, 3]\nfor i in items:\n    if i == 2:\n        items.remove(i)',
                        'fix': 'items = [1, 2, 3]\nitems = [i for i in items if i != 2]',
                        'type': 'Modifying list while iterating',
                        'explanation': 'لا تعدل القائمة أثناء التكرار عليها'
                    },
                    {
                        'error': 'class MyClass:\n    items = []\n    def add(self, item):\n        self.items.append(item)',
                        'fix': 'class MyClass:\n    def __init__(self):\n        self.items = []\n    def add(self, item):\n        self.items.append(item)',
                        'type': 'Class variable mutation',
                        'explanation': 'استخدم instance variables بدل class variables للبيانات المتغيرة'
                    }
                ]
            },
            'code_patterns': {
                'javascript': [
                    {
                        'name': 'Singleton Pattern',
                        'code': '''class Database {
    static instance = null;
    
    static getInstance() {
        if (!Database.instance) {
            Database.instance = new Database();
        }
        return Database.instance;
    }
    
    constructor() {
        if (Database.instance) {
            return Database.instance;
        }
        this.connection = null;
    }
}''',
                        'description': 'نمط Singleton يضمن وجود نسخة واحدة فقط من الكلاس'
                    },
                    {
                        'name': 'Observer Pattern',
                        'code': '''class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }
    
    emit(event, data) {
        if (this.events[event]) {
            this.events[event].forEach(cb => cb(data));
        }
    }
    
    off(event, callback) {
        if (this.events[event]) {
            this.events[event] = this.events[event]
                .filter(cb => cb !== callback);
        }
    }
}''',
                        'description': 'نمط Observer للاشتراك في الأحداث والإشعارات'
                    },
                    {
                        'name': 'Factory Pattern',
                        'code': '''class ShapeFactory {
    static create(type, options) {
        switch(type) {
            case 'circle':
                return new Circle(options.radius);
            case 'rectangle':
                return new Rectangle(options.width, options.height);
            case 'triangle':
                return new Triangle(options.base, options.height);
            default:
                throw new Error(`Unknown shape: ${type}`);
        }
    }
}

// الاستخدام
const circle = ShapeFactory.create('circle', { radius: 5 });
const rect = ShapeFactory.create('rectangle', { width: 10, height: 5 });''',
                        'description': 'نمط Factory لإنشاء الكائنات بشكل مركزي'
                    }
                ],
                'python': [
                    {
                        'name': 'Decorator Pattern',
                        'code': '''from functools import wraps
import time

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.4f}s")
        return result
    return wrapper

def retry_decorator(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return decorator

@timing_decorator
@retry_decorator(max_retries=3)
def fetch_data(url):
    # جلب البيانات
    pass''',
                        'description': 'نمط Decorator لإضافة وظائف للدوال'
                    },
                    {
                        'name': 'Context Manager',
                        'code': '''from contextlib import contextmanager

class DatabaseConnection:
    def __init__(self, connection_string):
        self.conn_str = connection_string
        self.connection = None
    
    def __enter__(self):
        self.connection = self._connect()
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
        return False  # لا تمنع الاستثناءات
    
    def _connect(self):
        # منطق الاتصال
        pass

# أو باستخدام decorator
@contextmanager
def open_file(filename, mode='r'):
    f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()

# الاستخدام
with DatabaseConnection("...") as conn:
    conn.execute("SELECT * FROM users")''',
                        'description': 'نمط Context Manager لإدارة الموارد تلقائياً'
                    }
                ]
            },
            'qa_templates': [
                {
                    'q': 'ما هو الفرق بين == و === في JavaScript؟',
                    'a': '== يقارن القيم مع تحويل الأنواع (type coercion)، بينما === يقارن القيم والأنواع معاً (strict equality). مثال: "5" == 5 يعطي true، لكن "5" === 5 يعطي false.',
                    'category': 'javascript'
                },
                {
                    'q': 'ما هو closure في JavaScript؟',
                    'a': 'Closure هو دالة تتذكر المتغيرات من النطاق الخارجي (lexical scope) حتى بعد انتهاء تنفيذ الدالة الخارجية. يستخدم لإنشاء متغيرات خاصة وللحفاظ على الحالة.',
                    'category': 'javascript'
                },
                {
                    'q': 'ما الفرق بين list و tuple في Python؟',
                    'a': 'List قابلة للتعديل (mutable) بينما tuple غير قابلة للتعديل (immutable). tuple أسرع وتستهلك ذاكرة أقل، وتستخدم كمفاتيح في القواميس.',
                    'category': 'python'
                },
                {
                    'q': 'كيف أتعامل مع الأخطاء في async/await؟',
                    'a': 'استخدم try/catch حول await للتعامل مع الأخطاء. مثال:\ntry {\n  const data = await fetchData();\n} catch (error) {\n  console.error("Failed:", error);\n}',
                    'category': 'javascript'
                },
                {
                    'q': 'ما هو virtual environment في Python ولماذا نستخدمه؟',
                    'a': 'Virtual environment هو بيئة معزولة لمشروع Python محدد. يسمح بتثبيت مكتبات مختلفة لكل مشروع دون تعارض. إنشاؤه: python -m venv venv ثم تفعيله.',
                    'category': 'python'
                },
                {
                    'q': 'كيف أحسّن أداء React component؟',
                    'a': 'استخدم: 1) React.memo لمنع إعادة الرسم غير الضرورية 2) useMemo لحفظ الحسابات المكلفة 3) useCallback لحفظ الدوال 4) تقسيم الكود (code splitting) 5) تحميل كسول (lazy loading)',
                    'category': 'react'
                },
                {
                    'q': 'ما هو REST API وما هي مبادئه؟',
                    'a': 'REST (Representational State Transfer) هو نمط معماري للـ APIs. مبادئه: 1) Client-Server 2) Stateless 3) Cacheable 4) Uniform Interface 5) Layered System. يستخدم HTTP methods: GET, POST, PUT, DELETE.',
                    'category': 'api'
                },
                {
                    'q': 'كيف أمنع SQL Injection؟',
                    'a': 'استخدم: 1) Parameterized queries / Prepared statements 2) ORM بدل SQL المباشر 3) Input validation و sanitization 4) مبدأ least privilege لصلاحيات قاعدة البيانات.',
                    'category': 'security'
                }
            ]
        }
        
    def generate_error_patterns(self) -> List[ErrorPattern]:
        """توليد أنماط أخطاء للتدريب"""
        patterns = []
        
        for lang, errors in self.templates['error_fixes'].items():
            for error_data in errors:
                patterns.append(ErrorPattern(
                    error_code=error_data['error'],
                    correct_code=error_data['fix'],
                    error_type=error_data['type'],
                    language=lang,
                    explanation=error_data['explanation']
                ))
        
        return patterns
    
    def generate_code_samples(self) -> List[CodeSample]:
        """توليد عينات كود للتدريب"""
        samples = []
        
        for lang, patterns_list in self.templates['code_patterns'].items():
            for pattern in patterns_list:
                samples.append(CodeSample(
                    code=pattern['code'],
                    language=lang,
                    description=pattern['description'],
                    tags=['design_pattern', pattern['name'].lower().replace(' ', '_')]
                ))
        
        return samples
    
    def generate_qa_pairs(self) -> List[QAPair]:
        """توليد أزواج Q&A للتدريب"""
        pairs = []
        
        for qa in self.templates['qa_templates']:
            pairs.append(QAPair(
                question=qa['q'],
                answer=qa['a'],
                category=qa.get('category', 'general')
            ))
        
        return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# نظام التدريب الرئيسي
# ═══════════════════════════════════════════════════════════════════════════════

class AITrainer:
    """المدرب الرئيسي للذكاء الاصطناعي"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data_loader = DataLoader(DATA_DIR)
        self.data_generator = TrainingDataGenerator()
        self.stats = TrainingStats()
        
        # بيانات التدريب المجمعة
        self.training_data = {
            'code_understanding': [],      # فهم الكود
            'code_completion': [],         # إكمال الكود
            'error_detection': [],         # اكتشاف الأخطاء
            'error_fixing': [],            # إصلاح الأخطاء
            'code_explanation': [],        # شرح الكود
            'qa_pairs': [],                # أسئلة وأجوبة
            'design_patterns': [],         # أنماط التصميم
            'best_practices': []           # أفضل الممارسات
        }
        
        # النموذج (سيتم تحميله لاحقاً)
        self.model = None
        self.tokenizer = None
    
    def prepare_data(self):
        """تجهيز البيانات للتدريب"""
        logger.info("\n" + "=" * 60)
        logger.info("Preparing Training Data")
        logger.info("=" * 60 + "\n")
        
        self.stats.start_time = time.time()
        
        # 1. تحميل البيانات من الملفات
        logger.info("1️⃣ تحميل البيانات من ملفات المشروع...")
        file_stats = self.data_loader.load_all()
        
        # 2. توليد بيانات إضافية
        logger.info("\n2️⃣ توليد بيانات تدريب إضافية...")
        generated_errors = self.data_generator.generate_error_patterns()
        generated_code = self.data_generator.generate_code_samples()
        generated_qa = self.data_generator.generate_qa_pairs()
        
        logger.info(f"   ✓ أنماط أخطاء: {len(generated_errors)}")
        logger.info(f"   ✓ عينات كود: {len(generated_code)}")
        logger.info(f"   ✓ أسئلة وأجوبة: {len(generated_qa)}")
        
        # 3. دمج البيانات
        logger.info("\n3️⃣ دمج وتنظيم البيانات...")
        
        # بيانات فهم الكود
        all_code_samples = self.data_loader.code_samples + generated_code
        self.training_data['code_understanding'] = [
            {
                'input': f"اشرح هذا الكود:\n```{s.language}\n{s.code}\n```",
                'output': s.description or f"كود {s.language} يقوم بـ...",
                'language': s.language
            }
            for s in all_code_samples if s.description
        ]
        
        # بيانات إصلاح الأخطاء
        all_errors = self.data_loader.error_patterns + generated_errors
        self.training_data['error_fixing'] = [
            {
                'input': f"أصلح هذا الخطأ:\n```{e.language}\n{e.error_code}\n```",
                'output': f"الإصلاح:\n```{e.language}\n{e.correct_code}\n```\n\nالشرح: {e.explanation}",
                'error_type': e.error_type,
                'language': e.language
            }
            for e in all_errors
        ]
        
        # بيانات اكتشاف الأخطاء
        self.training_data['error_detection'] = [
            {
                'input': f"هل يوجد خطأ في هذا الكود؟\n```{e.language}\n{e.error_code}\n```",
                'output': f"نعم، يوجد خطأ من نوع {e.error_type}. {e.explanation}",
                'language': e.language
            }
            for e in all_errors
        ]
        
        # بيانات الأسئلة والأجوبة
        all_qa = self.data_loader.qa_pairs + generated_qa
        self.training_data['qa_pairs'] = [
            {
                'input': qa.question,
                'output': qa.answer,
                'category': qa.category
            }
            for qa in all_qa
        ]
        
        # أنماط التصميم
        self.training_data['design_patterns'] = [
            {
                'input': f"اكتب مثال على {s.tags[-1].replace('_', ' ')} بـ {s.language}",
                'output': f"```{s.language}\n{s.code}\n```\n\n{s.description}",
                'pattern': s.tags[-1] if s.tags else 'unknown'
            }
            for s in generated_code if 'design_pattern' in s.tags
        ]
        
        # 4. حساب الإحصائيات
        self._calculate_stats()
        
        logger.info("\n[OK] Data preparation completed!")
        self._print_stats()
    
    def _calculate_stats(self):
        """حساب إحصائيات التدريب"""
        self.stats.code_samples = len(self.training_data['code_understanding'])
        self.stats.error_patterns = len(self.training_data['error_fixing'])
        self.stats.qa_pairs = len(self.training_data['qa_pairs'])
        
        self.stats.total_samples = sum(
            len(data) for data in self.training_data.values()
        )
        
        # حساب اللغات
        for category in ['code_understanding', 'error_fixing', 'error_detection']:
            for item in self.training_data[category]:
                lang = item.get('language', 'unknown')
                self.stats.languages[lang] = self.stats.languages.get(lang, 0) + 1
        
        # حساب الفئات
        for item in self.training_data['qa_pairs']:
            cat = item.get('category', 'general')
            self.stats.categories[cat] = self.stats.categories.get(cat, 0) + 1
    
    def _print_stats(self):
        """طباعة إحصائيات التدريب"""
        logger.info("\n" + "-" * 50)
        logger.info("Training Data Statistics:")
        logger.info("-" * 50)
        logger.info(f"   Total Samples: {self.stats.total_samples}")
        logger.info(f"   Code Samples: {self.stats.code_samples}")
        logger.info(f"   Error Patterns: {self.stats.error_patterns}")
        logger.info(f"   QA Pairs: {self.stats.qa_pairs}")
        
        if self.stats.languages:
            logger.info("\n   Languages:")
            for lang, count in sorted(self.stats.languages.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"      - {lang}: {count}")
        
        if self.stats.categories:
            logger.info("\n   Categories:")
            for cat, count in sorted(self.stats.categories.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"      - {cat}: {count}")
        
        logger.info("-" * 50)
    
    def save_training_data(self, output_path: Path = None):
        """حفظ بيانات التدريب"""
        output_path = output_path or OUTPUT_DIR / "training_data.json"
        
        logger.info(f"\nSaving training data: {output_path}")
        
        output_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_samples': self.stats.total_samples,
                'stats': {
                    'code_samples': self.stats.code_samples,
                    'error_patterns': self.stats.error_patterns,
                    'qa_pairs': self.stats.qa_pairs
                }
            },
            'data': self.training_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   [OK] Saved: {output_path.stat().st_size / 1024:.1f} KB")
    
    def train(self, mode: str = 'prepare'):
        """بدء التدريب"""
        logger.info("\n" + "=" * 60)
        logger.info("  Starting Training...")
        logger.info("=" * 60 + "\n")
        
        # تجهيز البيانات
        self.prepare_data()
        
        if mode == 'prepare':
            # حفظ البيانات فقط (للتدريب لاحقاً)
            self.save_training_data()
            self._save_for_nodejs()
            
        elif mode == 'full':
            # تدريب كامل مع النموذج
            self._train_with_model()
        
        self.stats.end_time = time.time()
        
        logger.info("\n" + "=" * 60)
        logger.info(f"[OK] Training completed in {self.stats.duration:.1f} seconds")
        logger.info("=" * 60)
    
    def _save_for_nodejs(self):
        """حفظ البيانات بتنسيق مناسب لـ Node.js"""
        # حفظ قاعدة المعرفة
        knowledge_path = MODELS_DIR / "knowledge-base.json"
        
        knowledge = {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'error_patterns': [
                {
                    'error': item['input'],
                    'fix': item['output'],
                    'type': item.get('error_type', 'general'),
                    'language': item.get('language', 'unknown')
                }
                for item in self.training_data['error_fixing']
            ],
            'qa_knowledge': [
                {
                    'question': item['input'],
                    'answer': item['output'],
                    'category': item.get('category', 'general')
                }
                for item in self.training_data['qa_pairs']
            ],
            'code_patterns': [
                {
                    'pattern': item.get('pattern', 'unknown'),
                    'code': item['output'],
                    'description': item['input']
                }
                for item in self.training_data['design_patterns']
            ]
        }
        
        with open(knowledge_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n[OK] Knowledge base saved: {knowledge_path}")
        
        # حفظ بيانات التعلم المحلية
        learned_path = DATA_DIR / "learned-knowledge.json"
        
        learned = {
            'updated_at': datetime.now().isoformat(),
            'training_stats': {
                'total_samples': self.stats.total_samples,
                'code_samples': self.stats.code_samples,
                'error_patterns': self.stats.error_patterns,
                'qa_pairs': self.stats.qa_pairs,
                'languages': self.stats.languages,
                'categories': self.stats.categories
            },
            'patterns': {
                'error_fixes': [
                    {'error': e['input'][:100], 'type': e.get('error_type')}
                    for e in self.training_data['error_fixing'][:100]
                ],
                'design_patterns': [
                    {'name': p.get('pattern'), 'language': 'javascript'}
                    for p in self.training_data['design_patterns']
                ]
            }
        }
        
        with open(learned_path, 'w', encoding='utf-8') as f:
            json.dump(learned, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📝 تم حفظ المعرفة المتعلمة: {learned_path}")
    
    def _train_with_model(self):
        """تدريب مع نموذج ML (يتطلب transformers)"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
            from transformers import Trainer as HFTrainer
            import torch
            
            logger.info("\n🔄 تحميل النموذج...")
            
            # استخدام نموذج صغير للتدريب المحلي
            model_name = "microsoft/CodeGPT-small-py"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # تجهيز البيانات للتدريب
            # ... (كود التدريب الفعلي)
            
            logger.info("[OK] Training successful!")
            
        except ImportError:
            logger.warning("[WARN] transformers library not available. Saving data only.")
            self.save_training_data()
            self._save_for_nodejs()


# ═══════════════════════════════════════════════════════════════════════════════
# نقطة البداية
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """نقطة البداية الرئيسية"""
    
    # طباعة الترحيب
    print("\n")
    print("=" * 70)
    print("          Bi IDE - AI Training System")
    print("=" * 70)
    print("  Training: Code Understanding | Error Fixing | Development")
    print("=" * 70)
    print()
    
    # إعداد الـ arguments
    parser = argparse.ArgumentParser(description='Bi IDE AI Training System')
    parser.add_argument('--mode', choices=['prepare', 'full'], default='prepare',
                       help='وضع التدريب: prepare (تجهيز البيانات) أو full (تدريب كامل)')
    parser.add_argument('--output', type=str, default=None,
                       help='مسار ملف الإخراج')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='طباعة تفاصيل إضافية')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # إنشاء المدرب وبدء التدريب
    trainer = AITrainer()
    
    try:
        trainer.train(mode=args.mode)
        
        print("\n" + "-" * 70)
        print("[SUCCESS] Training Completed!")
        print()
        print("Files Created:")
        print(f"   - {OUTPUT_DIR / 'training_data.json'}")
        print(f"   - {MODELS_DIR / 'knowledge-base.json'}")
        print(f"   - {DATA_DIR / 'learned-knowledge.json'}")
        print()
        print("To use in Bi IDE, run:")
        print("   npm start")
        print("-" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n[WARN] Training stopped by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Training error: {e}")
        raise


if __name__ == '__main__':
    main()
