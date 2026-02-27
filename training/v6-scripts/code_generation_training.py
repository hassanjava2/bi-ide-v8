#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Bi IDE - التدريب على توليد وإكمال الكود                           ║
║                   Code Generation & Completion Training                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  هذا السكريبت يولّد بيانات تدريب لـ:                                        ║
║    • إكمال الكود التلقائي                                                    ║
║    • توليد الدوال والكلاسات                                                  ║
║    • تحويل التعليقات إلى كود                                                 ║
║    • إنشاء القوالب البرمجية                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import re

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
# قوالب توليد الكود
# ═══════════════════════════════════════════════════════════════════════════════

CODE_TEMPLATES = {
    'javascript': {
        'functions': [
            {
                'description': 'دالة لجلب البيانات من API',
                'template': '''async function fetchData(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Fetch error:', error);
        throw error;
    }
}'''
            },
            {
                'description': 'دالة للتحقق من صحة البريد الإلكتروني',
                'template': '''function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}'''
            },
            {
                'description': 'دالة لتنسيق التاريخ',
                'template': '''function formatDate(date, locale = 'ar-SA') {
    return new Intl.DateTimeFormat(locale, {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    }).format(new Date(date));
}'''
            },
            {
                'description': 'دالة Debounce لتأخير التنفيذ',
                'template': '''function debounce(func, wait = 300) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            func.apply(this, args);
        }, wait);
    };
}'''
            },
            {
                'description': 'دالة Throttle للتحكم في معدل التنفيذ',
                'template': '''function throttle(func, limit = 300) {
    let inThrottle;
    return function (...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}'''
            },
            {
                'description': 'دالة Deep Clone لنسخ الكائنات',
                'template': '''function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }
    
    if (Array.isArray(obj)) {
        return obj.map(item => deepClone(item));
    }
    
    return Object.fromEntries(
        Object.entries(obj).map(([key, value]) => [key, deepClone(value)])
    );
}'''
            },
            {
                'description': 'دالة لتجميع العناصر حسب خاصية',
                'template': '''function groupBy(array, key) {
    return array.reduce((result, item) => {
        const groupKey = typeof key === 'function' ? key(item) : item[key];
        (result[groupKey] = result[groupKey] || []).push(item);
        return result;
    }, {});
}'''
            },
            {
                'description': 'دالة لإزالة التكرارات من مصفوفة',
                'template': '''function uniqueBy(array, key) {
    const seen = new Set();
    return array.filter(item => {
        const value = typeof key === 'function' ? key(item) : item[key];
        if (seen.has(value)) return false;
        seen.add(value);
        return true;
    });
}'''
            }
        ],
        'classes': [
            {
                'description': 'كلاس EventEmitter للأحداث',
                'template': '''class EventEmitter {
    constructor() {
        this.events = new Map();
    }
    
    on(event, callback) {
        if (!this.events.has(event)) {
            this.events.set(event, new Set());
        }
        this.events.get(event).add(callback);
        return () => this.off(event, callback);
    }
    
    off(event, callback) {
        if (this.events.has(event)) {
            this.events.get(event).delete(callback);
        }
    }
    
    emit(event, ...args) {
        if (this.events.has(event)) {
            this.events.get(event).forEach(cb => cb(...args));
        }
    }
    
    once(event, callback) {
        const wrapper = (...args) => {
            callback(...args);
            this.off(event, wrapper);
        };
        return this.on(event, wrapper);
    }
}'''
            },
            {
                'description': 'كلاس State Manager بسيط',
                'template': '''class StateManager {
    constructor(initialState = {}) {
        this.state = initialState;
        this.listeners = new Set();
    }
    
    getState() {
        return { ...this.state };
    }
    
    setState(updates) {
        this.state = { ...this.state, ...updates };
        this.notify();
    }
    
    subscribe(listener) {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    }
    
    notify() {
        this.listeners.forEach(listener => listener(this.state));
    }
}'''
            },
            {
                'description': 'كلاس LocalStorage Wrapper',
                'template': '''class Storage {
    constructor(prefix = 'app_') {
        this.prefix = prefix;
    }
    
    _key(key) {
        return `${this.prefix}${key}`;
    }
    
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(this._key(key));
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            return defaultValue;
        }
    }
    
    set(key, value) {
        try {
            localStorage.setItem(this._key(key), JSON.stringify(value));
            return true;
        } catch (e) {
            return false;
        }
    }
    
    remove(key) {
        localStorage.removeItem(this._key(key));
    }
    
    clear() {
        Object.keys(localStorage)
            .filter(k => k.startsWith(this.prefix))
            .forEach(k => localStorage.removeItem(k));
    }
}'''
            },
            {
                'description': 'كلاس HTTP Client',
                'template': '''class HttpClient {
    constructor(baseURL = '', defaultHeaders = {}) {
        this.baseURL = baseURL;
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            ...defaultHeaders
        };
    }
    
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            ...options,
            headers: { ...this.defaultHeaders, ...options.headers }
        };
        
        if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }
        
        const response = await fetch(url, config);
        
        if (!response.ok) {
            const error = new Error(`HTTP ${response.status}`);
            error.response = response;
            throw error;
        }
        
        return response.json();
    }
    
    get(endpoint, options = {}) {
        return this.request(endpoint, { ...options, method: 'GET' });
    }
    
    post(endpoint, data, options = {}) {
        return this.request(endpoint, { ...options, method: 'POST', body: data });
    }
    
    put(endpoint, data, options = {}) {
        return this.request(endpoint, { ...options, method: 'PUT', body: data });
    }
    
    delete(endpoint, options = {}) {
        return this.request(endpoint, { ...options, method: 'DELETE' });
    }
}'''
            }
        ],
        'react_components': [
            {
                'description': 'مكون Button قابل لإعادة الاستخدام',
                'template': '''function Button({ 
    children, 
    variant = 'primary', 
    size = 'medium',
    disabled = false,
    loading = false,
    onClick,
    ...props 
}) {
    const baseClasses = 'btn';
    const variantClasses = {
        primary: 'btn-primary',
        secondary: 'btn-secondary',
        danger: 'btn-danger'
    };
    const sizeClasses = {
        small: 'btn-sm',
        medium: 'btn-md',
        large: 'btn-lg'
    };
    
    return (
        <button
            className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]}`}
            disabled={disabled || loading}
            onClick={onClick}
            {...props}
        >
            {loading ? <Spinner size="small" /> : children}
        </button>
    );
}'''
            },
            {
                'description': 'مكون Modal',
                'template': '''function Modal({ isOpen, onClose, title, children }) {
    if (!isOpen) return null;
    
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2>{title}</h2>
                    <button className="modal-close" onClick={onClose}>
                        &times;
                    </button>
                </div>
                <div className="modal-body">
                    {children}
                </div>
            </div>
        </div>
    );
}'''
            },
            {
                'description': 'Custom Hook useLocalStorage',
                'template': '''function useLocalStorage(key, initialValue) {
    const [storedValue, setStoredValue] = useState(() => {
        try {
            const item = window.localStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        } catch (error) {
            return initialValue;
        }
    });
    
    const setValue = (value) => {
        try {
            const valueToStore = value instanceof Function 
                ? value(storedValue) 
                : value;
            setStoredValue(valueToStore);
            window.localStorage.setItem(key, JSON.stringify(valueToStore));
        } catch (error) {
            console.error('Error saving to localStorage:', error);
        }
    };
    
    return [storedValue, setValue];
}'''
            },
            {
                'description': 'Custom Hook useFetch',
                'template': '''function useFetch(url, options = {}) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    useEffect(() => {
        const controller = new AbortController();
        
        const fetchData = async () => {
            try {
                setLoading(true);
                const response = await fetch(url, {
                    ...options,
                    signal: controller.signal
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                setData(result);
                setError(null);
            } catch (err) {
                if (err.name !== 'AbortError') {
                    setError(err.message);
                }
            } finally {
                setLoading(false);
            }
        };
        
        fetchData();
        
        return () => controller.abort();
    }, [url]);
    
    return { data, loading, error };
}'''
            }
        ]
    },
    'python': {
        'functions': [
            {
                'description': 'دالة Decorator للتوقيت',
                'template': '''from functools import wraps
import time

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper'''
            },
            {
                'description': 'دالة Decorator للتخزين المؤقت',
                'template': '''from functools import wraps

def memoize(func):
    cache = {}
    
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    wrapper.cache = cache
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper'''
            },
            {
                'description': 'دالة لقراءة ملف JSON بأمان',
                'template': '''import json
from pathlib import Path
from typing import Any, Optional

def read_json(filepath: str, default: Optional[Any] = None) -> Any:
    """قراءة ملف JSON بأمان مع قيمة افتراضية"""
    try:
        path = Path(filepath)
        if not path.exists():
            return default
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filepath}: {e}")
        return default'''
            },
            {
                'description': 'دالة لكتابة ملف JSON بأمان',
                'template': '''import json
from pathlib import Path
from typing import Any

def write_json(filepath: str, data: Any, indent: int = 2) -> bool:
    """كتابة بيانات إلى ملف JSON"""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return True
    except IOError as e:
        print(f"Error writing to {filepath}: {e}")
        return False'''
            },
            {
                'description': 'دالة Retry للمحاولة المتكررة',
                'template': '''from functools import wraps
import time
from typing import Type, Tuple

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Decorator للمحاولة المتكررة عند الفشل"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator'''
            }
        ],
        'classes': [
            {
                'description': 'كلاس Singleton',
                'template': '''class Singleton:
    """كلاس Singleton - نسخة واحدة فقط"""
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance'''
            },
            {
                'description': 'كلاس Configuration Manager',
                'template': '''from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
from pathlib import Path

@dataclass
class Config:
    """إدارة إعدادات التطبيق"""
    _data: Dict[str, Any] = field(default_factory=dict)
    _filepath: Optional[str] = None
    
    def load(self, filepath: str) -> 'Config':
        """تحميل الإعدادات من ملف"""
        self._filepath = filepath
        path = Path(filepath)
        
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        
        return self
    
    def save(self) -> bool:
        """حفظ الإعدادات"""
        if not self._filepath:
            return False
        
        with open(self._filepath, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """الحصول على قيمة"""
        keys = key.split('.')
        value = self._data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """تعيين قيمة"""
        keys = key.split('.')
        data = self._data
        
        for k in keys[:-1]:
            data = data.setdefault(k, {})
        
        data[keys[-1]] = value'''
            },
            {
                'description': 'كلاس Logger مخصص',
                'template': '''import logging
from pathlib import Path
from datetime import datetime

class Logger:
    """نظام تسجيل مخصص"""
    
    def __init__(self, name: str, log_dir: str = 'logs'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # إنشاء مجلد السجلات
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # ملف السجل
        log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Handler للملف
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Handler للطرفية
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # التنسيق
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, msg): self.logger.debug(msg)
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def critical(self, msg): self.logger.critical(msg)'''
            }
        ]
    },
    'sql': {
        'queries': [
            {
                'description': 'استعلام مع Pagination',
                'template': '''SELECT 
    id,
    name,
    email,
    created_at
FROM users
WHERE is_active = true
ORDER BY created_at DESC
LIMIT :limit
OFFSET :offset;'''
            },
            {
                'description': 'استعلام مع تجميع وإحصائيات',
                'template': '''SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_orders,
    SUM(total_amount) as revenue,
    AVG(total_amount) as avg_order_value
FROM orders
WHERE created_at >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
GROUP BY DATE(created_at)
ORDER BY date DESC;'''
            },
            {
                'description': 'إنشاء جدول مستخدمين',
                'template': '''CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    email_verified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);'''
            }
        ]
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# بيانات إكمال الكود
# ═══════════════════════════════════════════════════════════════════════════════

CODE_COMPLETIONS = {
    'javascript': [
        {
            'prefix': 'async function fetchUsers() {\n    const response = await fetch',
            'completion': "('/api/users');\n    if (!response.ok) throw new Error('Failed to fetch');\n    return await response.json();\n}"
        },
        {
            'prefix': 'const handleSubmit = (e) => {\n    e.preventDefault',
            'completion': "();\n    const formData = new FormData(e.target);\n    // معالجة البيانات\n}"
        },
        {
            'prefix': 'useEffect(() => {\n    const controller = new AbortController',
            'completion': "();\n    \n    async function fetchData() {\n        try {\n            const response = await fetch(url, { signal: controller.signal });\n            const data = await response.json();\n            setData(data);\n        } catch (error) {\n            if (error.name !== 'AbortError') {\n                setError(error);\n            }\n        }\n    }\n    \n    fetchData();\n    return () => controller.abort();\n}, [url]);"
        },
        {
            'prefix': 'const sortedItems = items.sort((a, b)',
            'completion': ' => a.name.localeCompare(b.name));'
        },
        {
            'prefix': 'const filteredUsers = users.filter(user',
            'completion': ' => user.isActive && user.role === "admin");'
        }
    ],
    'python': [
        {
            'prefix': 'def read_file(filepath):\n    with open(filepath',
            'completion': ", 'r', encoding='utf-8') as f:\n        return f.read()"
        },
        {
            'prefix': 'class User:\n    def __init__(self',
            'completion': ", name: str, email: str):\n        self.name = name\n        self.email = email\n        self.created_at = datetime.now()"
        },
        {
            'prefix': 'async def fetch_data(url: str)',
            'completion': " -> dict:\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            response.raise_for_status()\n            return await response.json()"
        },
        {
            'prefix': 'items = [item for item in',
            'completion': ' data if item.get("active") and item["price"] > 0]'
        }
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
# بيانات تحويل التعليقات إلى كود
# ═══════════════════════════════════════════════════════════════════════════════

COMMENT_TO_CODE = {
    'javascript': [
        {
            'comment': '// دالة لحساب المجموع',
            'code': '''function sum(numbers) {
    return numbers.reduce((acc, num) => acc + num, 0);
}'''
        },
        {
            'comment': '// فرز المصفوفة تصاعدياً',
            'code': 'const sorted = array.sort((a, b) => a - b);'
        },
        {
            'comment': '// التحقق من أن المستخدم مسجل دخوله',
            'code': '''function isAuthenticated() {
    const token = localStorage.getItem('authToken');
    return token && !isTokenExpired(token);
}'''
        },
        {
            'comment': '// إنشاء رقم عشوائي بين min و max',
            'code': '''function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}'''
        },
        {
            'comment': '// تحويل النص إلى slug',
            'code': '''function toSlug(text) {
    return text
        .toLowerCase()
        .trim()
        .replace(/[^\\w\\s-]/g, '')
        .replace(/[\\s_-]+/g, '-')
        .replace(/^-+|-+$/g, '');
}'''
        }
    ],
    'python': [
        {
            'comment': '# دالة لحساب العاملي (factorial)',
            'code': '''def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)'''
        },
        {
            'comment': '# فرز قاموس حسب القيم',
            'code': 'sorted_dict = dict(sorted(my_dict.items(), key=lambda x: x[1]))'
        },
        {
            'comment': '# التحقق من صحة رقم الهاتف',
            'code': '''import re

def is_valid_phone(phone: str) -> bool:
    pattern = r'^\\+?[0-9]{10,14}$'
    return bool(re.match(pattern, phone))'''
        },
        {
            'comment': '# تحويل قائمة إلى chunks',
            'code': '''def chunk_list(lst: list, size: int) -> list:
    return [lst[i:i + size] for i in range(0, len(lst), size)]'''
        }
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
# مولّد بيانات التدريب
# ═══════════════════════════════════════════════════════════════════════════════

class CodeGenerationTrainer:
    """توليد بيانات تدريب لتوليد الكود"""
    
    def __init__(self):
        self.templates = CODE_TEMPLATES
        self.completions = CODE_COMPLETIONS
        self.comment_to_code = COMMENT_TO_CODE
    
    def generate_all(self) -> Dict[str, List]:
        """توليد جميع بيانات التدريب"""
        return {
            'code_generation': self._generate_code_generation_data(),
            'code_completion': self._generate_completion_data(),
            'comment_to_code': self._generate_comment_to_code_data(),
            'template_usage': self._generate_template_usage_data()
        }
    
    def _generate_code_generation_data(self) -> List[Dict]:
        """توليد بيانات إنشاء الكود"""
        data = []
        
        for lang, categories in self.templates.items():
            for category, templates in categories.items():
                for template in templates:
                    data.append({
                        'instruction': f'اكتب {category[:-1] if category.endswith("s") else category} بـ {lang}:',
                        'input': template['description'],
                        'output': template['template'],
                        'metadata': {
                            'language': lang,
                            'category': category,
                            'type': 'code_generation'
                        }
                    })
        
        return data
    
    def _generate_completion_data(self) -> List[Dict]:
        """توليد بيانات إكمال الكود"""
        data = []
        
        for lang, completions in self.completions.items():
            for comp in completions:
                data.append({
                    'instruction': f'أكمل هذا الكود ({lang}):',
                    'input': comp['prefix'],
                    'output': comp['prefix'] + comp['completion'],
                    'metadata': {
                        'language': lang,
                        'type': 'code_completion'
                    }
                })
        
        return data
    
    def _generate_comment_to_code_data(self) -> List[Dict]:
        """توليد بيانات تحويل التعليقات لكود"""
        data = []
        
        for lang, examples in self.comment_to_code.items():
            for ex in examples:
                data.append({
                    'instruction': f'حوّل هذا التعليق إلى كود ({lang}):',
                    'input': ex['comment'],
                    'output': ex['code'],
                    'metadata': {
                        'language': lang,
                        'type': 'comment_to_code'
                    }
                })
        
        return data
    
    def _generate_template_usage_data(self) -> List[Dict]:
        """توليد بيانات استخدام القوالب"""
        data = []
        
        # أمثلة على طلبات المستخدم والقوالب المناسبة
        user_requests = [
            ('أريد دالة لجلب بيانات من API', 'javascript', 'functions', 0),
            ('اكتب كلاس للتعامل مع الأحداث', 'javascript', 'classes', 0),
            ('أحتاج Hook للتخزين المحلي', 'javascript', 'react_components', 2),
            ('دالة Python للتوقيت', 'python', 'functions', 0),
            ('كلاس إعدادات', 'python', 'classes', 1),
        ]
        
        for request, lang, category, index in user_requests:
            if lang in self.templates and category in self.templates[lang]:
                template = self.templates[lang][category][index]
                data.append({
                    'instruction': 'اكتب كود حسب الطلب:',
                    'input': request,
                    'output': template['template'],
                    'metadata': {
                        'language': lang,
                        'category': category,
                        'type': 'template_usage'
                    }
                })
        
        return data
    
    def save_training_data(self, output_dir: Path):
        """حفظ بيانات التدريب"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_data = self.generate_all()
        
        # حفظ كل نوع
        total = 0
        for data_type, data in all_data.items():
            filepath = output_dir / f"{data_type}_training.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ {filepath.name}: {len(data)} عينة")
            total += len(data)
        
        # حفظ مجمع
        combined = []
        for data in all_data.values():
            combined.extend(data)
        
        combined_path = output_dir / "code_generation_combined.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        
        print(f"\n📦 المجموع: {total} عينة")
        
        return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# نقطة البداية
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n")
    print("╔" + "═" * 70 + "╗")
    print("║" + " " * 12 + "💻 Bi IDE - التدريب على توليد الكود 💻" + " " * 12 + "║")
    print("╚" + "═" * 70 + "╝")
    print()
    
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / "training" / "output"
    
    print("📂 مجلد الإخراج:", OUTPUT_DIR)
    print()
    
    print("=" * 60)
    print("🔄 توليد بيانات التدريب...")
    print("=" * 60)
    print()
    
    trainer = CodeGenerationTrainer()
    trainer.save_training_data(OUTPUT_DIR)
    
    print()
    print("=" * 60)
    print("✅ اكتمل التدريب على توليد الكود!")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
