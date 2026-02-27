#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              Bi IDE - التدريب المتقدم على إصلاح الأخطاء                     ║
║                    Advanced Error Fixing Training                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  هذا السكريبت يركز على:                                                     ║
║    • اكتشاف الأخطاء الشائعة في الكود                                        ║
║    • تعلم أنماط الإصلاح                                                      ║
║    • فهم رسائل الخطأ                                                         ║
║    • تقديم اقتراحات إصلاح ذكية                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time

# إعداد الترميز
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════════════════════
# قاعدة بيانات الأخطاء الشاملة
# ═══════════════════════════════════════════════════════════════════════════════

ERROR_DATABASE = {
    'javascript': {
        'syntax_errors': [
            {
                'pattern': r'Unexpected token',
                'causes': [
                    'قوس مفقود أو زائد',
                    'فاصلة منقوطة مفقودة',
                    'علامة اقتباس غير مغلقة'
                ],
                'examples': [
                    {
                        'error': 'const arr = [1, 2, 3',
                        'fix': 'const arr = [1, 2, 3];',
                        'explanation': 'القوس المربع غير مغلق'
                    },
                    {
                        'error': 'function test() { return "hello }',
                        'fix': 'function test() { return "hello"; }',
                        'explanation': 'علامة الاقتباس غير مغلقة'
                    }
                ]
            },
            {
                'pattern': r'Unexpected end of input',
                'causes': [
                    'قوس معقوص مفقود }',
                    'نهاية غير متوقعة للملف'
                ],
                'examples': [
                    {
                        'error': 'if (condition) {\n  doSomething();',
                        'fix': 'if (condition) {\n  doSomething();\n}',
                        'explanation': 'القوس المعقوص } مفقود'
                    }
                ]
            }
        ],
        'reference_errors': [
            {
                'pattern': r'(\w+) is not defined',
                'causes': [
                    'المتغير غير معرف',
                    'خطأ إملائي في اسم المتغير',
                    'المتغير خارج النطاق (scope)'
                ],
                'examples': [
                    {
                        'error': 'console.log(userName);',
                        'fix': 'const userName = "Ahmed";\nconsole.log(userName);',
                        'explanation': 'يجب تعريف المتغير قبل استخدامه'
                    },
                    {
                        'error': 'funtion test() {}',
                        'fix': 'function test() {}',
                        'explanation': 'خطأ إملائي: funtion بدلاً من function'
                    }
                ]
            },
            {
                'pattern': r'Cannot access.*before initialization',
                'causes': [
                    'استخدام let/const قبل التعريف (temporal dead zone)'
                ],
                'examples': [
                    {
                        'error': 'console.log(x);\nlet x = 5;',
                        'fix': 'let x = 5;\nconsole.log(x);',
                        'explanation': 'let و const لا يمكن الوصول إليهما قبل التعريف'
                    }
                ]
            }
        ],
        'type_errors': [
            {
                'pattern': r'Cannot read propert.*of undefined',
                'causes': [
                    'الكائن غير معرف',
                    'المسار للخاصية خاطئ',
                    'البيانات لم تُحمّل بعد'
                ],
                'examples': [
                    {
                        'error': 'const user = undefined;\nconsole.log(user.name);',
                        'fix': 'const user = { name: "Ahmed" };\nconsole.log(user?.name);',
                        'explanation': 'استخدم optional chaining (?.) للتحقق من وجود الكائن'
                    },
                    {
                        'error': 'const data = response.data.users[0].name;',
                        'fix': 'const data = response?.data?.users?.[0]?.name ?? "default";',
                        'explanation': 'تأكد من وجود كل مستوى من البيانات'
                    }
                ]
            },
            {
                'pattern': r'is not a function',
                'causes': [
                    'استدعاء شيء ليس دالة',
                    'الدالة غير موجودة',
                    'this context خاطئ'
                ],
                'examples': [
                    {
                        'error': 'const num = 5;\nnum.map(x => x * 2);',
                        'fix': 'const arr = [5];\narr.map(x => x * 2);',
                        'explanation': 'map() تعمل فقط مع المصفوفات'
                    },
                    {
                        'error': 'const obj = { onClick: "click" };\nobj.onClick();',
                        'fix': 'const obj = { onClick: () => console.log("clicked") };\nobj.onClick();',
                        'explanation': 'onClick يجب أن تكون دالة وليس نص'
                    }
                ]
            },
            {
                'pattern': r'Cannot set propert.*of undefined',
                'causes': [
                    'محاولة تعيين خاصية لكائن غير موجود'
                ],
                'examples': [
                    {
                        'error': 'const obj = {};\nobj.nested.value = 5;',
                        'fix': 'const obj = { nested: {} };\nobj.nested.value = 5;',
                        'explanation': 'يجب إنشاء الكائن المتداخل أولاً'
                    }
                ]
            }
        ],
        'async_errors': [
            {
                'pattern': r'await is only valid in async function',
                'causes': [
                    'استخدام await خارج دالة async'
                ],
                'examples': [
                    {
                        'error': 'function getData() {\n  const data = await fetch(url);\n  return data;\n}',
                        'fix': 'async function getData() {\n  const data = await fetch(url);\n  return data;\n}',
                        'explanation': 'أضف async قبل function'
                    }
                ]
            },
            {
                'pattern': r'Promise.*then.*is not a function',
                'causes': [
                    'القيمة ليست Promise',
                    'نسيان return في سلسلة Promises'
                ],
                'examples': [
                    {
                        'error': 'getData()\n  .then(data => processData(data))\n  .then(result => result.json())',
                        'fix': 'getData()\n  .then(data => processData(data))\n  .then(result => result)',
                        'explanation': 'processData قد لا تُرجع Promise'
                    }
                ]
            }
        ],
        'common_mistakes': [
            {
                'pattern': 'assignment_in_condition',
                'description': 'استخدام = بدلاً من == أو ===',
                'examples': [
                    {
                        'error': 'if (x = 5) { }',
                        'fix': 'if (x === 5) { }',
                        'explanation': '= للتعيين، === للمقارنة'
                    }
                ]
            },
            {
                'pattern': 'missing_return',
                'description': 'نسيان return في arrow function',
                'examples': [
                    {
                        'error': 'const double = (x) => { x * 2; };',
                        'fix': 'const double = (x) => x * 2;\n// أو\nconst double = (x) => { return x * 2; };',
                        'explanation': 'arrow function مع {} تحتاج return صريح'
                    }
                ]
            },
            {
                'pattern': 'array_mutation',
                'description': 'تعديل المصفوفة أثناء التكرار',
                'examples': [
                    {
                        'error': 'const arr = [1, 2, 3];\narr.forEach((item, i) => {\n  if (item === 2) arr.splice(i, 1);\n});',
                        'fix': 'const arr = [1, 2, 3];\nconst filtered = arr.filter(item => item !== 2);',
                        'explanation': 'لا تعدل المصفوفة أثناء التكرار، استخدم filter بدلاً'
                    }
                ]
            },
            {
                'pattern': 'floating_point',
                'description': 'مشاكل الأرقام العشرية',
                'examples': [
                    {
                        'error': 'console.log(0.1 + 0.2 === 0.3); // false!',
                        'fix': 'console.log(Math.abs((0.1 + 0.2) - 0.3) < 0.0001); // true',
                        'explanation': 'الأرقام العشرية في JavaScript ليست دقيقة 100%'
                    }
                ]
            }
        ]
    },
    'python': {
        'syntax_errors': [
            {
                'pattern': r'IndentationError',
                'causes': [
                    'مسافات بادئة غير صحيحة',
                    'خلط بين tabs و spaces'
                ],
                'examples': [
                    {
                        'error': 'def test():\nprint("hello")',
                        'fix': 'def test():\n    print("hello")',
                        'explanation': 'Python يتطلب مسافات بادئة صحيحة'
                    }
                ]
            },
            {
                'pattern': r'SyntaxError: invalid syntax',
                'causes': [
                    'نسيان النقطتين :',
                    'أقواس غير متطابقة',
                    'كلمات محجوزة'
                ],
                'examples': [
                    {
                        'error': 'if x == 5\n    print("yes")',
                        'fix': 'if x == 5:\n    print("yes")',
                        'explanation': 'if يحتاج : في نهاية السطر'
                    },
                    {
                        'error': 'class = "math"',
                        'fix': 'course = "math"',
                        'explanation': 'class كلمة محجوزة في Python'
                    }
                ]
            }
        ],
        'type_errors': [
            {
                'pattern': r"unsupported operand type",
                'causes': [
                    'عملية بين أنواع غير متوافقة'
                ],
                'examples': [
                    {
                        'error': '"Hello " + 5',
                        'fix': '"Hello " + str(5)',
                        'explanation': 'لا يمكن جمع نص ورقم مباشرة'
                    }
                ]
            },
            {
                'pattern': r"'NoneType' object",
                'causes': [
                    'الدالة تُرجع None',
                    'المتغير None'
                ],
                'examples': [
                    {
                        'error': 'result = some_list.sort()\nprint(result[0])',
                        'fix': 'some_list.sort()\nprint(some_list[0])\n# أو\nresult = sorted(some_list)',
                        'explanation': 'sort() تُعدل القائمة وتُرجع None، استخدم sorted() للحصول على قائمة جديدة'
                    }
                ]
            }
        ],
        'common_mistakes': [
            {
                'pattern': 'mutable_default_argument',
                'description': 'استخدام كائن قابل للتغيير كقيمة افتراضية',
                'examples': [
                    {
                        'error': 'def add_item(item, items=[]):\n    items.append(item)\n    return items',
                        'fix': 'def add_item(item, items=None):\n    if items is None:\n        items = []\n    items.append(item)\n    return items',
                        'explanation': 'القيم الافتراضية تُنشأ مرة واحدة وتُشارك بين الاستدعاءات'
                    }
                ]
            },
            {
                'pattern': 'late_binding_closure',
                'description': 'مشكلة الربط المتأخر في الحلقات',
                'examples': [
                    {
                        'error': 'funcs = [lambda: i for i in range(5)]\nprint([f() for f in funcs])  # [4,4,4,4,4]!',
                        'fix': 'funcs = [lambda i=i: i for i in range(5)]\nprint([f() for f in funcs])  # [0,1,2,3,4]',
                        'explanation': 'استخدم i=i لالتقاط القيمة في وقت الإنشاء'
                    }
                ]
            },
            {
                'pattern': 'modifying_list_while_iterating',
                'description': 'تعديل القائمة أثناء التكرار',
                'examples': [
                    {
                        'error': 'items = [1, 2, 3, 4, 5]\nfor item in items:\n    if item % 2 == 0:\n        items.remove(item)',
                        'fix': 'items = [1, 2, 3, 4, 5]\nitems = [item for item in items if item % 2 != 0]',
                        'explanation': 'تعديل القائمة أثناء التكرار يؤدي لنتائج غير متوقعة'
                    }
                ]
            }
        ]
    },
    'react': {
        'common_errors': [
            {
                'pattern': r'Invalid hook call',
                'causes': [
                    'استدعاء hook خارج component',
                    'استدعاء hook داخل شرط أو حلقة'
                ],
                'examples': [
                    {
                        'error': 'function Component() {\n  if (condition) {\n    const [state, setState] = useState(0);\n  }\n}',
                        'fix': 'function Component() {\n  const [state, setState] = useState(0);\n  if (condition) {\n    // استخدم state هنا\n  }\n}',
                        'explanation': 'Hooks يجب أن تُستدعى في أعلى مستوى من الcomponent'
                    }
                ]
            },
            {
                'pattern': r'Maximum update depth exceeded',
                'causes': [
                    'setState داخل useEffect بدون dependencies صحيحة',
                    'حلقة لا نهائية من التحديثات'
                ],
                'examples': [
                    {
                        'error': 'useEffect(() => {\n  setCount(count + 1);\n});',
                        'fix': 'useEffect(() => {\n  setCount(c => c + 1);\n}, []); // تشغيل مرة واحدة',
                        'explanation': 'أضف dependency array لمنع إعادة التشغيل المستمرة'
                    }
                ]
            },
            {
                'pattern': r'Each child.*should have a unique "key"',
                'causes': [
                    'عناصر القائمة بدون مفتاح فريد'
                ],
                'examples': [
                    {
                        'error': 'items.map(item => <li>{item.name}</li>)',
                        'fix': 'items.map(item => <li key={item.id}>{item.name}</li>)',
                        'explanation': 'كل عنصر في القائمة يحتاج key فريد'
                    }
                ]
            },
            {
                'pattern': r"Cannot update.*component.*while rendering",
                'causes': [
                    'تحديث state أثناء الرسم'
                ],
                'examples': [
                    {
                        'error': 'function Component() {\n  const [count, setCount] = useState(0);\n  setCount(count + 1); // خطأ!\n  return <div>{count}</div>;\n}',
                        'fix': 'function Component() {\n  const [count, setCount] = useState(0);\n  useEffect(() => {\n    setCount(c => c + 1);\n  }, []);\n  return <div>{count}</div>;\n}',
                        'explanation': 'استخدم useEffect لتحديث state'
                    }
                ]
            }
        ]
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# قاعدة بيانات الممارسات الأفضل
# ═══════════════════════════════════════════════════════════════════════════════

BEST_PRACTICES = {
    'javascript': [
        {
            'category': 'Variables',
            'practices': [
                'استخدم const افتراضياً، let عند الحاجة للتغيير',
                'تجنب var لمشاكل الscope',
                'استخدم أسماء واضحة ووصفية',
                'تجنب المتغيرات العامة (global)'
            ]
        },
        {
            'category': 'Functions',
            'practices': [
                'اجعل الدوال صغيرة ومركزة (single responsibility)',
                'استخدم arrow functions للدوال المجهولة',
                'تجنب الدوال المتداخلة العميقة',
                'استخدم default parameters'
            ]
        },
        {
            'category': 'Async',
            'practices': [
                'استخدم async/await بدلاً من .then() chains',
                'دائماً تعامل مع الأخطاء (try/catch)',
                'استخدم Promise.all للعمليات المتوازية',
                'تجنب callback hell'
            ]
        },
        {
            'category': 'Error Handling',
            'practices': [
                'لا تبتلع الأخطاء (empty catch)',
                'أنشئ custom errors للحالات الخاصة',
                'سجل الأخطاء مع context كافي',
                'استخدم error boundaries في React'
            ]
        }
    ],
    'python': [
        {
            'category': 'Code Style',
            'practices': [
                'اتبع PEP 8 للتنسيق',
                'استخدم snake_case للمتغيرات والدوال',
                'استخدم PascalCase للكلاسات',
                'اكتب docstrings للدوال والكلاسات'
            ]
        },
        {
            'category': 'Functions',
            'practices': [
                'استخدم type hints',
                'تجنب الarguments الكثيرة (max 5)',
                'استخدم *args و **kwargs بحكمة',
                'أرجع None صراحة عند الحاجة'
            ]
        },
        {
            'category': 'Data Structures',
            'practices': [
                'استخدم list comprehensions',
                'استخدم generators للبيانات الكبيرة',
                'استخدم dataclasses للstructures',
                'فضّل dict.get() على []'
            ]
        }
    ],
    'general': [
        {
            'category': 'Security',
            'practices': [
                'لا تخزن كلمات المرور كنص عادي',
                'استخدم parameterized queries',
                'تحقق من المدخلات (input validation)',
                'استخدم HTTPS دائماً'
            ]
        },
        {
            'category': 'Performance',
            'practices': [
                'تجنب الحلقات المتداخلة الكثيرة',
                'استخدم caching عند الحاجة',
                'قم بـ lazy loading للبيانات الكبيرة',
                'قس الأداء قبل التحسين'
            ]
        }
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
# محلل الأخطاء
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorAnalyzer:
    """تحليل واكتشاف الأخطاء في الكود"""
    
    def __init__(self):
        self.error_db = ERROR_DATABASE
        self.best_practices = BEST_PRACTICES
    
    def analyze_code(self, code: str, language: str = 'javascript') -> Dict[str, Any]:
        """تحليل الكود واكتشاف الأخطاء المحتملة"""
        results = {
            'language': language,
            'potential_errors': [],
            'suggestions': [],
            'best_practices': [],
            'code_quality': 0
        }
        
        if language not in self.error_db:
            language = 'javascript'  # افتراضي
        
        # فحص أنماط الأخطاء
        for category, patterns in self.error_db[language].items():
            for pattern_info in patterns:
                for example in pattern_info.get('examples', []):
                    if self._code_contains_pattern(code, example.get('error', '')):
                        results['potential_errors'].append({
                            'category': category,
                            'issue': pattern_info.get('description', example.get('explanation', '')),
                            'suggestion': example.get('fix', ''),
                            'explanation': example.get('explanation', '')
                        })
        
        # إضافة أفضل الممارسات
        if language in self.best_practices:
            for practice_category in self.best_practices[language]:
                results['best_practices'].append(practice_category)
        
        # حساب جودة الكود (مبسط)
        results['code_quality'] = self._calculate_quality(code, results['potential_errors'])
        
        return results
    
    def _code_contains_pattern(self, code: str, pattern: str) -> bool:
        """فحص إذا كان الكود يحتوي على نمط معين"""
        if not pattern:
            return False
        
        # تنظيف النمط للمقارنة
        pattern_clean = re.sub(r'\s+', '', pattern.lower())
        code_clean = re.sub(r'\s+', '', code.lower())
        
        return pattern_clean in code_clean
    
    def _calculate_quality(self, code: str, errors: List) -> int:
        """حساب نقاط جودة الكود"""
        base_score = 100
        
        # خصم نقاط للأخطاء
        base_score -= len(errors) * 10
        
        # خصم للكود الطويل جداً بدون تقسيم
        lines = code.split('\n')
        if len(lines) > 100:
            base_score -= 10
        
        # خصم للدوال الطويلة
        # (تبسيط: نفترض وجود دالة طويلة إذا كان هناك أكثر من 50 سطر متتالي بدون تعريف دالة جديد)
        
        return max(0, min(100, base_score))
    
    def get_fix_suggestion(self, error_message: str, language: str = 'javascript') -> Dict[str, Any]:
        """الحصول على اقتراح إصلاح بناءً على رسالة الخطأ"""
        if language not in self.error_db:
            language = 'javascript'
        
        for category, patterns in self.error_db[language].items():
            for pattern_info in patterns:
                pattern = pattern_info.get('pattern', '')
                if pattern and re.search(pattern, error_message, re.IGNORECASE):
                    return {
                        'found': True,
                        'category': category,
                        'causes': pattern_info.get('causes', []),
                        'examples': pattern_info.get('examples', []),
                        'suggestions': [
                            ex.get('fix') for ex in pattern_info.get('examples', [])
                            if ex.get('fix')
                        ]
                    }
        
        return {
            'found': False,
            'message': 'لم يتم العثور على إصلاح مقترح لهذا الخطأ'
        }


# ═══════════════════════════════════════════════════════════════════════════════
# مولّد بيانات التدريب المتقدم
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedTrainingGenerator:
    """توليد بيانات تدريب متقدمة لإصلاح الأخطاء"""
    
    def __init__(self):
        self.error_db = ERROR_DATABASE
        self.best_practices = BEST_PRACTICES
    
    def generate_all(self) -> Dict[str, List]:
        """توليد جميع بيانات التدريب"""
        return {
            'error_detection': self._generate_error_detection_data(),
            'error_fixing': self._generate_error_fixing_data(),
            'code_review': self._generate_code_review_data(),
            'best_practices_qa': self._generate_best_practices_qa()
        }
    
    def _generate_error_detection_data(self) -> List[Dict]:
        """توليد بيانات اكتشاف الأخطاء"""
        data = []
        
        for lang, categories in self.error_db.items():
            for category, patterns in categories.items():
                for pattern_info in patterns:
                    for example in pattern_info.get('examples', []):
                        if example.get('error'):
                            data.append({
                                'instruction': f'هل يوجد خطأ في هذا الكود ({lang})؟',
                                'input': example['error'],
                                'output': f"نعم، يوجد خطأ. {example.get('explanation', '')}",
                                'metadata': {
                                    'language': lang,
                                    'category': category,
                                    'type': 'error_detection'
                                }
                            })
                            
                            # إضافة مثال صحيح أيضاً
                            if example.get('fix'):
                                data.append({
                                    'instruction': f'هل يوجد خطأ في هذا الكود ({lang})؟',
                                    'input': example['fix'],
                                    'output': "لا، هذا الكود صحيح.",
                                    'metadata': {
                                        'language': lang,
                                        'category': category,
                                        'type': 'correct_code'
                                    }
                                })
        
        return data
    
    def _generate_error_fixing_data(self) -> List[Dict]:
        """توليد بيانات إصلاح الأخطاء"""
        data = []
        
        for lang, categories in self.error_db.items():
            for category, patterns in categories.items():
                for pattern_info in patterns:
                    for example in pattern_info.get('examples', []):
                        if example.get('error') and example.get('fix'):
                            data.append({
                                'instruction': f'أصلح الخطأ في هذا الكود ({lang}):',
                                'input': example['error'],
                                'output': f"{example['fix']}\n\n📝 الشرح: {example.get('explanation', '')}",
                                'metadata': {
                                    'language': lang,
                                    'category': category,
                                    'type': 'error_fixing'
                                }
                            })
        
        return data
    
    def _generate_code_review_data(self) -> List[Dict]:
        """توليد بيانات مراجعة الكود"""
        data = []
        
        # أمثلة على مراجعة كود جيد وسيء
        review_examples = [
            {
                'code': '''function getUserData(userId) {
    return fetch('/api/users/' + userId)
        .then(res => res.json())
        .then(data => {
            console.log(data);
            return data;
        });
}''',
                'language': 'javascript',
                'review': '''✅ الإيجابيات:
- دالة واضحة الهدف
- استخدام Promise chain

⚠️ يمكن تحسينه:
- استخدم async/await للوضوح
- أضف معالجة للأخطاء (try/catch)
- استخدم template literals بدلاً من concatenation
- أزل console.log في الإنتاج

📝 النسخة المحسنة:
```javascript
async function getUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        if (!response.ok) throw new Error('Failed to fetch user');
        return await response.json();
    } catch (error) {
        console.error('Error fetching user:', error);
        throw error;
    }
}
```'''
            },
            {
                'code': '''def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] != None:
            result.append(data[i] * 2)
    return result''',
                'language': 'python',
                'review': '''✅ الإيجابيات:
- الدالة بسيطة ومباشرة

⚠️ يمكن تحسينه:
- استخدم `for item in data` بدلاً من `range(len(data))`
- استخدم `is not None` بدلاً من `!= None`
- يمكن اختصارها بـ list comprehension
- أضف type hints

📝 النسخة المحسنة:
```python
def process_data(data: list) -> list:
    return [item * 2 for item in data if item is not None]
```'''
            }
        ]
        
        for example in review_examples:
            data.append({
                'instruction': f"راجع هذا الكود ({example['language']}) واقترح تحسينات:",
                'input': example['code'],
                'output': example['review'],
                'metadata': {
                    'language': example['language'],
                    'type': 'code_review'
                }
            })
        
        return data
    
    def _generate_best_practices_qa(self) -> List[Dict]:
        """توليد أسئلة وأجوبة عن أفضل الممارسات"""
        data = []
        
        for lang, categories in self.best_practices.items():
            for category_info in categories:
                category = category_info['category']
                practices = category_info['practices']
                
                # سؤال عن أفضل الممارسات
                data.append({
                    'instruction': 'أجب على السؤال:',
                    'input': f"ما هي أفضل الممارسات في {category} لـ {lang}؟",
                    'output': "\n".join([f"• {p}" for p in practices]),
                    'metadata': {
                        'language': lang,
                        'category': category,
                        'type': 'best_practices'
                    }
                })
        
        return data
    
    def save_training_data(self, output_dir: Path):
        """حفظ بيانات التدريب"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_data = self.generate_all()
        
        # حفظ كل نوع منفصلاً
        for data_type, data in all_data.items():
            filepath = output_dir / f"{data_type}_training.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ {filepath.name}: {len(data)} عينة")
        
        # حفظ ملف مجمع
        combined = []
        for data_type, data in all_data.items():
            combined.extend(data)
        
        combined_path = output_dir / "combined_training.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"\n📦 المجموع: {combined_path.name}: {len(combined)} عينة")
        
        return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# نقطة البداية
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """نقطة البداية الرئيسية"""
    
    print("\n")
    print("╔" + "═" * 70 + "╗")
    print("║" + " " * 10 + "🔧 Bi IDE - التدريب المتقدم على إصلاح الأخطاء 🔧" + " " * 10 + "║")
    print("╚" + "═" * 70 + "╝")
    print()
    
    # إعداد المسارات
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / "training" / "output"
    
    print("📂 مجلد الإخراج:", OUTPUT_DIR)
    print()
    
    # توليد البيانات
    print("=" * 60)
    print("🔄 توليد بيانات التدريب المتقدمة...")
    print("=" * 60)
    print()
    
    generator = AdvancedTrainingGenerator()
    data = generator.save_training_data(OUTPUT_DIR)
    
    # اختبار محلل الأخطاء
    print()
    print("=" * 60)
    print("🧪 اختبار محلل الأخطاء...")
    print("=" * 60)
    print()
    
    analyzer = ErrorAnalyzer()
    
    # اختبار كود به خطأ
    test_code = '''
const user = undefined;
console.log(user.name);

for (var i = 0; i < 5; i++) {
    setTimeout(() => console.log(i), 100);
}
'''
    
    print("📝 الكود المُختبر:")
    print("-" * 40)
    print(test_code)
    print("-" * 40)
    
    analysis = analyzer.analyze_code(test_code, 'javascript')
    
    print("\n📊 نتائج التحليل:")
    print(f"   جودة الكود: {analysis['code_quality']}%")
    print(f"   الأخطاء المحتملة: {len(analysis['potential_errors'])}")
    
    for i, error in enumerate(analysis['potential_errors'], 1):
        print(f"\n   {i}. {error['category']}:")
        print(f"      - {error['issue']}")
        if error['suggestion']:
            print(f"      - الإصلاح: {error['suggestion'][:100]}...")
    
    print()
    print("=" * 60)
    print("✅ اكتمل التدريب المتقدم!")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
