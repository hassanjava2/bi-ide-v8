#!/usr/bin/env python3
"""
generate_capsule_data.py — توليد بيانات تدريب لـ 20 كبسولة (المرحلة 1)

يستخدم Ollama لتوليد بيانات Q&A متخصصة لكل كبسولة.
كل كبسولة تحصل على بيانات خاصة بتخصصها فقط.
"""

import json, time, os, sys, logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [GEN] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
              logging.StreamHandler()])
logger = logging.getLogger("gen")

# ═══════════════════════════════════════════════════════════
# 20 كبسولة — المرحلة الأولى
# ═══════════════════════════════════════════════════════════

CAPSULES = {
    # ─── المحادثة والتواصل (3) ─────────────────────────
    "conversation_ar": {
        "name": "المحادثة العربية",
        "lang": "ar",
        "topics": ["السلام عليكم", "شلونك؟", "ساعدني بـ", "اشرحلي", "شنو رأيك بـ",
                   "اريد أتعلم", "شلون أقدر", "شنو أفضل طريقة", "عندي مشكلة", "ممكن تساعدني"],
        "vars": ["البرمجة", "تصميم المواقع", "الذكاء الاصطناعي", "إدارة المشاريع",
                 "بناء تطبيق", "حل مشاكل الكمبيوتر", "أمن المعلومات", "الشبكات",
                 "قواعد البيانات", "تعلم الآلة", "العمل الحر", "بدء مشروع"],
    },
    "iraqi_dialect": {
        "name": "اللهجة العراقية",
        "lang": "ar",
        "topics": ["هلا شلونك", "شمدريني", "يمعود شنو هذا", "اريد اسوي", "شگد يكلف",
                   "وين القى", "ليش ميشتغل", "شنو الفرق بين", "هسه شسوي", "ابوك شسوى"],
        "vars": ["تطبيق موبايل", "واجهة مستخدم", "سيرفر", "كود بايثون", "ري أكت",
                 "قاعدة بيانات", "ساعة ابل", "لابتوب كيمنك", "سيارة كهربائية",
                 "طبخ دولمة", "سفر تركيا", "شغل اونلاين"],
    },
    "translator": {
        "name": "المترجم",
        "lang": "mixed",
        "topics": ["Translate to Arabic:", "ترجم للإنجليزية:", "Translate this code comment:",
                   "ترجم هذا المصطلح التقني:", "What does this mean in Arabic:",
                   "Translate this error message:", "ترجم هذه الرسالة:", "Translate formally:"],
        "vars": ["machine learning", "الذكاء الاصطناعي", "database schema", "API endpoint",
                 "authentication token", "deployment pipeline", "user interface",
                 "continuous integration", "error handling", "dependency injection"],
    },

    # ─── البرمجة (7) ──────────────────────────────────
    "code_python": {
        "name": "Python",
        "lang": "en",
        "topics": ["Write a Python function that", "Create a Python class for",
                   "Implement in Python:", "Write a Python decorator for",
                   "Create a FastAPI endpoint for", "Write async Python code for",
                   "Implement error handling for", "Write a Python script to"],
        "vars": ["sorting", "binary search", "linked list", "hash map", "file I/O",
                 "web scraping", "database queries", "API authentication", "caching",
                 "logging", "data validation", "CSV parsing", "JSON handling",
                 "encryption", "websockets", "email sending", "image processing",
                 "multithreading", "configuration management", "CLI tool"],
    },
    "code_typescript": {
        "name": "TypeScript + React",
        "lang": "en",
        "topics": ["Create a React component for", "Write TypeScript code for",
                   "Build a Next.js page for", "Create a custom React hook for",
                   "Write a TypeScript interface for", "Implement state management for",
                   "Create a form component for", "Build a responsive layout for"],
        "vars": ["user dashboard", "login form", "product listing", "shopping cart",
                 "chat interface", "notification system", "file upload", "search bar",
                 "data table", "chart/graph", "user profile", "settings page",
                 "admin panel", "blog editor", "comment section", "payment form"],
    },
    "code_rust": {
        "name": "Rust + Tauri",
        "lang": "en",
        "topics": ["Write a Rust function for", "Create a Tauri command for",
                   "Implement in Rust:", "Write a Rust struct for",
                   "Create a Tauri plugin for", "Handle errors in Rust for",
                   "Write Rust code to", "Implement a Tauri event for"],
        "vars": ["file system operations", "HTTP requests", "JSON parsing",
                 "database connection", "system tray", "keyboard shortcuts",
                 "window management", "IPC communication", "notifications",
                 "auto updater", "clipboard access", "process management"],
    },
    "code_sql": {
        "name": "SQL",
        "lang": "en",
        "topics": ["Write a SQL query to", "Create a database table for",
                   "Write a JOIN query for", "Optimize this SQL query:",
                   "Create an index for", "Write a stored procedure for",
                   "Create a database migration for", "Write a SQL trigger for"],
        "vars": ["user accounts", "order management", "inventory tracking",
                 "financial transactions", "employee records", "product catalog",
                 "customer relationships", "audit logs", "reporting",
                 "access control", "sales analytics", "invoice management"],
    },
    "code_css": {
        "name": "CSS + HTML",
        "lang": "en",
        "topics": ["Create a CSS layout for", "Style a component for",
                   "Write responsive CSS for", "Create CSS animations for",
                   "Build an HTML structure for", "Create a CSS grid for",
                   "Style a form for", "Create a dark mode theme for"],
        "vars": ["navigation bar", "card layout", "modal dialog", "sidebar",
                 "footer", "hero section", "pricing table", "timeline",
                 "profile page", "login page", "dashboard", "landing page"],
    },
    "code_testing": {
        "name": "Testing",
        "lang": "en",
        "topics": ["Write a unit test for", "Create integration tests for",
                   "Write a pytest fixture for", "Create a test mock for",
                   "Write E2E tests for", "Create test data for",
                   "Write a test helper for", "Test edge cases for"],
        "vars": ["API endpoint", "database operations", "authentication flow",
                 "payment processing", "file upload", "user registration",
                 "search functionality", "notification sending", "data export",
                 "error handling", "rate limiting", "input validation"],
    },
    "code_debugging": {
        "name": "Debugging",
        "lang": "en",
        "topics": ["Debug this error:", "Why does this code fail:",
                   "Fix this bug:", "This returns wrong results:",
                   "Memory leak in:", "Performance issue in:",
                   "This crashes when:", "Unexpected behavior:"],
        "vars": ["TypeError: undefined is not a function", "ConnectionRefusedError",
                 "CORS policy error", "500 Internal Server Error",
                 "infinite loop", "race condition", "null pointer exception",
                 "stack overflow", "deadlock", "slow database query",
                 "memory out of bounds", "authentication failed"],
    },

    # ─── ERP (5) ──────────────────────────────────────
    "erp_accounting": {
        "name": "المحاسبة",
        "lang": "ar",
        "topics": ["اشرح القيد المحاسبي لـ", "كيف أسجل", "ما هو الفرق بين",
                   "كيف أحسب", "اعمل ميزانية لـ", "ما هي المعادلة المحاسبية لـ",
                   "اشرح مبدأ", "كيف أدقق"],
        "vars": ["فاتورة مبيعات", "مشتريات آجلة", "رواتب الموظفين", "ضريبة القيمة المضافة",
                 "إهلاك الأصول", "جرد المخزون", "حسابات القبض", "حسابات الدفع",
                 "قائمة الدخل", "الميزانية العمومية", "التدفقات النقدية", "المراجعة"],
    },
    "erp_inventory": {
        "name": "المخزون",
        "lang": "ar",
        "topics": ["كيف أدير مخزون", "ما هي طريقة", "اشرح نظام",
                   "كيف أحسب تكلفة", "ما هو الحد الأدنى لـ", "كيف أسوي جرد",
                   "اشرح حركة", "كيف أتعامل مع"],
        "vars": ["FIFO", "LIFO", "المتوسط المرجح", "نقطة إعادة الطلب",
                 "مخزون الأمان", "دورة المخزون", "التالف والمنتهي",
                 "التحويل بين المستودعات", "الباركود", "تتبع الشحنات"],
    },
    "erp_hr": {
        "name": "الموارد البشرية",
        "lang": "ar",
        "topics": ["كيف أحسب راتب", "ما هي سياسة", "اشرح نظام",
                   "كيف أدير", "ما هو قانون", "كيف أسوي تقييم",
                   "اشرح إجراء", "كيف أعالج"],
        "vars": ["الرواتب والأجور", "الإجازات السنوية", "التأمين الصحي",
                 "نهاية الخدمة", "التوظيف والاختيار", "التدريب والتطوير",
                 "تقييم الأداء", "الحضور والانصراف", "العمل الإضافي",
                 "الاستقالة والفصل", "السياسات الداخلية"],
    },
    "erp_sales": {
        "name": "المبيعات",
        "lang": "ar",
        "topics": ["كيف أسوي فاتورة", "اشرح عملية", "ما هو نظام",
                   "كيف أحسب", "طريقة إدارة", "كيف أتابع",
                   "اشرح سياسة", "كيف أعمل تقرير"],
        "vars": ["فاتورة ضريبية", "عرض سعر", "أمر بيع", "مرتجع مبيعات",
                 "خصم كمية", "عمولة المبيعات", "حد الائتمان",
                 "تقرير المبيعات", "تحليل العملاء", "CRM"],
    },
    "erp_purchasing": {
        "name": "المشتريات",
        "lang": "ar",
        "topics": ["كيف أسوي طلب شراء", "اشرح عملية", "ما هو نظام",
                   "كيف أقيّم", "طريقة مقارنة", "كيف أتابع",
                   "اشرح سياسة", "كيف أعمل"],
        "vars": ["طلب شراء", "أمر شراء", "استلام بضاعة", "فاتورة مورد",
                 "مرتجع مشتريات", "تقييم الموردين", "العقود",
                 "المناقصات", "التفاوض", "الاستيراد"],
    },

    # ─── البنية التحتية (3) ────────────────────────────
    "security": {
        "name": "الأمن السيبراني",
        "lang": "en",
        "topics": ["How to protect against", "Explain the vulnerability",
                   "Implement secure", "Best practices for",
                   "How to detect", "Create a security audit for",
                   "Harden this system:", "Encrypt data for"],
        "vars": ["SQL injection", "XSS", "CSRF", "authentication bypass",
                 "DDoS", "man-in-the-middle", "privilege escalation",
                 "API security", "JWT tokens", "password hashing",
                 "SSL/TLS", "firewall rules", "container security"],
    },
    "devops": {
        "name": "DevOps",
        "lang": "en",
        "topics": ["Write a Docker config for", "Create a CI/CD pipeline for",
                   "Set up nginx for", "Deploy to production:",
                   "Write a systemd service for", "Configure SSL for",
                   "Set up monitoring for", "Create a backup script for"],
        "vars": ["Node.js app", "Python API", "React frontend", "PostgreSQL",
                 "Redis cache", "WebSocket server", "microservices",
                 "load balancer", "auto-scaling", "log rotation"],
    },
    "database_design": {
        "name": "تصميم قواعد البيانات",
        "lang": "en",
        "topics": ["Design a database schema for", "Create an ERD for",
                   "Normalize this schema:", "Design relationships for",
                   "Create indexes for", "Design a migration for",
                   "Optimize schema for", "Design audit tables for"],
        "vars": ["e-commerce platform", "ERP system", "social media",
                 "hospital management", "school management", "banking system",
                 "inventory system", "CRM", "project management",
                 "real estate", "restaurant POS", "HR system"],
    },

    # ─── المجلس (2) ───────────────────────────────────
    "sage": {
        "name": "حكيم الاستراتيجية",
        "lang": "ar",
        "topics": ["من منظور استراتيجي،", "حلل المخاطر في", "ما هي الخطة المثلى لـ",
                   "قيّم هذا القرار:", "ما هي البدائل لـ", "اقترح استراتيجية لـ",
                   "حلل نقاط القوة والضعف في", "ما هو التأثير طويل المدى لـ"],
        "vars": ["إطلاق منتج جديد", "توسع الشركة", "تغيير التكنولوجيا",
                 "دخول سوق جديد", "تقليل التكاليف", "زيادة الإنتاجية",
                 "إدارة الأزمات", "الاستثمار في AI", "الاندماج والاستحواذ"],
    },
    "rebel": {
        "name": "المتمرد",
        "lang": "ar",
        "topics": ["لا أوافق! المشكلة هي", "هذا خطأ لأن", "الجواب الحقيقي هو",
                   "كلكم غلط لأن", "فكروا بشكل مختلف:", "ليش ماحد يفكر بـ",
                   "التحدي الحقيقي هو", "اعترضوا الطريقة هاي لأن"],
        "vars": ["استخدام AI", "تصميم النظام", "اختيار التكنولوجيا",
                 "خطة المشروع", "أمن النظام", "تجربة المستخدم",
                 "أداء التطبيق", "تكلفة التشغيل", "الاعتماد على السحابة"],
    },
}


def ollama_generate(model, prompt, timeout=90):
    import requests
    try:
        r = requests.post("http://127.0.0.1:11434/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "stream": False, "options": {"temperature": 0.7, "num_predict": 500}},
            timeout=timeout)
        if r.status_code == 200:
            return r.json().get("message", {}).get("content", "")
    except Exception as e:
        logger.warning(f"Ollama: {e}")
    return ""


def get_model():
    import requests
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if models:
            return models[0]
    except:
        pass
    return None


def generate_for_capsule(capsule_id, config, model, target=200):
    out_dir = PROJECT_ROOT / "capsules" / capsule_id / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generated_train.jsonl"

    logger.info(f"🧠 {capsule_id}: generating {target} samples with {model}")
    count = 0
    with open(out_path, "w") as f:
        for topic in config["topics"]:
            for var in config["vars"]:
                if count >= target:
                    break
                q = f"{topic} {var}"
                a = ollama_generate(model, q)
                if a and len(a) > 20:
                    f.write(json.dumps({"input_text": q, "output_text": a}, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 20 == 0:
                        logger.info(f"  [{capsule_id}] {count}/{target}")
            if count >= target:
                break
    logger.info(f"✅ {capsule_id}: {count} samples")
    return count


def main():
    logger.info("=" * 60)
    logger.info(f"DATA GENERATOR — 20 capsules")
    logger.info("=" * 60)

    model = get_model()
    if not model:
        logger.error("No Ollama model! Run: ollama serve && ollama pull qwen2.5:1.5b")
        sys.exit(1)
    logger.info(f"Using model: {model}")

    target = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    total = 0
    for cid, cfg in CAPSULES.items():
        total += generate_for_capsule(cid, cfg, model, target)

    logger.info(f"\nTOTAL: {total} samples across {len(CAPSULES)} capsules")


if __name__ == "__main__":
    main()
