#!/usr/bin/env python3
"""
generate_capsule_data.py — توليد بيانات تدريب متخصصة لكل كبسولة

يستخدم Ollama models الموجودة على RTX (190GB) لتوليد بيانات تدريب
عالية الجودة — ثم كل كبسولة تتدرب على بياناتها الخاصة.

الموديلات المتاحة: qwen3, gemma3, deepseek-r1, llama3, codellama, mistral

لكل كبسولة:
1. يولّد أسئلة متخصصة
2. يجيب عليها بموديل Ollama ذكي
3. يحفظها كـ JSONL
4. يشغّل التدريب عليها
"""

import json
import time
import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"data_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("data_generator")

# ─── Capsule Definitions ─────────────────────────────────────
# كل كبسولة = تخصص + أسئلة خاصة بي
CAPSULES = {
    "code_python": {
        "name": "كبسولة البرمجة — Python",
        "ollama_model": "qwen3-coder:30b",
        "fallback_model": "codellama:7b",
        "language": "en",
        "topics": [
            "Write a Python function that",
            "Create a Python class for",
            "Implement a data structure in Python for",
            "Write a Python script to",
            "Debug this Python code:",
            "Optimize this Python function:",
            "Write unit tests for",
            "Create a REST API endpoint using FastAPI for",
            "Write a Python decorator that",
            "Implement error handling for",
        ],
        "variations": [
            "sorting algorithms", "binary search", "linked list", "hash map",
            "file I/O", "web scraping", "database queries", "async operations",
            "regular expressions", "data validation", "API authentication",
            "caching", "logging", "configuration management", "CLI tool",
            "image processing", "CSV parsing", "JSON handling", "encryption",
            "multithreading", "websockets", "email sending", "PDF generation",
            "machine learning pipeline", "data cleaning", "graph traversal",
            "tree operations", "matrix operations", "string manipulation",
            "date/time handling", "network requests", "queue management",
        ],
    },
    "code_web": {
        "name": "كبسولة تطوير الويب",
        "ollama_model": "qwen3-coder:30b",
        "fallback_model": "codellama:7b",
        "language": "en",
        "topics": [
            "Create a React component for",
            "Write TypeScript code for",
            "Build an HTML/CSS layout for",
            "Create a Next.js page for",
            "Write a REST API route for",
            "Implement authentication using",
            "Create a database schema for",
            "Write SQL queries for",
            "Build a responsive design for",
            "Create a WebSocket handler for",
        ],
        "variations": [
            "user dashboard", "login form", "product listing", "shopping cart",
            "chat interface", "notification system", "file upload", "search bar",
            "data table", "chart/graph", "user profile", "settings page",
            "admin panel", "blog post editor", "comment section", "payment form",
            "navigation menu", "modal dialog", "toast notifications", "sidebar",
        ],
    },
    "knowledge_arabic": {
        "name": "كبسولة المعرفة العربية",
        "ollama_model": "gemma3:27b",
        "fallback_model": "gemma3:12b",
        "language": "ar",
        "topics": [
            "اشرح بالتفصيل",
            "ما هو",
            "كيف يعمل",
            "ما الفرق بين",
            "اذكر أهم",
            "لماذا يعتبر",
            "ما هي مراحل",
            "قارن بين",
            "ما هي فوائد",
            "كيف يمكن تطبيق",
        ],
        "variations": [
            "الذكاء الاصطناعي", "تعلم الآلة", "الشبكات العصبية",
            "أمن المعلومات", "التشفير", "قواعد البيانات",
            "نظام التشغيل", "الحوسبة السحابية", "إنترنت الأشياء",
            "البلوكتشين", "العملات الرقمية", "تطوير البرمجيات",
            "إدارة المشاريع", "الخوارزميات", "هياكل البيانات",
            "الفيزياء الكمية", "الرياضيات التطبيقية", "الهندسة الكهربائية",
            "الطاقة المتجددة", "الروبوتات", "الطباعة ثلاثية الأبعاد",
        ],
    },
    "security": {
        "name": "كبسولة الأمن السيبراني",
        "ollama_model": "deepseek-r1:8b",
        "fallback_model": "qwen3:8b",
        "language": "en",
        "topics": [
            "Explain the security vulnerability",
            "How to protect against",
            "Write a security audit for",
            "Implement secure",
            "What are the risks of",
            "How to detect",
            "Create a firewall rule for",
            "Explain the attack vector",
            "How to harden",
            "Best practices for",
        ],
        "variations": [
            "SQL injection", "XSS attacks", "CSRF", "buffer overflow",
            "privilege escalation", "man-in-the-middle", "DDoS",
            "password cracking", "social engineering", "phishing",
            "network scanning", "port security", "SSL/TLS",
            "API security", "JWT tokens", "OAuth implementation",
            "container security", "Linux hardening", "Windows security",
        ],
    },
    "conversation_ar": {
        "name": "كبسولة المحادثة العربية",
        "ollama_model": "gemma3:27b",
        "fallback_model": "gemma3:12b",
        "language": "ar",
        "topics": [
            "السلام عليكم",
            "شلونك اليوم؟",
            "شنو رأيك بـ",
            "ساعدني بـ",
            "اريد أتعلم عن",
            "وين أقدر ألقى",
            "شلون أسوي",
            "اشرحلي بسهولة",
            "شنو أفضل طريقة لـ",
            "عندي مشكلة بـ",
        ],
        "variations": [
            "البرمجة", "تصميم المواقع", "الذكاء الاصطناعي",
            "إدارة المشاريع", "العمل الحر", "التعلم الذاتي",
            "بناء تطبيق", "حل مشاكل الكمبيوتر", "اختيار لغة برمجة",
            "تعلم الإنجليزية", "بدء مشروع", "كتابة السيرة الذاتية",
        ],
    },
}


def ollama_generate(model: str, prompt: str, timeout: int = 60) -> str:
    """توليد نص من Ollama"""
    import requests
    try:
        resp = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 500},
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "")
    except Exception as e:
        logger.warning(f"Ollama error ({model}): {e}")
    return ""


def check_ollama_model(model: str) -> bool:
    """تأكد الموديل موجود"""
    import requests
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return model in models or model.split(":")[0] in [m.split(":")[0] for m in models]
    except:
        pass
    return False


def generate_capsule_data(capsule_id: str, config: dict, target_samples: int = 500):
    """توليد بيانات لكبسولة واحدة"""
    output_dir = PROJECT_ROOT / "capsules" / capsule_id / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generated_train.jsonl"
    
    # Pick available model
    model = config["ollama_model"]
    if not check_ollama_model(model):
        model = config["fallback_model"]
        if not check_ollama_model(model):
            # Try any available model
            import requests
            try:
                resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
                models = [m["name"] for m in resp.json().get("models", [])]
                if models:
                    model = models[0]
                    logger.info(f"Using fallback model: {model}")
                else:
                    logger.error(f"❌ [{capsule_id}] No Ollama models available!")
                    return 0
            except:
                logger.error(f"❌ [{capsule_id}] Ollama not running!")
                return 0
    
    logger.info(f"🧠 [{capsule_id}] Generating {target_samples} samples with {model}")
    
    generated = 0
    with open(output_path, "w") as f:
        for topic in config["topics"]:
            for variation in config["variations"]:
                if generated >= target_samples:
                    break
                
                question = f"{topic} {variation}"
                answer = ollama_generate(model, question)
                
                if answer and len(answer) > 20:
                    sample = {
                        "input_text": question,
                        "output_text": answer,
                        "source": f"ollama_{model}",
                        "kind": "generated",
                        "language": config["language"],
                        "capsule": capsule_id,
                        "timestamp": datetime.now().isoformat(),
                    }
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    generated += 1
                    
                    if generated % 10 == 0:
                        logger.info(f"  [{capsule_id}] {generated}/{target_samples}")
            
            if generated >= target_samples:
                break
    
    logger.info(f"✅ [{capsule_id}] Generated {generated} samples → {output_path}")
    return generated


def main():
    logger.info("=" * 60)
    logger.info("🏭 Capsule Data Generator — بسم الله")
    logger.info("=" * 60)
    
    # Check Ollama
    import requests
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        logger.info(f"📦 Ollama models available: {models}")
    except:
        logger.error("❌ Ollama not running! Start: ollama serve")
        sys.exit(1)
    
    # Generate data for each capsule
    target_per_capsule = 200  # Start with 200, increase later
    
    for capsule_id, config in CAPSULES.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"🎯 {config['name']} ({capsule_id})")
        count = generate_capsule_data(capsule_id, config, target_per_capsule)
        logger.info(f"📊 Result: {count} samples")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Data generation complete!")
    logger.info("Next: run start_capsule_training.py for each capsule")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
