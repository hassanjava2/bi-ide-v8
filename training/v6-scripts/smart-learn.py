"""
Bi IDE – التعلم الذكي الشامل
Smart Learning Engine - يبحث أونلاين + يولّد training data + يبني embeddings + يدرّب

الاستخدام:
  python training/smart-learn.py                    # تعلم كل المواضيع بالطابور
  python training/smart-learn.py --topic "React"    # تعلم موضوع واحد
  python training/smart-learn.py --curriculum web    # تعلم منهج كامل
  python training/smart-learn.py --train             # تدريب النموذج فقط
  python training/smart-learn.py --status            # عرض الحالة

يكتب التقدم في data/learning/smart-learn-progress.json (Bi IDE يقرأه)
"""

import json
import os
import sys
import time
import re
import hashlib
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════
# المسارات
# ══════════════════════════════════════

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LEARNING_DIR = DATA_DIR / "learning"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
TRAINING_DIR = BASE_DIR / "training" / "output"
MODELS_DIR = BASE_DIR / "models"

QUEUE_FILE = LEARNING_DIR / "learn-queue.json"
PROGRESS_FILE = LEARNING_DIR / "smart-learn-progress.json"
LEARNED_FILE = KNOWLEDGE_DIR / "smart-learned-data.json"
EMBEDDINGS_FILE = KNOWLEDGE_DIR / "smart-embeddings.json"

# إنشاء المجلدات
LEARNING_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════
# المناهج
# ══════════════════════════════════════

CURRICULA = {
    "js": {
        "name": "JavaScript Mastery",
        "topics": [
            "JavaScript variables let const var scope",
            "JavaScript functions arrow closures hoisting",
            "JavaScript promises async await error handling",
            "JavaScript classes inheritance prototypes OOP",
            "JavaScript modules import export ES6",
            "JavaScript array methods map filter reduce forEach",
            "JavaScript destructuring spread rest operator",
            "JavaScript event loop microtasks macrotasks",
            "JavaScript DOM manipulation events",
            "JavaScript fetch API HTTP requests",
            "JavaScript error handling try catch finally",
            "JavaScript regular expressions patterns",
            "JavaScript generators iterators symbols",
            "JavaScript proxy reflect metaprogramming",
            "JavaScript web workers service workers",
            "Node.js express REST API middleware",
            "Node.js file system streams buffers",
            "Node.js authentication JWT bcrypt sessions",
            "Node.js MongoDB Mongoose database",
            "Node.js testing Jest Mocha Chai",
        ]
    },
    "python": {
        "name": "Python Complete",
        "topics": [
            "Python data types variables strings lists dicts",
            "Python functions lambda decorators generators",
            "Python OOP classes inheritance polymorphism",
            "Python file handling reading writing CSV JSON",
            "Python exception handling try except finally",
            "Python list comprehension dict comprehension",
            "Python async programming asyncio await",
            "Python regular expressions re module",
            "Python NumPy arrays operations broadcasting",
            "Python Pandas DataFrame Series operations",
            "Python Flask web framework routing templates",
            "Python Django models views templates ORM",
            "Python FastAPI REST API async endpoints",
            "Python SQLAlchemy database ORM queries",
            "Python testing pytest unittest mocking",
            "Python machine learning scikit-learn basics",
            "Python PyTorch tensors neural networks",
            "Python web scraping BeautifulSoup requests",
            "Python automation scripts os subprocess",
            "Python packaging pip virtualenv setup",
        ]
    },
    "web": {
        "name": "Full Stack Web Development",
        "topics": [
            "HTML5 semantic elements forms accessibility",
            "CSS3 flexbox grid layout responsive design",
            "CSS animations transitions transforms",
            "React components JSX props state hooks",
            "React useState useEffect useContext useReducer",
            "React Router navigation dynamic routes",
            "React performance memo useMemo useCallback",
            "Vue.js components reactivity computed watchers",
            "Next.js SSR SSG ISR API routes",
            "TypeScript types interfaces generics enums",
            "REST API design best practices versioning",
            "GraphQL queries mutations subscriptions",
            "WebSocket real-time Socket.io implementation",
            "Authentication OAuth2 JWT session cookies",
            "MongoDB NoSQL document database queries",
            "PostgreSQL SQL joins indexes transactions",
            "Redis caching pub/sub session store",
            "Docker containers Dockerfile compose networking",
            "CI/CD GitHub Actions deployment pipelines",
            "Web security XSS CSRF SQL injection CORS",
        ]
    },
    "laravel": {
        "name": "Laravel PHP Framework",
        "topics": [
            "PHP 8 types match named arguments fibers",
            "Laravel routing controllers middleware groups",
            "Laravel Blade templates components layouts",
            "Laravel Eloquent ORM models relationships",
            "Laravel migrations seeders factories",
            "Laravel authentication Breeze Sanctum Passport",
            "Laravel authorization gates policies",
            "Laravel validation form requests rules",
            "Laravel file storage upload S3 local",
            "Laravel API resources JSON responses",
            "Laravel queues jobs workers scheduling",
            "Laravel events listeners notifications mail",
            "Laravel testing feature unit browser PHPUnit",
            "Laravel Livewire real-time components",
            "Laravel deployment optimization caching",
        ]
    },
    "security": {
        "name": "Security & DevOps",
        "topics": [
            "Linux command line bash scripting permissions",
            "Network security TCP/IP DNS firewalls",
            "Cryptography hashing encryption AES RSA",
            "Web security OWASP top 10 vulnerabilities",
            "Penetration testing methodology tools Kali",
            "Docker security best practices scanning",
            "Kubernetes basics pods services deployments",
            "Cloud security AWS IAM VPC security groups",
            "SSL TLS HTTPS certificate management",
            "Security logging monitoring SIEM tools",
        ]
    },
    "ai": {
        "name": "AI & Machine Learning",
        "topics": [
            "Machine learning supervised unsupervised basics",
            "Neural networks perceptron backpropagation",
            "Deep learning CNN image classification",
            "RNN LSTM sequence modeling text generation",
            "Transformer architecture attention mechanism",
            "NLP tokenization embeddings word2vec",
            "Large Language Models GPT BERT training",
            "Fine-tuning LLM LoRA QLoRA PEFT",
            "RAG retrieval augmented generation vector DB",
            "Model deployment ONNX TensorRT optimization",
            "PyTorch training pipeline DataLoader optimizer",
            "Hugging Face transformers pipeline usage",
            "Computer vision object detection YOLO",
            "Reinforcement learning Q-learning policy gradient",
            "AI ethics bias fairness safety alignment",
        ]
    },
    "survival": {
        "name": "البقاء بعد الكارثة",
        "topics": [
            "emergency survival shelter building techniques",
            "water purification filtration methods survival",
            "food preservation drying salting fermentation",
            "herbal medicine natural remedies first aid",
            "fire starting methods primitive tools survival",
            "navigation without GPS stars compass natural",
            "seed saving crop cultivation subsistence farming",
            "animal husbandry raising livestock chickens goats",
            "solar power DIY panels battery systems",
            "wind turbine construction alternative energy",
            "well digging water collection rainwater harvesting",
            "brick making construction mud stone building",
            "blacksmithing metal working tools weapons",
            "textile production weaving spinning cloth",
            "soap making candle making basic chemistry",
            "radio communication emergency broadcasting",
            "community governance conflict resolution",
            "basic surgery wound care field medicine",
            "food storage long term preservation techniques",
            "defense security perimeter protection",
        ]
    },
    "factory": {
        "name": "التصنيع من الصفر",
        "topics": [
            "PCB design manufacturing etching soldering",
            "laptop motherboard assembly production line",
            "display screen LCD OLED manufacturing",
            "battery lithium ion cell assembly pack",
            "thermal management heatsink cooling design",
            "chassis mechanical design CNC machining",
            "BIOS firmware embedded programming",
            "quality testing reliability burn-in testing",
            "smartphone manufacturing assembly process",
            "solar panel manufacturing silicon wafer",
            "LED lighting manufacturing assembly",
            "electric motor winding assembly testing",
            "transformer manufacturing core winding",
            "Lean manufacturing Six Sigma quality",
            "factory ERP MES PLM WMS systems",
        ]
    }
}

# ══════════════════════════════════════
# بحث أونلاين
# ══════════════════════════════════════

def search_online(query, max_results=10):
    """بحث أونلاين في عدة مصادر"""
    results = []
    
    # 1. StackOverflow
    try:
        import urllib.request
        import urllib.parse
        url = f"https://api.stackexchange.com/2.3/search?order=desc&sort=votes&intitle={urllib.parse.quote(query)}&site=stackoverflow&pagesize=5"
        req = urllib.request.Request(url, headers={'User-Agent': 'BiIDE/5.0', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for item in (data.get('items') or [])[:5]:
                results.append({
                    'source': 'stackoverflow',
                    'title': item.get('title', ''),
                    'content': item.get('title', ''),
                    'url': item.get('link', ''),
                    'score': item.get('score', 0),
                    'tags': item.get('tags', []),
                })
        print(f"  ✅ StackOverflow: {len(data.get('items', [])[:5])} نتائج")
    except Exception as e:
        print(f"  ⚠️ StackOverflow: {e}")
    
    # 2. npm
    try:
        url = f"https://registry.npmjs.org/-/v1/search?text={urllib.parse.quote(query)}&size=5"
        req = urllib.request.Request(url, headers={'User-Agent': 'BiIDE/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for obj in (data.get('objects') or [])[:5]:
                pkg = obj.get('package', {})
                results.append({
                    'source': 'npm',
                    'title': pkg.get('name', ''),
                    'content': f"{pkg.get('description', '')}\nKeywords: {', '.join(pkg.get('keywords', []))}",
                    'url': f"https://www.npmjs.com/package/{pkg.get('name', '')}",
                    'score': obj.get('score', {}).get('final', 0),
                })
        print(f"  ✅ npm: {len(data.get('objects', [])[:5])} نتائج")
    except Exception as e:
        print(f"  ⚠️ npm: {e}")
    
    # 3. GitHub
    try:
        url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&per_page=5"
        req = urllib.request.Request(url, headers={'User-Agent': 'BiIDE/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for repo in (data.get('items') or [])[:5]:
                results.append({
                    'source': 'github',
                    'title': repo.get('full_name', ''),
                    'content': f"{repo.get('description', '')}\nStars: {repo.get('stargazers_count', 0)}\nLanguage: {repo.get('language', 'N/A')}",
                    'url': repo.get('html_url', ''),
                    'score': repo.get('stargazers_count', 0),
                })
        print(f"  ✅ GitHub: {len(data.get('items', [])[:5])} نتائج")
    except Exception as e:
        print(f"  ⚠️ GitHub: {e}")
    
    return results

# ══════════════════════════════════════
# توليد بيانات التدريب
# ══════════════════════════════════════

def generate_training_data(topic, search_results):
    """يولّد أزواج instruction/output من نتائج البحث"""
    pairs = []
    
    for result in search_results:
        text = f"{result['title']}\n{result['content']}".strip()
        if len(text) < 20:
            continue
        
        # نمط 1: اشرح الموضوع
        pairs.append({
            'instruction': f"اشرح {topic}",
            'input': '',
            'output': text[:500],
            'metadata': {'source': result['source'], 'topic': topic, 'type': 'explain'}
        })
        
        # نمط 2: ما هو
        pairs.append({
            'instruction': f"What is {topic}?",
            'input': '',
            'output': text[:500],
            'metadata': {'source': result['source'], 'topic': topic, 'type': 'what-is'}
        })
        
        # نمط 3: أمثلة
        if result.get('tags'):
            pairs.append({
                'instruction': f"ما هي المكتبات المتعلقة بـ {topic}?",
                'input': '',
                'output': f"المكتبات والأدوات: {', '.join(result['tags'][:10])}",
                'metadata': {'source': result['source'], 'topic': topic, 'type': 'related'}
            })
        
        # نمط 4: مقارنة
        if result['source'] == 'github' and result.get('score', 0) > 100:
            pairs.append({
                'instruction': f"ما أفضل مشروع مفتوح المصدر لـ {topic}?",
                'input': '',
                'output': f"{result['title']} - {result['content'][:300]}",
                'metadata': {'source': 'github', 'topic': topic, 'type': 'recommendation'}
            })
    
    return pairs

# ══════════════════════════════════════
# تعلم موضوع واحد
# ══════════════════════════════════════

def learn_topic(topic, progress_callback=None):
    """تعلم موضوع واحد: بحث → توليد → حفظ"""
    print(f"\n📖 تعلّم: {topic}")
    
    # 1. بحث أونلاين
    print("  🌐 بحث أونلاين...")
    results = search_online(topic)
    
    if not results:
        print("  ⚠️ لا نتائج - تخطي")
        return {'topic': topic, 'status': 'no_results', 'pairs': 0}
    
    # 2. توليد بيانات تدريب
    print("  📝 توليد بيانات تدريب...")
    pairs = generate_training_data(topic, results)
    
    # 3. حفظ
    save_training_data(pairs)
    
    print(f"  ✅ {len(pairs)} زوج تدريبي + {len(results)} نتيجة بحث")
    
    return {'topic': topic, 'status': 'completed', 'pairs': len(pairs), 'results': len(results)}

# ══════════════════════════════════════
# حفظ البيانات
# ══════════════════════════════════════

def save_training_data(new_pairs):
    """إضافة بيانات تدريب جديدة"""
    existing = []
    if LEARNED_FILE.exists():
        try:
            with open(LEARNED_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except:
            existing = []
    
    # إزالة التكرار
    existing_keys = set()
    for item in existing:
        key = hashlib.md5(f"{item.get('instruction','')}|{item.get('output','')[:50]}".encode()).hexdigest()
        existing_keys.add(key)
    
    added = 0
    for pair in new_pairs:
        key = hashlib.md5(f"{pair.get('instruction','')}|{pair.get('output','')[:50]}".encode()).hexdigest()
        if key not in existing_keys:
            existing.append(pair)
            existing_keys.add(key)
            added += 1
    
    # حد أقصى 50000
    if len(existing) > 50000:
        existing = existing[-50000:]
    
    with open(LEARNED_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=1)
    
    # نسخة لـ training/output أيضاً
    training_copy = TRAINING_DIR / "smart_learned_training.json"
    with open(training_copy, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=1)
    
    return added

def update_progress(state):
    """تحديث ملف التقدم (Bi IDE يقرأه)"""
    state['updated_at'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ══════════════════════════════════════
# حفظ/استعادة المواضيع المكتملة (resume)
# ══════════════════════════════════════

COMPLETED_FILE = LEARNING_DIR / "completed-topics.json"

def load_completed_topics():
    """تحميل المواضيع المكتملة سابقاً"""
    if COMPLETED_FILE.exists():
        try:
            with open(COMPLETED_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except:
            pass
    return set()

def save_completed_topic(topic, completed_set):
    """حفظ موضوع مكتمل"""
    completed_set.add(topic)
    with open(COMPLETED_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(completed_set), f, ensure_ascii=False)

# ══════════════════════════════════════
# التدريب الفعلي
# ══════════════════════════════════════

def train_model():
    """تدريب النموذج بكل البيانات المجمعة"""
    print("\n" + "=" * 50)
    print("🚀 بدء تدريب النموذج")
    print("=" * 50)
    
    # جمع كل البيانات
    all_data = []
    
    for json_file in TRAINING_DIR.glob("*.json"):
        if json_file.name == "training_report.json":
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                valid = [d for d in data if d.get('instruction') and d.get('output')]
                all_data.extend(valid)
        except:
            pass
    
    # بيانات المعرفة
    for kb_file in ['rag-knowledge-base.json', 'smart-learned-data.json']:
        kb_path = KNOWLEDGE_DIR / kb_file
        if kb_path.exists():
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for d in data:
                        if d.get('instruction') and d.get('output'):
                            all_data.append(d)
                        elif d.get('text') and d.get('answer'):
                            all_data.append({'instruction': d['text'][:200], 'output': d['answer'][:500]})
            except:
                pass
    
    # إزالة التكرار
    seen = set()
    unique = []
    for item in all_data:
        key = f"{item['instruction'][:30]}|{item['output'][:30]}"
        if key not in seen:
            seen.add(key)
            unique.append(item)
    
    print(f"📊 بيانات التدريب: {len(unique)} عينة فريدة")
    
    if len(unique) < 50:
        print("❌ بيانات قليلة جداً. تعلم مواضيع أكثر أولاً.")
        return False
    
    # استدعاء auto-finetune
    try:
        import subprocess
        script = BASE_DIR / "training" / "auto-finetune.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE_DIR),
            capture_output=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ خطأ بالتدريب: {e}")
        return False

# ══════════════════════════════════════
# Main
# ══════════════════════════════════════

def main():
    args = sys.argv[1:]
    
    print("═" * 50)
    print("🧠 Bi IDE - التعلم الذكي الشامل")
    print("═" * 50)
    
    # عرض الحالة فقط
    if '--status' in args:
        if LEARNED_FILE.exists():
            with open(LEARNED_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"📊 بيانات متعلّمة: {len(data)} عينة")
        else:
            print("📊 لا توجد بيانات متعلّمة بعد")
        
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            print(f"📈 آخر تحديث: {progress.get('updated_at', 'N/A')}")
            print(f"   مواضيع مكتملة: {progress.get('completed', 0)}")
            print(f"   بيانات تدريب: {progress.get('total_pairs', 0)}")
        
        print(f"\nالمناهج المتاحة:")
        for cid, cur in CURRICULA.items():
            print(f"  {cid}: {cur['name']} ({len(cur['topics'])} موضوع)")
        return
    
    # تدريب النموذج فقط
    if '--train' in args:
        train_model()
        return
    
    # تحديد المواضيع
    topics = []
    
    # موضوع واحد
    if '--topic' in args:
        idx = args.index('--topic')
        if idx + 1 < len(args):
            topics = [args[idx + 1]]
    
    # منهج كامل
    elif '--curriculum' in args:
        idx = args.index('--curriculum')
        if idx + 1 < len(args):
            cur_id = args[idx + 1]
            if cur_id == 'all':
                for cur in CURRICULA.values():
                    topics.extend(cur['topics'])
                print(f"📚 كل المناهج: {len(topics)} موضوع")
            elif cur_id in CURRICULA:
                topics = CURRICULA[cur_id]['topics']
                print(f"📚 منهج: {CURRICULA[cur_id]['name']} ({len(topics)} موضوع)")
            else:
                print(f"❌ منهج غير موجود: {cur_id}")
                print(f"   المتاح: {', '.join(CURRICULA.keys())}, all")
                return
    
    # بدون arguments → أول منهج
    else:
        # تحميل أول منهج كافتراضي
        first = list(CURRICULA.keys())[0]
        topics = CURRICULA[first]['topics']
        print(f"📚 افتراضي: {CURRICULA[first]['name']} ({len(topics)} موضوع)")
        print(f"   استخدم --curriculum <id> لمنهج آخر")
        print(f"   المتاح: {', '.join(CURRICULA.keys())}, all")
    
    if not topics:
        print("❌ لا مواضيع محددة")
        return
    
    # تحميل المواضيع المكتملة سابقاً (resume)
    completed_set = load_completed_topics()
    remaining = [t for t in topics if t not in completed_set]
    skipped = len(topics) - len(remaining)
    
    if skipped > 0:
        print(f"⏭️ تخطي {skipped} موضوع مكتمل سابقاً")
    
    if not remaining:
        print("✅ كل المواضيع مكتملة! استخدم --reset لإعادة البدء")
        if '--reset' in args:
            completed_set = set()
            if COMPLETED_FILE.exists():
                COMPLETED_FILE.unlink()
            remaining = topics
            print("🔄 تم إعادة التعيين")
        else:
            return
    
    # بدء التعلم
    progress = {
        'state': 'running',
        'total': len(topics),
        'completed': skipped,
        'failed': 0,
        'total_pairs': 0,
        'total_results': 0,
        'current_topic': '',
        'remaining': len(remaining),
        'started_at': datetime.now().isoformat(),
    }
    update_progress(progress)
    
    print(f"\n🚀 بدء تعلّم {len(remaining)} موضوع ({skipped} مكتمل سابقاً)...\n")
    
    for i, topic in enumerate(remaining):
        progress['current_topic'] = topic
        progress['current_index'] = skipped + i + 1
        update_progress(progress)
        
        try:
            result = learn_topic(topic)
            
            if result['status'] == 'completed':
                progress['completed'] += 1
                progress['total_pairs'] += result['pairs']
                progress['total_results'] += result['results']
                save_completed_topic(topic, completed_set)
            else:
                progress['failed'] += 1
            
        except KeyboardInterrupt:
            print("\n\n⏹️ أوقف بواسطة المستخدم - التقدم محفوظ!")
            print(f"   مكتمل: {progress['completed']}/{progress['total']}")
            print(f"   شغّل نفس الأمر مرة ثانية ليكمل من هنا")
            progress['state'] = 'stopped'
            update_progress(progress)
            break
        except Exception as e:
            print(f"  ❌ خطأ: {e}")
            progress['failed'] += 1
        
        # تقدم
        total_done = skipped + i + 1
        pct = round((total_done / len(topics)) * 100)
        print(f"  📊 التقدم: {pct}% ({total_done}/{len(topics)})")
        
        # راحة بين المواضيع (rate limiting)
        if i < len(remaining) - 1:
            time.sleep(2)
    
    # اكتمال
    if progress['state'] != 'stopped':
        progress['state'] = 'completed'
        progress['completed_at'] = datetime.now().isoformat()
    update_progress(progress)
    
    print("\n" + "=" * 50)
    print(f"✅ اكتمل: {progress['completed']}/{progress['total']} موضوع")
    print(f"📝 بيانات تدريب: {progress['total_pairs']} زوج")
    print(f"🌐 نتائج بحث: {progress['total_results']}")
    print("=" * 50)
    
    # سؤال عن التدريب
    if progress['total_pairs'] > 100:
        print(f"\n💡 عندك {progress['total_pairs']} بيانات تدريب جديدة.")
        print(f"   شغّل: python training/smart-learn.py --train")
        print(f"   أو: python training/auto-finetune.py")

if __name__ == "__main__":
    main()
