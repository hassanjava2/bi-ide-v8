# BI IDE v8 🚀

**BI IDE v8** - منصة متكاملة للتطوير وإدارة الموارد المؤسسية مدعومة بالذكاء الاصطناعي

**ERP + IDE + AI Hierarchy (10 Layers + 100+ AI Entities)**

[English](#english) | [العربية](#arabic)

---

<a name="english"></a>
## 🌟 English

### Features

- **🧠 Smart Council**: 16 AI Wise Men for strategic decisions
- **🏛️ AI Hierarchy**: 10-layer hierarchical AI system
- **💻 IDE**: Full-featured development environment with AI Copilot
- **🏢 ERP**: Enterprise Resource Planning (Invoices, Inventory, HR, Sales)
- **📚 Autonomous Learning**: Self-learning from user interactions
- **⚡ Real-time Inference**: Connected to RTX 4090 server

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python start.py
# or
start.bat  # Windows

# Run with Docker
docker-compose up -d
```

### API Endpoints

- `GET /health` - Health check
- `GET /docs` - API Documentation (Swagger)
- `GET /api/v1/status` - System status
- `POST /api/v1/council/message` - Send message to AI Council
- `POST /api/v1/ide/analyze` - Analyze code
- `GET /api/v1/erp/invoices` - List invoices

### Project Structure

```
bi-ide-v8/
├── api/app.py              # Main API entry point
├── core/                   # Core modules
│   ├── logging_config.py  # Centralized logging
│   ├── database.py        # Database layer
│   ├── cache.py           # Caching layer
│   └── config.py          # Configuration
├── hierarchy/             # AI Hierarchy (10 layers)
├── ide/                   # IDE Service
├── erp/                   # ERP Service
├── ui/                    # React Frontend
├── tests/                 # Test suite
├── docker-compose.yml     # Docker orchestration
└── requirements.txt       # Python dependencies
```

---

<a name="arabic"></a>
## 🌟 العربية

### الميزات

- **🧠 المجلس الذكي**: 16 حكيم AI للقرارات الاستراتيجية
- **🏛️ النظام الهرمي**: 10 طبقات من الذكاء الاصطناعي
- **💻 بيئة التطوير**: IDE متكامل مع AI Copilot
- **🏢 نظام ERP**: إدارة الموارد (فواتير، مخزون، موارد بشرية، مبيعات)
- **📚 التعلم الذاتي**: يتعلم تلقائياً من تفاعلات المستخدم
- **⚡ الاستدلال الفوري**: متصل بخادم RTX 4090

### الهيكل التنظيمي للـ AI

```
الرئيس (المستخدم)
    ↓
البعد السابع (4 مخططون - 100 سنة)
    ↓
مجلس الحكماء (16 حكيم - 24/7)
    ↓
فرق الظل والنور (8 متوازنون)
    ↓
الكشافة (4 كشافة)
    ↓
الفريق الميتا (16 مدير)
    ↓
خبراء المجالات (12 خبير)
    ↓
فرق التنفيذ (مؤقتة)
```

### البدء السريع

```bash
# تثبيت المتطلبات
pip install -r requirements.txt

# التشغيل محلياً
python start.py
# أو
start.bat  # Windows

# التشغيل بـ Docker
docker-compose up -d
```

### نقاط النهاية للـ API

- `GET /health` - فحص صحة النظام
- `GET /docs` - توثيق API (Swagger)
- `GET /api/v1/status` - حالة النظام
- `POST /api/v1/council/message` - إرسال رسالة للمجلس
- `POST /api/v1/ide/analyze` - تحليل الكود
- `GET /api/v1/erp/invoices` - قائمة الفواتير

### هيكل المشروع

```
bi-ide-v8/
├── api/app.py              # نقطة الدخول الرئيسية
├── core/                   # الوحدات الأساسية
│   ├── logging_config.py  # التسجيل المركزي
│   ├── database.py        # طبقة قاعدة البيانات
│   ├── cache.py           # طبقة التخزين المؤقت
│   └── config.py          # الإعدادات
├── hierarchy/             # النظام الهرمي (10 طبقات)
├── ide/                   # خدمة IDE
├── erp/                   # خدمة ERP
├── ui/                    # واجهة React
├── tests/                 - اختبارات
├── docker-compose.yml     # تنسيق Docker
└── requirements.txt       # تبعيات Python
```

---

## 🔧 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=hierarchy

# Run specific test file
pytest tests/test_api.py -v
```

### Linting & Formatting

```bash
# Format code
black .

# Lint
ruff check .
mypy core/ hierarchy/
```

---

## 🐳 Docker Deployment

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- API: http://localhost:8000
- UI: http://localhost:3000
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090

---

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines.

---

**Built with ❤️ and AI**
