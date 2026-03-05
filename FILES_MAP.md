# FILES_MAP.md — فهرس المشروع الشامل
> آخر تحديث: 2026-03-06

---

## البنية الرئيسية

| المسار | النوع | الوصف |
|--------|-------|-------|
| `api/` | مجلد | FastAPI backend + routers |
| `api/app.py` | ملف | نقطة الدخول الرئيسية (FastAPI) |
| `api/routers/` | مجلد | Routers: rtx5090, network, brain, notifications |
| `ai/` | مجلد | AI services + training_data_sync |
| `brain/` | مجلد | الدماغ: bi_brain, scheduler, evaluator, config |
| `hierarchy/` | مجلد | 29+ ملف — البنية الهرمية الكاملة |
| `hierarchy/__init__.py` | ملف | نقطة الدخول + تسجيل كل الطبقات |
| `hierarchy/autonomous_council.py` | ملف | المجلس المستقل 24/7 |
| `hierarchy/real_life_layer.py` | ملف | طبقة الحياة الواقعية |
| `hierarchy/auto_programming.py` | ملف | البرمجة الأوتوماتيكية |
| `core/` | مجلد | database, user_service, auth |
| `services/` | مجلد | ai_service, notification_service, training_service |
| `apps/desktop-tauri/` | مجلد | Tauri v2 Desktop App |
| `apps/desktop-tauri/src/` | مجلد | React frontend (TSX) |
| `apps/desktop-tauri/src-tauri/` | مجلد | Rust backend |
| `rtx4090_machine/` | مجلد | RTX API server + services |
| `rtx4090_machine/rtx_api_server.py` | ملف | RTX API v8.0.3 (port 8090) |
| `training/` | مجلد | LoRA training modules |
| `tests/` | مجلد | 20+ test files, 51 passing |
| `deploy/` | مجلد | deployment scripts |
| `deploy_hostinger.sh` | ملف | VPS deployment (549 lines) |
| `scripts/` | مجلد | utility scripts |
| `docs/` | مجلد | VISION_MASTER, MASTER_PLAN, etc |
| `.agent/rules.md` | ملف | القوانين الإلزامية (418 سطر) |
| `FILES_MAP.md` | ملف | هذا الملف — الفهرس الشامل |

## الأجهزة والمنافذ

| الجهاز | IP | المنفذ | الحالة |
|--------|------|--------|--------|
| Mac | localhost | 5173/8000 | ✅ |
| RTX 5090 | 100.104.35.44 | 8090 | ✅ |
| VPS (bi-iq.com) | 76.13.154.123 | 8010 | ✅ |

## النسخ

| الملف | النسخة |
|-------|--------|
| package.json | 8.0.4 |
| Cargo.toml | 8.0.4 |
| tauri.conf.json | 8.0.4 |
| Title Bar | BI-IDE Desktop v8.0.4 |
