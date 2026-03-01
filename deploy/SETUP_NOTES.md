# 📋 BI-IDE v8 — ملاحظات الإعداد والنشر
> آخر تحديث: 2026-03-01

---

## 🌐 Tailscale — الاتصال عن بعد

### الحالة:
| الجهاز | Tailscale | IP | ملاحظات |
|--------|-----------|-----|---------|
| **RTX 5090** | ✅ مفعّل | `100.104.35.44` | شغّال ومتصل |
| **Mac** | ⚡ مثبت | — | يحتاج "Allow System Extension" + تسجيل دخول |
| **Windows** | ✅ مفعّل | `100.78.169.110` | شغّال ومتصل |

### الحساب:
- **Email:** `alshrefihassan@gmail.com`
- **Tailnet:** `alshrefihassan@gmail.com`

### 🔑 Auth Key (لأجهزة جديدة — ما يحتاج براوزر):
```
tskey-auth-kxKSpafJg311CNTRL-nMvrngeSFd9jLBcfg8oic95mFbxcV2iB8
```
- **ينتهي:** May 30, 2026
- **نوع:** Reusable ♻️
- **استخدم هيجي:**
  ```bash
  sudo tailscale up --authkey tskey-auth-kxKSpafJg311CNTRL-nMvrngeSFd9jLBcfg8oic95mFbxcV2iB8
  ```

### إكمال إعداد Mac:
1. اضغط **"Install Now"** بنافذة "Allow System Extension"
2. يفتح macOS **System Settings → Privacy & Security**
3. اضغط **"Allow"** بأسفل الصفحة
4. بعدها اضغط أيقونة Tailscale بالـ menu bar ← **Log in**
5. سجل بحساب `alshrefihassan@gmail.com`

### إعداد Windows:
1. حمّل: https://tailscale.com/download/windows
2. ثبّت
3. سجل دخول بـ `alshrefihassan@gmail.com`

### بعد ما الماك يتصل:
```bash
# اتصل بالـ 5090 من أي مكان:
ssh bi@100.104.35.44

# النشر:
./deploy/auto_deploy.sh --5090
```

---

## 🚀 النشر (Deployment)

### الأجهزة:
| الجهاز | Host (LAN) | Host (Tailscale) | SSH |
|--------|-----------|------------------|-----|
| **Mac** | localhost | — | — |
| **RTX 5090** | `bi@192.168.1.164` | `bi@100.104.35.44` | ✅ key |
| **VPS** | `root@bi-iq.com` | — | 🔐 password |
| **Windows** | — | — | ❌ manual |

### سكربتات النشر:
```bash
./deploy/auto_deploy.sh              # كل الأجهزة
./deploy/auto_deploy.sh --5090       # 5090 فقط
./deploy/auto_deploy.sh --vps        # VPS فقط
./deploy/auto_deploy.sh --push-only  # git push فقط
```

### إعداد النشر التلقائي (مرة وحدة):
```bash
bash deploy/setup_ssh_keys.sh
```

---

## 🔐 كلمات المرور
| الجهاز | المستخدم | كلمة المرور |
|--------|----------|-------------|
| RTX 5090 | bi | 353631 |
| Mac | bi | 353631 |
| Windows | bi | 353631 |
| VPS | root | يحتاج ssh-copy-id |

---

## 📡 API

### الحالة (على الماك):
- **URL:** `http://localhost:8000`
- **Version:** 8.1.0
- **Routes:** 187
- **Docs:** `http://localhost:8000/docs`
- **Database:** SQLite at `data/bi_ide.db`

### تشغيل API:
```bash
cd /Users/bi/Documents/bi-ide-v8
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Endpoints:
- `/health` — فحص صحة السيرفر
- `/ready` — فحص جاهزية الخدمات
- `/docs` — Swagger UI
- `/api/v1/auth/login` — تسجيل دخول
- `/api/v1/council/query` — مجلس الحكماء

---

## 🛠 إصلاحات اليوم (2026-03-01):
1. ✅ `api/schemas.py` — حذف `from __future__ import annotations` + إضافة `Dict, List` imports + `model_rebuild()` لـ 53 class
2. ✅ `monitoring/metrics_exporter.py` — نقل import `asynccontextmanager` للأعلى
3. ✅ `tests/test_security.py` — إصلاح regex patterns + GitHub push protection
4. ✅ `_run_tests.py` — cross-platform runner
5. ✅ Deploy scripts: `auto_deploy.sh`, `setup_ssh_keys.sh`, `post-push.sh`
6. ✅ Tailscale على 5090
7. ✅ Deploy workflow: `.agent/workflows/deploy.md`

---

## 📊 الاختبارات:
- **Passed:** 401
- **Failed:** 39 (معظمها تحتاج database/GPU)
- **Skipped:** 31
- **Errors:** 78 (بيئة/hardware)

---

## 🔗 الروابط:
- **GitHub:** https://github.com/hassanjava2/bi-ide-v8
- **Domain:** https://bi-iq.com
- **App:** https://app.bi-iq.com
- **Community:** https://biiraq.com
- **Tailscale Admin:** https://login.tailscale.com/admin
