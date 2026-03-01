---
description: Deploy BI-IDE v8 to all machines (VPS, RTX 5090, Windows, Domain)
---

# /deploy — نشر التحديثات على جميع الأجهزة

## الأجهزة والمعلومات

### 1. Mac (Development — الجهاز الحالي)
- **المسار:** `/Users/bi/Documents/bi-ide-v8`
- **الدور:** جهاز تطوير رئيسي
- **Git Remote:** `https://github.com/hassanjava2/bi-ide-v8.git`
- **Branch:** `main`

### 2. VPS Server (bi-iq.com)
- **Host:** `root@bi-iq.com`
- **SSH:** يحتاج كلمة مرور (حتى يتم تشغيل `setup_ssh_keys.sh`)
- **المسار على السيرفر:** `/root/bi-ide-v8` (يحتاج تأكيد من المستخدم)
- **الدومين:** `bi-iq.com`
- **الخدمات:** `systemctl restart bi-ide-api` (يحتاج تأكيد)
- **ملفات الويب:** `/var/www/bi-iq.com/` (يحتاج تأكيد)
- **ملاحظة:** GitHub Push Protection تمنع `sk_live_*` — استخدم `sk_test_*` بالاختبارات
- **ملاحظة:** PAT ما عنده `workflow` scope — لا تعدل `.github/workflows/*.yml` بالـ commit

### 3. RTX 5090 (Ubuntu — 192.168.1.164)
- **Host (LAN):** `bi@192.168.1.164`
- **Host (Tailscale):** `bi@100.104.35.44`
- **SSH:** ✅ key-based (يشتغل بدون كلمة مرور)
- **الريبو:** `/home/bi/bi-ide-v8`
- **مجلد العامل:** `/home/bi/.bi-ide-worker`
- **venv:** `/home/bi/.bi-ide-worker/venv`
- **الخدمة:** `bi-ide-worker`
- **sudo:** يحتاج كلمة مرور (حتى يتم إضافة `NOPASSWD` rule)
- **GPU:** NVIDIA RTX 5090 (32GB VRAM)
- **نظام:** Ubuntu 24, 3.7TB NVMe, 5% مستخدم
- **ملاحظة:** المجلد كان manually deployed — الملفات تنسخ من `bi-ide-v8/` → `.bi-ide-worker/`

### 4. Windows (RTX 4050)
- **الوصول:** غير متصل عبر SSH حالياً
- **النشر:** يدوي عبر `git pull` أو PowerShell script `scripts/deploy_windows.ps1`
- **ملاحظة:** يحتاج إعداد SSH أو Remote Desktop

## خطوات النشر

// turbo-all

### الخطوة 1: Commit & Push
```bash
cd /Users/bi/Documents/bi-ide-v8 && git add -A && git status --short | wc -l && git commit -m "deploy: update $(date +%Y-%m-%d)" && git push origin main
```
- إذا ظهر خطأ `GH013 push protection` → ابحث عن `sk_live` وغيّرها لـ `sk_test`
- إذا ظهر خطأ `workflow scope` → أزل `.github/workflows/` من الـ commit: `git checkout HEAD~1 -- .github/workflows/`

### الخطوة 2: Deploy to RTX 5090
```bash
# استخدم Tailscale IP خارج الشبكة المحلية
# RTX_HOST="bi@100.104.35.44"
RTX_HOST="bi@192.168.1.164"
ssh $RTX_HOST "cd /home/bi/bi-ide-v8 && git pull origin main && cp -r worker/* /home/bi/.bi-ide-worker/ 2>/dev/null; cp -r hierarchy/ core/ api/ monitoring/ ai/ requirements.txt /home/bi/.bi-ide-worker/ 2>/dev/null; cd /home/bi/.bi-ide-worker && source venv/bin/activate && pip install -r requirements.txt --quiet 2>&1 | tail -3 && echo '✅ 5090 updated'"
```
- بعدها العامل يحتاج restart: المستخدم يسوي `sudo systemctl restart bi-ide-worker` على الـ 5090

### الخطوة 3: Deploy to VPS
```bash
ssh root@bi-iq.com "cd /root/bi-ide-v8 && git pull origin main && pip3 install -r requirements.txt --quiet && systemctl restart bi-ide-api && echo '✅ VPS updated'"
```
- ملاحظة: يحتاج كلمة مرور حالياً

### الخطوة 4: Health Check
```bash
echo "5090:" && ssh bi@100.104.35.44 "uptime" 2>/dev/null || ssh bi@192.168.1.164 "uptime" && echo "VPS:" && curl -s https://bi-iq.com/health 2>/dev/null || echo "VPS unreachable"
```

## سكربت النشر السريع
```bash
./deploy/auto_deploy.sh              # كل الأجهزة
./deploy/auto_deploy.sh --5090       # 5090 فقط
./deploy/auto_deploy.sh --vps        # VPS فقط
./deploy/auto_deploy.sh --push-only  # git push فقط
```

## إعداد التحديث الذاتي (Self-Update) — مرة وحدة لكل جهاز

### RTX 5090 (Ubuntu)
```bash
ssh bi@192.168.1.164 "cd /home/bi/bi-ide-v8 && bash deploy/setup-auto-update.sh"
```

### VPS (bi-iq.com)
```bash
ssh root@bi-iq.com "cd /root/bi-ide-v8 && bash deploy/setup-auto-update.sh --vps"
```

### Windows (RTX 4050)
```powershell
cd C:\Users\BI\bi-ide-v8
.\deploy\bi-auto-update.ps1 -Install
```

### فحص حالة التحديث الذاتي
```bash
# RTX 5090
ssh bi@192.168.1.164 "bash /home/bi/bi-ide-v8/deploy/bi-auto-update.sh status"

# VPS
ssh root@bi-iq.com "bash /root/bi-ide-v8/deploy/bi-auto-update.sh status"

# Log
ssh bi@192.168.1.164 "tail -20 /var/log/bi-auto-update.log"
```

## إعداد SSH keys (مرة وحدة)
```bash
bash deploy/setup_ssh_keys.sh
```

## ملاحظات مهمة
1. **لا تضيف `sk_live_*`** بأي ملف — GitHub يمنع الـ push
2. **لا تعدل `.github/workflows/`** — PAT ما عنده `workflow` scope
3. **الـ 5090 sudo** يحتاج كلمة مرور — خلي المستخدم يسوي: `sudo bash -c 'echo "bi ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart bi-ide-worker" >> /etc/sudoers.d/bi-ide'`
4. **الـ VPS SSH** يحتاج `ssh-copy-id root@bi-iq.com` مرة وحدة
5. **Windows** — غير متصل SSH حالياً، النشر يدوي
