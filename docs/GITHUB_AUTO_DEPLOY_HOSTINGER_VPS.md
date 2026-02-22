# GitHub Auto-Deploy to Hostinger VPS (bi-iq.com)

هذا الدليل يخلي أي `push` على فرع `main` يحدّث السيرفر تلقائياً.

## 0) المتطلبات

- لازم يكون عندك **Hostinger VPS** (مو Shared Hosting).
- الدومين `bi-iq.com` لازم يشير إلى IP السيرفر.
- الخدمة `bi-ide-api` موجودة (عن طريق `deploy/vps/install.sh`).

---

## 1) إعداد DNS للدومين

داخل Hostinger DNS:

- `A` record:
  - Name: `@`
  - Value: `VPS_PUBLIC_IP`
- `A` record:
  - Name: `www`
  - Value: `VPS_PUBLIC_IP`

انتظر انتشار DNS (غالباً دقائق إلى ساعة).

---

## 2) تجهيز مستخدم السيرفر للـdeploy

نفّذ على VPS (كمستخدم عادي، مو root):

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

أضف public key الخاص بك إلى `~/.ssh/authorized_keys`.

للسماح بإعادة تشغيل الخدمة من GitHub Actions بدون كلمة مرور:

```bash
sudo visudo
```

وأضف السطر التالي (بدّل `youruser`):

```text
youruser ALL=(ALL) NOPASSWD:/bin/systemctl restart bi-ide-api,/bin/systemctl status bi-ide-api
```

---

## 3) إعداد GitHub Secrets

في GitHub repo:
`Settings` → `Secrets and variables` → `Actions` → `New repository secret`

أضف القيم التالية:

- `VPS_HOST` = IP السيرفر
- `VPS_PORT` = `22`
- `VPS_USER` = اسم المستخدم على السيرفر
- `VPS_SSH_KEY` = المفتاح الخاص (private key) بصيغة OpenSSH
- `VPS_APP_DIR` = مسار المشروع على السيرفر (مثال: `/home/youruser/bi-ide-v8`)
- `VPS_REPO_URL` = `https://github.com/hassanjava2/bi-ide-v8.git`
- `VPS_BRANCH` = `main`

> ملاحظة: ملف workflow المستخدم هو:
> `.github/workflows/deploy-vps.yml`

---

## 4) تفعيل أول Deploy

بعد إضافة الـSecrets:

1. ادخل تبويب `Actions` في GitHub.
2. اختَر workflow: **Deploy to Hostinger VPS**.
3. شغله يدويًا (Run workflow) مرة أولى.
4. بعد نجاحه، أي `git push origin main` يصير deploy تلقائي.

---

## 5) فحص النجاح

على السيرفر:

```bash
sudo systemctl status bi-ide-api
curl -I https://bi-iq.com/health
curl -I https://bi-iq.com/api/v1/orchestrator/health
```

لازم ترجع استجابة `200` أو `307/308` متبوعة بـ `200`.

---

## 6) تسلسل العمل اليومي

من جهازك:

```bash
git add .
git commit -m "update"
git push origin main
```

GitHub Actions راح:

1. يتصل بالسيرفر عبر SSH.
2. يسوي `git fetch/reset` على آخر نسخة من `main`.
3. يثبت dependencies.
4. يشغل migrations (إذا موجودة).
5. يعيد تشغيل `bi-ide-api`.
