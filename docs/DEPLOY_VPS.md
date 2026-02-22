# Deploy BI-IDE on VPS (Online Control)

هذا الدليل ينشر السيرفر أونلاين مع HTTPS ويخلي أجهزة الـworkers تتصل من أي مكان.

## المتطلبات
- VPS Ubuntu 22.04 أو أحدث
- Domain يشير إلى IP السيرفر
- مستخدم لديه `sudo`

## 1) رفع المشروع
```bash
git clone <your-repo-url> bi-ide-v8
cd bi-ide-v8
```

## 2) نشر تلقائي كامل (Nginx + HTTPS + systemd)
```bash
chmod +x deploy/vps/install.sh
./deploy/vps/install.sh "$PWD" your-domain.com YOUR_STRONG_TOKEN admin@your-domain.com
```

السكربت ينفذ:
- إنشاء virtualenv وتثبيت المتطلبات
- إنشاء ملف `.env` وتفعيل `ORCHESTRATOR_TOKEN`
- إنشاء خدمة `bi-ide-api` عبر systemd
- إعداد Nginx Reverse Proxy
- إصدار شهادة SSL عبر Certbot
- فتح الجدار الناري المناسب

## 3) التحقق
```bash
curl https://your-domain.com/health
curl https://your-domain.com/api/v1/orchestrator/health
sudo systemctl status bi-ide-api
```

## 4) ربط أي جهاز Worker
- Windows:
```powershell
iwr https://your-domain.com/api/v1/orchestrator/download/windows -OutFile install_windows.ps1
./install_windows.ps1 -ServerUrl "https://your-domain.com" -Token "YOUR_STRONG_TOKEN" -Labels "gpu,rtx4090"
```

- Linux:
```bash
curl -fsSL https://your-domain.com/api/v1/orchestrator/download/linux -o install_linux.sh
chmod +x install_linux.sh
./install_linux.sh https://your-domain.com YOUR_STRONG_TOKEN worker-linux "gpu"
```

- macOS:
```bash
curl -fsSL https://your-domain.com/api/v1/orchestrator/download/macos -o install_macos.sh
chmod +x install_macos.sh
./install_macos.sh https://your-domain.com YOUR_STRONG_TOKEN worker-mac "cpu"
```

## 5) إنشاء Job تدريب
```bash
curl -X POST https://your-domain.com/api/v1/orchestrator/jobs \
  -H "Content-Type: application/json" \
  -H "X-Orchestrator-Token: YOUR_STRONG_TOKEN" \
  -d '{
    "name": "Training Job",
    "command": "python rtx4090_machine/rtx4090_server.py",
    "shell": true,
    "target_labels": ["gpu"]
  }'
```

## 6) متابعة الحالة
```bash
curl -H "X-Orchestrator-Token: YOUR_STRONG_TOKEN" \
  https://your-domain.com/api/v1/orchestrator/workers

curl -H "X-Orchestrator-Token: YOUR_STRONG_TOKEN" \
  https://your-domain.com/api/v1/orchestrator/jobs
```
