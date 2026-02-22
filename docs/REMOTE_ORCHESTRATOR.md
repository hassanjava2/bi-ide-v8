# Remote Orchestrator (Online Control)

هذا الدليل يحول المشروع إلى **نظام تحكم مركزي أونلاين**:
- سيرفر واحد (Public URL) يدير المهام
- أي جهاز (Windows / Linux / macOS) ينزل Agent
- Agent يسجل نفسه تلقائياً ويسحب مهام التدريب ويشغلها محلياً

---

## 1) تشغيل السيرفر المركزي

على الجهاز/السيرفر الرئيسي:

```bash
# Windows PowerShell
$env:ORCHESTRATOR_TOKEN="CHANGE_ME_STRONG_TOKEN"
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

ارفعه أونلاين (VPS أو Cloud VM) وخليه خلف HTTPS (Nginx/Caddy أو Cloudflare Tunnel).

---

## 2) تنزيل وتثبيت الأداة على أي جهاز

> استبدل `https://your-server.com` بالرابط الفعلي.

### Windows

```powershell
iwr https://your-server.com/api/v1/orchestrator/download/windows -OutFile install_windows.ps1
./install_windows.ps1 -ServerUrl "https://your-server.com" -Token "CHANGE_ME_STRONG_TOKEN" -Labels "gpu,rtx4090"
```

### Linux

```bash
curl -fsSL https://your-server.com/api/v1/orchestrator/download/linux -o install_linux.sh
chmod +x install_linux.sh
./install_linux.sh https://your-server.com CHANGE_ME_STRONG_TOKEN worker-linux "gpu"
```

### macOS

```bash
curl -fsSL https://your-server.com/api/v1/orchestrator/download/macos -o install_macos.sh
chmod +x install_macos.sh
./install_macos.sh https://your-server.com CHANGE_ME_STRONG_TOKEN worker-mac "cpu"
```

بعد التثبيت، الـAgent يبقى شغال Auto-start ويطلب مهام من السيرفر.

> ملاحظة: الـAgent الآن يرفع حالة الـJob والـlogs بشكل دوري أثناء التشغيل (near real-time) وليس فقط عند النهاية.
> ومهم: بعد انتهاء كل Job (حتى لو فشل/توقف) الـAgent يرفع checkpoints/ملفات التدريب تلقائياً إلى السيرفر المركزي.

---

## 3) إنشاء مهمة تدريب من أي مكان

### مثال (PowerShell)

```powershell
$headers = @{ "X-Orchestrator-Token" = "CHANGE_ME_STRONG_TOKEN"; "Content-Type" = "application/json" }
$body = @{
  name = "RTX4090 Training Job"
  command = "python rtx4090_machine/rtx4090_server.py"
  shell = $true
  target_labels = @("gpu")
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://your-server.com/api/v1/orchestrator/jobs" -Method POST -Headers $headers -Body $body
```

### متابعة الحالة

```bash
curl -H "X-Orchestrator-Token: CHANGE_ME_STRONG_TOKEN" \
  https://your-server.com/api/v1/orchestrator/jobs
```

### تشغيل 20 حاسبة بنفس الوقت

- ثبت الـAgent على كل الحواسب (20 جهاز)
- تأكد أنها طالعة في `/api/v1/orchestrator/workers`
- من صفحة الموبايل `.../api/v1/orchestrator/mobile`:
  - حدد `Labels` مثلاً `gpu`
  - حدد `عدد المهام = 20`
  - اضغط إنشاء Job

> النظام يوزع الـJobs على العمال المتاحين بنفس الـlabels بشكل تلقائي.

---

## 4) أهم endpoints

- `GET /api/v1/orchestrator/health`
- `GET /api/v1/orchestrator/workers`
- `POST /api/v1/orchestrator/jobs`
- `POST /api/v1/orchestrator/jobs/batch`
- `GET /api/v1/orchestrator/jobs`
- `GET /api/v1/orchestrator/jobs/{job_id}`
- `POST /api/v1/orchestrator/jobs/{job_id}/cancel`
- `POST /api/v1/orchestrator/jobs/{job_id}/artifacts/upload`
- `GET /api/v1/orchestrator/jobs/{job_id}/artifacts`
- `GET /api/v1/orchestrator/jobs/{job_id}/artifacts/{artifact_id}/download`
- `GET /api/v1/orchestrator/mobile` (لوحة موبايل للمراقبة + دردشة المجلس)

---

## 6) المراقبة من التلفون + التحدث مع الطبقات

افتح من الموبايل:

```
https://your-server.com/api/v1/orchestrator/mobile
```

- أدخل `ORCHESTRATOR_TOKEN` داخل الصفحة لمراقبة workers/jobs
- تقدر ترسل رسالة مباشرة للمجلس من نفس الصفحة (يروح إلى `/api/v1/council/message`)
- تقدر تنشئ Job تدريب مباشرة من نفس الصفحة (بدون curl/PowerShell)
- تقدر توقف Job شغّال (يرسل Stop Signal للـWorker) أو تسوي Restart لنفس الـJob من نفس الصفحة
- الصفحة ترسل تنبيه متصفح + صوت عند تغير حالة Job إلى `completed` أو `failed` أو `stopping`

---

## 5) ملاحظات أمان ضرورية

- فعّل `ORCHESTRATOR_TOKEN` دائماً على السيرفر الأونلاين.
- لا تفتح السيرفر بدون HTTPS.
- خليه خلف Reverse Proxy مع rate-limit.
- استخدم firewall وقصر المنافذ المفتوحة.
