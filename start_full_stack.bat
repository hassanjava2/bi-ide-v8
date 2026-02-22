@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "PYTHONIOENCODING=utf-8"
if not defined AI_CORE_HOST set "AI_CORE_HOST=192.168.68.111"
if not defined AI_CORE_PORT set "AI_CORE_PORT=8080"
if not defined AI_CORE_PORTS set "AI_CORE_PORTS=8080,9090"
if not defined AUTO_SYNC_CHECKPOINTS set "AUTO_SYNC_CHECKPOINTS=1"
if not defined AUTO_SYNC_INTERVAL_SEC set "AUTO_SYNC_INTERVAL_SEC=60"
if not defined MIN_CHECKPOINT_SIZE_MB set "MIN_CHECKPOINT_SIZE_MB=1"

echo ===========================================
echo 🚀 BI-IDE Full Stack Launcher
echo ===========================================
echo.
echo سيبدأ الآن:
echo   1) API على http://localhost:8000
echo   2) UI على http://localhost:3000 (أو منفذ بديل إذا مستخدم)
echo   3) Checkpoint Auto-Sync من RTX4090 كل %AUTO_SYNC_INTERVAL_SEC% ثانية
echo.

set "PY_CMD=python"
if exist "%ROOT%venv\Scripts\python.exe" set "PY_CMD=%ROOT%venv\Scripts\python.exe"

start "BI-IDE API" cmd /k "cd /d "%ROOT%" && "%PY_CMD%" -m api.app"
start "BI-IDE UI" cmd /k "cd /d "%ROOT%ui" && npm run dev"

echo ✅ تم فتح نافذتين للتشغيل.
echo إذا أول مرة تشغّل الواجهة نفّذ داخل ui: npm install
echo.
pause
