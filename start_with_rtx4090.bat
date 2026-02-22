@echo off
chcp 65001 >nul
setlocal

echo ===========================================
echo 🔥 BI IDE + RTX 4090 Remote Training
echo ===========================================
echo.
echo 📡 الاتصال بـ AI Core Server:
echo    Host: 192.168.68.125
echo    Port: 8080
echo.
echo 💡 يتعلم AI على الـ 4090 بشكل منفصل
echo 🌐 Orchestrator API: Enabled
echo ===========================================
echo.

rem تعديل IP حسب حاسبتك
set AI_CORE_HOST=192.168.68.125
set AI_CORE_PORT=8080
set AUTO_SYNC_CHECKPOINTS=1
set AUTO_SYNC_INTERVAL_SEC=60
set MIN_CHECKPOINT_SIZE_MB=1
set ORCHESTRATOR_HEARTBEAT_TIMEOUT_SEC=45
if "%ORCHESTRATOR_TOKEN%"=="" set ORCHESTRATOR_TOKEN=CHANGE_ME_STRONG_TOKEN
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"

echo 🔗 Connecting to RTX 4090 Server...
echo 🔐 Orchestrator token is set (change ORCHESTRATOR_TOKEN for production)
python -m api.app

pause
