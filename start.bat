@echo off
chcp 65001 >nul
setlocal

echo ===========================================
echo 🔥 BI IDE - Autonomous Learning System
echo ===========================================
echo.
echo 💡 هذا النظام يتعلم من شغلك أوتماتيكياً
echo    مايحتاج تسوي ملفات تدريب
echo.
echo 📚 كل ما تسوي:
echo    • تكتب كود ← AI يتعلم
echo    • تسوي فاتورة ← AI يتعلم  
echo    • تخطئ و تصحح ← AI يتعلم
echo    • تنفذ أمر ← AI يتعلم
echo.
pause

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

python start.py

pause
