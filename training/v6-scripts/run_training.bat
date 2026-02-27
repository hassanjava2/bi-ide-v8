@echo off
chcp 65001 >nul
title Bi IDE - AI Training System

echo.
echo ================================================================
echo              Bi IDE - AI Training System
echo ================================================================
echo.

:: Check for Python
py --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.11 from:
    echo    https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
py --version
echo.

:: Show options
echo ================================================================
echo Select Training Mode:
echo ================================================================
echo.
echo   [1] Prepare Data Only (Fast - Recommended)
echo   [2] Full Training (Requires ML libraries)
echo   [3] Install Requirements then Train
echo   [4] Exit
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto prepare
if "%choice%"=="2" goto full
if "%choice%"=="3" goto install_all
if "%choice%"=="4" goto end

:prepare
echo.
echo ================================================================
echo Preparing Training Data...
echo ================================================================
echo.
py train_ai.py --mode prepare
goto done

:full
echo.
echo ================================================================
echo Starting Full Training...
echo ================================================================
echo.
py train_ai.py --mode full --verbose
goto done

:install_all
echo.
echo ================================================================
echo Installing Requirements...
echo ================================================================
echo.
py -m pip install -r requirements.txt
echo.
echo [OK] Requirements installed. Starting training...
echo.
py train_ai.py --mode full --verbose
goto done

:done
echo.
echo ================================================================
echo Training Complete!
echo ================================================================
echo.

:end
pause
