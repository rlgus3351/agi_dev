@echo off
chcp 65001 >nul
title [Data Validation App] Launcher
echo ============================================
echo     CHU_LOCAL - Data Validation Application
echo ============================================
echo.

REM ✅ Move to current directory
cd /d "%~dp0"

REM ✅ Check virtual environment
if not exist venv_gpu (
    echo ❌ Virtual environment not found.
    echo ▶ Please run "setup_env.bat" first.
    pause
    exit /b
)

REM ✅ Activate virtual environment
call venv_gpu\Scripts\activate

REM ✅ Run main.py
if exist main.py (
    echo 🎬 Launching data validation program...
    python main.py
) else (
    echo ❌ main.py not found.
)

echo.
pause
