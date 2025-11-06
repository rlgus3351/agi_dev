@echo off
chcp 65001 >nul
title [Data Anonymization App] Launcher
echo ============================================
echo     CU_LOCAL - Data Anonymization Program
echo ============================================
echo.

REM ✅ Move to current directory
cd /d "%~dp0"

REM ✅ Check virtual environment
if not exist venv (
    echo ❌ Virtual environment not found.
    echo ▶ Please run "setup_env.bat" first.
    pause
    exit /b
)

REM ✅ Activate virtual environment
call venv\Scripts\activate

REM ✅ Run main program
if exist main.py (
    echo 🎬 Launching data anonymization process...
    python main.py
) else (
    echo ❌ main.py not found.
)

echo.
pause
