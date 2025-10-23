@echo off
chcp 65001 >nul
title [Data Input App] Launcher
echo ============================================
echo     WiseHeal CHU - Data Input Application
echo ============================================
echo.

REM ✅ 1. Move to current directory
cd /d "%~dp0"

REM ✅ 2. Check if virtual environment exists
if not exist venv (
    echo ❌ Virtual environment not found.
    echo ▶ Please run "setup_env.bat" first to set up the environment.
    pause
    exit /b
)

REM ✅ 3. Activate virtual environment
call venv\Scripts\activate

REM ✅ 4. Run main program
if exist ui\main.py (
    echo 🎬 Launching Data Input UI...
    python ui\main.py
) else (
    echo ❌ ui\main.py not found.
)

echo.
echo ============================================
echo ✅ Program closed.
echo ============================================
pause
