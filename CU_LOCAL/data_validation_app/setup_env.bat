@echo off
chcp 65001 >nul
title [Data Validation App] Environment Setup
echo ============================================
echo     CHU_LOCAL - Data Validation Environment
echo ============================================
echo.

REM ✅ 1. Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python is not installed.
    echo ▶ Please install Python first: https://www.python.org/downloads
    pause
    exit /b
)

REM ✅ 2. Move to current directory
cd /d "%~dp0"

REM ✅ 3. Create venv if missing
if not exist venv (
    echo 🧱 Creating virtual environment...
    python -m venv venv
) else (
    echo ✅ Virtual environment already exists.
)

REM ✅ 4. Activate venv
call venv\Scripts\activate

REM ✅ 5. Upgrade pip and tools
echo 🚀 Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel

REM ✅ 6. Install requirements
if exist requirements.txt (
    echo 📦 Installing dependencies from requirements.txt...
    pip install -r requirements.txt --upgrade
) else (
    echo ⚠️ requirements.txt not found.
)

echo.
echo ✅ Environment setup completed successfully!
echo Run "run_app.bat" to start the program.
echo.
pause
