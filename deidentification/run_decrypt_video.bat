@echo off
chcp 65001 >nul
title [Decrypt Video Batch] Launcher
echo ============================================
echo     WiseHeal AGI - Face ROI 복호화 배치
echo ============================================
echo.

REM ✅ 1. Move to current directory (script folder)
cd /d "%~dp0"

REM ✅ 2. Check if virtual environment exists
if not exist venv (
    echo ❌ Virtual environment not found.
    echo ▶ Please run "setup_env.bat" first to create venv.
    pause
    exit /b
)

REM ✅ 3. Activate virtual environment
echo 🔹 Activating virtual environment...
call venv\Scripts\activate

REM ✅ 4. Check decrypt_video.py existence
if not exist decrypt_video.py (
    echo ❌ decrypt_video.py not found in current directory.
    pause
    exit /b
)

REM ✅ 5. Run Python script
echo.
echo 🚀 Starting decryption batch...
python decrypt_video.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Python script encountered an error. Exit code: %errorlevel%
    echo 🔍 Please check the console log above for details.
    pause
    exit /b
)

REM ✅ 6. Finished
echo.
echo ============================================
echo ✅ All processes completed successfully.
echo ============================================
pause
