@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ===== Encoding setup (avoid garbling) =====
REM If you will save this file as UTF-8 without BOM, keeping chcp 65001 is fine.
chcp 65001 >nul

title Torch 2.5.1 + Ultralytics Installer
echo ============================================
echo  Torch 2.5.1 + Ultralytics Installer
echo  (CUDA 12.1 auto/manual selection)
echo ============================================

REM 0) Move to script directory
cd /d "%~dp0"

REM 1) Create/activate venv
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating venv...
    py -3 -m venv venv 2>nul || python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv. Check Python installation.
        pause & exit /b 1
    )
) else (
    echo [INFO] Using existing venv.
)

echo [INFO] Activating venv...
call "venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate venv.
    pause & exit /b 1
)

REM 2) Upgrade pip toolchain
echo.
echo [STEP] Upgrading pip/setuptools/wheel
python -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
    echo [ERROR] Failed to upgrade pip/setuptools/wheel
    pause & exit /b 1
)

REM 3) Choose GPU/CPU mode (auto-detect + manual confirm)
set "TORCH_VER=2.5.1"
set "VISION_VER=0.20.1"
set "AUDIO_VER=2.5.1"
set "CUDA_TAG=cu121"
set "WHL_CUDA_URL=https://download.pytorch.org/whl/%CUDA_TAG%"
set "WHL_CPU_URL=https://download.pytorch.org/whl/cpu"

echo.
echo [STEP] Checking NVIDIA GPU (nvidia-smi)...
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
    echo   - GPU detected.
    set "AUTO_HAVE_GPU=1"
) else (
    echo   - GPU not detected or driver missing.
    set "AUTO_HAVE_GPU=0"
)

echo.
echo [SELECT] Install mode:
echo   1) GPU (CUDA 12.1)
echo   2) CPU only
if "%AUTO_HAVE_GPU%"=="1" (
    set "DEFAULT_CHOICE=1"
) else (
    set "DEFAULT_CHOICE=2"
)
set /p USER_CHOICE="Enter number (default: %DEFAULT_CHOICE%): "
if "%USER_CHOICE%"=="" set "USER_CHOICE=%DEFAULT_CHOICE%"
if "%USER_CHOICE%"=="1" ( set "USE_GPU=1" ) else ( set "USE_GPU=0" )

REM 4) Uninstall existing torch/vision/audio/ultralytics
echo.
echo [STEP] Uninstalling existing torch/vision/audio/ultralytics (if any)
python -m pip uninstall -y torch torchvision torchaudio ultralytics

REM 5) Install Torch (GPU or CPU)
echo.
if "%USE_GPU%"=="1" (
    echo [STEP] Installing Torch %TORCH_VER% (CUDA 12.1)
    python -m pip install --index-url %WHL_CUDA_URL% ^
      torch==%TORCH_VER% torchvision==%VISION_VER% torchaudio==%AUDIO_VER%
) else (
    echo [STEP] Installing Torch %TORCH_VER% (CPU)
    python -m pip install --index-url %WHL_CPU_URL% ^
      torch==%TORCH_VER% torchvision==%VISION_VER% torchaudio==%AUDIO_VER%
)
if %errorlevel% neq 0 (
    echo [ERROR] Torch install failed.
    pause & exit /b 1
)

REM 6) Install Ultralytics and deps
echo.
echo [STEP] Installing Ultralytics and dependencies
python -m pip install -U ultralytics opencv-python tqdm cryptography
REM Non-fatal warnings may occur, continue anyway.

REM 7) Verify
echo.
echo [CHECK] Version info
python - <<PYEND
import sys, torch, torchvision
print("Python         :", sys.version.split()[0])
print("Torch          :", torch.__version__)
print("Torch CUDA     :", torch.version.cuda)
print("CUDA Available :", torch.cuda.is_available())
print("torchvision    :", torchvision.__version__)
PYEND

python - <<PYEND
import ultralytics
print("Ultralytics    :", ultralytics.__version__)
PYEND

echo.
echo Done.
echo ============================================
pause
endlocal
