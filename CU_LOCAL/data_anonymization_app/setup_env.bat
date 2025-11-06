@echo off
setlocal ENABLEDELAYEDEXPANSION
chcp 65001 >nul
title [Data Anonymization App] Environment Setup (Py38 + GPU cu121)

set "LOG=%~dp0setup_env.log"
> "%LOG%" echo ============================================
>>"%LOG%" echo  Setup started: %DATE% %TIME%
>>"%LOG%" echo ============================================

echo ============================================
echo   CHU_LOCAL - Data Anonymization Setup
echo   (Py3.8 + GPU Torch 2.2.2 cu121)
echo   Log: %LOG%
echo ============================================
echo.

REM 1) Python check
where python >nul 2>&1
if errorlevel 1 (
  echo ❌ Python is not installed.
  echo ▶ https://www.python.org/downloads
  >>"%LOG%" echo ERROR: python.exe not found
  pause & exit /b 1
)

REM 2) CD to script dir
cd /d "%~dp0"

REM 3) Create venv if missing
if not exist venv\Scripts\python.exe (
  echo 🧱 Creating virtual environment...
  python -m venv venv >>"%LOG%" 2>&1
  if errorlevel 1 (
    echo ❌ Failed to create venv. See %LOG%
    pause & exit /b 1
  )
) else (
  echo ✅ Virtual environment already exists.
)

REM 4) Activate venv
call venv\Scripts\activate >>"%LOG%" 2>&1
if errorlevel 1 (
  echo ❌ Failed to activate venv. See %LOG%
  pause & exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python -V') do set "VENV_PY=%%v"
echo 🐍 Venv Python: %VENV_PY%
>>"%LOG%" echo Venv Python: %VENV_PY%

REM 5) Upgrade pip toolchain
echo 🚀 Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel >>"%LOG%" 2>&1
if errorlevel 1 (
  echo ❌ pip/setuptools/wheel upgrade failed. See %LOG%
  pause & exit /b 1
)

REM 6) Install base requirements (may pull CPU torch via ultralytics)
if exist requirements.txt (
  echo 📦 Installing dependencies from requirements.txt...
  pip install -r requirements.txt --upgrade >>"%LOG%" 2>&1
  if errorlevel 1 (
    echo ⚠️ Some dependencies failed. See %LOG% (continuing) 
  )
) else (
  echo ⚠️ requirements.txt not found.
  >>"%LOG%" echo WARN: requirements.txt missing
)

REM 7) Ensure GPU torch stack (remove anything pulled by ultralytics)
echo 🧹 Removing existing torch/vision/audio...
python -m pip uninstall -y torch torchvision torchaudio >>"%LOG%" 2>&1

echo ⚙️ Installing GPU Torch 2.2.2 (cu121) stack...
python -m pip install --no-cache-dir ^
  torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 ^
  -f https://download.pytorch.org/whl/torch_stable.html >>"%LOG%" 2>&1

if errorlevel 1 (
  echo 🔁 GPU wheels failed. Trying CPU fallback (2.2.2+cpu)...
  python -m pip install --no-cache-dir ^
    torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu ^
    -f https://download.pytorch.org/whl/torch_stable.html >>"%LOG%" 2>&1
  if errorlevel 1 (
    echo ❌ Torch install failed. See %LOG%
    type "%LOG%" | more
    pause & exit /b 1
  ) else (
    echo ⚠️ Installed CPU torch fallback (no GPU). Check driver/CUDA later.
  )
)

REM 8) Reinstall ultralytics without deps (so it won't drag wrong torch)
echo 🔄 Reinstalling ultralytics --no-deps...
python -m pip install --upgrade --no-deps ultralytics==8.0.196 >>"%LOG%" 2>&1

REM 9) Verify environment
echo 🔎 Verifying torch/cuDNN/GPU...
python - <<PY >>"%LOG%" 2>&1
import sys, torch, importlib, ctypes
print("Python        :", sys.version.split()[0])
print("Torch         :", getattr(torch,'__version__','n/a'))
print("CUDA runtime  :", getattr(getattr(torch,'version',None),'cuda',None))
print("CUDA avail    :", torch.cuda.is_available() if hasattr(torch,'cuda') else None)
try:
    print("GPU count     :", torch.cuda.device_count())
    print("GPU name[0]   :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
except Exception as e:
    print("GPU inspect   :", e)
for m in ("torchvision","torchaudio","ultralytics","opencv_python_headless","numpy","tqdm","cryptography","psycopg2"):
    try:
        mod = importlib.import_module(m.replace('-', '_'))
        print(f"{m:20s}:", getattr(mod,"__version__","OK"))
    except Exception as e:
        print(f"{m:20s}: not found ->", e)
try:
    ctypes.WinDLL("nvcuda.dll")
    print("nvcuda.dll    : OK")
except Exception as e:
    print("nvcuda.dll    : FAIL ->", e)
PY

echo --------------------------------------------
type "%LOG%" | more
echo --------------------------------------------

echo.
echo ✅ Environment setup completed!
echo • If CUDA avail=False, update NVIDIA driver (CUDA 12.1+), ensure 64-bit Python, and check CUDA_VISIBLE_DEVICES.
echo.
pause
endlocal
