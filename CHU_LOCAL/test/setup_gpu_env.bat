@echo off
chcp 65001 >nul
title [AIHOPS GPU Environment Setup]
echo ============================================
echo   Setting up YOLO GPU Environment
echo ============================================
echo.

REM 1️⃣ 가상환경 생성
if not exist venv_gpu (
    echo 🧱 Creating GPU virtual environment...
    python -m venv venv_gpu
) else (
    echo ✅ GPU venv already exists.
)

REM 2️⃣ 가상환경 활성화
call venv_gpu\Scripts\activate

REM 3️⃣ pip 및 setuptools 업그레이드
echo 🚀 Upgrading pip, setuptools, and wheel...
python.exe -m pip install --upgrade pip setuptools wheel

REM 4️⃣ requirements.txt 패키지 설치
if exist requirements.txt (
    echo 📦 Installing dependencies from requirements.txt ...
    python.exe -m pip install -r requirements.txt
) else (
    echo ⚠️ requirements.txt not found. Please create one before running this script.
    pause
    exit /b
)

REM 5️⃣ GPU 사용 가능 여부 테스트
echo 🔍 Checking CUDA availability...
python - <<END
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA GPU not detected. CPU mode will be used.")
END

REM 6️⃣ YOLO 테스트 (모델 자동 다운로드)
echo 🎯 Testing YOLO on GPU ...
python - <<END
from ultralytics import YOLO
import torch
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
try:
    model = YOLO('yolov8n.pt')
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print('✅ YOLO model loaded successfully.')
except Exception as e:
    print('❌ YOLO test failed:', e)
END

echo.
echo ============================================
echo ✅ GPU environment setup complete!
echo ============================================
pause
