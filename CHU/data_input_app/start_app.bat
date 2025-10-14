@echo off
title AGI Data Input Program
echo ================================
echo   AIHOPS Data Input Program
echo ================================
echo.

REM 가상환경이 없으면 자동 생성
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM 가상환경 활성화
call venv\Scripts\activate

REM 필요한 패키지 설치
echo Installing dependencies...
pip install --upgrade pip >nul
pip install -r requirements.txt >nul

REM 프로그램 실행
echo Launching program...
python ui\main.py

REM 프로그램 종료 대기
pause
