@echo off
title AGI Data Input Program

where python >nul 2>nul || (
    echo Python not found. Please install it first.
    pause
    exit /b
)

if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate


python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --upgrade


python ui\main.py

pause
