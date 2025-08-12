@echo off
REM ----------------------------
REM Gradio 앱 실행 배치 파일
REM ----------------------------

REM 가상환경 활성화
call C:\Users\user\Desktop\frameSecurity\venv\Scripts\activate.bat

REM Gradio 앱 실행 (자동으로 브라우저 열림)
python C:\Users\user\Desktop\frameSecurity\src\PD\gradio_app.py

REM 명령창 유지
pause
