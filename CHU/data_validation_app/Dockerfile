FROM python:3.8-slim

# ✅ ffmpeg(ffprobe) + 시스템 기본 툴 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 🔹 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 🔹 앱 코드 복사
COPY . /app


# ✅ 기본 실행 명령
CMD ["python", "main.py"]
