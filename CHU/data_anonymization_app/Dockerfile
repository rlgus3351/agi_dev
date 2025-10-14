# ✅ data_anonymization_app/Dockerfile
FROM python:3.8-slim

# Debian 미러를 한국 서버로 변경 (sources.list 없을 경우 새로 생성)
RUN echo "deb http://ftp.kr.debian.org/debian bookworm main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://ftp.kr.debian.org/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://ftp.kr.debian.org/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --fix-missing \
        ffmpeg \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["python", "main.py"]