FROM python:3.8-slim

# ✅ 필요한 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y \
    python3-tk \
    libgl1-mesa-glx \        
    libglib2.0-0 \           
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "ui.main"]
