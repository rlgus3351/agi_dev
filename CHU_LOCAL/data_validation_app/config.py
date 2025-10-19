"""
config.py
----------------------------------------
로컬 환경용 PostgreSQL 설정 파일
(인터넷 없이도 완전 독립 동작)
----------------------------------------
"""

import os

# ============================================================
# 🗄️ PostgreSQL DB 설정
# ============================================================

DB_HOST = "127.0.0.1"        # 로컬 DB 서버
DB_PORT = 5432               # 기본 포트
DB_NAME = "agi_dev"          # 실제 DB 이름
DB_USER = "postgres"         # DB 사용자명
DB_PASSWORD = "rkskekfk1!"   # PostgreSQL 비밀번호

# ✅ SQLAlchemy / psycopg2 연결용 URL
DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?options=-csearch_path%3Ddev_kkh"
)

DB_CONFIG = {
    "host": DB_HOST,
    "port": DB_PORT,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
}

# ============================================================
# ⚙️ 로컬 앱 기본 설정
# ============================================================

INSTITUTION = "CNU"
DEBUG_MODE = True
LOG_FILE = "app.log"

# ============================================================
# 📁 로컬 파일 경로 설정
# ============================================================

# ✅ 비디오 기본 저장 경로
VIDEO_SAVE_BASE = r"C:\Users\user\Desktop\DEV_AGI\parkinson\video"

# ✅ 업로드 및 출력 경로
LOCAL_UPLOAD_DIR = r"C:\Users\user\Desktop\DEV_AGI\parkinson\output\video"

# ✅ 비디오 비식별화 / 검증 경로
WINDOW_PREFIX = LOCAL_UPLOAD_DIR.lower()
CONTAINER_PREFIX = "/app/input_videos"

# 디렉토리 자동 생성 (없으면 생성)
os.makedirs(LOCAL_UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEO_SAVE_BASE, exist_ok=True)

# ============================================================
# 🧩 환경 확인 로그
# ============================================================

print("✅ CONFIG 로드 완료 — 로컬 PostgreSQL 환경")
print(f"🗄️ DB: {DB_NAME} ({DB_HOST}:{DB_PORT}, schema=dev_kkh)")
print(f"📁 VIDEO_SAVE_BASE={VIDEO_SAVE_BASE}")
print(f"📁 LOCAL_UPLOAD_DIR={LOCAL_UPLOAD_DIR}")
