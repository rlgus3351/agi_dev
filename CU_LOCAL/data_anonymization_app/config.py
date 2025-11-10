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

# DB_HOST = "121.178.59.41"     # 로컬 DB 서버
# DB_PORT = 45432            # 기본 포트
# DB_NAME = "agi_dev"     # 실제 DB 이름
# DB_USER = "kkh"      # DB 사용자명
# DB_PASSWORD = "Rkskekfk1!"  # PostgreSQL 비밀번호

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
INSTITUTION = "CU"
DEBUG_MODE = True
LOG_FILE = "app.log"

# ============================================================
# 📁 로컬 파일 경로 설정
# ============================================================

OUTPUT_BASE = r"C:\Users\user\Desktop\DEV_AGI\MDD\output"

# (이미 설문 JSON 번들에 쓰던 경로가 있다면 그대로 유지하거나 OUTPUT_BASE 아래로 두어도 됨)
LOCAL_JSON_DIR = os.path.join(OUTPUT_BASE, "json")

# 디렉토리 자동 생성
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(LOCAL_JSON_DIR, exist_ok=True)

print(f"📁 OUTPUT_BASE={OUTPUT_BASE}")
print(f"📁 LOCAL_JSON_DIR={LOCAL_JSON_DIR}")
