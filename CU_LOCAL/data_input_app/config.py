"""
config.py
----------------------------------------
로컬 환경용 PostgreSQL 설정 파일
(인터넷 없이도 동작)
----------------------------------------
"""

# ============================================================
# 🗄️ PostgreSQL DB 설정
# ============================================================

# DB_HOST = "127.0.0.1"     # 로컬 DB 서버
# DB_PORT = 5432            # 기본 포트
# DB_NAME = "agi_dev"     # 실제 DB 이름
# DB_USER = "postgres"      # DB 사용자명
# DB_PASSWORD = "rkskekfk1!"  # PostgreSQL 비밀번호

DB_HOST = "121.178.59.41"     # 로컬 DB 서버
DB_PORT = 45432            # 기본 포트
DB_NAME = "agi_dev"     # 실제 DB 이름
DB_USER = "kkh"      # DB 사용자명
DB_PASSWORD = "Rkskekfk1!"  # PostgreSQL 비밀번호


# SQLAlchemy URL
DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?options=-csearch_path%3Ddev_kkh"
)


# ============================================================
# ⚙️ 로컬 앱 설정
# ============================================================

INSTITUTION = "CU"
DEBUG_MODE = True
LOG_FILE = "app.log"

# 로컬 경로
LOCAL_UPLOAD_DIR = r"C:\Users\user\Desktop\DEV_AGI\parkinson\output\video"
VIDEO_SAVE_BASE = r"C:\Users\user\Desktop\DEV_AGI\parkinson\video"