"""
health_check.py
----------------------------------------
로컬 환경에서 DB 연결 상태를 확인하는 모듈
(인터넷, 외부 서버 불필요)
----------------------------------------
"""
from sqlalchemy import text
from utils.db_utils import get_connection, release_connection   # 혹은 get_cursor() 사용 가능
from sqlalchemy.orm import Session

# ============================================================
# ✅ 1️⃣ DB 헬스체크 함수 (psycopg2 직접)
# ============================================================
def local_db_check_psycopg2():
    """로컬 DB 연결 상태 확인 (psycopg2 기반)"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        cur.close()
        release_connection(conn)
        if result and result[0] == 1:
            return {"status": "ok", "message": "Local DB connection is healthy"}
        else:
            return {"status": "fail", "message": "DB query returned invalid result"}
    except Exception as e:
        return {"status": "fail", "message": str(e)}

# ============================================================
# ✅ 2️⃣ DB 헬스체크 함수 (SQLAlchemy 기반)
# ============================================================

def local_db_check_sqlalchemy(db: Session):
    """SQLAlchemy Session을 통한 DB 상태 확인"""
    try:
        result = db.execute(text("SELECT 1")).fetchone()
        if result and result[0] == 1:
            return {"status": "ok", "message": "Local DB connection is healthy"}
        else:
            return {"status": "fail", "message": "DB returned unexpected result"}
    except Exception as e:
        return {"status": "fail", "message": str(e)}
