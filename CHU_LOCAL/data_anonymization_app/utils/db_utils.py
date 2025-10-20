import psycopg2
from psycopg2 import pool
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# ============================================================
# 🗄️ PostgreSQL 연결 풀 (Connection Pool)
# ============================================================

db_pool = None

def init_db_pool():
    """PostgreSQL 연결 풀 초기화 (앱 시작 시 1회만 호출)"""
    global db_pool
    if db_pool is None:
        db_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            options='-c search_path=dev_kkh'  # ✅ 스키마 지정
        )
        print("✅ DB 연결 풀 초기화 완료 (스키마: dev_kkh)")

def get_connection():
    """연결 풀에서 커넥션 1개 가져오기"""
    if db_pool is None:
        raise Exception("❌ DB 연결 풀이 초기화되지 않았습니다. init_db_pool()을 먼저 호출하세요.")
    return db_pool.getconn()

def release_connection(conn):
    """커넥션 반환"""
    if db_pool:
        db_pool.putconn(conn)

def close_all_connections():
    """앱 종료 시 모든 연결 닫기"""
    if db_pool:
        db_pool.closeall()
        print("🔒 모든 DB 연결 종료 완료")

# ✅ 앱 실행 시 자동 초기화
init_db_pool()
