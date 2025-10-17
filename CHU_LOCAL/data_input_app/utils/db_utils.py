import psycopg2
from psycopg2 import pool
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# ============================================================
# 🗄️ DB 연결 풀(Connection Pool) 초기화
# ============================================================

db_pool = None

def init_db_pool():
    """PostgreSQL 연결 풀 생성 (앱 시작 시 1회만 호출)"""
    global db_pool
    if db_pool is None:
        db_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("✅ DB 연결 풀 초기화 완료")

def get_connection():
    """연결 풀에서 연결 가져오기"""
    if db_pool is None:
        raise Exception("❌ DB 연결 풀이 초기화되지 않았습니다. 먼저 init_db_pool()을 호출하세요.")
    return db_pool.getconn()

def release_connection(conn):
    """연결 반환"""
    if db_pool:
        db_pool.putconn(conn)

def close_all_connections():
    """모든 연결 종료 (앱 종료 시 호출)"""
    if db_pool:
        db_pool.closeall()
        print("🔒 모든 DB 연결 종료 완료")
        
