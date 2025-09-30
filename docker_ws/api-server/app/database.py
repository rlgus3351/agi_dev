import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 환경 변수에서 DB 접속 정보 불러오기
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("POSTGRES_DB")
DB_SCHEMA = os.getenv("POSTGRES_SCHEMA", "public")

# SQLAlchemy 연결 문자열
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?options=-csearch_path%3D{DB_SCHEMA}"

# SQLAlchemy 엔진 및 세션 설정
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# FastAPI Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
