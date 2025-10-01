from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db  # 본인 프로젝트에 맞게 import 수정
from sqlalchemy import text

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)

# API 통신 확인용
@router.get("")
def health_check():
    return {"status": "ok", "message": "API server is running"}


# API <-> DB 통신 확인용
@router.get("/db")
def db_health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "message": "Database connection is healthy"}
    except Exception as e:
        return {"status": "fail", "message": str(e)}
