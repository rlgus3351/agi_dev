from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from uuid import UUID
from typing import List
from database import get_db
import schemas

router = APIRouter(
    prefix="/meta",
    tags=["Meta"],
)

# 전체 비디오 데이터 목록
@router.get("/", response_model=List[schemas.VideoMeta])
def read_patients(db: Session = Depends(get_db)):
    query = text("""
        SELECT * FROM tb_video_metadata
        ORDER BY created_ts DESC;
    """)
    result = db.execute(query)
    return result.fetchall()
