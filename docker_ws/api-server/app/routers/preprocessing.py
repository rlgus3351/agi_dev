from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from uuid import UUID
from typing import List
from database import get_db
import schemas

router = APIRouter(
    prefix="/processing",
    tags=["Processing"],
)


@router.get("/next", response_model=schemas.VideoMeta)
def get_next_video_to_process(db: Session = Depends(get_db)):
    """
    처리 대기 중인 영상 중 가장 오래된 1건을 반환
    조건:
    - needs_anonymization = true
    - is_anonymized is null 또는 false
    - data_category = 'PD'
    """
    query = text("""
        SELECT v.*
        FROM tb_video_metadata v
        WHERE v.needs_anonymization = true
          AND (v.is_anonymized IS NULL OR v.is_anonymized = false)
          AND v.data_category = 'PD'
        ORDER BY v.created_ts ASC
        LIMIT 1
    """)

    result = db.execute(query).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="처리할 영상이 없습니다.")

    return dict(result._mapping)
