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
        JOIN tb_data_validation d
          ON v.item_id = d.item_id
        WHERE v.needs_anonymization = true
          AND (v.is_anonymized IS NULL OR v.is_anonymized = false)
          AND v.data_category = 'PD'
          AND v.validation_description LIKE '%PASS%'
        ORDER BY v.created_ts ASC
        LIMIT 1;
    """)

    result = db.execute(query).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="처리할 영상이 없습니다.")

    return dict(result._mapping)

@router.put("/update")
def update_processed_video(video_meta: schemas.DataProcessingVideoMetaUpdate, db: Session = Depends(get_db)):
    """
    처리 완료 영상 업데이트
    """
    query = text("""
        UPDATE tb_video_metadata
        SET is_anonymized = :is_anonymized,
            anonymized_ts = :anonymized_ts
        WHERE video_metadata_id = :video_metadata_id
    """)

    result = db.execute(query, {
        "is_anonymized": video_meta.is_anonymized,
        "anonymized_ts": video_meta.anonymized_ts,
        "video_metadata_id": video_meta.video_metadata_id
    })

    db.commit()  # ✅ 꼭 커밋 필요

    if result.rowcount == 0:  # ✅ 영향을 받은 행이 없을 때
        raise HTTPException(status_code=404, detail="처리할 영상이 없습니다.")

    return {"message": "업데이트 완료", "video_metadata_id": video_meta.video_metadata_id}

@router.post("/", summary="비식별화 처리 결과 저장")
def insert_preprocessing_record(payload: schemas.PreprocessingCreate, db: Session = Depends(get_db)):
    query = text("""
        INSERT INTO tb_data_preprocessing (
            item_id, data_category, original_file_path,
            json_file_path, encrypted_file_path,
            processing_started_at, processing_ended_at,
            processing_duration_sec, total_frames,
            encrypted_frames, detected_face_frames, success_rate,
            preprocessing_type, description
        )
        VALUES (
            :item_id, :data_category, :original_file_path,
            :json_file_path, :encrypted_file_path,
            :processing_started_at, :processing_ended_at,
            :processing_duration_sec, :total_frames,
            :encrypted_frames, :detected_face_frames, :success_rate,
            :preprocessing_type, :description
        )
    """)
    db.execute(query, payload.dict())
    db.commit()

    return {"ok": True, "message": "✅ 전처리 결과 저장 완료"}