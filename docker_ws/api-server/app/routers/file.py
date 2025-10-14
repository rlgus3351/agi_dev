# routers/video_api.py

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
from database import get_db
import schemas
import os

router = APIRouter(
    prefix="/video",
    tags=["Video Metadata"],
)


# -------------------------------------------------------------
# 1. 특정 item_id에 대한 비디오 메타데이터 전체 조회
# -------------------------------------------------------------
@router.get("/{item_id}", response_model=List[schemas.VideoMeta])
def read_videos_by_item_id(item_id: int, db: Session = Depends(get_db)):
    """
    특정 수집 항목(item_id)에 연결된 모든 비디오 메타데이터를 조회합니다.
    """
    query = text("""
        SELECT 
            video_metadata_id, item_id, file_name, file_path, file_ext,
            file_size_mb, duration_seconds, resolution, frame_rate,
            needs_anonymization,is_anonymized, created_ts, shooting_ts,anonymized_ts, data_category
        FROM tb_Video_Metadata
        WHERE item_id = :item_id
    """)

    result = db.execute(query, {"item_id": item_id}).fetchall()

    if not result:
        raise HTTPException(status_code=404, detail="해당 item_id에 대한 비디오 메타데이터가 없습니다.")

    return [dict(row._mapping) for row in result]


# -------------------------------------------------------------
# 2. 비디오 메타데이터 다중 등록
# -------------------------------------------------------------
@router.post("/{item_id}", status_code=status.HTTP_201_CREATED)
def create_video_metadata(
    item_id: int,
    request: schemas.VideoMetasCreate,
    db: Session = Depends(get_db)
):
    """
    특정 item_id에 대해 여러 비디오 메타데이터를 한 번에 등록합니다.
    (파일 업로드는 별도 엔드포인트에서 수행)
    """
    if not request.videos:
        raise HTTPException(status_code=400, detail="videos 리스트가 비어 있습니다.")

    query = text("""
        INSERT INTO tb_Video_Metadata (
            item_id,  file_name, file_path, file_size_mb, file_ext,
            duration_seconds, resolution, frame_rate,
            needs_anonymization, shooting_ts, data_category
        ) VALUES (
            :item_id, :file_name, :file_path, :file_size_mb, :file_ext,
            :duration_seconds, :resolution, :frame_rate,
            :needs_anonymization, :shooting_ts, :data_category
        )
    """)

    try:
        params_list = []
        for video in request.videos:
            params_list.append({
                "item_id": item_id,
                "file_name": video.file_name,
                "file_path": video.file_path,
                "file_size_mb": video.file_size_mb,
                "duration_seconds": video.duration_seconds,
                "resolution": video.resolution,
                "frame_rate": video.frame_rate,
                "file_ext": video.file_ext,
                "needs_anonymization": video.needs_anonymization,
                "shooting_ts": video.shooting_ts,
                "data_category": video.data_category
            })

        db.execute(query, params_list)
        db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB 삽입 오류: {str(e)}")

    return JSONResponse(
        status_code=201,
        content={"message": f"{len(request.videos)}개의 비디오 메타데이터가 등록되었습니다."}
    )


# -------------------------------------------------------------
# 3. 비디오 메타데이터 개별 수정 (is_anonymized 등)
# -------------------------------------------------------------
@router.put("/update", status_code=status.HTTP_200_OK)
def update_video_metadata(
    request: List[schemas.VideoMetaUpdate],
    db: Session = Depends(get_db)
):
    """
    비디오 메타데이터의 일부 필드 (예: is_anonymized, note 등) 수정
    """
    if not request:
        raise HTTPException(status_code=400, detail="수정할 데이터가 없습니다.")

    query = text("""
        UPDATE tb_Video_Metadata
        SET 
            file_name = COALESCE(:file_name, file_name),
            file_path = COALESCE(:file_path, file_path),
            file_size_mb = COALESCE(:file_size_mb, file_size_mb),
            duration_seconds = COALESCE(:duration_seconds, duration_seconds),
            resolution = COALESCE(:resolution, resolution),
            frame_rate = COALESCE(:frame_rate, frame_rate),
            needs_anonymization = COALESCE(:needs_anonymization, needs_anonymization),
            file_ext = COALESCE(:file_ext, file_ext),
            shooting_ts = COALESCE(:shooting_ts, shooting_ts)
        WHERE video_metadata_id = :video_metadata_id
    """)

    try:
        params_list = []
        for idx, video in enumerate(request):
            if not hasattr(video, "video_metadata_id"):
                raise HTTPException(status_code=400, detail=f"{idx+1}번째 요청에 video_metadata_id가 없습니다.")
            params = video.dict()
            params_list.append(params)

        db.execute(query, params_list)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"업데이트 실패: {str(e)}")

    return {"message": f"{len(request)}개의 비디오 메타데이터가 수정되었습니다."}



# -------------------------------------------------------------
# 4. 단일 파일 업로드 (옵션)
# -------------------------------------------------------------
# @router.post("/upload/{patient_id}")
# async def upload_video_file(
#     patient_id: str,
#     file: UploadFile = File(...),
# ):
#     """
#     실제 비디오 파일 업로드용 (물리적 저장)
#     """
#     upload_dir = f"./uploads/{patient_id}"
#     os.makedirs(upload_dir, exist_ok=True)
#     file_path = os.path.join(upload_dir, file.filename)

#     with open(file_path, "wb") as buffer:
#         buffer.write(await file.read())

#     file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)

#     return {
#         "message": "업로드 완료",
#         "file_name": file.filename,
#         "file_path": file_path,
#         "file_size_mb": file_size_mb
#     }

# -------------------------------------------------------------
# 5. 비식별화 처리 후 상태 업데이트 전용
# -------------------------------------------------------------
@router.put("/anonymization/update", status_code=status.HTTP_200_OK)
def update_video_anonymization_status(
    request: List[schemas.DataProcessingVideoMetaUpdate],
    db: Session = Depends(get_db)
):
    """
    비식별화 처리 후 needs_anonymization, is_anonymized, anonymized_ts 필드만 업데이트합니다.
    """
    if not request:
        raise HTTPException(status_code=400, detail="수정할 데이터가 없습니다.")

    query = text("""
        UPDATE tb_video_metadata
        SET 
            needs_anonymization = COALESCE(:needs_anonymization, needs_anonymization),
            is_anonymized = COALESCE(:is_anonymized, is_anonymized),
            anonymized_ts = COALESCE(:anonymized_ts, anonymized_ts)
        WHERE video_metadata_id = :video_metadata_id
    """)

    try:
        params_list = [video.dict() for video in request]
        db.execute(query, params_list)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"업데이트 실패: {str(e)}")

    return {
        "message": f"{len(request)}개의 비식별화 상태가 업데이트되었습니다."
    }