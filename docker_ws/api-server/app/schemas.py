from pydantic import BaseModel, Field
from typing import Optional,List
from datetime import date, datetime
from uuid import UUID
from typing import Union

# ▶ 등록용
class PatientCreate(BaseModel):
    patient_initials: Optional[str]
    birth_date: Optional[date]
    institution: Optional[str]
    gender: Optional[str]
    is_data_complete: Optional[bool] = False
    completion_date: Optional[datetime]

# ▶ 조회용
class Patient(PatientCreate):
    patient_id: UUID
    display_id: Optional[str]
    created_ts: Optional[datetime]
    update_ts: Optional[datetime]

    class Config:
        orm_mode = True

# ▶ 수정용 (모든 필드 optional)
class PatientUpdate(BaseModel):
    patient_initials: Optional[str]
    birth_date: Optional[date]
    institution: Optional[str]
    gender: Optional[str]
    is_data_complete: Optional[bool]
    completion_date: Optional[datetime]


# 수집 항목 등록용
class ItemCreate(BaseModel):
    data_category: Optional[str]
    data_type: Optional[str]
    seq: Optional[int]
    description: Optional[str]

# 수집 항목 조회용
class Item(BaseModel):
    item_id: Optional[int]
    patient_id: UUID
    data_category: Optional[str]
    data_type: Optional[str]
    seq: Optional[int]
    description: Optional[str]
    collected_at: Optional[datetime]
    is_deleted: Optional[bool] = False
    deleted_at: Optional[datetime] = None
    is_updated: Optional[bool] = None
    updated_at: Optional[datetime] = None
    class Config:
        orm_mode = True
        
# 여러 항목 등록용 DTO
class ItemsCreate(BaseModel):
    items: List[ItemCreate]

# 수집 항목 수정용
class ItemUpdate(BaseModel):
    item_id: Optional[int]
    data_category: Optional[str]
    data_type: Optional[str]
    seq: Optional[int]
    description: Optional[str]
    collected_at: Optional[datetime]
    is_deleted: Optional[bool]           # ✅ 필요 시 상태 업데이트 가능
    deleted_at: Optional[datetime]


#  _   _  _____ ______  _____  _____ ___  ___ _____  _____   ___  
# | | | ||_   _||  _  \|  ___||  _  ||  \/  ||  ___||_   _| / _ \ 
# | | | |  | |  | | | || |__  | | | || .  . || |__    | |  / /_\ \
# | | | |  | |  | | | ||  __| | | | || |\/| ||  __|   | |  |  _  |
# \ \_/ / _| |_ | |/ / | |___ \ \_/ /| |  | || |___   | |  | | | |
#  \___/  \___/ |___/  \____/  \___/ \_|  |_/\____/   \_/  \_| |_/
                                                                
                                                            
# ▶ 단일 비디오 메타데이터 등록용
class VideoMetaCreate(BaseModel):
    item_id: Optional[int]                 # 상위 tb_items 외래키
    file_name: Optional[str]               # 실제 파일명
    file_path: Optional[str]               # 저장 경로
    file_ext: Optional[str]                # 파일 확장자
    file_size_mb: Optional[float]          # 파일 크기 (MB)
    duration_seconds: Optional[int]        # 길이 (초)
    resolution: Optional[str]              # 해상도 (예: '1920x1080')
    frame_rate: Optional[int]              # FPS
    needs_anonymization: Optional[bool]
    shooting_ts: Optional[datetime]
    data_category: Optional[str]

# ▶ 비디오 메타데이터 조회용
class VideoMeta(BaseModel):
    video_metadata_id: int
    item_id: int
    file_name: Optional[str]
    file_path: Optional[str]
    file_ext: Optional[str]
    file_size_mb: Optional[float]
    duration_seconds: Optional[int]
    resolution: Optional[str]
    frame_rate: Optional[int]
    needs_anonymization: Optional[bool]
    is_anonymized: Optional[bool]
    created_ts: Optional[datetime]
    shooting_ts: Optional[datetime]
    anonymized_ts: Optional[datetime]
    data_category: Optional[str]


    class Config:
        orm_mode = True


# ▶ 여러 영상 등록용 (배치)
class VideoMetasCreate(BaseModel):
    videos: List[VideoMetaCreate]

# ▶ 영상 메타데이터 수정용
class VideoMetaUpdate(BaseModel):
    video_metadata_id: int # 👈 이 필드를 추가해야 함
    file_name: Optional[str]              # 파일명
    file_path: Optional[str]             # 파일 경로
    file_size_mb: Optional[float]        # 파일 크기 (MB 단위)
    file_ext: Optional[str]              # 파일 확장자
    duration_seconds: Optional[int]     # 영상 길이 (초)
    resolution: Optional[str]           # 해상도 (예: 1920x1080)
    frame_rate: Optional[int]            # 프레임 레이트 (fps)
    needs_anonymization: Optional[bool]
    shooting_ts: Optional[datetime]

class DataProcessingVideoMetaUpdate(BaseModel):
    video_metadata_id: int # 👈 이 필드를 추가해야 함
    needs_anonymization: Optional[bool]
    is_anonymized: Optional[bool]
    anonymized_ts: Optional[datetime]

class MDSFormCreate(BaseModel):
    # item_id: Optional[int] # 경로나 통합 제출에서 처리되므로 제거
    question_id: int # Optional이 아닌 필수 항목으로 변경
    answer_component: Optional[str] = None # grouped-inputs와 같은 세부 항목
    answer_value: Union[str,int] # 응답 값

# 📋 MDS 설문 응답 조회 스키마 (응답으로 반환)
class MDSForm(BaseModel):
    # db 테이블 컬럼명에 맞춰 answer_id로 변경
    answer_id: int 
    item_id: int
    question_id: int
    answer_component: Optional[str]
    answer_value: str
    submission_datetime: datetime # 테이블 컬럼명에 맞춤

    class Config:
        orm_mode = True
        
# 📦 다중 등록 요청을 위한 컨테이너 스키마
class MDSFormsCreate(BaseModel):
    answers: List[MDSFormCreate]

class MDSAnswerValueUpdate(BaseModel):
    answer_id: int
    answer_value: Union[str, int]  # 수정할 값만 받음

class MDSAnswerValueUpdateRequest(BaseModel):
    answers: List[MDSAnswerValueUpdate]


class DataValidationCreate(BaseModel):
    patient_id: Optional[UUID]
    item_id: int
    validation_method: Optional[str]
    validation_description: Optional[str]
    validation_datetime: Optional[datetime]

# ▶ 조회용
class DataValidation(DataValidationCreate):
    validation_id: int

    class Config:
        orm_mode = True

# ▶ 수정용
class DataValidationUpdate(BaseModel):
    validation_method: Optional[str]
    validation_description: Optional[str]
    validation_datetime: Optional[datetime]