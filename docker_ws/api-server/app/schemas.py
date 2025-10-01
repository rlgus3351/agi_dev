from pydantic import BaseModel, Field
from typing import Optional,List
from datetime import date, datetime
from uuid import UUID

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
    patient_id: UUID = Field(..., example="46cd05ef-bc85-4a7f-8432-6827f706708b")
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