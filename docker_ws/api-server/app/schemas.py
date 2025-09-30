from pydantic import BaseModel
from typing import Optional
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
