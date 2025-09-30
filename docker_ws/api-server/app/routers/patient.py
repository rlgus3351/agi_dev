from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from uuid import UUID
from typing import List
from database import get_db
import schemas

router = APIRouter(
    prefix="/patients",
    tags=["Patients"],
)

# 환자 등록
@router.post("/", response_model=schemas.Patient)
def create_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    query = text("""
        INSERT INTO tb_patient_info (
            patient_initials, birth_date, institution,
            gender, is_data_complete, completion_date
        )
        VALUES (
            :patient_initials, :birth_date, :institution,
            :gender, :is_data_complete, :completion_date
        )
        RETURNING *;
    """)
    result = db.execute(query, patient.dict())
    db.commit()
    return result.fetchone()

# 전체 환자 목록 (페이징 포함)
@router.get("/", response_model=List[schemas.Patient])
def read_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    query = text("""
        SELECT * FROM tb_patient_info
        ORDER BY created_ts DESC
        LIMIT :limit OFFSET :skip;
    """)
    result = db.execute(query, {"limit": limit, "skip": skip})
    return result.fetchall()

# 특정 환자 조회
@router.get("/{patient_id}", response_model=schemas.Patient)
def read_patient(patient_id: UUID, db: Session = Depends(get_db)):
    query = text("SELECT * FROM tb_patient_info WHERE patient_id = :patient_id")
    result = db.execute(query, {"patient_id": str(patient_id)}).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Patient not found")
    return result

# 환자 수정
@router.put("/{patient_id}", response_model=schemas.Patient)
def update_patient(patient_id: UUID, patient: schemas.PatientUpdate, db: Session = Depends(get_db)):
    query = text("""
        UPDATE tb_patient_info
        SET
            patient_initials = :patient_initials,
            birth_date = :birth_date,
            institution = :institution,
            gender = :gender,
            is_data_complete = :is_data_complete,
            completion_date = :completion_date,
            update_ts = now()
        WHERE patient_id = :patient_id
        RETURNING *;
    """)
    values = patient.dict()
    values["patient_id"] = str(patient_id)
    result = db.execute(query, values)
    db.commit()
    updated = result.fetchone()
    if not updated:
        raise HTTPException(status_code=404, detail="Patient not found")
    return updated

# 환자 삭제
@router.delete("/{patient_id}")
def delete_patient(patient_id: UUID, db: Session = Depends(get_db)):
    query = text("DELETE FROM tb_patient_info WHERE patient_id = :patient_id RETURNING *;")
    result = db.execute(query, {"patient_id": str(patient_id)}).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Patient not found")
    db.commit()
    return {"ok": True}
