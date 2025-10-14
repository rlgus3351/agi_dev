# routers/items_api.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from uuid import UUID
from typing import List
from database import get_db
import schemas


router = APIRouter(
    prefix="/items",
    tags=["Items"],
)


# 특정 환자의 수집 항목 목록 조회
@router.get("/{patient_id}", response_model=List[schemas.Item])
def read_items(patient_id: UUID, db: Session = Depends(get_db)):
    query = text("""
        SELECT * FROM tb_items
        WHERE patient_id = :patient_id
          AND is_deleted = FALSE              -- ✅ 소프트 삭제된 항목 제외
        ORDER BY data_category, data_type, seq
    """)
    result = db.execute(query, {"patient_id": str(patient_id)})
    rows = result.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="해당 환자의 수집 항목이 없습니다.")

    return [dict(row._mapping) for row in rows]

# 환자 수집 항목 등록
# 환자 수집 항목 단건 등록
@router.post("/{patient_id}/item", response_model=schemas.Item)
def create_item(patient_id: UUID, item: schemas.ItemCreate, db: Session = Depends(get_db)):
    query = text("""
        INSERT INTO tb_items (patient_id, data_category, data_type, seq, description, collected_at)
        VALUES (:patient_id, :data_category, :data_type, :seq, :description, NOW())
        RETURNING item_id, patient_id, data_category, data_type, seq, description, collected_at, is_deleted, deleted_at
    """)
    result = db.execute(query, {
        "patient_id": str(patient_id),
        "data_category": item.data_category,
        "data_type": item.data_type,
        "seq": item.seq,
        "description": item.description,
    })
    row = result.fetchone()
    db.commit()

    if not row:
        raise HTTPException(status_code=400, detail="항목 등록 실패")
    return dict(row._mapping)


# 환자 수집 항목 다중 등록

@router.post("/{patient_id}/items", status_code=204)
def create_items(patient_id: UUID, request: schemas.ItemsCreate, db: Session = Depends(get_db)):
    if not request.items:
        raise HTTPException(status_code=400, detail="items 리스트가 비어 있습니다.")

    query = text("""
        INSERT INTO tb_items (patient_id, data_category, data_type, seq, description, collected_at)
        VALUES (:patient_id, :data_category, :data_type, :seq, :description, NOW())
    """)

    try:
        for item in request.items:
            db.execute(query, {
                "patient_id": str(patient_id),
                "data_category": item.data_category,
                "data_type": item.data_type,
                "seq": item.seq,
                "description": item.description,
            })
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")

    return JSONResponse(status_code=204, content=None)

# 환자 수집 항목 소프트 삭제
@router.delete("/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    query = text("""
        UPDATE tb_items
        SET is_deleted = TRUE, deleted_at = NOW()
        WHERE item_id = :item_id
        RETURNING item_id
    """)
    result = db.execute(query, {"item_id": item_id}).fetchone()
    db.commit()

    if not result:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True, "item_id": result.item_id}


@router.put("/{item_id}/mark-updated")
def mark_item_as_updated(item_id: int, db: Session = Depends(get_db)):
    db.execute(text("""
        UPDATE tb_items
        SET is_updated = TRUE,
            updated_at = NOW()
        WHERE item_id = :item_id
    """), {"item_id": str(item_id)})
    db.commit()
    return {"ok": True}


@router.get("/by-id/{item_id}", response_model=schemas.Item)
def get_item_by_id(item_id: int, db: Session = Depends(get_db)):
    """
    특정 item_id로 단일 아이템 조회
    """
    query = text("""
        SELECT * FROM tb_items
        WHERE item_id = :item_id
        LIMIT 1
    """)
    result = db.execute(query, {"item_id": item_id}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Item not found")

    row = dict(result._mapping)

    # ✅ UUID 타입을 문자열로 변환
    if "patient_id" in row and not isinstance(row["patient_id"], str):
        row["patient_id"] = str(row["patient_id"])

    return row