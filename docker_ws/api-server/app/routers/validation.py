# 데이터 검증 라우터
# validation.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from uuid import UUID
from typing import List
from database import get_db
import schemas

router = APIRouter(
    prefix="/validations",
    tags=["DataValidation"],
)

# -------------------------------------------------------------------
# 1️⃣ 데이터 검증 등록 (UPSERT 방식: 이미 있으면 업데이트)
# -------------------------------------------------------------------
@router.post("/", response_model=schemas.DataValidation)
def create_or_update_validation(validation: schemas.DataValidationCreate, db: Session = Depends(get_db)):
    """
    ✅ 환자별(item_id 단위) 검증 데이터 등록 또는 갱신
    """
    query = text("""
        INSERT INTO tb_data_validation (
            patient_id, item_id, validation_method,
            validation_description, validation_datetime
        )
        VALUES (
            :patient_id, :item_id, :validation_method,
            :validation_description, :validation_datetime
        )
        ON CONFLICT (patient_id, item_id)
        DO UPDATE SET
            validation_method = EXCLUDED.validation_method,
            validation_description = EXCLUDED.validation_description,
            validation_datetime = EXCLUDED.validation_datetime
        RETURNING *;
    """)

    result = db.execute(query, validation.dict())
    db.commit()
    return result.fetchone()

# -------------------------------------------------------------------
# 2️⃣ 전체 검증 목록 조회 (필요 시 patient_id, item_id 별 필터링)
# -------------------------------------------------------------------
@router.get("/", response_model=List[schemas.DataValidation])
def read_validations(patient_id: UUID = None, item_id: int = None, db: Session = Depends(get_db)):
    base_query = "SELECT * FROM tb_data_validation WHERE 1=1"
    params = {}

    if patient_id:
        base_query += " AND patient_id = :patient_id"
        params["patient_id"] = str(patient_id)
    if item_id:
        base_query += " AND item_id = :item_id"
        params["item_id"] = item_id

    base_query += " ORDER BY validation_datetime DESC"
    result = db.execute(text(base_query), params)
    return result.fetchall()

# -------------------------------------------------------------------
# 3️⃣ 특정 item_id 검증 조회
# -------------------------------------------------------------------
@router.get("/item/{item_id}", response_model=schemas.DataValidation)
def read_validation_by_item(item_id: int, db: Session = Depends(get_db)):
    query = text("SELECT * FROM tb_data_validation WHERE item_id = :item_id")
    result = db.execute(query, {"item_id": item_id}).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Validation not found for this item")
    return result

# -------------------------------------------------------------------
# 4️⃣ 검증 정보 수정 (validation_id 기준)
# -------------------------------------------------------------------
@router.put("/{validation_id}", response_model=schemas.DataValidation)
def update_validation(validation_id: int, validation: schemas.DataValidationUpdate, db: Session = Depends(get_db)):
    query = text("""
        UPDATE tb_data_validation
        SET
            validation_method = :validation_method,
            validation_description = :validation_description,
            validation_datetime = :validation_datetime
        WHERE validation_id = :validation_id
        RETURNING *;
    """)
    values = validation.dict()
    values["validation_id"] = validation_id
    result = db.execute(query, values)
    db.commit()
    updated = result.fetchone()
    if not updated:
        raise HTTPException(status_code=404, detail="Validation not found")
    return updated

# -------------------------------------------------------------------
# 5️⃣ 검증 데이터 삭제 (validation_id 기준)
# -------------------------------------------------------------------
@router.delete("/{validation_id}")
def delete_validation(validation_id: int, db: Session = Depends(get_db)):
    query = text("""
        DELETE FROM tb_data_validation
        WHERE validation_id = :validation_id
        RETURNING validation_id;
    """)
    result = db.execute(query, {"validation_id": validation_id}).fetchone()
    db.commit()
    if not result:
        raise HTTPException(status_code=404, detail="Validation not found")
    return {"ok": True, "validation_id": validation_id}


# 🧠 1. 파킨슨병 설문지 (MDS-UPDRS Part 3) 신규 데이터 조회
@router.get("/pd-new-items/survey/{patient_id}", response_model=List[schemas.Item])
def read_new_pd_survey_items(patient_id: UUID, db: Session = Depends(get_db)):
    """
    🧠 파킨슨병 설문지(MDS-UPDRS Part 3) 중,
    검증 시점 이후 새로 수집된 item 데이터만 조회
    """
    query = text("""
            SELECT i.*
            FROM tb_items i
            LEFT JOIN tb_data_validation v ON i.item_id = v.item_id
            WHERE i.patient_id = :patient_id
              AND i.is_deleted = FALSE
              AND i.data_category = 'PD'
              AND i.data_type = 'MDS-UPDRS Part 3'
              AND (
                  v.validation_datetime IS NULL
                  OR GREATEST(i.collected_at, i.updated_at) > v.validation_datetime
              )
            ORDER BY GREATEST(i.collected_at, i.updated_at) DESC;
    """)
    rows = db.execute(query, {"patient_id": str(patient_id)}).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="검증 이후 새로 수집된 PD 설문 데이터가 없습니다.")
    return [dict(r._mapping) for r in rows]


# 🎥 2. 파킨슨병 영상(VIDEO) 신규 데이터 조회
# @router.get("/pd-new-items/video/{patient_id}", response_model=List[schemas.Item])
# def read_new_pd_video_items(patient_id: UUID, db: Session = Depends(get_db)):
#     """
#     🎥 파킨슨병 영상(VIDEO) 중,
#     검증 시점 이후 새로 수집된 item 데이터만 조회
#     """
#     query = text("""
#         SELECT i.*
#         FROM tb_items i
#         LEFT JOIN tb_data_validation v
#           ON i.item_id = v.item_id
#         WHERE i.patient_id = :patient_id
#           AND i.is_deleted = FALSE
#           AND i.data_category = 'PD'
#           AND i.data_type = 'VIDEO'              -- ✅ 영상 구분
#           AND (
#               v.validation_datetime IS NULL
#               OR i.collected_at > v.validation_datetime
#           )
#         ORDER BY i.collected_at DESC;
#     """)
#     rows = db.execute(query, {"patient_id": str(patient_id)}).fetchall()
#     if not rows:
#         raise HTTPException(status_code=404, detail="검증 이후 새로 수집된 PD 영상 데이터가 없습니다.")
#     return [dict(r._mapping) for r in rows]
# 🎥 2. 파킨슨병 영상(VIDEO) 신규 데이터 조회 (수정 버전)

@router.get("/pd-new-items/video/{patient_id}", response_model=List[schemas.Item])
def read_new_pd_video_items(patient_id: UUID, db: Session = Depends(get_db)):
    """
    🎥 파킨슨병 영상(VIDEO) 중,
    검증 시점 이후 새로 수집되거나 수정된 item 데이터만 조회
    """
    query = text("""
        SELECT i.*
        FROM tb_items i
        LEFT JOIN tb_data_validation v
          ON i.item_id = v.item_id
        WHERE i.patient_id = :patient_id
          AND i.is_deleted = FALSE
          AND i.data_category = 'PD'
          AND i.data_type = 'VIDEO'
          AND (
              v.validation_datetime IS NULL
              OR GREATEST(i.collected_at, i.updated_at) > v.validation_datetime
          )
        ORDER BY GREATEST(i.collected_at, i.updated_at) DESC;
    """)
    rows = db.execute(query, {"patient_id": str(patient_id)}).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="검증 이후 새로 수집된 PD 영상 데이터가 없습니다.")
    return [dict(r._mapping) for r in rows]



@router.post("/pd-survey-check/{item_id}")
def pd_validate_survey_completeness(item_id: int, db: Session = Depends(get_db)):
    """
    🧠 Parkinson's Disease 설문(item_id) 응답 누락 검증
    tb_questionnaire_answers 기반으로 answer_value 누락 여부 검사
    """
    # ---------------------------
    # 1️⃣ 설문 응답 상태 조회
    # ---------------------------
    query = text("""
        SELECT
            CASE
                WHEN COUNT(*) FILTER (WHERE COALESCE(answer_value, '') = '') > 0
                THEN 'FAIL'
                ELSE 'PASS'
            END AS validation_status,
            COUNT(*) FILTER (WHERE COALESCE(answer_value, '') = '') AS missing_count,
            COUNT(*) AS total_count
        FROM tb_questionnaire_answers
        WHERE item_id = :item_id;
    """)
    result = db.execute(query, {"item_id": item_id}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="해당 item의 설문 응답 데이터가 없습니다.")

    validation_status = result.validation_status
    missing = result.missing_count
    total = result.total_count

    # ---------------------------
    # 2️⃣ 누락 항목 번호 확인
    # ---------------------------
    missing_query = text("""
        SELECT question_id
        FROM tb_questionnaire_answers
        WHERE item_id = :item_id
          AND (answer_value IS NULL OR TRIM(answer_value) = '');
    """)
    missing_rows = db.execute(missing_query, {"item_id": item_id}).fetchall()
    missing_questions = [r.question_id for r in missing_rows]

    # ---------------------------
    # 3️⃣ 검증 결과 설명
    # ---------------------------
    desc = (
        f"총 {total}문항 중 {missing}개 누락됨 (누락 문항: {missing_questions})"
        if missing > 0
        else "모든 문항 응답 완료"
    )

    # ---------------------------
    # 4️⃣ item_id로 patient_id 조회
    # ---------------------------
    patient_row = db.execute(
        text("SELECT patient_id FROM tb_items WHERE item_id = :item_id"),
        {"item_id": item_id}
    ).fetchone()
    if not patient_row:
        raise HTTPException(status_code=404, detail="해당 item에 연결된 환자를 찾을 수 없습니다.")
    patient_id = str(patient_row.patient_id)

    # ---------------------------
    # 5️⃣ tb_data_validation UPSERT 저장
    # ---------------------------
    insert_query = text("""
        INSERT INTO tb_data_validation (
            patient_id, item_id, validation_method,
            validation_description, validation_datetime
        )
        VALUES (
            :patient_id, :item_id, :method,
            :desc, NOW()
        )
        ON CONFLICT (patient_id, item_id)
        DO UPDATE SET
            validation_method = EXCLUDED.validation_method,
            validation_description = EXCLUDED.validation_description,
            validation_datetime = NOW()
        RETURNING *;
    """)

    db.execute(insert_query, {
        "patient_id": patient_id,
        "item_id": item_id,
        "method": "AutoCheck_PD_SURVEY",
        "desc": desc
    })
    db.commit()

    return {
        "item_id": item_id,
        "patient_id": patient_id,
        "status": validation_status,
        "missing_count": missing,
        "total_questions": total,
        "missing_questions": missing_questions,
        "description": desc
    }


@router.post("/pd-stage-calc/{item_id}")
def calculate_parkinson_stage(item_id: int, db: Session = Depends(get_db)):
    """
    🧠 설문(question_id=8)을 기반으로 파킨슨병 중증도 저장
    """
    # 1️⃣ 해당 응답 가져오기
    query = text("""
        SELECT 
            i.patient_id,
            a.answer_value::numeric AS stage_value
        FROM tb_questionnaire_answers a
        JOIN tb_items i ON a.item_id = i.item_id
        WHERE a.item_id = :item_id
          AND a.question_id = 8
          AND TRIM(a.answer_value) != ''
          AND i.data_category = 'PD'
          AND i.data_type = 'MDS-UPDRS Part 3';
    """)
    row = db.execute(query, {"item_id": item_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="중증도 설문 응답 없음")

    stage_val = row.stage_value
    patient_id = row.patient_id

    # 2️⃣ 중증도 설명 자동 해석 (선택 사항)
    def get_stage_description(value: float) -> str:
        if value == 0:
            return "질병의 증후가 없음"
        elif value == 1:
            return "일측성 상하지 장애"
        elif value == 1.5:
            return "일측성 상하지 장애와 체간 장애가 있음"
        elif value == 2:
            return "양측성 장애이나 균형장애는 전혀 없음"
        elif value == 2.5:
            return "양측성 장애이며, 몸을 잡아당기는 검사에서 균형을 잡을 수는 있음"
        elif value == 3:
            return "경도 및 중등도의 양측성 장애, 균형이 불안정, 그러나 독립적인 활동 가능"
        elif value == 4:
            return "걷고 서기는 할 수 있으나 심각한 무능력 상태"
        elif value == 5:
            return "휠체어를 타거나 침대에 누워 있어야만 하는 상태"
        else:
            return "기타 (직접 입력)"

    desc = get_stage_description(stage_val)

    # 3️⃣ INSERT (혹은 UPSERT) 기록
    insert = text("""
        INSERT INTO tb_parkinson_stage (patient_id, item_id, stage_value, stage_description)
        VALUES (:patient_id, :item_id, :val, :desc)
        ON CONFLICT (patient_id, item_id) DO UPDATE SET
        stage_value = EXCLUDED.stage_value,
        stage_description = EXCLUDED.stage_description
        RETURNING *;
    """)
    result = db.execute(insert, {
        "patient_id": patient_id,
        "item_id": item_id,
        "val": stage_val,
        "desc": desc
    })
    db.commit()

    return {
        "patient_id": patient_id,
        "item_id": item_id,
        "stage_value": float(stage_val),
        "stage_description": desc
    }