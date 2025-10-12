# routers/mds_forms_api.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
from database import get_db
import schemas 

router = APIRouter(
    prefix="/mds",
    tags=["MDS Forms Answers"],
)

# -------------------------------------------------------------
# 1. 특정 item_id에 대한 설문 응답 전체 조회
# -------------------------------------------------------------
@router.get("/{item_id}", response_model=List[schemas.MDSForm])
def read_mds_answers_by_item_id(item_id: int, db: Session = Depends(get_db)):
    """
    특정 수집 항목(item_id)에 등록된 모든 MDS 설문 응답을 조회합니다.
    """
    query = text("""
        SELECT 
            answer_id, item_id, question_id, answer_component, answer_value, submission_datetime
        FROM tb_Questionnaire_Answers
        WHERE item_id = :item_id
        ORDER BY question_id, answer_component
    """)
    result = db.execute(query, {"item_id": item_id})
    rows = result.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="해당 item_id에 대한 MDS 설문 응답이 없습니다.")

    # 조회 스키마(schemas.MDSForm)에 맞춰 데이터를 반환합니다.
    return [dict(row._mapping) for row in rows]


# -------------------------------------------------------------
# 2. 설문 응답 다중 등록 (하나의 설문지 제출)
# -------------------------------------------------------------
@router.post("/{item_id}", status_code=status.HTTP_201_CREATED)
def create_mds_form_answers(
    item_id: int, 
    request: schemas.MDSFormsCreate, 
    db: Session = Depends(get_db)
):
    """
    특정 수집 항목(item_id)에 대해 여러 MDS 질문 응답을 한 번에 등록합니다.
    """
    if not request.answers:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="응답(answers) 리스트가 비어 있습니다.")

    query = text("""
        INSERT INTO tb_Questionnaire_Answers 
        (item_id, question_id, answer_component, answer_value)
        VALUES (:item_id, :question_id, :answer_component, :answer_value)
    """)

    try:
        # 데이터베이스에 효율적으로 다중 삽입하기 위한 파라미터 리스트 생성
        params_list = []
        for answer in request.answers:
            params_list.append({
                "item_id": item_id,
                "question_id": answer.question_id,
                "answer_component": answer.answer_component,
                "answer_value": answer.answer_value,
            })
        
        db.execute(query, params_list) 
        db.commit()
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"DB 오류: {str(e)}")

    return JSONResponse(
        status_code=status.HTTP_201_CREATED, 
        content={"message": f"{len(request.answers)}개의 MDS 설문 응답이 item_id={item_id}에 성공적으로 등록되었습니다."}
    )


@router.put("/answers", status_code=status.HTTP_200_OK)
def update_answer_values(
    request: schemas.MDSAnswerValueUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    answer_id 기준으로 answer_value만 수정하는 경량 업데이트 API
    """
    if not request.answers:
        raise HTTPException(status_code=400, detail="수정할 응답이 없습니다.")

    query = text("""
        UPDATE tb_Questionnaire_Answers
        SET answer_value = :answer_value
        WHERE answer_id = :answer_id
    """)

    try:
        params = [{"answer_id": ans.answer_id, "answer_value": ans.answer_value} for ans in request.answers]
        db.execute(query, params)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"업데이트 실패: {str(e)}")

    return {"message": f"{len(request.answers)}개 항목이 성공적으로 수정되었습니다."}
