from typing import Union, Tuple, List
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback

# ============================================================
# 🧩 MDS 질문 매핑 (로컬 DB용)
# ============================================================
MDS_QUESTION_MAPPING = {
    # DB 삽입 순서: 1~8번 (기초 정보)
    "a": 1, "b": 2, "c": 3, "c1": 4, "d": 5, "d1": 6, "d2": 7, "e": 8,
    # DB 삽입 순서: 9~26번 (운동 항목별 평가)
    "1": 9, "2": 10, "3": 11, "4": 12, "5": 13, "6": 14, "7": 15, "8": 16,
    "9": 17, "10": 18, "11": 19, "12": 20, "13": 21, "14": 22, "15": 23,
    "16": 24, "17": 25, "18": 26,
}


# ============================================================
# 1️⃣ 수집 항목 등록 (기존: /items/{patient_id}/item)
# ============================================================
def create_new_item_and_get_id(target_patient_id: str) -> Union[int, None]:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO dev_kkh.tb_items (
                    patient_id, data_category, data_type, seq, description, collected_at
                )
                VALUES (%s, %s, %s, %s, %s, NOW())
                RETURNING item_id;
            """, (
                target_patient_id,
                "PD",
                "MDS-UPDRS Part 3",
                1,
                "MDS-UPDRS Part 3 설문 응답"
            ))
            item = cur.fetchone()
            conn.commit()
            return item["item_id"] if item else None

    except Exception as e:
        print(f"❌ 수집 항목 등록 실패: {e}")
        traceback.print_exc()
        return None
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 2️⃣ 설문 응답 등록 (기존: /mds/{item_id})
# ============================================================
def save_mds_answers(item_id: int, answers_list: list) -> Tuple[bool, Union[str, None]]:
    """
    answers_list = [
        {"question_id": 1, "answer_component": "a", "answer_value": "5"},
        {"question_id": 2, "answer_component": None, "answer_value": "2"},
    ]
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            for ans in answers_list:
                cur.execute("""
                    INSERT INTO dev_kkh.tb_Questionnaire_Answers
                    (item_id, question_id, answer_component, answer_value)
                    VALUES (%s, %s, %s, %s);
                """, (
                    item_id,
                    ans.get("question_id"),
                    ans.get("answer_component"),
                    ans.get("answer_value"),
                ))
            conn.commit()

        return True, None

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ 설문 응답 등록 실패: {e}")
        traceback.print_exc()
        return False, str(e)

    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 3️⃣ 설문 응답 수정 (기존: /mds/answers PUT)
# ============================================================
def update_mds_answers(answers_list: list) -> Tuple[bool, Union[str, None]]:
    """
    answers_list = [
        {"answer_id": 10, "answer_value": "3"},
        {"answer_id": 11, "answer_value": "2"},
    ]
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            for ans in answers_list:
                cur.execute("""
                    UPDATE dev_kkh.tb_Questionnaire_Answers
                    SET answer_value = %s
                    WHERE answer_id = %s;
                """, (
                    ans.get("answer_value"),
                    ans.get("answer_id"),
                ))
            conn.commit()

        return True, None

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ 설문 응답 수정 실패: {e}")
        traceback.print_exc()
        return False, str(e)

    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 4️⃣ 설문 응답 조회 (기존: /mds/{item_id} GET)
# ============================================================
def fetch_mds_answers(item_id: int) -> List[dict]:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    answer_id, item_id, question_id,
                    answer_component, answer_value, submission_datetime
                FROM dev_kkh.tb_Questionnaire_Answers
                WHERE item_id = %s
                ORDER BY question_id, answer_component;
            """, (item_id,))
            return cur.fetchall()

    except Exception as e:
        print(f"❌ 설문 응답 조회 실패: {e}")
        traceback.print_exc()
        return []

    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 5️⃣ item 상태 업데이트 (기존: /items/{item_id}/mark-updated)
# ============================================================
def mark_item_updated(item_id: str) -> bool:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE dev_kkh.tb_items
                SET is_updated = TRUE,
                    updated_at = NOW()
                WHERE item_id = %s;
            """, (item_id,))
            conn.commit()
        print(f"✅ item_id={item_id} 메타정보 갱신 완료")
        return True

    except Exception as e:
        print(f"❌ item_id={item_id} 업데이트 실패: {e}")
        traceback.print_exc()
        return False

    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 6️⃣ CTk Raw 데이터 변환 → DB용 포맷
# ============================================================
def transform_to_api_format(raw_data: dict) -> list:
    """
    CTk StringVar에서 추출한 raw_data를 DB 입력 형식으로 변환합니다.
    """
    answers = []
    for key, value in raw_data.items():
        value = value.strip()
        if not value:
            continue

        if "_" in key:
            json_id, component = key.split("_", 1)
        else:
            json_id = key
            component = None

        question_db_id = MDS_QUESTION_MAPPING.get(json_id)

        if question_db_id is not None:
            answer = {
                "question_id": question_db_id,
                "answer_component": component if component else None,
                # 숫자 변환 시도 (int 변환 가능하면)
                "answer_value": int(value) if value.isdigit() else value
            }
            answers.append(answer)

    return answers
