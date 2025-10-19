import traceback
from datetime import datetime
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection

# ============================================================
# 1️⃣ 검증 결과 저장 (UPSERT)
# ============================================================
def add_or_update_validation(payload: dict):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO dev_kkh.tb_data_validation (
                    patient_id, item_id, validation_method,
                    validation_description, validation_datetime
                )
                VALUES (%(patient_id)s, %(item_id)s, %(validation_method)s,
                        %(validation_description)s, %(validation_datetime)s)
                ON CONFLICT (patient_id, item_id)
                DO UPDATE SET
                    validation_method = EXCLUDED.validation_method,
                    validation_description = EXCLUDED.validation_description,
                    validation_datetime = EXCLUDED.validation_datetime
                RETURNING *;
            """, payload)
            conn.commit()
            return True, cur.fetchone()
    except Exception as e:
        print("❌ add_or_update_validation 실패:", e)
        traceback.print_exc()
        return False, str(e)
    finally:
        if conn:
            release_connection(conn)

# ============================================================
# 2️⃣ 신규 설문 항목 조회
# ============================================================
def fetch_new_pd_survey_items(patient_id: str):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT i.*
                FROM dev_kkh.tb_items i
                LEFT JOIN dev_kkh.tb_data_validation v ON i.item_id = v.item_id
                WHERE i.patient_id = %s
                  AND i.is_deleted = FALSE
                  AND i.data_category = 'PD'
                  AND i.data_type = 'MDS-UPDRS Part 3'
                  AND (
                      v.validation_datetime IS NULL
                      OR GREATEST(i.collected_at, i.updated_at) > v.validation_datetime
                  )
                ORDER BY GREATEST(i.collected_at, i.updated_at) DESC;
            """, (patient_id,))
            return cur.fetchall()
    except Exception as e:
        print("❌ fetch_new_pd_survey_items 실패:", e)
        traceback.print_exc()
        return []
    finally:
        if conn:
            release_connection(conn)

# ============================================================
# 3️⃣ 신규 영상 항목 조회
# ============================================================
def fetch_new_pd_video_items(patient_id: str):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT i.*
                FROM dev_kkh.tb_items i
                LEFT JOIN dev_kkh.tb_data_validation v ON i.item_id = v.item_id
                WHERE i.patient_id = %s
                  AND i.is_deleted = FALSE
                  AND i.data_category = 'PD'
                  AND i.data_type = 'VIDEO'
                  AND (
                      v.validation_datetime IS NULL
                      OR GREATEST(i.collected_at, i.updated_at) > v.validation_datetime
                  )
                ORDER BY GREATEST(i.collected_at, i.updated_at) DESC;
            """, (patient_id,))
            return cur.fetchall()
    except Exception as e:
        print("❌ fetch_new_pd_video_items 실패:", e)
        traceback.print_exc()
        return []
    finally:
        if conn:
            release_connection(conn)

# ============================================================
# 4️⃣ 설문 검증 (응답 누락)
# ============================================================
def validate_survey(item_id: int):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 응답 검증
            cur.execute("""
                SELECT
                    CASE
                        WHEN COUNT(*) FILTER (WHERE COALESCE(answer_value, '') = '') > 0
                        THEN 'FAIL'
                        ELSE 'PASS'
                    END AS validation_status,
                    COUNT(*) FILTER (WHERE COALESCE(answer_value, '') = '') AS missing_count,
                    COUNT(*) AS total_count
                FROM dev_kkh.tb_questionnaire_answers
                WHERE item_id = %s;
            """, (item_id,))
            result = cur.fetchone()
            if not result:
                return None, "응답 데이터 없음"

            status = result["validation_status"]
            missing = result["missing_count"]
            total = result["total_count"]

            # 누락 문항
            cur.execute("""
                SELECT question_id
                FROM dev_kkh.tb_questionnaire_answers
                WHERE item_id = %s
                  AND (answer_value IS NULL OR TRIM(answer_value) = '');
            """, (item_id,))
            missing_qs = [r["question_id"] for r in cur.fetchall()]

            desc = (
                f"총 {total}문항 중 {missing}개 누락 ({missing_qs})"
                if missing > 0
                else "모든 문항 응답 완료"
            )

            # patient_id 찾기
            cur.execute("SELECT patient_id FROM dev_kkh.tb_items WHERE item_id = %s;", (item_id,))
            row = cur.fetchone()
            patient_id = row["patient_id"] if row else None

            # 결과 저장
            payload = {
                "patient_id": patient_id,
                "item_id": item_id,
                "validation_method": "Local_PD_SURVEY",
                "validation_description": desc,
                "validation_datetime": datetime.now(),
            }
            add_or_update_validation(payload)

            return {
                "item_id": item_id,
                "patient_id": patient_id,
                "status": status,
                "missing_count": missing,
                "total_questions": total,
                "missing_questions": missing_qs,
                "description": desc,
            }, None
    except Exception as e:
        print("❌ validate_survey 실패:", e)
        traceback.print_exc()
        return None, str(e)
    finally:
        if conn:
            release_connection(conn)

def calculate_parkinson_stage(item_id: int):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1️⃣ 설문 응답 조회
            cur.execute("""
                SELECT i.patient_id, a.answer_value::numeric AS stage_value
                FROM dev_kkh.tb_questionnaire_answers a
                JOIN dev_kkh.tb_items i ON a.item_id = i.item_id
                WHERE a.item_id = %s
                  AND a.question_id = 8
                  AND TRIM(a.answer_value) != ''
                  AND i.data_category = 'PD'
                  AND i.data_type = 'MDS-UPDRS Part 3';
            """, (item_id,))
            row = cur.fetchone()
            if not row:
                return None, "중증도 설문 응답 없음"

            stage_val = row["stage_value"]
            patient_id = row["patient_id"]

            # 2️⃣ 중증도 단계 설명 매핑
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

            # 3️⃣ INSERT 또는 UPSERT
            cur.execute("""
                INSERT INTO dev_kkh.tb_parkinson_stage (patient_id, item_id, stage_value, stage_description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (patient_id, item_id)
                DO UPDATE SET
                    stage_value = EXCLUDED.stage_value,
                    stage_description = EXCLUDED.stage_description
                RETURNING *;
            """, (patient_id, item_id, stage_val, desc))

            conn.commit()

            return {
                "patient_id": patient_id,
                "item_id": item_id,
                "stage_value": float(stage_val),
                "stage_description": desc
            }, None

    except Exception as e:
        print("❌ calculate_parkinson_stage 실패:", e)
        traceback.print_exc()
        return None, str(e)
    finally:
        if conn:
            release_connection(conn)