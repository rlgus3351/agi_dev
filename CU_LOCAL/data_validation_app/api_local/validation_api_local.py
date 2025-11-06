import traceback
from datetime import datetime
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
from typing import Optional

FORM_NAME_MAP = {
    "E-SURVEY": {1: "PHQ-9", 2: "MADRS", 3: "ANXIETY"},
    "S-SURVEY": {1: "ISI",   2: "PSQI",  3: "KESS", 4: "MEQ-K"},
}

def _form_name(dtype: str, seq: Optional[int]) -> str:
    if not dtype:
        return "UNKNOWN"
    d = dtype.upper()
    if isinstance(seq, int) and d in FORM_NAME_MAP and seq in FORM_NAME_MAP[d]:
        return FORM_NAME_MAP[d][seq]
    # 매핑 없으면 dtype-seq 레이블로라도 명시
    return f"{d}-{seq if seq is not None else '?'}"

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
def fetch_new_mdd_surveys_7(patient_id: str):
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
                  AND i.data_category = 'MDD'
                  AND i.data_type iN ('S-SURVEY','E-SURVEY')
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
def validate_survey(item_id: int, data_type: Optional[str] = None, seq: Optional[int] = None):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 🔎 필요시 data_type/seq 보강
            if not data_type or seq is None:
                cur.execute("""
                    SELECT patient_id, UPPER(data_type) AS data_type, seq
                    FROM dev_kkh.tb_items
                    WHERE item_id = %s;
                """, (item_id,))
                meta_row = cur.fetchone()
                if meta_row:
                    data_type = meta_row.get("data_type", data_type)
                    seq = meta_row.get("seq", seq)
                patient_id = meta_row["patient_id"] if meta_row else None
            else:
                # 이미 patient_id도 함께 얻자
                cur.execute("SELECT patient_id FROM dev_kkh.tb_items WHERE item_id = %s;", (item_id,))
                row_pid = cur.fetchone()
                patient_id = row_pid["patient_id"] if row_pid else None

            form_name = _form_name(data_type or "", seq)

            # ✅ 응답 검증
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

            status  = result["validation_status"]
            missing = result["missing_count"]
            total   = result["total_count"]

            # 누락 문항 목록
            cur.execute("""
                SELECT question_id
                FROM dev_kkh.tb_questionnaire_answers
                WHERE item_id = %s
                  AND (answer_value IS NULL OR TRIM(answer_value) = '');
            """, (item_id,))
            missing_qs = [r["question_id"] for r in cur.fetchall()]

            # ✅ 설명/메서드에 폼명 반영
            desc = (
                f"[{form_name}] 총 {total}문항 중 {missing}개 누락 ({missing_qs})"
                if missing > 0
                else f"[{form_name}] 모든 문항 응답 완료"
            )
            validation_method = f"Local_MDD_SURVEY:{form_name}"

            # 결과 저장
            payload = {
                "patient_id": patient_id,
                "item_id": item_id,
                "validation_method": validation_method,
                "validation_description": desc,
                "validation_datetime": datetime.now(),
            }
            add_or_update_validation(payload)

            return {
                "item_id": item_id,
                "patient_id": patient_id,
                "data_type": data_type,
                "seq": seq,
                "form_name": form_name,
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

def calculate_depression_stage(item_id: int):
    """
    MADRS(E-SURVEY, seq=2)의 각 문항 점수를 합산해 총점으로 등급을 산정하고
    dev_kkh.tb_depression_stage에 (patient_id, item_id) 기준으로 UPSERT합니다.
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1) item 메타 확인 (환자, 유형, seq)
            cur.execute("""
                SELECT i.patient_id, i.data_category, i.data_type, COALESCE(i.seq, 0) AS seq
                FROM dev_kkh.tb_items i
                WHERE i.item_id = %s;
            """, (item_id,))
            meta = cur.fetchone()
            if not meta:
                return None, "해당 item_id가 존재하지 않습니다."
            
            patient_id = meta["patient_id"]
            data_type  = (meta["data_type"] or "").upper()
            data_cat   = (meta["data_category"] or "").upper()
            seq        = int(meta["seq"] or 0)

            # MADRS인지 가드(필요 없다면 제거 가능)
            if not (data_cat == "MDD" and data_type == "E-SURVEY" and seq == 2):
                # MADRS 항목이 아닐 수 있음 (원치 않으면 주석 처리해도 됨)
                return None, "MADRS(E-SURVEY, seq=2) 항목이 아닙니다."

            # 2) MADRS 총점 계산 (빈값/공백은 0 취급)
            cur.execute("""
                SELECT COALESCE(SUM(
                    CASE
                        WHEN TRIM(COALESCE(answer_value, '')) = '' THEN 0
                        ELSE (answer_value)::int
                    END
                ), 0) AS total_score
                FROM dev_kkh.tb_questionnaire_answers
                WHERE item_id = %s;
            """, (item_id,))
            row = cur.fetchone()
            if not row:
                return None, "설문 응답 데이터가 없습니다."
            total_score = int(row["total_score"] or 0)

            # 3) 구간 매핑
            if   0 <= total_score <= 6:   desc = "정상"
            elif 7 <= total_score <= 19:  desc = "경도"
            elif 20 <= total_score <= 34: desc = "중등도"
            elif 35 <= total_score <= 60: desc = "고도"
            else:
                desc = "범위 밖"

            # 4) UPSERT 저장 (stage_value에는 총점을 그대로 기록)
            cur.execute("""
                INSERT INTO dev_kkh.tb_depression_stage (patient_id, item_id, stage_value, stage_description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (patient_id, item_id)
                DO UPDATE SET
                    stage_value = EXCLUDED.stage_value,
                    stage_description = EXCLUDED.stage_description
                RETURNING *;
            """, (patient_id, item_id, float(total_score), desc))
            saved = cur.fetchone()
            conn.commit()

            return {
                "patient_id": patient_id,
                "item_id": item_id,
                "total_score": total_score,          # MADRS 총점
                "stage_value": float(total_score),   # 테이블 필드명에 맞춰 저장한 값
                "stage_description": desc
            }, None

    except Exception as e:
        print("❌ calculate_depression_stage 실패:", e)
        traceback.print_exc()
        return None, str(e)
    finally:
        if conn:
            release_connection(conn)


def fetch_new_media_items(patient_id: str, data_types: list, category: str = 'MDD'):
    """
    data_types 예: ['MOBILE','WEBCAM'] / ['VOICE','AUDIO'] / ['TXT','TEXT','FILE']
    """
    if not data_types:
        return []
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = f"""
                SELECT i.*
                FROM dev_kkh.tb_items i
                LEFT JOIN dev_kkh.tb_data_validation v ON i.item_id = v.item_id
                WHERE i.patient_id = %s
                  AND i.is_deleted = FALSE
                  AND i.data_category = %s
                  AND i.data_type = ANY(%s)
                  AND (
                      v.validation_datetime IS NULL
                      OR GREATEST(i.collected_at, i.updated_at) > v.validation_datetime
                  )
                ORDER BY GREATEST(i.collected_at, i.updated_at) DESC;
            """
            cur.execute(sql, (patient_id, category, data_types))
            return cur.fetchall()
    except Exception as e:
        print("❌ fetch_new_media_items 실패:", e)
        traceback.print_exc()
        return []
    finally:
        if conn:
            release_connection(conn)