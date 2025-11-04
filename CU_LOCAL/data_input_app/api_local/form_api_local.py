from typing import Union, Tuple, List
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback

# ============================================================
# 🧩 MDS 질문 매핑 (로컬 DB용)
# ============================================================




BASIC_QUESTION_MAPPING = {
    "결혼 여부": 104,
    "교육년수": 105,
    "직업": 106,
    "교대근무 여부": 107,
    "기타 항목 입력": 108,
    "신장": 109,
    "몸무게": 110,
    "카페인 음료 섭취": 111,
    "섭취 여부": 112,
    "월 섭취 횟수": 113,
    "기타 음료 입력": 114,
    "음주에 대한 질문": 115,
    "음주 종류": 117,
    "기타 주류 입력": 118,
    "흡연 여부": 119,
    "하루 흡연량": 120,
    "흡연 기간": 121,
    "운동 여부": 122,
    "운동 종류": 124,
    "운동 시간": 126,
    "현병력": 127,
    "우울증": 128,
    "불면증": 129,
    "기타": 131,
    "발병 시점": 132,
    "발병 년수": 133,
    "발병 월수": 134,
}

PHQ9_QUESTION_MAPPING ={
    "기분이 가라앉거나, 우울하거나 희망이 없다고 느꼈다." : 27,
    "평소 하던 일에 대한 흥미가 없어지거나 즐거움을 느끼지 못했다." : 28,
    "잠들기 어렵거나 자주 깼다. 혹은 너무 많이 잤다." : 29,
    "평소보다 식욕이 줄었다. 혹은 평소보다 많이 먹었다." : 30,
    "다른 사람들이 눈치 챌 정도로 평소보다 말과 행동이 느려졌다.\n혹은 너무 안절부절 못해서 가만히 앉아 있을 수 없었다." : 31,
    "피곤하고 기운이 없었다." : 32,
    "내가 잘못했거나 실패했다는 생각이 들었다.\n혹은 자신과 가족을 실망시켰다고 생각했다." : 33,
    "학교 공부, 독서, TV시청 같은 일상적인 일에도 집중할 수가 없었다." : 34,
    "차라리 죽는 것이 더 낫겠다고 생각했다. 혹은 자해할 생각을 했다." : 35,
}
MADRS_QUESTION_MAPPING={
    "겉으로 드러나는 슬픔": 36,
    "스스로 보고하는 슬픔": 37,
    "내적 긴장감": 38,
    "수면 저하": 39,
    "식욕 저하": 40,
    "집중의 어려움": 41,
    "나태(권태)": 42,
    "느낌의 상실": 43,
    "비관적 사고": 44,
    "자살사고": 45,
}

ANXIETY_DISORDER_QUESTION_MAPPING = {
    "초조하거나 불안하거나 조마조마하게 느낀다." : 46, 
    "걱정하는 것을 멈추거나 조절할 수가 없다." :  47,
    "여러 가지 것들에 대해 걱정을 너무 많이 한다." :  48,
    "편하게 있기가 어렵다." :  49,
    "쉽게 짜증이 나거나 쉽게 성을 내게 된다." :  50,
    "너무 안절부절 못해서 가만히 앉아 있을 수 없음" :  51,
    "마치 끔찍한 일이 생길 것처럼 두렵게 느껴진다." :  52
}




EMOTION_QMAP_BY_SEQ = {
    1: PHQ9_QUESTION_MAPPING,
    2: MADRS_QUESTION_MAPPING,
    3: ANXIETY_DISORDER_QUESTION_MAPPING,
}


ISI_QUESTION_MAPPING={
    "잠들기 어렵다.":53,
    "잠을 유지하기 어렵다.":54,
    "원하는 시간보다 일찍 깬다.":55,
    "당신의 현재 수면 양상에 관하여\n얼마나 만족하고 있습니까?":56,
    "당신이 생각하기에 당신의 수면 장애가\n어느 정도 당신의 낮 활동을 방해한다고 생각합니까?":57,
    "수면장애로 인한 당신의 삶의 질의 \n손상 정도는 다른 사람들이 보기에 어떻다고 생각합니까?":58,
    "당신은 현재 당신의 수면장애에 \n관하여 얼마나 걱정하고 있습니까?":59,
}

KESS_QUESTION_MAPPING={
    "앉아서 책(신문,잡지,서류 등)을 읽을 때":83,
    "TV를 볼 때":84,
    "공공장소(모임,극장 등)에서 가만히 앉아 있을 때":85,
    "정차없이 1시간 동안 운행 중인\n 차에 승객으로 앉아 있을 때":86,
    "오후에 주의상황이 허락되어 쉬려고 누워 있을 때":87,
    "앉아서 상대방과 이야기할 때":88,
    "반주를 곁들이지 않은 점심식사 후 조용히 앉아 있을 때":89,
    "교통 혼잡으로 몇 분 동안 멈춰선 차 안에서":90,
}
PSQI_QUESTION_MAPPING={
    "보통 몇시에 잠자리에 듭니까?"
    "보통 잠 들 때까지 평균 얼마나 걸립니까?"
    "보통 몇 시에 일어납니까?"
    "당신은 실제로 하루에 몇 시간 잡니까?"
    "밤에 30분 이내에 잠들지 못해서"
    "중간에 깨거나 너무 일찍 깨서",
    "화장실을 다녀오려고 일어나서",
    "수면 중 숨을 쉬기가 불편해서",
    "기침을 하거나 크게 코를 골아서",
    "수면 중 너무 춥다고 느껴서",
    "수면 중 너무 덥다고 느껴서",
    "나쁜 꿈을 꿔서",
    "통증이 있어서",
    "위에 적혀진 이유 외에 잠을 못 잔 다른 이유",
}

MEQK_QUESTION_MAPPING={
    "보통 몇시에 잠자리에 듭니까?"
    "보통 잠 들 때까지 평균 얼마나 걸립니까?"
    "보통 몇 시에 일어납니까?"
    "당신은 실제로 하루에 몇 시간 잡니까?"
    "밤에 30분 이내에 잠들지 못해서"
    "중간에 깨거나 너무 일찍 깨서",
    "화장실을 다녀오려고 일어나서",
    "수면 중 숨을 쉬기가 불편해서",
    "기침을 하거나 크게 코를 골아서",
    "수면 중 너무 춥다고 느껴서",
    "수면 중 너무 덥다고 느껴서",
    "나쁜 꿈을 꿔서",
    "통증이 있어서",
    "위에 적혀진 이유 외에 잠을 못 잔 다른 이유",
}




SLEEP_QMAP_BY_SEQ = {
    1: ISI_QUESTION_MAPPING,
    2: KESS_QUESTION_MAPPING,
    3: PSQI_QUESTION_MAPPING,
    4: MEQK_QUESTION_MAPPING
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

def create_new_item_and_get_id_generic(
    target_patient_id: str,
    data_category: str = "MDD",
    data_type: str = "B-SURVEY",
    seq: int = 1,
    description: str = "기초평가"
) -> Union[int, None]:
    """
    ✅ 설문 item 생성 (기초평가 / 정서 / 수면 등 공통)
    Args:
        target_patient_id (str): 환자 UUID
        data_category (str): 질환 구분 (예: 'MDD', 'PD')
        data_type (str): 데이터 유형 (예: 'B-SURVEY', 'E-SURVEY', 'S-SURVEY')
        seq (int): 일련번호
        description (str): 데이터 설명
    Returns:
        item_id (int) 또는 None
    """
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
                data_category,
                data_type,
                seq,
                description
            ))
            item = cur.fetchone()
            conn.commit()
            if item:
                print(f"✅ 새 item 생성 완료: item_id={item['item_id']} ({data_type})")
                return item["item_id"]
            else:
                print("⚠️ item 생성 결과 없음")
                return None

    except Exception as e:
        print(f"❌ 설문 item 생성 실패: {e}")
        traceback.print_exc()
        return None

    finally:
        if conn:
            release_connection(conn)




# ============================================================
# 2️⃣ 설문 응답 등록 (기존: /mds/{item_id})
# ============================================================
def save_answers(item_id: int, answers_list: list) -> Tuple[bool, Union[str, None]]:
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

# api_local/form_api_local.py
def update_existing_survey_answers_by_id(answers: List[dict]) -> Tuple[bool, Union[str, None]]:
    """
    ✅ answer_id(PK) 기준으로 설문 응답 수정
    Parameters
    ----------
    answers : List[dict]
        예) [{"answer_id": 10, "answer_value": "3"},
             {"answer_id": 11, "answer_value": "true"}]
    Returns
    -------
    (ok: bool, err: Optional[str])
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            for ans in answers:
                aid = ans.get("answer_id")
                val = ans.get("answer_value")

                if aid is None:
                    # 잘못된 payload는 무시하거나 예외 처리
                    continue

                cur.execute(
                    """
                    UPDATE dev_kkh.tb_Questionnaire_Answers
                    SET answer_value = %s
                    WHERE answer_id = %s;
                    """,
                    (val, aid)
                )
        conn.commit()
        return True, None

    except Exception as e:
        if conn:
            conn.rollback()
        return False, str(e)

    finally:
        if conn:
            release_connection(conn)



