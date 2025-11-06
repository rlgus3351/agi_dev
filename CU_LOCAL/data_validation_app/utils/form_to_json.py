# api_local/mds_updrs_part3_json.py
from typing import Tuple, Optional, Dict, Any
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import json
import os

# ─────────────────────────────────────────────────────────────
# 매핑 정의
# ─────────────────────────────────────────────────────────────

# question_id  → 필드 매핑 (기초/약물/중증도)
BASE_MAPPING = {
    1: ("medication.is_on_medication", "bool_yes_no"),           # 예/아니오 → bool
    2: ("medication.clinical_effect", "pos_neg_ko"),             # 긍정적/부정적 → positive/negative
    3: ("medication.levodopa_taken", "bool_yes_no"),
    4: ("medication.levodopa_elapsed_hours", "minutes_to_hours"),# 분 → 시간(float)
    # dyskinesia 관련은 5,6,7이 섞여 들어오는 경우가 있어 안전하게 후처리로 결정
    8: ("stage.hoehn_yahr", "number"),                           # 0/1/1.5/…/5
}

# 항목(9~26) question_id → (항목ID, 항목명, 그룹키셋 혹은 None)
ITEMS_MAPPING = {
    9:  (1,  "말하기", None),
    10: (2,  "얼굴 표정", None),
    11: (3,  "관절의 뻣뻣함", ["Neck", "RA", "LA", "RL", "LL"]),
    12: (4,  "손가락 부딪치기", ["L", "R"]),
    13: (5,  "손 동작", ["L", "R"]),
    14: (6,  "손 내전/외전 움직임", ["L", "R"]),
    15: (7,  "발가락으로 두드리기", ["L", "R"]),
    16: (8,  "다리 민첩성", ["L", "R"]),
    17: (9,  "의자에서 일어나기", None),
    18: (10, "걷는 자세", None),
    19: (11, "걷는 중 몸의 굳어짐", None),
    20: (12, "자세의 안정", None),
    21: (13, "자세", None),
    22: (14, "움직임에서 전반적인 자연스러움", None),
    23: (15, "자세 유지시 손의 떨림", ["L", "R"]),
    24: (16, "움직일 때 손의 떨림", ["L", "R"]),
    25: (17, "가만 있을 때 떨림의 폭", ["RA", "LA", "RL", "LL", "LJ"]),
    26: (18, "가만 있을 때 떨림의 지속시간", None),
}

YES_SET = {"예", "yes", "y", "true", "True", True}
POS_KO_TO_EN = {"긍정적": "positive", "부정적": "negative"}

def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any):
    """ 'a.b.c' 키 경로에 값 설정 """
    cur = d
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _coerce(value: Optional[str], how: str):
    if value is None:
        return None
    s = str(value).strip()
    if how == "bool_yes_no":
        return s in YES_SET
    if how == "pos_neg_ko":
        return POS_KO_TO_EN.get(s, s)  # 모르면 원문 유지
    if how == "minutes_to_hours":
        try:
            minutes = float(s)
            return round(minutes / 60.0, 2)
        except Exception:
            return None
    if how == "number":
        try:
            return float(s) if "." in s else int(s)
        except Exception:
            return None
    # 기본: 원문 반환
    return s

# ─────────────────────────────────────────────────────────────
# 메인 함수: DB → 원하는 JSON 구조
# ─────────────────────────────────────────────────────────────
def build_mds_updrs_part3_json(item_id: int) -> Tuple[Optional[dict], Optional[str]]:
    """
    tb_questionnaire_answers의 (item_id) 응답을 읽어 Part3 JSON을 만든다.
    반환: (dict or None, error_message or None)
    """
    from psycopg2.extras import RealDictCursor
    from utils.db_utils import get_connection, release_connection

    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT answer_id, item_id, question_id, answer_component, answer_value, submission_datetime
                FROM dev_kkh.tb_questionnaire_answers
                WHERE item_id = %s
                ORDER BY answer_id ASC
            """, (item_id,))
            rows = cur.fetchall()

        if not rows:
            return None, "해당 item_id의 응답이 없습니다."

        # 결과 스켈레톤
        result = {
            "mds_updrs_part3": {
                "medication": {
                    "is_on_medication": None,
                    "clinical_effect": None,
                    "levodopa_taken": None,
                    "levodopa_elapsed_hours": None,
                },
                "dyskinesia_impact": {
                    "dyskinesia_present": None,
                    "dyskinesia_interfered": None,
                },
                "items": []
            }
        }

        # 항목 채우기용 버퍼: item_id(1~18) → dict
        items_buf: Dict[int, Dict[str, Any]] = {}

        # dyskinesia 판단을 위한 임시 플래그
        dys_flags = {
            5: None,  # (폼상) DYSKINESIA 질문
            6: None,  # 검사지 영향 여부
            7: None,  # 검사 중 dyskinesia 유무
        }

        # 1차 스캔
        for r in rows:
            qid = r["question_id"]
            comp = r["answer_component"]  # RA/LA/L/R/Neck...
            val  = r["answer_value"]

            # 1) 베이스 매핑 (1~4, 8)
            if qid in BASE_MAPPING:
                key, how = BASE_MAPPING[qid]
                coerced = _coerce(val, how)
                _set_nested(result["mds_updrs_part3"], key, coerced)
                continue

            # 2) dyskinesia 관련(5/6/7) — 후처리
            if qid in dys_flags:
                dys_flags[qid] = (val in YES_SET)
                continue

            # 3) 항목 점수 (9~26)
            if qid in ITEMS_MAPPING:
                sub_id, sub_name, sides = ITEMS_MAPPING[qid]
                if sub_id not in items_buf:
                    items_buf[sub_id] = {"item_id": sub_id, "item_name": sub_name, "scores": {} if sides else None}

                if sides:
                    # 그룹형: comp별 점수 축적
                    if comp is None or comp == "":
                        continue  # 그룹인데 comp 없으면 스킵
                    try:
                        score = int(val)
                    except Exception:
                        continue
                    items_buf[sub_id]["scores"][comp] = score
                else:
                    # 단일 점수
                    try:
                        score = int(val)
                    except Exception:
                        score = None
                    items_buf[sub_id]["scores"] = score

        # dyskinesia 최종 결정 (데이터가 5/6/7로 들어올 수 있어 유연하게 처리)
        present = any(flag is True for q, flag in dys_flags.items() if q in (5, 7))
        interfered = (dys_flags.get(6) is True) or (dys_flags.get(5) is True and dys_flags.get(7) is True)
        result["mds_updrs_part3"]["dyskinesia_impact"]["dyskinesia_present"] = present
        result["mds_updrs_part3"]["dyskinesia_impact"]["dyskinesia_interfered"] = interfered

        # items 정렬(1~18) 후 리스트로 삽입
        for k in sorted(items_buf.keys()):
            result["mds_updrs_part3"]["items"].append(items_buf[k])

        return result, None

    except Exception as e:
        return None, f"JSON 빌드 실패: {e}"
    finally:
        if conn:
            release_connection(conn)

# 파일 저장 헬퍼(선택)
def save_json_to_file(data: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
