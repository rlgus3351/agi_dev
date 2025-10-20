# api_local/mds_updrs_part3_json.py
from typing import Dict, Any, Tuple, Optional
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback, json

# -----------------------------
# 기초정보/중증도 question_id 매핑 (당신 DB 기준)
# -----------------------------
MED_Q = {
    "is_on_medication":   1,  # 예/아니오
    "clinical_effect":    2,  # 긍정적/부정적 -> positive/negative
    "levodopa_taken":     3,  # 예/아니오
    "levodopa_minutes":   4,  # 분 단위
    "dyskinesia_present": 6,  # 예/아니오
    "dyskinesia_interfered": 7,  # 예/아니오
    "hoehn_yahr_stage":   8,  # (필요시 사용)
}

# 항목 이름 (item_id = question_id - 8)
ITEM_NAMES: Dict[int, str] = {
    1: "말하기",
    2: "얼굴 표정",
    3: "관절의 뻣뻣함",
    4: "손가락 부딪치기",
    5: "손 동작",
    6: "손 내전/외전 움직임",
    7: "발가락으로 두드리기",
    8: "다리 민첩성",
    9: "의자에서 일어나기",
    10: "걷는 자세",
    11: "걷는 중 몸의 굳어짐",
    12: "자세의 안정",
    13: "자세",
    14: "움직임에서 전반적인 자연스러움",
    15: "자세 유지시 손의 떨림",
    16: "움직일 때 손의 떨림",
    17: "가만 있을 때 떨림의 폭",
    18: "가만 있을 때 떨림의 지속시간",
}

def _to_int(v) -> Optional[int]:
    try:
        return int(str(v).strip())
    except Exception:
        return None

def _to_float(v) -> Optional[float]:
    try:
        return float(str(v).strip())
    except Exception:
        return None

def _to_bool(v) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "예"}:
        return True
    if s in {"0", "false", "f", "no", "n", "아니오"}:
        return False
    return None

def _clinical_to_en(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"긍정적", "positive", "pos"}:
        return "positive"
    if s in {"부정적", "negative", "neg"}:
        return "negative"
    return None

def build_mds_updrs_part3_json(item_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    dev_kkh.tb_questionnaire_answers의 (item_id) 응답을
    원하는 JSON 포맷으로 변환.
    반환: (json_dict | None, error | None)
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT question_id, answer_component, answer_value, submission_datetime
                FROM dev_kkh.tb_questionnaire_answers
                WHERE item_id = %s
                ORDER BY COALESCE(submission_datetime, NOW()) ASC, answer_id ASC;
            """, (item_id,))
            rows = cur.fetchall()

        if not rows:
            return None, "해당 item_id 설문 응답 없음"

        # (question_id, component) 기준 최신값만 유지
        latest: Dict[Tuple[int, Optional[str]], Dict[str, Any]] = {}
        for r in rows:
            key = (r["question_id"], (r["answer_component"] or None))
            latest[key] = r

        # -------------------------
        # A) medication / dyskinesia
        # -------------------------
        med = {}
        dysk = {}

        v = latest.get((MED_Q["is_on_medication"], None), {}).get("answer_value")
        b = _to_bool(v)
        if b is not None:
            med["is_on_medication"] = b

        v = latest.get((MED_Q["clinical_effect"], None), {}).get("answer_value")
        ce = _clinical_to_en(v)
        if ce is not None:
            med["clinical_effect"] = ce

        v = latest.get((MED_Q["levodopa_taken"], None), {}).get("answer_value")
        b = _to_bool(v)
        if b is not None:
            med["levodopa_taken"] = b

        v = latest.get((MED_Q["levodopa_minutes"], None), {}).get("answer_value")
        m = _to_float(v)
        if m is not None:
            med["levodopa_elapsed_hours"] = round(m / 60.0, 2)

        v = latest.get((MED_Q["dyskinesia_present"], None), {}).get("answer_value")
        b = _to_bool(v)
        if b is not None:
            dysk["dyskinesia_present"] = b

        v = latest.get((MED_Q["dyskinesia_interfered"], None), {}).get("answer_value")
        b = _to_bool(v)
        if b is not None:
            dysk["dyskinesia_interfered"] = b

        # (참고) 필요하면 stage도 여기서 읽어올 수 있음:
        # stage_raw = latest.get((MED_Q["hoehn_yahr_stage"], None), {}).get("answer_value")

        # -------------------------
        # B) 9~26 → item 1~18 매핑
        # -------------------------
        items = []
        for qid in range(9, 27):                # 9..26
            item_idx = qid - 8                  # 1..18
            item_name = ITEM_NAMES.get(item_idx, f"항목 {item_idx}")

            # 후보 키 찾기
            comps = [(qq, comp) for (qq, comp) in latest.keys() if qq == qid]
            if not comps:
                continue

            comp_scores: Dict[str, int] = {}
            single_score: Optional[int] = None

            for (_qq, comp) in comps:
                val = latest.get((_qq, comp), {}).get("answer_value")
                score = _to_int(val)
                if score is None:
                    continue
                if comp is None:
                    single_score = score
                else:
                    comp_scores[comp] = score

            entry = {"item_id": item_idx, "item_name": item_name}
            if comp_scores:
                entry["scores"] = comp_scores
            elif single_score is not None:
                entry["scores"] = single_score
            else:
                continue

            items.append(entry)

        result = {
            "mds_updrs_part3": {
                "medication": med,
                "dyskinesia_impact": dysk,
                "items": items
            }
        }
        return result, None

    except Exception as e:
        print("❌ build_mds_updrs_part3_json 실패:", e)
        traceback.print_exc()
        return None, str(e)
    finally:
        if conn:
            release_connection(conn)

# 옵션: 파일로 저장하고 싶을 때
def save_json_to_file(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
