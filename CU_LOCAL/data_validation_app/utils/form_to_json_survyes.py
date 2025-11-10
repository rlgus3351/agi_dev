# utils/form_to_json_surveys.py
# E-SURVEY(PHQ-9/MADRS/불안척도) + S-SURVEY(ISI/PSQI/KESS/MEQ-K) JSON Exporter
# - per-item 저장 + 환자별 번들 저장(E/S 각각 1파일)

import os
import re
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from psycopg2.extras import RealDictCursor

from utils.db_utils import get_connection, release_connection

# 네가 가진 매핑들을 그대로 사용
from api_local.form_api_local import (
    EMOTION_QMAP_BY_SEQ,   # { seq: {question_text: question_id} }
    SLEEP_QMAP_BY_SEQ,     # { seq: {question_text: question_id} }
    PHQ9_QUESTION_MAPPING,
    MADRS_QUESTION_MAPPING,
    ANXIETY_DISORDER_QUESTION_MAPPING,
    ISI_QUESTION_MAPPING,
    PSQI_QUESTION_MAPPING,
    KESS_QUESTION_MAPPING,
    MEQK_QUESTION_MAPPING
)

# -----------------------------
# 공통 유틸
# -----------------------------

QUESTION_REGISTRY: Dict[Tuple[str, int], Dict[str, int]] = {
    ("E_SURVEY", 1): PHQ9_QUESTION_MAPPING,
    ("E_SURVEY", 2): MADRS_QUESTION_MAPPING,
    ("E_SURVEY", 3): ANXIETY_DISORDER_QUESTION_MAPPING,
    ("S_SURVEY", 1): ISI_QUESTION_MAPPING,
    ("S_SURVEY", 2): PSQI_QUESTION_MAPPING,
    ("S_SURVEY", 3): KESS_QUESTION_MAPPING,
    ("S_SURVEY", 4): MEQK_QUESTION_MAPPING,
}

_EMO_TITLES = {1: "PHQ-9", 2: "MADRS", 3: "불안척도"}
_SLEEP_TITLES = {1: "ISI", 2: "PSQI", 3: "KESS", 4: "MEQ-K"}

def _norm_dtype(dt: Any) -> str:
    return str(dt or "").strip().upper().replace("-", "_")

def _safe(s: Any) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_\-\.]+", "_", str(s or ""))

def _invert_qmap(qmap: Dict[str, int]) -> Dict[int, str]:
    return {qid: qtext for qtext, qid in qmap.items()}

def _fmt_ts(ts: Any) -> Optional[str]:
    if not ts:
        return None
    try:
        if isinstance(ts, datetime):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        return str(ts).split(".")[0].replace("T", " ")
    except Exception:
        return str(ts)

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _fetch_item(cur, item_id: int) -> Optional[Dict[str, Any]]:
    cur.execute("""
        SELECT i.item_id, i.patient_id, i.data_category, i.data_type, i.seq,
               i.collected_at, i.updated_at, i.is_deleted
        FROM dev_kkh.tb_items i
        WHERE i.item_id = %s
          AND i.is_deleted = FALSE;
    """, (item_id,))
    return cur.fetchone()

def _fetch_answers(cur, item_id: int) -> List[Dict[str, Any]]:
    cur.execute("""
        SELECT question_id, answer_value
        FROM dev_kkh.tb_questionnaire_answers
        WHERE item_id = %s
        ORDER BY question_id;
    """, (item_id,))
    return cur.fetchall() or []

def _fetch_patient_items(cur, patient_id: str, dtype_norm: str) -> List[Dict[str, Any]]:
    cur.execute(f"""
        SELECT item_id, patient_id, data_category, data_type, seq, collected_at, updated_at
        FROM dev_kkh.tb_items
        WHERE patient_id = %s
          AND is_deleted = FALSE
          AND UPPER(REPLACE(data_type, '-', '_')) = %s
        ORDER BY COALESCE(updated_at, collected_at) DESC, item_id DESC;
    """, (patient_id, dtype_norm))
    return cur.fetchall() or []

def save_json_to_file(data: Dict[str, Any], out_path: str) -> Tuple[bool, Optional[str]]:
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True, None
    except Exception as e:
        return False, str(e)

# -----------------------------
# per-item JSON (기존 기능 유지)
# -----------------------------

def build_emotion_survey_json(item_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            item = _fetch_item(cur, item_id)
            if not item:
                return None, "item을 찾을 수 없음"

            if _norm_dtype(item["data_type"]) != "E_SURVEY":
                return None, f"E_SURVEY 항목이 아님(data_type={item.get('data_type')})"

            seq_raw = item.get("seq")
            try:
                seq = int(seq_raw) if seq_raw is not None else None
            except Exception:
                seq = None
            if not seq:
                return None, f"E_SURVEY seq 감지 실패 (seq={seq_raw})"

            qmap_src = EMOTION_QMAP_BY_SEQ.get(seq) or QUESTION_REGISTRY.get(("E_SURVEY", seq)) or {}
            if not qmap_src:
                return None, f"E_SURVEY 질문 매핑 없음 (seq={seq})"

            id_to_text = _invert_qmap(qmap_src)
            rows = _fetch_answers(cur, item_id)
            answers = [{
                "question_id": r["question_id"],
                "question": id_to_text.get(r["question_id"], f"(정의없음 #{r['question_id']})"),
                "answer": r["answer_value"],
            } for r in rows]

            json_obj = {
                "meta": {
                    "patient_id": str(item["patient_id"]),
                    "item_id": item["item_id"],
                    "data_category": item.get("data_category"),
                    "data_type": _norm_dtype(item.get("data_type")),
                    "seq": seq,
                    "survey_name": _EMO_TITLES.get(seq, f"E_SURVEY-{seq}"),
                    "collected_at": _fmt_ts(item.get("collected_at")),
                    "updated_at": _fmt_ts(item.get("updated_at")),
                },
                "answers": answers
            }
            return json_obj, None
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, str(e)
    finally:
        if conn:
            release_connection(conn)

def export_emotion_json_by_item(item_id: int, out_dir: str = "./json") -> Tuple[Optional[str], Optional[str]]:
    data, err = build_emotion_survey_json(item_id)
    if err:
        return None, err
    meta = data["meta"]
    out_dir_patient = os.path.join(out_dir, _safe(meta['patient_id']))  # ⬅️ 환자 폴더
    out_path = os.path.join(out_dir_patient, f"{_safe(meta['patient_id'])}_E{meta['seq']}_{_safe(meta['survey_name'])}_item{meta['item_id']}.json")
    ok, err2 = save_json_to_file(data, out_path)
    return (out_path, None) if ok else (None, err2)

def export_emotion_jsons_for_patient(patient_id: str, out_dir: str = "./json") -> Tuple[List[str], List[str]]:
    saved, errors = [], []
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            items = _fetch_patient_items(cur, patient_id, "E_SURVEY")
        for it in items:
            p, err = export_emotion_json_by_item(it["item_id"], out_dir=out_dir)
            (saved.append(p) if p else errors.append(f"item_id={it['item_id']}: {err}"))
    except Exception as e:
        import traceback; traceback.print_exc()
        errors.append(str(e))
    finally:
        if conn:
            release_connection(conn)
    return saved, errors

def build_sleep_survey_json(item_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            item = _fetch_item(cur, item_id)
            if not item:
                return None, "item을 찾을 수 없음"

            if _norm_dtype(item["data_type"]) != "S_SURVEY":
                return None, f"S_SURVEY 항목이 아님(data_type={item.get('data_type')})"

            seq_raw = item.get("seq")
            try:
                seq = int(seq_raw) if seq_raw is not None else None
            except Exception:
                seq = None
            if not seq:
                return None, f"SURVEY seq 감지 실패 (seq={seq_raw})"

            qmap_src = SLEEP_QMAP_BY_SEQ.get(seq) or QUESTION_REGISTRY.get(("S_SURVEY", seq)) or {}
            if not qmap_src:
                return None, f"SURVEY 질문 매핑 없음 (seq={seq})"

            id_to_text = _invert_qmap(qmap_src)
            rows = _fetch_answers(cur, item_id)
            answers = [{
                "question_id": r["question_id"],
                "question": id_to_text.get(r["question_id"], f"(정의없음 #{r['question_id']})"),
                "answer": r["answer_value"],
            } for r in rows]

            json_obj = {
                "meta": {
                    "patient_id": str(item["patient_id"]),
                    "item_id": item["item_id"],
                    "data_category": item.get("data_category"),
                    "data_type": _norm_dtype(item.get("data_type")),
                    "seq": seq,
                    "survey_name": _SLEEP_TITLES.get(seq, f"S_SURVEY-{seq}"),
                    "collected_at": _fmt_ts(item.get("collected_at")),
                    "updated_at": _fmt_ts(item.get("updated_at")),
                },
                "answers": answers
            }
            return json_obj, None
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, str(e)
    finally:
        if conn:
            release_connection(conn)

def export_sleep_json_by_item(item_id: int, out_dir: str = "./json") -> Tuple[Optional[str], Optional[str]]:
    data, err = build_sleep_survey_json(item_id)
    if err:
        return None, err
    meta = data["meta"]
    out_dir_patient = os.path.join(out_dir, _safe(meta['patient_id']))  # ⬅️ 환자 폴더
    out_path = os.path.join(out_dir_patient, f"{_safe(meta['patient_id'])}_S{meta['seq']}_{_safe(meta['survey_name'])}_item{meta['item_id']}.json")
    ok, err2 = save_json_to_file(data, out_path)
    return (out_path, None) if ok else (None, err2)

def export_sleep_jsons_for_patient(patient_id: str, out_dir: str = "./json") -> Tuple[List[str], List[str]]:
    saved, errors = [], []
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            items = _fetch_patient_items(cur, patient_id, "S_SURVEY")
        for it in items:
            p, err = export_sleep_json_by_item(it["item_id"], out_dir=out_dir)
            (saved.append(p) if p else errors.append(f"item_id={it['item_id']}: {err}"))
    except Exception as e:
        import traceback; traceback.print_exc()
        errors.append(str(e))
    finally:
        if conn:
            release_connection(conn)
    return saved, errors

# -----------------------------
# 환자별 번들 JSON (E/S 각각 1파일로 묶기)
# -----------------------------

def build_emotion_bundle_for_patient(patient_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    환자 한 명의 모든 E_SURVEY를 한 JSON으로 묶어 반환
    구조:
    {
      "patient_id": "...",
      "bundle_type": "E_SURVEY",
      "generated_at": "YYYY-MM-DD HH:MM:SS",
      "surveys": [
        {
          "item_id": ...,
          "seq": 1|2|3,
          "survey_name": "...",
          "collected_at": "...",
          "updated_at": "...",
          "answers": [{question_id, question, answer}, ...]
        }, ...
      ]
    }
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            items = _fetch_patient_items(cur, patient_id, "E_SURVEY")
            if not items:
                return None, "E_SURVEY 항목 없음"

            surveys = []
            for it in items:
                seq_raw = it.get("seq")
                try:
                    seq = int(seq_raw) if seq_raw is not None else None
                except Exception:
                    seq = None
                if not seq:
                    continue
                qmap_src = EMOTION_QMAP_BY_SEQ.get(seq) or QUESTION_REGISTRY.get(("E_SURVEY", seq)) or {}
                id_to_text = _invert_qmap(qmap_src)

                rows = _fetch_answers(cur, it["item_id"])
                answers = [{
                    "question_id": r["question_id"],
                    "question": id_to_text.get(r["question_id"], f"(정의없음 #{r['question_id']})"),
                    "answer": r["answer_value"]
                } for r in rows]

                surveys.append({
                    "item_id": it["item_id"],
                    "seq": seq,
                    "survey_name": _EMO_TITLES.get(seq, f"E_SURVEY-{seq}"),
                    "collected_at": _fmt_ts(it.get("collected_at")),
                    "updated_at": _fmt_ts(it.get("updated_at")),
                    "answers": answers
                })

            if not surveys:
                return None, "E_SURVEY 항목의 유효한 설문이 없음"

            bundle = {
                "patient_id": str(patient_id),
                "bundle_type": "E_SURVEY",
                "generated_at": _fmt_ts(datetime.now()),
                "surveys": surveys
            }
            return bundle, None
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, str(e)
    finally:
        if conn:
            release_connection(conn)

def export_emotion_bundle_for_patient(patient_id: str, out_dir: str = "./json") -> Tuple[Optional[str], Optional[str]]:
    data, err = build_emotion_bundle_for_patient(patient_id)
    if err:
        return None, err
    out_dir_patient = os.path.join(out_dir, _safe(patient_id))  # ⬅️ 환자 폴더
    # 🔻 시간 태그 제거, 항상 같은 파일명
    out_path = os.path.join(out_dir_patient, f"{_safe(patient_id)}_E_bundle.json")
    ok, err2 = save_json_to_file(data, out_path)
    return (out_path, None) if ok else (None, err2)

def build_sleep_bundle_for_patient(patient_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            items = _fetch_patient_items(cur, patient_id, "S_SURVEY")
            if not items:
                return None, "S_SURVEY 항목 없음"

            surveys = []
            for it in items:
                seq_raw = it.get("seq")
                try:
                    seq = int(seq_raw) if seq_raw is not None else None
                except Exception:
                    seq = None
                if not seq:
                    continue
                qmap_src = SLEEP_QMAP_BY_SEQ.get(seq) or QUESTION_REGISTRY.get(("S_SURVEY", seq)) or {}
                id_to_text = _invert_qmap(qmap_src)

                rows = _fetch_answers(cur, it["item_id"])
                answers = [{
                    "question_id": r["question_id"],
                    "question": id_to_text.get(r["question_id"], f"(정의없음 #{r['question_id']})"),
                    "answer": r["answer_value"]
                } for r in rows]

                surveys.append({
                    "item_id": it["item_id"],
                    "seq": seq,
                    "survey_name": _SLEEP_TITLES.get(seq, f"S_SURVEY-{seq}"),
                    "collected_at": _fmt_ts(it.get("collected_at")),
                    "updated_at": _fmt_ts(it.get("updated_at")),
                    "answers": answers
                })

            if not surveys:
                return None, "S_SURVEY 항목의 유효한 설문이 없음"

            bundle = {
                "patient_id": str(patient_id),
                "bundle_type": "S_SURVEY",
                "generated_at": _fmt_ts(datetime.now()),
                "surveys": surveys
            }
            return bundle, None
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, str(e)
    finally:
        if conn:
            release_connection(conn)

def export_sleep_bundle_for_patient(patient_id: str, out_dir: str = "./json") -> Tuple[Optional[str], Optional[str]]:
    data, err = build_sleep_bundle_for_patient(patient_id)
    if err:
        return None, err
    out_dir_patient = os.path.join(out_dir, _safe(patient_id))  # ⬅️ 환자 폴더
    out_path = os.path.join(out_dir_patient, f"{_safe(patient_id)}_S_bundle.json")
    ok, err2 = save_json_to_file(data, out_path)
    return (out_path, None) if ok else (None, err2)

# -----------------------------
# (옵션) 환자 단일 호출로 모두 내보내기 (번들)
# -----------------------------

def export_all_bundles_for_patient(patient_id: str, out_dir: str = "./json") -> Dict[str, Any]:
    e_path, e_err = export_emotion_bundle_for_patient(patient_id, out_dir)
    s_path, s_err = export_sleep_bundle_for_patient(patient_id, out_dir)
    return {
        "emotion_bundle_path": e_path,
        "emotion_bundle_error": e_err,
        "sleep_bundle_path": s_path,
        "sleep_bundle_error": s_err,
    }
