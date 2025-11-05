# utils/meqk.py
from typing import Dict, List, Tuple, Any

def _hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)

def _score_time_by_rules(hour_float: float, rules: List[List[Any]]) -> int:
    """
    hour_float: 0~24 float 시각 (예: 21.5 = 21:30)
    rules: [["HH:MM","HH:MM",score], ...]
    자정 교차 포함 규칙 가능. 구간은 [start, end)로 계산, 마지막 구간은 end 포함 허용.
    """
    minutes = int(hour_float) * 60 + int(round((hour_float % 1) * 60))  # 0~1439
    for idx, (s_str, e_str, score) in enumerate(rules):
        s = _hhmm_to_minutes(s_str) % 1440
        e = _hhmm_to_minutes(e_str) % 1440
        last = (idx == len(rules) - 1)

        if s == e:  # 전구간
            return int(score)

        if s < e:
            inside = (s <= minutes < e) or (last and minutes == e)
        else:
            # 자정 교차 (예: 21:00 ~ 03:00)
            inside = (minutes >= s or minutes < e) or (last and minutes == e)

        if inside:
            return int(score)
    # 매칭 실패 시 0점
    return 0

def compute_meqk(
    answers: Dict[str, float],
    time_rules: Dict[str, List[List[Any]]],
    range_scoring: Dict[str, str],
) -> Dict[str, Any]:
    """
    answers: { "1": 7.5, "2": 23.0, "3": 4, ... }  # slider-time은 float-hour(0~24), radio는 정수 점수
    time_rules: { "1": [["05:00","06:30",5], ...], "2": [...], ... }
    range_scoring: { "16-30":"극단적 저녁형", ... }

    return: { "total": int, "bucket": {"range":"59-69","label":"보통 아침형"} }
    """
    total = 0

    # 1) 시간형 문항 점수 반영
    for qid, rules in time_rules.items():
        q = str(qid)
        if q in answers:
            total += _score_time_by_rules(float(answers[q]), rules)

    # 2) 나머지(라디오/수치형)는 그대로 더함
    for q, v in answers.items():
        if q not in time_rules:
            try:
                total += int(v)
            except Exception:
                total += 0

    # 3) 총점 → 범주 매핑
    bucket = {"range": None, "label": None}
    for rng, label in range_scoring.items():
        if "-" in rng:
            lo, hi = rng.split("-", 1)
            try:
                loi, hii = int(lo), int(hi)
            except Exception:
                continue
            if loi <= total <= hii:
                bucket = {"range": rng, "label": label}
                break

    return {"total": total, "bucket": bucket}
