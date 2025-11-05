# utils/psqi.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

# 총점 : 
# Q1 : id 1번 항목 00:00 ~ 23:30 30분단위 -> 23.5이렇게 들어옴.
# Q2 : id 2번항목 : 0 ~ 60분 사이
# Q3 : id 3번항목 : 시간단위 0 ~ 23
# Q4 : id 4번항목 : 실제 수면 시간 0 ~ 23
# Q5a : id 5a번항목 : 0,1,2,3
# Q5b : id 5b번항목 : 0,1,2,3
# Q5c : id 5c번항목 : 0,1,2,3
# Q5d : id 5d번항목 : 0,1,2,3
# Q5e : id 5e번항목 : 0,1,2,3
# Q5f : id 5f번항목 : 0,1,2,3
# Q5g : id 5g번항목 : 0,1,2,3
# Q5h : id 5h번항목 : 0,1,2,3
# Q5i : id 5i번항목 : 0,1,2,3
# Q5j : id 5j번항목 : 0,1,2,3
# Q6 :  id 6번항목 : 0,1,2,3
# Q7 :  id 7번항목 : 0,1,2,3
# Q8 :  id 8번항목 : 0,1,2,3
# Q9 :  id 9번항목 : 0,1,2,3
# TIB : (Q3+24)- Q2
# VSE : (Q4/TIB) * 100
# C2 = { Q3 <=15 : 0 , 15< Q3 <31 : 1 , 31 <= Q3 <=60 : 2 ,Q3>60 : 3}
# 5A+Q2 = {Q5a + c2}
# 5b:5j = {Q5b+Q5c+Q5d+Q5e+Q5f+Q5g+Q5h+Q5i+Q5j}
# day_time = { Q7+Q8 }
# SQ = Q9
# SL = { (5A+Q2) <1 : 0 , 0 < (5A+Q2) <3 : 1 , 2< (5A+Q2) < 5 : 2 ,4<(5A+Q2)<7 : 3}
# SD = { Q4 > 7 : 0 , 6 < Q4 <= 7 : 1, 5<= Q4 <= 6 : 2, Q4 < 5 : 3}
# SE = {VSE>85 : 0, 75<=VSE<85 : 1, 65<=VSE<75 : 2, VSE < 65 : 3}
# DS = {(5b:5j) < 1 : 0 , 0 <(5b:5j)<10 : 1, 9 < (5b:5j) <19 : 2, 18 < (5b:5j) < 28 : 3 }
# MED = Q6
# DAY = { (day_time) < 1 : 0, 0 < (day_time) < 3 : 1, 2 < (day_time) < 5 :2, 4 < (day_time) < 7 : 3}
# GPSQI = (SQ+SL+SD+SE+DS+MED+DAY)

# =========================
# Helpers
# =========================
def _to_float_hour(x: Any) -> float:
    """
    23.5 혹은 '23:30' → 시간(float)로 변환.
    예) 23.5, '23:00', '07:30'
    """
    if x is None or x == "":
        raise ValueError("missing time value")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if ":" in s:
        hh, mm = s.split(":")
        return int(hh) + int(mm) / 60.0
    return float(s)


def _safe_int(x: Any, default: int = 0) -> int:
    if x in (None, ""):
        return default
    try:
        return int(x)
    except Exception:
        return default


# =========================
# Result container
# =========================
@dataclass
class PSQIResult:
    # 입력 원자료
    Q1: float  # 취침시각(시간)
    Q2: float  # 잠들기까지(분)
    Q3: float  # 기상시각(시간)
    Q4: float  # 실제수면(시간)
    Q5a: int; Q5b: int; Q5c: int; Q5d: int; Q5e: int
    Q5f: int; Q5g: int; Q5h: int; Q5i: int; Q5j: int
    Q6: int; Q7: int; Q8: int; Q9: int

    # 파생지표
    TIB_hours: float
    VSE_percent: float
    C2_from_Q2: int
    fiveA_plus_Q2: int
    sum_5b_to_5j: int
    day_time_sum: int

    # 도메인 점수
    SQ: int
    SL: int
    SD: int
    SE: int
    DS: int
    MED: int
    DAY: int

    # 최종
    GPSQI: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "inputs": {
                "Q1": self.Q1, "Q2": self.Q2, "Q3": self.Q3, "Q4": self.Q4,
                "Q5a": self.Q5a, "Q5b": self.Q5b, "Q5c": self.Q5c, "Q5d": self.Q5d, "Q5e": self.Q5e,
                "Q5f": self.Q5f, "Q5g": self.Q5g, "Q5h": self.Q5h, "Q5i": self.Q5i, "Q5j": self.Q5j,
                "Q6": self.Q6, "Q7": self.Q7, "Q8": self.Q8, "Q9": self.Q9,
            },
            "derived": {
                "TIB_hours": round(self.TIB_hours, 2),
                "VSE_percent": round(self.VSE_percent, 2),
                "C2_from_Q2": self.C2_from_Q2,
                "fiveA_plus_Q2": self.fiveA_plus_Q2,
                "sum_5b_to_5j": self.sum_5b_to_5j,
                "day_time_sum": self.day_time_sum,
            },
            "domains": {
                "SQ": self.SQ, "SL": self.SL, "SD": self.SD,
                "SE": self.SE, "DS": self.DS, "MED": self.MED, "DAY": self.DAY,
            },
            "GPSQI": self.GPSQI,
        }


# =========================
# Core
# =========================
def compute_psqi(answers: Dict[str, Any]) -> PSQIResult:
    """
    PSQI(피츠버그 수면질 척도) 계산기.

    Parameters
    ----------
    answers : dict
        {
          "1": 23.5 or "23:30",   # 취침시각(시간)
          "2": 30,                # 잠들기까지(분)
          "3": 7 or "07:00",      # 기상시각(시간)
          "4": 6,                 # 실제 수면시간(시간)
          "5-a": 0..3, ..., "5-j": 0..3,
          "6": 0..3, "7": 0..3, "8": 0..3, "9": 0..3
        }

    Returns
    -------
    PSQIResult
    """

    # ---- 원자료 파싱 ----
    q1 = _to_float_hour(answers.get("1"))
    q2 = float(answers.get("2"))
    q3 = _to_float_hour(answers.get("3"))
    q4 = float(answers.get("4"))

    q5a = _safe_int(answers.get("5-a")); q5b = _safe_int(answers.get("5-b"))
    q5c = _safe_int(answers.get("5-c")); q5d = _safe_int(answers.get("5-d"))
    q5e = _safe_int(answers.get("5-e")); q5f = _safe_int(answers.get("5-f"))
    q5g = _safe_int(answers.get("5-g")); q5h = _safe_int(answers.get("5-h"))
    q5i = _safe_int(answers.get("5-i")); q5j = _safe_int(answers.get("5-j"))

    q6 = _safe_int(answers.get("6"))
    q7 = _safe_int(answers.get("7"))
    q8 = _safe_int(answers.get("8"))
    q9 = _safe_int(answers.get("9"))

    # ---- 파생지표 ----
    # TIB: 침대에 누워있는 총 시간(자정 넘김 고려)
    tib = (q3 + 24) - q1
    vse = (q4 / tib) * 100.0 if tib > 0 else 0.0

    # C2: 잠들기까지 지연 분류(분)
    # {<=15:0, 16~30:1, 31~60:2, >60:3}
    if q2 <= 15:
        c2 = 0
    elif q2 <= 30:
        c2 = 1
    elif q2 <= 60:
        c2 = 2
    else:
        c2 = 3

    fiveA_plus_Q2 = q5a + c2
    five_b_to_j = q5b + q5c + q5d + q5e + q5f + q5g + q5h + q5i + q5j
    day_time_sum = q7 + q8

    # ---- 도메인 점수 ----
    SQ = q9

    # SL: {(5A+Q2)<=0:0, 1~2:1, 3~4:2, >=5:3}
    if fiveA_plus_Q2 <= 0:
        SL = 0
    elif fiveA_plus_Q2 <= 2:
        SL = 1
    elif fiveA_plus_Q2 <= 4:
        SL = 2
    else:
        SL = 3

    # SD: {>7:0, (6,7]:1, [5,6]:2, <5:3}
    if q4 > 7:
        SD = 0
    elif q4 > 6:
        SD = 1
    elif q4 >= 5:
        SD = 2
    else:
        SD = 3

    # SE: {>85:0, 75~85:1, 65~75:2, <65:3}
    if vse > 85:
        SE = 0
    elif vse >= 75:
        SE = 1
    elif vse >= 65:
        SE = 2
    else:
        SE = 3

    # DS: {(sum)<1:0, 1~9:1, 10~18:2, 19~27:3}
    if five_b_to_j <= 0:
        DS = 0
    elif five_b_to_j <= 9:
        DS = 1
    elif five_b_to_j <= 18:
        DS = 2
    else:
        DS = 3

    MED = q6

    # DAY: {<=0:0, 1~2:1, 3~4:2, >=5:3}
    if day_time_sum <= 0:
        DAY = 0
    elif day_time_sum <= 2:
        DAY = 1
    elif day_time_sum <= 4:
        DAY = 2
    else:
        DAY = 3

    gpsqi = SQ + SL + SD + SE + DS + MED + DAY

    return PSQIResult(
        Q1=q1, Q2=q2, Q3=q3, Q4=q4,
        Q5a=q5a, Q5b=q5b, Q5c=q5c, Q5d=q5d, Q5e=q5e,
        Q5f=q5f, Q5g=q5g, Q5h=q5h, Q5i=q5i, Q5j=q5j,
        Q6=q6, Q7=q7, Q8=q8, Q9=q9,
        TIB_hours=round(tib, 2),
        VSE_percent=round(vse, 2),
        C2_from_Q2=c2,
        fiveA_plus_Q2=fiveA_plus_Q2,
        sum_5b_to_5j=five_b_to_j,
        day_time_sum=day_time_sum,
        SQ=SQ, SL=SL, SD=SD, SE=SE, DS=DS, MED=MED, DAY=DAY,
        GPSQI=int(gpsqi),
    )
