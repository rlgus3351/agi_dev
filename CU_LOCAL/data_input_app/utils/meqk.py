from typing import Dict, Any

MEQK_REVERSE_MAP = {
    87: 1,
    88: 2,
    89: 3,
    90: 4,
    91: 5,
    92: 6,
    93: 7,
    94: 8,
    95: 9,
    96: 10,
    97: 11,
    98: 12,
    99: 13,
    100: 14,
    101: 15,
    102: 16,
    103: 17,
    104: 18,
    105: 19,
}

Q1_RULE = [
    (5.00, 6.50, 5),
    (6.50, 7.75, 4),
    (7.75, 9.75, 3),
    (9.75, 11.00, 2),
    (11.00, 12.00, 1),
]

Q2_RULE = [
    (20.00, 21.00, 5),
    (21.00, 22.25, 4),
    (22.25, 24.50, 3),
    (24.50, 25.75, 2),
    (25.75, 27.00, 1),
]

Q10_RULE = Q2_RULE

Q17_RULE = [
    (0.00, 4.00, 1),
    (4.00, 8.00, 5),
    (8.00, 9.00, 4),
    (9.00, 14.00, 3),
    (14.00, 17.00, 2),
    (17.00, 24.00, 1),
]

Q18_RULE = [
    (0.00, 5.00, 1),
    (5.00, 8.00, 5),
    (8.00, 10.00, 4),
    (10.00, 17.00, 3),
    (17.00, 22.00, 2),
    (22.00, 24.00, 1),
]


def score_from_ranges(val: float, rules, wrap_midnight=True):
    v = val
    if wrap_midnight and v < 4:
        v += 24
    for s, e, sc in rules:
        if s <= v < e:
            return sc
    return 0


def compute_meqk(answers: Dict[str, Any]) -> int:
    """
    answers: {"1": 11.0, "3": 4, ...}
    - 슬라이더(Q1,2,10,17,18) → 규칙으로 점수 계산
    - 나머지(Q3~Q16,19) → 이미 점수로 들어온 값 그대로 합산
    """
    total = 0

    for q, raw in answers.items():
        original_qid = int(q)
        qid = MEQK_REVERSE_MAP.get(original_qid, original_qid)
        v = float(raw)

        # 슬라이더 5개
        if qid == 1:
            total += score_from_ranges(v, Q1_RULE)
            continue
        if qid == 2:
            total += score_from_ranges(v, Q2_RULE)
            continue
        if qid == 10:
            total += score_from_ranges(v, Q10_RULE)
            continue
        if qid == 17:
            total += score_from_ranges(v, Q17_RULE, wrap_midnight=False)
            continue
        if qid == 18:
            total += score_from_ranges(v, Q18_RULE, wrap_midnight=False)
            continue

        # 나머지는 이미 점수로 변환된 상태 → 그대로 더함
        total += int(v)

    return total


def debug_meqk(answers: Dict[str, Any]) -> Dict[str, Any]:
    results = {}
    total = 0

    for q, raw in answers.items():
        original_qid = int(q)
        qid = MEQK_REVERSE_MAP.get(original_qid, original_qid)
        v = float(raw)

        if qid == 1:
            score = score_from_ranges(v, Q1_RULE)
            results[qid] = {"value": v, "type": "slider(Q1)", "score": score}
            total += score
            continue
        if qid == 2:
            score = score_from_ranges(v, Q2_RULE)
            results[qid] = {"value": v, "type": "slider(Q2)", "score": score}
            total += score
            continue
        if qid == 10:
            score = score_from_ranges(v, Q10_RULE)
            results[qid] = {"value": v, "type": "slider(Q10)", "score": score}
            total += score
            continue
        if qid == 17:
            score = score_from_ranges(v, Q17_RULE, wrap_midnight=False)
            results[qid] = {"value": v, "type": "slider(Q17)", "score": score}
            total += score
            continue
        if qid == 18:
            score = score_from_ranges(v, Q18_RULE, wrap_midnight=False)
            results[qid] = {"value": v, "type": "slider(Q18)", "score": score}
            total += score
            continue

        score = int(v)
        results[qid] = {"value": score, "type": "direct", "score": score}
        total += score

    return {"detail": results, "total": total}
