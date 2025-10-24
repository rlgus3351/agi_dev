import json

def format_mds_answers(answers_raw, survey_type=None):
    """설문 유형별로 맞춤 요약"""
    if survey_type == "BASIC":
        return format_basic_survey(answers_raw)
    elif survey_type == "EMOTION":
        return format_emotion_survey(answers_raw)
    elif survey_type == "SLEEP":
        return format_sleep_survey(answers_raw)
    else:
        return format_generic_survey(answers_raw)


# 🩺 1️⃣ 기초 평가
def format_basic_survey(answers_raw):
    """기초 평가 (B-SURVEY): 단답/숫자형 중심"""
    summary_lines = []
    for ans in answers_raw:
        q = ans.get("question_name") or ans.get("question_id")
        val = ans.get("answer_value", "")
        if q and val:
            summary_lines.append(f"{q}: {val}")
    return "\n".join(summary_lines)


# 💭 2️⃣ 정서 설문 (PHQ-9, MADRS 등)
def format_emotion_survey(answers_raw):
    """정서 설문 (E-SURVEY): 점수 기반"""
    total_score = 0
    count = 0
    lines = []

    for ans in answers_raw:
        q = ans.get("question_name", "")
        val = str(ans.get("answer_value", "")).strip()
        lines.append(f"{q}: {val}")
        if val.isdigit():
            total_score += int(val)
            count += 1

    # 평균/총점 요약
    avg_score = total_score / count if count > 0 else 0
    level = emotion_level(total_score)
    lines.append(f"\n총점: {total_score}점 ({level}) / 평균: {avg_score:.1f}")
    return "\n".join(lines)


def emotion_level(score):
    """간단한 PHQ9/MADRS 기준 예시"""
    if score < 5:
        return "정상"
    elif score < 10:
        return "경도"
    elif score < 15:
        return "중등도"
    elif score < 20:
        return "중증"
    else:
        return "극중증"


# 🌙 3️⃣ 수면 설문 (PSQI, ISI, KESS 등)
def format_sleep_survey(answers_raw):
    """수면 설문 (S-SURVEY): 총점 + 수준"""
    total_score = 0
    lines = []

    for ans in answers_raw:
        q = ans.get("question_name", "")
        val = str(ans.get("answer_value", "")).strip()
        lines.append(f"{q}: {val}")
        if val.isdigit():
            total_score += int(val)

    # 예시 ISI 기준으로 분류
    if total_score < 8:
        level = "정상"
    elif total_score < 15:
        level = "경도 수면장애"
    elif total_score < 22:
        level = "중등도 수면장애"
    else:
        level = "중증 수면장애"

    lines.append(f"\n총점: {total_score}점 / 수면 상태: {level}")
    return "\n".join(lines)


# 🧩 4️⃣ 기본형 (MDS 등)
def format_generic_survey(answers_raw):
    """일반 설문 / 디버깅용"""
    try:
        return json.dumps(answers_raw, ensure_ascii=False, indent=2)
    except Exception:
        return str(answers_raw)
