import os
import json
import psycopg2
from datetime import datetime, timedelta
import re

# ============================================
# 1️⃣ DB 연결 설정
# ============================================
DB_CONFIG = {
    "host": "121.178.59.41",
    "port": "45432",
    "dbname": "agi_dev",
    "user": "kkh",
    "password": "Rkskekfk1!"
}

# ✅ 폴더 경로 (질환별로 바꿔서 실행)
base_path = r"C:\TeamGit\agi_dev\external_data_scheduler\Depression"
data_category = "MDD"  # Parkinson = PD / Depression = MDD


# ============================================
# 2️⃣ 수집 소요시간 파싱 함수
# ============================================
def parse_latency(latency_value):
    """collection_latency 필드가 JSON(dict) 또는 문자열일 수 있음."""
    if isinstance(latency_value, dict):
        h = latency_value.get("hours", 0)
        m = latency_value.get("minutes", 0)
        s = latency_value.get("seconds", 0)
        return timedelta(hours=h, minutes=m, seconds=s)

    elif isinstance(latency_value, str):
        h = m = s = 0
        h_match = re.search(r"(\d+)h", latency_value)
        m_match = re.search(r"(\d+)m", latency_value)
        s_match = re.search(r"(\d+)s", latency_value)
        if h_match: h = int(h_match.group(1))
        if m_match: m = int(m_match.group(1))
        if s_match: s = int(s_match.group(1))
        return timedelta(hours=h, minutes=m, seconds=s)
    return None


# ============================================
# 3️⃣ 종료 시각 파싱 함수 (collected_date → checked_at)
# ============================================
def parse_checked_at(collected_date_str):
    """
    collected_date 예: '2025-09-04 09:37 ~ 2025-09-05 11:28'
    → 종료 시간(오른쪽 값)을 datetime으로 반환
    """
    if not collected_date_str:
        return None

    try:
        if "~" in collected_date_str:
            _, end_str = [s.strip() for s in collected_date_str.split("~")]
            return datetime.strptime(end_str, "%Y-%m-%d %H:%M")
        else:
            return datetime.strptime(collected_date_str.strip(), "%Y-%m-%d %H:%M")
    except Exception:
        return None


# ============================================
# 4️⃣ 검증 데이터 삽입 함수
# ============================================
def insert_validation_metadata(cur, folder_path, code_name):
    merged_dir = os.path.join(folder_path, "merged_jsons")

    if not os.path.exists(merged_dir):
        print(f"⚠️ {code_name}: merged_jsons 폴더 없음")
        return

    for meta_file in os.listdir(merged_dir):
        if not meta_file.endswith("_metadata.json"):
            continue

        meta_path = os.path.join(merged_dir, meta_file)

        # ✅ 매칭되는 실제 데이터 파일 찾기
        data_file = meta_file.replace("_metadata.json", ".json")
        data_path = os.path.join(merged_dir, data_file)

        # ✅ 데이터 타입 자동 판별
        if "_corpus" in meta_file.lower():
            data_type = "corpus"
        elif "_qna" in meta_file.lower():
            data_type = "qna"
        else:
            data_type = "unknown"

        # ✅ 메타데이터 읽기
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        reference_missing_rate = meta.get("reference_missing_rate", 0.0)
        doi_missing_rate = meta.get("doi_missing_rate", 0.0)
        verification_missing_rate = meta.get("verification_missing_rate", 0.0)
        latency_raw = meta.get("collection_latency")
        collection_latency = parse_latency(latency_raw)

        # ✅ 종료 시각 (검증 완료 시간)
        collected_date_str = meta.get("collected_date")
        checked_at = parse_checked_at(collected_date_str)

        # ✅ 파일 크기 (메타데이터 + 실제 데이터 합산)
        total_bytes = 0
        if os.path.exists(meta_path):
            total_bytes += os.path.getsize(meta_path)
        if os.path.exists(data_path):
            total_bytes += os.path.getsize(data_path)
        file_size_mb = total_bytes / (1024 * 1024)

        # ✅ 검증 점수 계산 (단순화: 누락률 평균 기반)
        try:
            total_missing = (
                reference_missing_rate + doi_missing_rate + verification_missing_rate
            ) / 3
            validation_score = round(max(0.0, 1.0 - total_missing), 4)
        except Exception:
            validation_score = None

        # ✅ 검증 결과 판정
        if validation_score is None:
            validation_result = "UNKNOWN"
        elif validation_score >= 0.95:
            validation_result = "PASS"
        elif validation_score >= 0.8:
            validation_result = "WARNING"
        else:
            validation_result = "FAIL"

        # ✅ DB INSERT
        cur.execute("""
            INSERT INTO dev_kkh.tb_external_validation
            (code_name, data_category, data_type,
             validation_type, validation_result, validation_score,
             verification_missing_rate, reference_missing_rate, doi_missing_rate,
             file_size_mb, collection_latency, checked_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            code_name,
            data_category,
            data_type,
            "auto_quality_check",  # 자동 검증
            validation_result,
            validation_score,
            verification_missing_rate,
            reference_missing_rate,
            doi_missing_rate,
            round(file_size_mb, 2),
            collection_latency,
            checked_at  # ✅ 수집 종료 시간 기록
        ))

        print(
            f"✅ {code_name} → {meta_file} 등록 완료 "
            f"({data_type}, {file_size_mb:.2f}MB, score={validation_score}, "
            f"result={validation_result}, checked_at={checked_at})"
        )


# ============================================
# 5️⃣ 메인 실행
# ============================================
if __name__ == "__main__":
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        print(f"🚀 외부 데이터 검증 메타데이터 DB 업로드 시작... ({data_category})\n")

        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if not (os.path.isdir(folder_path) and folder.startswith(("P-", "D-"))):
                continue
            insert_validation_metadata(cur, folder_path, folder)

        conn.commit()
        print("\n🎉 모든 검증 메타데이터 DB 등록 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        conn.rollback()

    finally:
        cur.close()
        conn.close()
