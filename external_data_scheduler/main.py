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
# base_path = r"C:\TeamGit\agi_dev\external_data_scheduler\Parkinson"
base_path = r"C:\TeamGit\agi_dev\external_data_scheduler\Depression"

data_category = "MDD"   # ✅ Parkinson = PD / Depression = MDD


# ============================================
# 2️⃣ 수집 소요시간 파싱 함수
# ============================================
def parse_latency(latency_value):
    """
    collection_latency 필드가 JSON(dict) 또는 문자열일 수 있음.
    dict: {"hours": 1, "minutes": 18, "seconds": 49}
    str : "1h 18m 49s"
    """
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

    else:
        return None


# ============================================
# 3️⃣ 데이터 삽입 함수
# ============================================
def insert_metadata(cur, folder_path, code_name):
    data_type = "qna"
    metadata_dir = os.path.join(folder_path, "qna_metadata")
    qna_dir = os.path.join(folder_path, "qna")

    if not os.path.exists(metadata_dir):
        print(f"⚠️ {code_name}: qna_metadata 폴더 없음")
        return

    for meta_file in os.listdir(metadata_dir):
        if not meta_file.endswith(".json"):
            continue

        meta_path = os.path.join(metadata_dir, meta_file)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # ✅ 메타데이터 파싱
        num_samples = meta.get("num_samples", 0)
        total_sentences = meta.get("total_sentences", 0)
        total_tokens = meta.get("total_tokens", 0)
        reference_missing_rate = meta.get("reference_missing_rate", 0.0)
        doi_missing_rate = meta.get("doi_missing_rate", 0.0)
        verification_missing_rate = meta.get("verification_missing_rate", 0.0)  # ✅ 추가
        collected_date_str = meta.get("collected_date")
        latency_raw = meta.get("collection_latency")

        # ✅ 수집일시 변환
        collected_date = None
        if collected_date_str:
            try:
                collected_date = datetime.strptime(collected_date_str, "%Y-%m-%d %H:%M")
            except ValueError:
                collected_date = None

        # ✅ 수집 소요시간 변환
        collection_latency = parse_latency(latency_raw)

        # ✅ 실제 qna 폴더 용량 계산
        total_bytes = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(qna_dir)
            for f in files
        ) if os.path.exists(qna_dir) else 0
        file_size_mb = total_bytes / (1024 * 1024)

        # ✅ DB insert
        cur.execute("""
            INSERT INTO dev_kkh.tb_external_collection
            (code_name, data_category, data_type, num_samples, total_sentences, total_tokens,
             reference_missing_rate, doi_missing_rate, verification_missing_rate,
             collected_date, collection_latency, file_size_mb)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            code_name,
            data_category,   # ✅ PD / MDD 구분
            data_type,
            num_samples,
            total_sentences,
            total_tokens,
            reference_missing_rate,
            doi_missing_rate,
            verification_missing_rate,  # ✅ 추가
            collected_date,
            collection_latency,
            round(file_size_mb, 2)
        ))

        print(f"✅ {code_name} → {meta_file} 등록 완료 ({file_size_mb:.2f}MB, latency={collection_latency}, verification={verification_missing_rate}, category={data_category})")


# ============================================
# 4️⃣ 메인 실행
# ============================================
if __name__ == "__main__":
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        print(f"🚀 외부 데이터 메타데이터 DB 업로드 시작... ({data_category}, QnA)\n")

        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if not (os.path.isdir(folder_path) and folder.startswith("D-")):  # ✅ Depression은 D- 코드
                continue
            insert_metadata(cur, folder_path, folder)

        conn.commit()
        print("\n🎉 모든 QnA 메타데이터 DB 등록 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        conn.rollback()

    finally:
        cur.close()
        conn.close()
