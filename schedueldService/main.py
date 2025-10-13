import os
import subprocess
import requests
from datetime import datetime
from config import API_URL

# -----------------------------
#  기본 설정
# -----------------------------
VALIDATION_BASE_URL = API_URL + "validations/"
PATIENTS_BASE_URL = API_URL + "patients/"
VIDEO_BASE_URL = API_URL + "video/"


# -----------------------------
#  서버 데이터 조회 함수
# -----------------------------
def fetch_all_patients():
    """전체 환자 목록 조회"""
    url = PATIENTS_BASE_URL
    res = requests.get(url)
    res.raise_for_status()
    return res.json()


def fetch_new_pd_survey_items(patient_id):
    """PD 설문 (MDS-UPDRS Part 3) 신규 데이터"""
    url = f"{VALIDATION_BASE_URL}pd-new-items/survey/{patient_id}"
    res = requests.get(url)
    if res.status_code == 404:
        return []
    res.raise_for_status()
    return res.json()


def fetch_new_pd_video_items(patient_id):
    """PD 영상 (VIDEO) 신규 데이터"""
    url = f"{VALIDATION_BASE_URL}pd-new-items/video/{patient_id}"
    res = requests.get(url)
    if res.status_code == 404:
        return []
    res.raise_for_status()
    return res.json()


def fetch_video_metadata(item_id):
    """해당 item의 비디오 메타데이터 가져오기"""
    url = f"{VIDEO_BASE_URL}{item_id}"
    res = requests.get(url)
    if res.status_code == 404:
        return []
    res.raise_for_status()
    return res.json()


# -----------------------------
#  검증 함수
# -----------------------------
def validate_survey(patient_id: str, item_id: int):
    """PD 설문 자동 검증 (서버에서 수행 후 결과 기록)"""
    url = f"{VALIDATION_BASE_URL}pd-survey-check/{item_id}"
    res = requests.post(url)
    res.raise_for_status()
    result = res.json()
    return result


def check_video_local(file_path: str, file_ext: str):
    """로컬 환경에서 영상 검증"""
    if not os.path.exists(file_path):
        return "FAIL", f"파일 없음: {file_path}"

    actual_ext = os.path.splitext(file_path)[1].lower().replace(".", "")
    if file_ext and actual_ext != file_ext.lower():
        return "FAIL", f"확장자 불일치 ({actual_ext} ≠ {file_ext})"

    try:
        # ffprobe로 재생 가능 여부 점검
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        return "FAIL", f"재생 불가 파일: {file_path}"

    return "PASS", f"정상 재생 가능 ({file_ext})"


def post_video_validation(patient_id: str, item_id: int, desc: str, status: str):
    """영상 검증 결과 서버로 전송"""
    payload = {
        "patient_id": patient_id,
        "item_id": item_id,
        "validation_method": "LocalCheck_PD_VIDEO",
        "validation_description": f"{status}: {desc}",
        "validation_datetime": datetime.now().isoformat()
    }
    post_validation_result(payload)


def post_validation_result(payload: dict):
    """공통: 서버에 검증 결과 등록 (UPSERT)"""
    res = requests.post(VALIDATION_BASE_URL, json=payload)
    res.raise_for_status()
    return res.json()


# -----------------------------
#  통합 실행 함수
# -----------------------------
def run_validation_pipeline():
    """전체 환자에 대해 설문 + 영상 자동 검증 실행"""
    print("🚀 파킨슨병(PD) 데이터 자동 검증 시작...\n")

    patients = fetch_all_patients()

    for p in patients:
        patient_id = p["patient_id"]
        patient_name = p.get("patient_initials", "Unknown")
        print(f"👤 환자: {patient_name} ({patient_id})")

        # 1️⃣ 설문 신규 데이터 확인 + 검증
        surveys = fetch_new_pd_survey_items(patient_id)
        for s in surveys:
            item_id = s["item_id"]
            print(f"🧠 설문 신규 데이터 → item_id={item_id}")
            result = validate_survey(patient_id, item_id)
            print("   ⮑ 검증 결과:", result["status"], "|", result["description"])

        # 2️⃣ 영상 신규 데이터 확인 + 로컬 검증
        videos = fetch_new_pd_video_items(patient_id)
        for v in videos:
            item_id = v["item_id"]
            print(f"🎥 영상 신규 데이터 → item_id={item_id}")

            # 메타데이터 조회 (여러 개 있을 수 있음)
            metadata_list = fetch_video_metadata(item_id)
            for m in metadata_list:
                file_path = m["file_path"]
                file_ext = m.get("file_ext", "")
                status, desc = check_video_local(file_path, file_ext)
                print(f"   ⮑ 검증 결과: {status} | {desc}")

                # 서버에 결과 기록
                post_video_validation(patient_id, item_id, desc, status)

        print("────────────────────────────\n")

    print("✅ 모든 검증 완료!")


# -----------------------------
#  실행
# -----------------------------
if __name__ == "__main__":
    run_validation_pipeline()
