import os
import subprocess
import requests
from datetime import datetime
from config import API_URL, WINDOW_PREFIX, CONTAINER_PREFIX

# -----------------------------
#  기본 설정
# -----------------------------
VALIDATION_BASE_URL = API_URL + "validations/"
PATIENTS_BASE_URL = API_URL + "patients/"
VIDEO_BASE_URL = API_URL + "video/"

def now_str():
    """현재 시각 문자열 (YYYY-MM-DD HH:MM:SS)"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------
#  서버 데이터 조회 함수
# -----------------------------
def fetch_all_patients():
    url = PATIENTS_BASE_URL
    res = requests.get(url)
    res.raise_for_status()
    return res.json()

def fetch_new_pd_survey_items(patient_id):
    url = f"{VALIDATION_BASE_URL}pd-new-items/survey/{patient_id}"
    res = requests.get(url)
    if res.status_code == 404:
        return []
    res.raise_for_status()
    return res.json()

def fetch_new_pd_video_items(patient_id):
    url = f"{VALIDATION_BASE_URL}pd-new-items/video/{patient_id}"
    res = requests.get(url)
    if res.status_code == 404:
        return []
    res.raise_for_status()
    return res.json()

def fetch_video_metadata(item_id):
    url = f"{VIDEO_BASE_URL}{item_id}"
    res = requests.get(url)
    if res.status_code == 404:
        return []
    res.raise_for_status()
    return res.json()

def extract_stage_from_survey(item_id: int):
    query_url = f"{VALIDATION_BASE_URL}pd-stage/{item_id}"
    res = requests.get(query_url)
    if res.status_code == 404:
        return None
    res.raise_for_status()
    return res.json()

def calculate_and_save_stage(item_id: int):
    url = f"{VALIDATION_BASE_URL}pd-stage-calc/{item_id}"
    res = requests.post(url)
    if res.status_code == 404:
        return None
    res.raise_for_status()
    return res.json()

# -----------------------------
#  검증 함수
# -----------------------------
def validate_survey(patient_id: str, item_id: int):
    url = f"{VALIDATION_BASE_URL}pd-survey-check/{item_id}"
    res = requests.post(url)
    res.raise_for_status()
    return res.json()

def normalize_input_path(raw_path: str) -> str:
    """
    윈도우 로컬 경로를 Docker 컨테이너 내부 경로로 변환합니다.
    """
    if not raw_path:
        return raw_path

    normalized = raw_path.replace("\\", "/")
    lower_path = normalized.lower()

    if WINDOW_PREFIX.lower().replace("\\", "/") in lower_path:
        return CONTAINER_PREFIX + normalized.split("parkinson/output/video", 1)[-1]

    return normalized

def check_video_local(file_path: str, file_ext: str):
    """Docker 컨테이너 환경에서도 경로 변환 후 영상 검증"""

    # ✅ 1. Windows 경로 → Docker 내부 경로로 변환 (config 기반)
    file_path = normalize_input_path(file_path)

    # ✅ 2. 파일 존재 여부 확인
    if not os.path.exists(file_path):
        return "FAIL", f"파일 없음: {file_path}"

    # ✅ 3. 확장자 확인
    actual_ext = os.path.splitext(file_path)[1].lower().replace(".", "")
    if file_ext and actual_ext != file_ext.lower():
        return "FAIL", f"확장자 불일치 ({actual_ext} ≠ {file_ext})"

    # ✅ 4. ffprobe로 영상 유효성 검사
    try:
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


# def check_video_local(file_path: str, file_ext: str):
#     """Docker 컨테이너 환경에서도 경로 변환 후 영상 검증"""

#     # ✅ 1. Windows 경로 → Docker 내부 경로로 변환
#     normalized = file_path.replace("\\", "/")  # 슬래시 통일 먼저
#     if "teamgit/agi_dev/parkinson/output/video" in normalized.lower():
#         # 대소문자 구분 없이 교체
#         file_path = "/app/input_videos" + normalized.split("parkinson/output/video", 1)[-1]
#     else:
#         file_path = normalized

#     # ✅ 2. 파일 존재 여부 확인
#     if not os.path.exists(file_path):
#         return "FAIL", f"파일 없음: {file_path}"

#     # ✅ 3. 확장자 확인
#     actual_ext = os.path.splitext(file_path)[1].lower().replace(".", "")
#     if file_ext and actual_ext != file_ext.lower():
#         return "FAIL", f"확장자 불일치 ({actual_ext} ≠ {file_ext})"

#     # ✅ 4. ffprobe로 영상 유효성 검사
#     try:
#         cmd = [
#             "ffprobe", "-v", "error",
#             "-show_entries", "format=duration",
#             "-of", "default=noprint_wrappers=1:nokey=1",
#             file_path
#         ]
#         subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
#     except Exception:
#         return "FAIL", f"재생 불가 파일: {file_path}"

#     return "PASS", f"정상 재생 가능 ({file_ext})"

def post_video_validation(patient_id: str, item_id: int, desc: str, status: str):
    payload = {
        "patient_id": patient_id,
        "item_id": item_id,
        "validation_method": "LocalCheck_PD_VIDEO",
        "validation_description": f"{status}: {desc}",
        "validation_datetime": datetime.now().isoformat()
    }
    post_validation_result(payload)

def post_validation_result(payload: dict):
    res = requests.post(VALIDATION_BASE_URL, json=payload)
    res.raise_for_status()
    return res.json()

# -----------------------------
#  통합 실행 함수
# -----------------------------
def run_validation_pipeline():
    start_time = now_str()
    print(f"\n[{start_time}] 🚀 파킨슨병(PD) 데이터 자동 검증 시작...\n")

    patients = fetch_all_patients()
    for p in patients:
        patient_id = p["patient_id"]
        patient_name = p.get("patient_initials", "Unknown")
        print(f"[{now_str()}] 👤 환자: {patient_name} ({patient_id})")

        # 설문 데이터
        surveys = fetch_new_pd_survey_items(patient_id)
        for s in surveys:
            item_id = s["item_id"]
            print(f"[{now_str()}] 🧠 설문 신규 데이터 → item_id={item_id}")
            result = validate_survey(patient_id, item_id)
            print(f"[{now_str()}]   ⮑ 검증 결과: {result['status']} | {result['description']}")
        
            if result["status"] == "PASS":
                stage = calculate_and_save_stage(item_id)
                if stage:
                    print(f"[{now_str()}]   ⮑ 중증도 저장됨: {stage['stage_value']} ({stage['stage_description']})")
                else:
                    print(f"[{now_str()}]   ⚠️ 설문 응답 없음 — 중증도 저장 생략")

        # 영상 데이터
        videos = fetch_new_pd_video_items(patient_id)
        for v in videos:
            item_id = v["item_id"]
            print(f"[{now_str()}] 🎥 영상 신규 데이터 → item_id={item_id}")
            metadata_list = fetch_video_metadata(item_id)

            for m in metadata_list:
                file_path = m["file_path"]
                file_ext = m.get("file_ext", "")
                status, desc = check_video_local(file_path, file_ext)
                print(f"[{now_str()}]   ⮑ 검증 결과: {status} | {desc}")
                post_video_validation(patient_id, item_id, desc, status)

        print(f"[{now_str()}] ────────────────────────────\n")

    end_time = now_str()
    print(f"[{end_time}] ✅ 모든 검증 완료!\n")

# -----------------------------
#  실행
# -----------------------------
if __name__ == "__main__":
    run_validation_pipeline()
