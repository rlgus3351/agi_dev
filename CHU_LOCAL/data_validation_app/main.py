import os
import subprocess
from datetime import datetime
from api_local.validation_api_local import (
    fetch_new_pd_survey_items,
    fetch_new_pd_video_items,
    validate_survey,
    calculate_parkinson_stage,
    add_or_update_validation
)
from api_local.video_api_local import fetch_video_metadata_by_item_id
from utils.db_utils import get_connection, release_connection
from api_local.patient_api_local import fetch_all_patients
from utils.form_to_json import save_json_to_file,build_mds_updrs_part3_json
from config import WINDOW_PREFIX

# -----------------------------
#  시각 문자열 포맷
# -----------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------
#  경로 변환 함수
# -----------------------------
def normalize_input_path(raw_path: str) -> str:
    if not raw_path:
        return raw_path
    normalized = raw_path.replace("\\", "/")
    lower_path = normalized.lower()
    if WINDOW_PREFIX.lower().replace("\\", "/") in lower_path:
        return normalized.split("parkinson/output/video", 1)[-1]
    return normalized

def post_video_validation(patient_id: str, item_id: int, desc: str, status: str):
    payload = {
        "patient_id": patient_id,
        "item_id": item_id,
        "validation_method": "LocalCheck_PD_VIDEO",
        "validation_description": f"{status}: {desc}",
        "validation_datetime": datetime.now()
    }
    success, result = add_or_update_validation(payload)
    if not success:
        print(f"❌ 검증 결과 저장 실패: {result}")

# -----------------------------
#  영상 로컬 파일 검증
# -----------------------------
def check_video_local(file_path: str, file_ext: str):
    if not os.path.exists(file_path):
        return "FAIL", f"파일 없음: {file_path}"
    actual_ext = os.path.splitext(file_path)[1].lower().replace(".", "")
    if file_ext and actual_ext != file_ext.lower():
        return "FAIL", f"확장자 불일치 ({actual_ext} ≠ {file_ext})"
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        return "FAIL", f"재생 불가 파일: {file_path}"
    return "PASS", f"정상 재생 가능 ({file_ext})"

# -----------------------------
#  실행 파이프라인
# -----------------------------
def run_validation_pipeline():
    start_time = now_str()
    print(f"\n[{start_time}] 🚀 파킨슨병(PD) 데이터 자동 검증 시작...\n")

    patients = fetch_all_patients()
    for p in patients:
        patient_id = p["patient_id"]
        patient_name = p.get("patient_initials", "Unknown")
        print(f"[{now_str()}] 👤 환자: {patient_name} ({patient_id})")

        surveys = fetch_new_pd_survey_items(patient_id)
        for s in surveys:
            item_id = s["item_id"]
            print(f"[{now_str()}] 🧠 설문 신규 데이터 → item_id={item_id}")
            result, err = validate_survey(item_id)
            if err:
                print(f"[{now_str()}]   ⚠️ 오류: {err}")
                continue
            print(f"[{now_str()}]   ⮑ 검증 결과: {result['status']} | {result['description']}")
            if result["status"] == "PASS":
                stage_result, err = calculate_parkinson_stage(item_id)
                if stage_result:
                    print(f"[{now_str()}]   ⮑ 중증도 저장됨: {stage_result['stage_value']} ({stage_result['stage_description']})")
                    json_obj, err = build_mds_updrs_part3_json(item_id)
                    if json_obj:
                    # 필요하면 파일로도 저장
                        JSON_PATH  = r"C:\Users\user\Desktop\DEV_AGI\parkinson\result\survey"
                        out_path = f"{JSON_PATH}/{patient_id}_mds_updrs_part3.json"
                        save_json_to_file(json_obj, out_path)
                        print(f"📝 설문 JSON 저장: {out_path}")
                    else:
                        print(f"⚠️ JSON 생성 실패: {err}")
                else:
                    print(f"[{now_str()}]   ⚠️ 중증도 저장 실패: {err}")

        videos = fetch_new_pd_video_items(patient_id)
        for v in videos:
            item_id = v["item_id"]
            print(f"[{now_str()}] 🎥 영상 신규 데이터 → item_id={item_id}")
            metadata_list = fetch_video_metadata_by_item_id(item_id)
            print(type(metadata_list))
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
