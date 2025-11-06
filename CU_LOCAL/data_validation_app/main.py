import os
import subprocess
from datetime import datetime
from api_local.validation_api_local import (
    fetch_new_mdd_surveys_7,
    validate_survey,
    add_or_update_validation,
    fetch_new_media_items,
    calculate_depression_stage
)
from api_local.video_api_local import fetch_video_metadata_by_item_id
from utils.db_utils import get_connection, release_connection
from utils.form_to_json_survyes import export_emotion_bundle_for_patient, export_sleep_bundle_for_patient
from api_local.patient_api_local import fetch_all_patients
from config import WINDOW_PREFIX, LOCAL_JSON_DIR


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
        return normalized.split("MDD/output/video", 1)[-1]
    return normalized

def post_video_validation(patient_id: str, item_id: int, desc: str, status: str):
    payload = {
        "patient_id": patient_id,
        "item_id": item_id,
        "validation_method": "LocalCheck_MDD_VIDEO",
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


def check_audio_local(file_path: str, file_ext: str):
    if not os.path.exists(file_path):
        return "FAIL", f"파일 없음: {file_path}"
    actual_ext = os.path.splitext(file_path)[1].lower().replace(".", "")
    if file_ext and actual_ext != file_ext.lower():
        return "FAIL", f"확장자 불일치 ({actual_ext} ≠ {file_ext})"
    try:
        # 오디오 스트림 유무 + duration 확인
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0",
            file_path
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        if not r.stdout.strip():
            return "FAIL", "오디오 스트림 없음"
        # duration 체크(있으면 가점)
        cmd2 = ["ffprobe","-v","error","-show_entries","format=duration",
                "-of","default=noprint_wrappers=1:nokey=1", file_path]
        subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        return "FAIL", f"재생 불가 파일: {file_path}"
    return "PASS", f"정상 재생 가능 ({file_ext})"


def check_text_local(file_path: str, file_ext: str):
    if not os.path.exists(file_path):
        return "FAIL", f"파일 없음: {file_path}"
    actual_ext = os.path.splitext(file_path)[1].lower().replace(".", "")
    if file_ext and actual_ext != file_ext.lower():
        return "FAIL", f"확장자 불일치 ({actual_ext} ≠ {file_ext})"
    try:
        size = os.path.getsize(file_path)
        if size <= 0:
            return "FAIL", "빈 파일(0 byte)"
        # 간단 라인수 체크
        lines = 0
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                lines += 1
        if lines == 0:
            return "FAIL", "텍스트 내용 없음"
    except Exception as e:
        return "FAIL", f"읽기 실패: {e}"
    return "PASS", f"정상 읽기 ({file_ext}), {lines}줄"

def _as_list_and_err(maybe):
    """
    fetch_xxx_by_item_id 가 (list, err) 또는 list 를 반환해도 안전 처리
    """
    if isinstance(maybe, tuple) and len(maybe) == 2:
        return maybe[0], maybe[1]
    return maybe, None
# -----------------------------
#  실행 파이프라인
# -----------------------------
def run_validation_pipeline():
    start_time = now_str()
    print(f"\n[{start_time}] 🚀 MDD 데이터 자동 검증 시작...\n")

    patients = fetch_all_patients()
    for p in patients:
        patient_id = p["patient_id"]
        patient_name = p.get("patient_initials", "Unknown")
        print(f"[{now_str()}] 👤 환자: {patient_name} ({patient_id})")

        # ---------------------------
        # 1) 설문 7건 (E 3종 + S 4종)
        # ---------------------------
        surveys = fetch_new_mdd_surveys_7(patient_id)
        for s in surveys:
            item_id = s["item_id"]
            print(f"[{now_str()}] 🧠 설문 신규/갱신 → item_id={item_id}")
            result, err = validate_survey(item_id)
            if err or not result:
                print(f"[{now_str()}]   ⚠️ 설문 검증 오류: {err or '결과 없음'}")
                continue
            print(f"[{now_str()}]   ⮑ 검증 결과: {result['status']} | {result['description']}")
                # ✅ MADRS만 중증도 저장: E-SURVEY & seq=2
            if result["status"] == "PASS" and (str(s.get("data_type","")).upper() == "E-SURVEY") and int(s.get("seq") or 0) == 2:
                dep_stage, derr = calculate_depression_stage(item_id)
                if dep_stage:
                    print(f"[{now_str()}]   ⮑ MADRS 저장: 총점 {dep_stage['total_score']} / {dep_stage['stage_description']}")
                else:
                    print(f"[{now_str()}]   ⚠️ MADRS 중증도 저장 실패: {derr}")
        # ---------------------------
        # 2) 영상 2건 (MOBILE, WEBCAM)
        # ---------------------------
        video_items = fetch_new_media_items(patient_id, ["MOBILE","WEBCAM"], category="MDD")
        for v in video_items:
            item_id = v["item_id"]
            print(f"[{now_str()}] 🎥 영상 신규/갱신 → item_id={item_id}")
            raw = fetch_video_metadata_by_item_id(item_id)
            metadata_list, err = _as_list_and_err(raw)
            if err:
                print(f"[{now_str()}]   ⚠️ 메타 조회 실패: {err}")
                continue
            if not metadata_list:
                print(f"[{now_str()}]   ⚠️ 메타 없음")
                continue
            for m in metadata_list:
                file_path = m.get("file_path", "")
                file_ext  = m.get("file_ext", "")  # 확장자는 DB에 점 없이 저장하는 현재 관례 유지
                status, desc = check_video_local(file_path, file_ext)
                print(f"[{now_str()}]   ⮑ 검증 결과: {status} | {desc}")
                post_video_validation(patient_id, item_id, desc, status)

        # ---------------------------
        # 3) 음성 1건 (VOICE/AUDIO)
        # ---------------------------
        audio_items = fetch_new_media_items(patient_id, ["VOICE","AUDIO"], category="MDD")
        for a in audio_items:
            item_id = a["item_id"]
            print(f"[{now_str()}] 🔊 음성 신규/갱신 → item_id={item_id}")

            from api_local.audio_api_local import fetch_audio_metadata_by_item_id
            raw = fetch_audio_metadata_by_item_id(item_id)
            metadata_list, err = _as_list_and_err(raw)
            if err:
                print(f"[{now_str()}]   ⚠️ 메타 조회 실패: {err}")
                continue
            if not metadata_list:
                print(f"[{now_str()}]   ⚠️ 메타 없음")
                continue

            for m in metadata_list:
                file_path = m.get("file_path", "")
                # audio_meta에는 보통 file_ext가 이미 들어가 있음 (없으면 경로에서 추출)
                file_ext = m.get("file_ext", "") or os.path.splitext(file_path)[1].lstrip(".").lower()
                status, desc = check_audio_local(file_path, file_ext)
                print(f"[{now_str()}]   ⮑ 검증 결과: {status} | {desc}")
                payload = {
                    "patient_id": patient_id,
                    "item_id": item_id,
                    "validation_method": "LocalCheck_MDD_AUDIO",
                    "validation_description": f"{status}: {desc}",
                    "validation_datetime": datetime.now()
                }
                ok, _res = add_or_update_validation(payload)
                if not ok:
                    print(f"[{now_str()}]   ⚠️ 검증 저장 실패")

        # ---------------------------
        # 4) 텍스트 1건 (TXT/TEXT/FILE)
        # ---------------------------
        text_items = fetch_new_media_items(patient_id, ["TXT","TEXT","FILE"], category="MDD")
        for t in text_items:
            item_id = t["item_id"]
            print(f"[{now_str()}] 📝 텍스트 신규/갱신 → item_id={item_id}")

            from api_local.file_api_local import fetch_file_metadata_by_item_id
            raw = fetch_file_metadata_by_item_id(item_id)
            metadata_list, err = _as_list_and_err(raw)
            if err:
                print(f"[{now_str()}]   ⚠️ 메타 조회 실패: {err}")
                continue
            if not metadata_list:
                print(f"[{now_str()}]   ⚠️ 메타 없음")
                continue

            for m in metadata_list:
                file_path = m.get("file_path", "")
                file_ext = m.get("file_ext", "") or os.path.splitext(file_path)[1].lstrip(".").lower()
                status, desc = check_text_local(file_path, file_ext)
                print(f"[{now_str()}]   ⮑ 검증 결과: {status} | {desc}")
                payload = {
                    "patient_id": patient_id,
                    "item_id": item_id,
                    "validation_method": "LocalCheck_MDD_TEXT",
                    "validation_description": f"{status}: {desc}",
                    "validation_datetime": datetime.now()
                }
                ok, _res = add_or_update_validation(payload)
                if not ok:
                    print(f"[{now_str()}]   ⚠️ 검증 저장 실패")

        print(f"[{now_str()}] ────────────────────────────\n")
        # 환자 p 처리 끝 무렵(텍스트 처리 for 루프가 끝난 직후)에 추가
        e_path, e_err = export_emotion_bundle_for_patient(patient_id, out_dir=LOCAL_JSON_DIR)
        print(f"[{now_str()}] 📦 E 번들: {e_path or e_err}")

        s_path, s_err = export_sleep_bundle_for_patient(patient_id, out_dir=LOCAL_JSON_DIR)
        print(f"[{now_str()}] 📦 S 번들: {s_path or s_err}")

    end_time = now_str()
    print(f"[{end_time}] ✅ 모든 검증 완료!\n")

# -----------------------------
#  실행
# -----------------------------
if __name__ == "__main__":
    run_validation_pipeline()
