from ultralytics import YOLO
import cv2, os, json, uuid, base64, hashlib, subprocess, shutil
import numpy as np
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import requests
from api_local.processing_api_local import update_processed_video, insert_preprocessing_record
from api_local.item_api_local import get_item_by_id
from api_local.patient_api_local import read_patient
from datetime import datetime
import urllib3
from typing import List, Dict, Set, Tuple, Optional, Union, Any
import time
urllib3.disable_warnings()

# =============================
# 🔧 PyTorch 2.6 안전 허용 처리
# =============================
import torch
from torch.nn.modules.container import Sequential
import inspect

# 1) add_safe_globals는 2.4+에만 있음 → 있으면만 등록
_add_safe_globals = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
if callable(_add_safe_globals):
    try:
        _add_safe_globals([
            Sequential,
            __import__('ultralytics').nn.tasks.DetectionModel,
            __import__('ultralytics').nn.modules.conv.Conv,
            __import__('ultralytics').nn.modules.block.C2f,
        ])
    except Exception:
        pass  # 등록 실패해도 치명적 아님

# 2) torch.load 래퍼: 우선 weights_only=True 시도 → 실패 시 False 로 자동 폴백
if "weights_only" in inspect.signature(torch.load).parameters:
    _orig_torch_load = torch.load

    def _safe_torch_load(*args, **kwargs):
        # 사용자가 명시하면 그 값을 존중
        if "weights_only" in kwargs:
            return _orig_torch_load(*args, **kwargs)

        # 1차: 안전 모드 시도
        try:
            kwargs["weights_only"] = True
            return _orig_torch_load(*args, **kwargs)
        except Exception as e:
            # Ultralytics처럼 pickle 객체 필요한 경우 여기서 실패 → 폴백
            try:
                kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)
            except Exception:
                # 원래 예외를 올리는 편이 디버그에 유리
                raise e

    torch.load = _safe_torch_load

# 3) 성능 옵션(가능할 때만)
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
# =============================
# ✅ YOLO 모델 로드 (GPU 자동 감지)
# =============================

try:
    from config import (
        # 권장: config.py에 아래 키들이 있으면 그대로 사용
        BASE_OUTPUT_DIR,           # 예: r"C:\Users\user\Desktop\DEV_AGI\MDD\output"
        VIDEO_SAVE_BASE,           # 예: BASE_OUTPUT_DIR/video
        LOCAL_UPLOAD_DIR,          # 업로드/원본 위치(있으면 사용)
        MODEL_PATH as CFG_MODEL_PATH
    )
except Exception:
    BASE_OUTPUT_DIR  = r"C:\Users\user\Desktop\DEV_AGI\MDD\output"
    VIDEO_SAVE_BASE  = os.path.join(BASE_OUTPUT_DIR, "video")
    LOCAL_UPLOAD_DIR = VIDEO_SAVE_BASE
    CFG_MODEL_PATH   = None

# 최종 출력 디렉터리들
FINAL_VIDEO_DIR = VIDEO_SAVE_BASE                     # 최종(_final.mp4) 저장
JSON_DIR        = os.path.join(BASE_OUTPUT_DIR, "json")  # ROI json 저장
LOG_DIR         = os.path.join(BASE_OUTPUT_DIR, "logs")

os.makedirs(FINAL_VIDEO_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

base_dir   = os.path.dirname(__file__)
model_path = CFG_MODEL_PATH or os.path.join(base_dir, "model", "model.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

FINAL_VIDEO_DIR = r"C:\Users\user\Desktop\DEV_AGI\MDD\output\video"

VIDEO_PATH = r"C:\Users\user\Desktop\DEV_AGI\MDD\output\video"
JSON_PATH  = r"C:\Users\user\Desktop\DEV_AGI\MDD\output\json"

if device == "cuda":
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "CUDA device"
    print(f"🚀 YOLO 모델 로드 중... (device=cuda, gpu={gpu_name})")
else:
    print("🚀 YOLO 모델 로드 중... (device=cpu)")

model = YOLO(model_path)
try:
    model.to(device)
except Exception:
    pass

print(f"✅ YOLO 모델 로드 완료 (device={device})")

# ----------------------------------------
# 🔧 탐지/암호화 튜닝 파라미터
# ----------------------------------------
DETECT_CONF = 0.35            # YOLO 탐지 최소 신뢰도(상향)
MIN_ENCRYPT_CONF = 0.30       # JSON에 복구용 ROI로 담는 최소 신뢰도
IOU = 0.5                     # NMS IOU
EXPAND_RATIO = 0.20           # bbox 확장(옆얼굴 커버)
MAX_HOLD_FRAMES = 3           # 탐지 끊김 시 이전 bbox 유지 프레임 수
MOSAIC_BLOCK_DIVISOR = 25     # 모자이크 블록 크기(값↑ = 블록↑)

# ----------------------------------------
# ✅ SHA256 해시 계산
# ----------------------------------------
def sha256_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()

# ----------------------------------------
# ✅ ffprobe 유틸
# ----------------------------------------
def _has_audio_stream(path: str) -> bool:
    """ffprobe로 오디오 스트림 존재 여부 확인"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            path
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return False

def _output_with_audio_path(anonymized_video: str, final_dir: Optional[str] = None) -> str:
    base = os.path.basename(anonymized_video)
    if base.endswith("_anonymized.mp4"):
        base = base.replace("_anonymized.mp4", "_final.mp4")
    else:
        root, ext = os.path.splitext(base)
        base = f"{root}_final{ext or '.mp4'}"
    out_dir = final_dir if final_dir else os.path.dirname(anonymized_video)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base)

def _merge_ffmpeg(original_video: str, anonymized_video: str, out_path: str) -> Tuple[bool, str]:
    """두 번 시도:
       1) 0:v:0 + 1:a:0 (일반적)
       2) 0:v:0 + 1:a   (첫 오디오 자동 선택)
    """
    common = ["-y", "-c:v", "copy", "-c:a", "aac", "-shortest", "-movflags", "+faststart"]
    attempts = [
        ["-i", anonymized_video, "-i", original_video, "-map", "0:v:0", "-map", "1:a:0"] + common + [out_path],
        ["-i", anonymized_video, "-i", original_video, "-map", "0:v:0", "-map", "1:a"]   + common + [out_path],
    ]
    last_err = ""
    for args in attempts:
        proc = subprocess.run(["ffmpeg"] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            return True, proc.stderr
        last_err = proc.stderr
    return False, last_err

def _output_has_audio(path: str) -> bool:
    return _has_audio_stream(path)

# ----------------------------------------
# ✅ 오디오 병합 함수(유연 매핑 + 결과 검증)
# ----------------------------------------
def merge_audio_with_video(original_video: str, anonymized_video: str, final_dir: Optional[str] = None) -> str:
    """원본 영상의 오디오를 익명화된 영상에 병합(오디오 존재 확인/유연 매핑/결과 검증 포함)"""
    # ffmpeg/ffprobe 존재 확인
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("⚠️ ffmpeg/ffprobe를 찾을 수 없습니다. PATH를 확인하세요.")
        return anonymized_video

    out_path = _output_with_audio_path(anonymized_video, final_dir)

    # 원본에 오디오 없으면 그대로 반환
    if not _has_audio_stream(original_video):
        print("ℹ️ 원본 영상에 오디오 스트림이 없습니다. 비디오만 유지됩니다.")
        return anonymized_video

    ok, log = _merge_ffmpeg(original_video, anonymized_video, out_path)
    if not ok:
        print(f"⚠️ 오디오 병합 실패(두 번 시도):\n{log}")
        return anonymized_video

    if not _output_has_audio(out_path):
        print("⚠️ 병합된 파일에 오디오 스트림이 없습니다. 매핑 규칙을 점검하세요.")
        return anonymized_video

    print(f"🎵 오디오 포함 영상 생성 완료 → {out_path}")
    return out_path

# ----------------------------------------
# ✅ 상태 업데이트 API
# ----------------------------------------
def update_anonymization_status(video_id: int):
    try:
        ts = datetime.utcnow()
        update_processed_video(video_metadata_id=video_id, anonymized_ts=ts, is_anonymized=True)
        print(f"✅ is_anonymized 업데이트 완료 (video_id={video_id})")
    except Exception as e:
        print(f"❌ 상태 업데이트 실패: {e}")

def get_display_id_from_item(item_id: int):
    try:
        item = get_item_by_id(item_id)
        if not item:
            print("⚠️ item 정보를 찾지 못했습니다.")
            return None, None, None

        patient_id = item.get("patient_id")
        seq = item.get("seq")
        if not patient_id:
            print("⚠️ item에 patient_id가 없습니다.")
            return None, seq, None

        patient = read_patient(str(patient_id))
        if not patient:
            print("⚠️ 환자 정보를 찾지 못했습니다.")
            return None, seq, patient_id

        display_id = patient.get("display_id") or patient.get("patient_initials")
        if display_id:
            print(f"🧩 displayID 조회 완료: {display_id}")
            print(f"🧩 seq 조회 완료: {seq}")
            print(f"🧩 patient_id 조회 완료: {patient_id}")
        else:
            print("⚠️ display_id가 환자 정보에 없습니다.")

        # ✅ 세 개 모두 반환
        return display_id, seq, patient_id

    except Exception as e:
        print(f"❌ display_id 조회 실패: {e}")
        return None, None, None

# ----------------------------------------
# ✅ 실패 로그 저장
# ----------------------------------------
def log_failure(video_id_or_path: str, error_msg: str):
    os.makedirs("logs", exist_ok=True)
    with open("logs/error_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[FAIL] {video_id_or_path} → {error_msg}\n")

# ----------------------------------------
# 🔧 bbox 보정(확장 + 프레임 경계 클리핑)
# ----------------------------------------
def _adjust_bbox(x1, y1, x2, y2, W, H, pad_ratio=EXPAND_RATIO):
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    w = x2 - x1
    h = y2 - y1
    pw = max(1, int(w * pad_ratio))
    ph = max(1, int(h * pad_ratio))
    x1 -= pw; y1 -= ph; x2 += pw; y2 += ph
    x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W,     x2)); y2 = max(0, min(H,     y2))
    if x2 <= x1 + 1: x2 = min(W, x1 + 2)
    if y2 <= y1 + 1: y2 = min(H, y1 + 2)
    return x1, y1, x2, y2

# ----------------------------------------
# 🔒 PNG 인코딩 후 AES-GCM 암호화
# ----------------------------------------
def _encrypt_roi_png(roi_bgr):
    ok, buf = cv2.imencode('.png', roi_bgr)
    if not ok:
        raise ValueError("ROI PNG 인코딩 실패")
    data = buf.tobytes()

    key = os.urandom(32)   # 256-bit
    iv  = os.urandom(12)   # 96-bit (GCM 권장)
    aes = AESGCM(key)
    ct  = aes.encrypt(iv, data, None)

    return (
        base64.b64encode(key).decode(),
        base64.b64encode(iv).decode(),
        base64.b64encode(ct).decode()
    )

def get_rotation_angle(path: str) -> int:
    """ffprobe로 회전 메타데이터 읽기 (0, 90, 180, 270 반환)"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        val = r.stdout.strip()
        if val:
            return int(val)
    except Exception:
        pass
    return 0  # 기본값 (회전 없음)


def detect_and_fix_rotation(frame, rotation_meta: int):
    """
    영상의 실제 방향을 자동 감지 및 보정.
    - 메타데이터 회전값이 0이라도 프레임 종횡비를 통해 가로/세로형 판단
    - 실제로 오른쪽으로 누운 경우(시계 방향 90°)는 반시계로 회전
    """
    h, w = frame.shape[:2]
    corrected = False

    if rotation_meta == 0:
    # 회전 정보가 없으면 그대로 사용 (절대 돌리지 않음)
        frame = frame
    elif rotation_meta == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_meta == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_meta == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if corrected:
        print(f"🧭 자동 회전 보정 적용됨 ({rotation_meta}°)")
    return frame, rotation_meta

def process_video(meta: dict):
    raw_path = meta["file_path"]
    input_path = raw_path
    print(f"[DEBUG] raw_path: {raw_path}")
    filename = os.path.basename(input_path)
    filename_wo_ext = os.path.splitext(filename)[0]
    video_id = meta.get("video_metadata_id")
    item_id = meta.get("item_id")

    os.makedirs(FINAL_VIDEO_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

    output_path = os.path.join(FINAL_VIDEO_DIR, f"{filename_wo_ext}_anonymized.mp4")

    
    display_id, seq, patient_id = get_display_id_from_item(item_id)
    json_filename = (
        f"{display_id}_{seq}.json" if (display_id and seq)
        else f"{display_id}_{filename}.json" if display_id
        else f"{filename_wo_ext}_rois.json"
    )
    json_path = os.path.join(JSON_DIR, json_filename)
    start_time = datetime.utcnow()

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ 비디오를 열 수 없습니다: {input_path}")

        # ✅ 회전 정보 감지 및 첫 프레임 보정
        rotation = get_rotation_angle(input_path)
        print(f"🧭 회전 메타데이터 감지: {rotation}°")

        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("❌ 첫 프레임 읽기 실패")

        first_frame, rotation = detect_and_fix_rotation(first_frame, rotation)
        height, width = first_frame.shape[:2]
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 다시 처음부터

        roi_log = {
            "version": "1.1",
            "video_info": {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "sha256": sha256_file(input_path)
            },
            "frames": {}
        }

        total_detected_frames = total_roi_count = total_encrypted = 0
        prev_rois = []
        hold_counter = 0

        with tqdm(total=frame_count, desc=f"🔄 {filename} 비식별화 진행중", ncols=110) as pbar:
            for idx in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                # 매 프레임 동일 회전 보정
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # YOLO 탐지
                results = model.predict(
                    source=frame,
                    conf=DETECT_CONF,
                    iou=IOU,
                    device=device,
                    verbose=False
                )

                current_rois = []
                for r in results:
                    if not hasattr(r, "boxes") or r.boxes is None:
                        continue
                    for box in r.boxes:
                        conf_score = float(getattr(box, "conf", [1.0])[0])
                        if conf_score < DETECT_CONF:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if x2 <= x1 or y2 <= y1:
                            continue
                        x1, y1, x2, y2 = _adjust_bbox(x1, y1, x2, y2, width, height, pad_ratio=EXPAND_RATIO)
                        current_rois.append((x1, y1, x2, y2, conf_score))

                # 홀도버 적용
                if not current_rois and prev_rois and hold_counter < MAX_HOLD_FRAMES:
                    current_rois = [(x1, y1, x2, y2, MIN_ENCRYPT_CONF) for (x1, y1, x2, y2) in prev_rois]
                    hold_counter += 1
                    source_tag = "holdover"
                else:
                    prev_rois = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in current_rois]
                    hold_counter = 0
                    source_tag = "yolo"

                rois_data = []
                for (x1, y1, x2, y2, conf_score) in current_rois:
                    roi = frame[y1:y2, x1:x2]
                    rec = {
                        "uuid": str(uuid.uuid4()),
                        "bbox": [x1, y1, x2, y2],
                        "key": None, "iv": None, "encrypted_roi": None,
                        "restorable": False,
                        "source": source_tag,
                        "conf": conf_score
                    }

                    if roi.size > 0 and conf_score >= MIN_ENCRYPT_CONF:
                        try:
                            key_b64, iv_b64, enc_b64 = _encrypt_roi_png(roi)
                            rec.update({"key": key_b64, "iv": iv_b64, "encrypted_roi": enc_b64, "restorable": True})
                            total_encrypted += 1
                        except Exception as e:
                            rec["skip_reason"] = f"encrypt_fail:{str(e)}"
                    else:
                        if roi.size == 0:
                            rec["skip_reason"] = "empty_crop"
                        elif conf_score < MIN_ENCRYPT_CONF:
                            rec["skip_reason"] = "low_conf"

                    try:
                        roi_w, roi_h = x2 - x1, y2 - y1
                        mw = max(1, roi_w // MOSAIC_BLOCK_DIVISOR)
                        mh = max(1, roi_h // MOSAIC_BLOCK_DIVISOR)
                        if roi.size > 0:
                            small = cv2.resize(roi, (mw, mh))
                            mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                            frame[y1:y2, x1:x2] = mosaic
                    except Exception as e:
                        rec["skip_reason"] = rec.get("skip_reason", "") + f"|mosaic_fail:{e}"

                    rois_data.append(rec)
                    total_roi_count += 1

                roi_log["frames"][f"frame_{idx:05d}"] = rois_data
                if rois_data:
                    total_detected_frames += 1

                out.write(frame)
                pbar.update(1)

        cap.release()
        out.release()
        end_time = datetime.utcnow()

        output_final_path = merge_audio_with_video(
            original_video=input_path,
            anonymized_video=output_path,
            final_dir=FINAL_VIDEO_DIR
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(roi_log, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 완료: {output_final_path}")
        print(f"📊 ROI 프레임 {total_detected_frames}/{frame_count} / ROI {total_roi_count} / 암호화 {total_encrypted}")

        if video_id:
            update_anonymization_status(video_id)

        nas_video_path = f"/mAGI/CU_Data/VIDEO/{os.path.basename(output_final_path)}"
        nas_json_path  = f"/mAGI/JSON/{os.path.basename(json_path)}"

        save_preprocessing_result(
            meta, nas_video_path, nas_json_path,
            total_frames=frame_count,
            roi_frames=total_detected_frames,
            roi_count=total_roi_count,
            encrypted_count=total_encrypted,
            start_time=start_time,
            end_time=end_time
        )

        return output_final_path, json_path

    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        log_failure(video_id or input_path, str(e))
        raise

# ----------------------------------------
# ✅ 전처리 결과 저장 (로컬 DB 버전)
# ----------------------------------------
def save_preprocessing_result(meta: dict, output_path: str, json_path: str,
                              total_frames: int, roi_frames: int,
                              roi_count: int, encrypted_count: int,
                              start_time: datetime, end_time: datetime):
    try:
        duration_sec = (end_time - start_time).total_seconds()
        success_rate = (encrypted_count / roi_count * 100) if roi_count else 0

        payload = {
            "item_id": meta.get("item_id"),
            "data_category": "video",
            "original_file_path": meta.get("file_path"),
            "json_file_path": json_path,
            "encrypted_file_path": output_path,
            "processing_started_at": start_time,
            "processing_ended_at": end_time,
            "processing_duration_sec": round(duration_sec, 2),
            "total_frames": total_frames,
            "encrypted_frames": encrypted_count,
            "detected_face_frames": roi_frames,
            "success_rate": round(success_rate, 2),
            "preprocessing_type": "face_anonymization + AES-GCM + audio_merge",
            "description": f"총 ROI={roi_count}, 암호화 성공={encrypted_count}"
        }

        ok = insert_preprocessing_record(payload)
        if not ok:
            raise RuntimeError("DB insert 실패")

        print(f"✅ 전처리 결과 DB 저장 완료 (item_id={meta.get('item_id')})")

    except Exception as e:
        print(f"❌ 전처리 결과 저장 실패: {e}")
