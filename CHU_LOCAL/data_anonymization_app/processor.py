from ultralytics import YOLO
import cv2, os, json, uuid, base64, hashlib, subprocess
import numpy as np
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import requests
from api_local.processing_api_local import update_processed_video, insert_preprocessing_record
from api_local.item_api_local import get_item_by_id
from api_local.patient_api_local import read_patient
from datetime import datetime
import urllib3
from typing import Optional
import time
urllib3.disable_warnings()

# =============================
# 🔧 PyTorch 2.6 안전 허용 처리
# =============================
import torch
from torch.nn.modules.container import Sequential
torch.serialization.add_safe_globals([
    Sequential,
    __import__('ultralytics').nn.tasks.DetectionModel,
    __import__('ultralytics').nn.modules.conv.Conv,
    __import__('ultralytics').nn.modules.block.C2f,
])
torch.backends.cudnn.benchmark = True

# =============================
# ✅ YOLO 모델 로드 (GPU 자동 감지)
# =============================
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "model", "model.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

FINAL_VIDEO_DIR = r"C:\Users\user\Desktop\DEV_AGI\parkinson\output\video"

VIDEO_PATH = r"C:\Users\user\Desktop\DEV_AGI\parkinson\output\video"
JSON_PATH  = r"C:\Users\user\Desktop\DEV_AGI\parkinson\output\json"

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
MOSAIC_BLOCK_DIVISOR = 20     # 모자이크 블록 크기(값↑ = 블록↑)

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
# ✅ 오디오 병합 함수
# ----------------------------------------
def merge_audio_with_video(original_video, anonymized_video):
    """원본 영상의 오디오를 익명화된 영상에 병합"""
    output_with_audio = anonymized_video.replace("_anonymized.mp4", "_final.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", anonymized_video,
        "-i", original_video,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0",
        output_with_audio
    ]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode == 0:
        print(f"🎵 오디오 포함 영상 생성 완료 → {output_with_audio}")
    else:
        print(f"⚠️ 오디오 병합 실패: {process.stderr.decode()}")
    return output_with_audio

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
            return None, None

        patient_id = item.get("patient_id")
        seq = item.get("seq")
        if not patient_id:
            print("⚠️ item에 patient_id가 없습니다.")
            return None, seq

        patient = read_patient(str(patient_id))
        if not patient:
            print("⚠️ 환자 정보를 찾지 못했습니다.")
            return None, seq

        display_id = patient.get("display_id") or patient.get("patient_initials")
        if display_id:
            print(f"🧩 displayID 조회 완료: {display_id}")
            print(f"🧩 seq 조회 완료: {seq}")
        else:
            print("⚠️ display_id가 환자 정보에 없습니다.")

        return display_id, seq
    except Exception as e:
        print(f"❌ display_id 조회 실패: {e}")
        return None, None

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

def process_video(meta: dict):
    raw_path = meta["file_path"]
    input_path = raw_path
    print(f"[DEBUG] raw_path: {raw_path}")
    print(f"[DEBUG] normalized to container path: {input_path}")
    filename = os.path.basename(input_path)
    filename_wo_ext = os.path.splitext(filename)[0]
    video_id = meta.get("video_metadata_id")
    item_id = meta.get("item_id")

    os.makedirs("data/output", exist_ok=True)
    os.makedirs("data/json", exist_ok=True)

    output_path = f"data/output/{filename_wo_ext}_anonymized.mp4"
    display_id, seq = get_display_id_from_item(item_id) if item_id else (None, None)

    if display_id and seq:
        json_filename = f"{display_id}_{seq}.json"
    elif display_id:
        json_filename = f"{display_id}_{filename}.json"
    else:
        json_filename = f"{filename_wo_ext}_rois.json"

    json_path = f"data/json/{json_filename}"
    start_time = datetime.utcnow()

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ 비디오를 열 수 없습니다: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(3)), int(cap.get(4))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

        total_detected_frames = 0
        total_roi_count = 0
        total_encrypted = 0

        # ✅ ROI 유지 로직 변수 (홀도버: 탐지 끊김 시 복구 보장)
        prev_rois = []            # [[x1,y1,x2,y2], ...]
        hold_counter = 0
        max_hold_frames = MAX_HOLD_FRAMES

        with tqdm(total=frame_count, desc=f"🔄 {filename} 비식별화 진행중", ncols=110) as pbar:
            for idx in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                # 1) YOLO 탐지 (conf 상향)
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
                        # 옆모습 커버를 위해 bbox 확장 + 프레임 경계 클리핑
                        x1, y1, x2, y2 = _adjust_bbox(x1, y1, x2, y2, width, height, pad_ratio=EXPAND_RATIO)
                        current_rois.append((x1, y1, x2, y2, conf_score))

                # 2) 감지 끊김 시 홀도버(현재 프레임에도 암호화/기록)
                source_tag = "yolo"
                if not current_rois and prev_rois and hold_counter < max_hold_frames:
                    current_rois = [(x1, y1, x2, y2, MIN_ENCRYPT_CONF) for (x1, y1, x2, y2) in prev_rois]
                    hold_counter += 1
                    source_tag = "holdover"
                else:
                    prev_rois = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in current_rois]
                    hold_counter = 0
                    source_tag = "yolo"

                # 3) 암호화/모자이크/로그
                rois_data = []
                for (x1, y1, x2, y2, conf_score) in current_rois:
                    roi = frame[y1:y2, x1:x2]
                    rec = {
                        "uuid": str(uuid.uuid4()),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "key": None,
                        "iv": None,
                        "encrypted_roi": None,
                        "restorable": False,
                        "source": source_tag,
                        "conf": float(conf_score)
                    }

                    if roi.size > 0 and conf_score >= MIN_ENCRYPT_CONF:
                        try:
                            key_b64, iv_b64, enc_b64 = _encrypt_roi_png(roi)
                            rec["key"] = key_b64
                            rec["iv"] = iv_b64
                            rec["encrypted_roi"] = enc_b64
                            rec["restorable"] = True
                            total_encrypted += 1
                        except Exception as e:
                            rec["restorable"] = False
                            rec["skip_reason"] = f"encrypt_fail:{str(e)}"
                    else:
                        if roi.size == 0:
                            rec["skip_reason"] = "empty_crop"
                        elif conf_score < MIN_ENCRYPT_CONF:
                            rec["skip_reason"] = "low_conf"

                    # 모자이크(영상 익명화)
                    try:
                        roi_w, roi_h = x2 - x1, y2 - y1
                        mw = max(1, roi_w // MOSAIC_BLOCK_DIVISOR)
                        mh = max(1, roi_h // MOSAIC_BLOCK_DIVISOR)
                        if roi.size > 0:
                            small = cv2.resize(roi, (mw, mh))
                            mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                            frame[y1:y2, x1:x2] = mosaic
                    except Exception as e:
                        rec.setdefault("skip_reason", "")
                        rec["skip_reason"] += f"|mosaic_fail:{str(e)}"

                    rois_data.append(rec)
                    total_roi_count += 1

                # 프레임별 기록 (빈 리스트라도 기록)
                roi_log["frames"][f"frame_{idx:05d}"] = rois_data
                if rois_data:
                    total_detected_frames += 1

                out.write(frame)
                pbar.update(1)

        cap.release()
        out.release()
        end_time = datetime.utcnow()

        # ✅ 오디오 병합 (원본 오디오 유지) — 호출 인자 수정: (원본, 익명화경로)
        output_final_path = merge_audio_with_video(original_video=input_path, anonymized_video=output_path)

        # ✅ JSON 저장
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(roi_log, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 완료: {output_final_path}")
        print(f"📊 ROI 프레임 {total_detected_frames} / 전체 {frame_count} / ROI {total_roi_count} / 암호화 성공 {total_encrypted}")

        if video_id:
            update_anonymization_status(video_id)

        # ✅ NAS 업로드 (필요 시 주석 해제)
        # sid = None
        # try:
        #     print(f"🛰️ NAS 업로드 준비 중... (video: {output_final_path}, json: {json_path})")
        #     sid = nas_login()
        #     success_video = upload_to_nas(sid, output_final_path, nas_folder="/mAGI/CNU_Data/VIDEO")
        #     success_json  = upload_to_nas(sid, json_path,       nas_folder="/mAGI/JSON")
        #     if not (success_video and success_json):
        #         print("⚠️ 일부 NAS 업로드 실패 발생")
        # except Exception as e:
        #     print(f"⚠️ NAS 업로드 중 오류 발생: {e}")
        # finally:
        #     if sid:
        #         nas_logout(sid)

        nas_video_path = f"/mAGI/CNU_Data/VIDEO/{os.path.basename(output_final_path)}"
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
