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

VIDEO_PATH =r"C:\Users\user\Desktop\DEV_AGI\parkinson\output\video"
JSON_PATH = r"C:\Users\user\Desktop\DEV_AGI\parkinson\output\json"

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
    # 일부 버전에선 내부적으로 무시될 수 있으므로, predict에서 device도 함께 전달함
    model.to(device)
except Exception:
    pass

print(f"✅ YOLO 모델 로드 완료 (device={device})")


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
        # FastAPI 호출 대신 로컬 DB 업데이트
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

        # ✅ ROI 유지 로직 변수 (복구 보장: holdover라도 현재 프레임의 ROI를 암호화해서 기록)
        prev_rois = []            # 직전 프레임 bbox 리스트 [[x1,y1,x2,y2], ...]
        hold_counter = 0
        max_hold_frames = 3       # 유지할 최대 프레임 수

        with tqdm(total=frame_count, desc=f"🔄 {filename} 비식별화 진행중", ncols=110) as pbar:
            for idx in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                # ✅ YOLO 감지 (조금 느슨하게)
                #  - conf 낮춰 옆모습 탐지율 ↑
                results = model.predict(source=frame, conf=0.15, device=device, verbose=False)

                # 후보 박스 수집
                current_rois = []
                for r in results:
                    if not hasattr(r, "boxes") or r.boxes is None:
                        continue
                    expand_ratio = 0.2
                    for box in r.boxes:
                        # 너무 낮은 건 제외 (극저신뢰 오탐 컷)
                        conf_score = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                        if conf_score < 0.10:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        if w <= 0 or h <= 0:
                            continue

                        # 약간 확장 (옆얼굴 커버)
                        expand_w, expand_h = int(w * expand_ratio), int(h * expand_ratio)
                        x1 = max(0, x1 - expand_w)
                        y1 = max(0, y1 - expand_h)
                        x2 = min(width,  x2 + expand_w)
                        y2 = min(height, y2 + expand_h)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        current_rois.append([x1, y1, x2, y2])

                # ✅ 감지 안되면 이전 ROI 유지(holdover) — 복구 가능하도록 현재 프레임 ROI를 암호화/기록함
                source_tag = "yolo"
                if not current_rois and prev_rois and hold_counter < max_hold_frames:
                    current_rois = [r.copy() for r in prev_rois]
                    hold_counter += 1
                    source_tag = "holdover"
                else:
                    prev_rois = [r.copy() for r in current_rois]
                    hold_counter = 0
                    source_tag = "yolo"

                rois_data = []
                for bbox in current_rois:
                    x1, y1, x2, y2 = bbox
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        # 그래도 “기록”은 남겨 복구 로그 일관성 유지
                        rois_data.append({
                            "uuid": str(uuid.uuid4()),
                            "bbox": [x1, y1, x2, y2],
                            "key": None,
                            "iv": None,
                            "encrypted_roi": None,
                            "restorable": False,
                            "source": source_tag
                        })
                        continue

                    total_roi_count += 1

                    roi_data = {
                        "uuid": str(uuid.uuid4()),
                        "bbox": [x1, y1, x2, y2],
                        "key": None,
                        "iv": None,
                        "encrypted_roi": None,
                        "restorable": False,
                        "source": source_tag
                    }

                    # ✅ 복구 보장: 현재 프레임의 ROI를 반드시 암호화해서 JSON에 저장
                    try:
                        key = os.urandom(32)   # 256-bit
                        iv  = os.urandom(12)   # 96-bit nonce
                        aes = AESGCM(key)

                        success, roi_bytes = cv2.imencode('.png', roi)
                        if not success:
                            raise ValueError("ROI 인코딩 실패")

                        encrypted = aes.encrypt(iv, roi_bytes.tobytes(), None)
                        total_encrypted += 1

                        roi_data["key"] = base64.b64encode(key).decode()
                        roi_data["iv"]  = base64.b64encode(iv).decode()
                        roi_data["encrypted_roi"] = base64.b64encode(encrypted).decode()
                        roi_data["restorable"] = True
                    except Exception as e:
                        print(f"⚠️ ROI 암호화 실패 (frame {idx}): {e}")

                    # ✅ 모자이크 적용 (복구 위해 JSON에 같은 bbox를 기록했으므로 복원 가능)
                    roi_w, roi_h = x2 - x1, y2 - y1
                    mosaic_w = max(1, roi_w // 20)
                    mosaic_h = max(1, roi_h // 20)
                    try:
                        small = cv2.resize(roi, (mosaic_w, mosaic_h))
                        mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                        frame[y1:y2, x1:x2] = mosaic
                    except Exception as e:
                        print(f"⚠️ 모자이크 적용 실패 (frame {idx}): {e}")

                    rois_data.append(roi_data)

                # ✅ 프레임별 기록 (빈 리스트라도 항상 기록해 복호화 시 프레임 정합성 유지)
                roi_log["frames"][f"frame_{idx:05d}"] = rois_data
                if rois_data:
                    total_detected_frames += 1

                out.write(frame)
                pbar.update(1)

        cap.release()
        out.release()
        end_time = datetime.utcnow()

        # ✅ 오디오 병합 (원본 오디오 유지)
        output_final_path = merge_audio_with_video(input_path, FINAL_VIDEO_DIR)

        # ✅ JSON 저장
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(roi_log, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 완료: {output_final_path}")
        print(f"📊 ROI 프레임 {total_detected_frames} / 전체 {frame_count} / ROI {total_roi_count} / 암호화 성공 {total_encrypted}")

        if video_id:
            update_anonymization_status(video_id)

        # ✅ NAS 업로드 (재시도 포함) — 외부 함수는 기존 코드/모듈에 정의되어 있어야 함
        sid = None
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
            "processing_started_at": start_time,      # datetime 객체 그대로 OK
            "processing_ended_at": end_time,          # datetime 객체 그대로 OK
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
