import cv2
import os
import json
import uuid
import base64
import hashlib
from tqdm import tqdm
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from ultralytics import YOLO
import torch

# ==== 경로 설정 ====
MODEL_PATH = r"C:\Users\user\Desktop\frameSecurity\data\model.pt"
VIDEO_PATH = r"C:\Users\user\Desktop\frameSecurity\data\pd\input1.mp4"
OUTPUT_VIDEO = r"C:\Users\user\Desktop\frameSecurity\data\AES\yolo_blur3.mp4"
OUTPUT_JSON = r"C:\Users\user\Desktop\frameSecurity\data\AES\yolo_blur_rois3.json"

# ==== SHA256 해시 계산 함수 ====
def sha256_file(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# ==== 모델 로딩 ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

model = YOLO(MODEL_PATH)
model.to(device)

# ==== 비디오 정보 ====
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_hash = sha256_file(VIDEO_PATH)

print(f"🎥 원본 영상 프레임: {total_frames}, SHA256: {video_hash[:16]}...")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError("❌ VideoWriter 초기화 실패")

# ==== JSON 로그 구조 ====
roi_data = {
    "video_info": {
        "frame_count": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "sha256": video_hash
    },
    "frames": {}
}

# ==== 프레임 처리 ====
for frame_idx in tqdm(range(total_frames), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.25, device=0, half=True, verbose=False)
    frame_rois = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # ROI 암호화
            success, roi_bytes = cv2.imencode(".png", roi)
            if not success:
                continue
            roi_bytes = roi_bytes.tobytes()

            key = os.urandom(32)
            aesgcm = AESGCM(key)
            iv = os.urandom(12)
            encrypted_roi = aesgcm.encrypt(iv, roi_bytes, None)

            obj_uuid = str(uuid.uuid4())
            frame_rois.append({
                "uuid": obj_uuid,
                "bbox": [x1, y1, x2, y2],
                "key": base64.b64encode(key).decode(),
                "iv": base64.b64encode(iv).decode(),
                "encrypted_roi": base64.b64encode(encrypted_roi).decode()
            })

            # 모자이크 처리
            mosaic_size = 10
            small = cv2.resize(roi, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = mosaic

    if frame_rois:
        roi_data["frames"][f"frame_{frame_idx:05d}"] = frame_rois

    out.write(frame)

cap.release()
out.release()

# ==== JSON 저장 ====
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(roi_data, f, indent=2, ensure_ascii=False)

print(f"✅ 완료! 모자이크 영상: {OUTPUT_VIDEO}")
print(f"✅ ROI JSON 저장: {OUTPUT_JSON}")
