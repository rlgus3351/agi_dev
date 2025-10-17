from ultralytics import YOLO
import cv2
import os
import json
import uuid
import base64
import hashlib
from tqdm import tqdm
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import torch

# ==== 기본 경로 설정 ====
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model", "model.pt")
input_path = os.path.join(base_dir, "data", "input", "sample_2.mp4")

# 출력 폴더 통일
output_dir = os.path.join(base_dir, "data", "output")
json_dir = os.path.join(base_dir, "data", "json")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

filename = os.path.splitext(os.path.basename(input_path))[0]
output_video = os.path.join(output_dir, f"{filename}_anonymized.mp4")
output_json = os.path.join(json_dir, f"{filename}_rois.json")

# ==== SHA256 해시 계산 ====
def sha256_file(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# ==== 모델 로드 ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

model = YOLO(model_path)
model.to(device)

# ==== 비디오 정보 ====
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_hash = sha256_file(input_path)

print(f"🎥 원본 영상 프레임: {total_frames}, SHA256: {video_hash[:16]}...")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError("❌ VideoWriter 초기화 실패")

# ==== JSON 구조 ====
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

    results = model.predict(source=frame, conf=0.25, device=device, half=(device == "cuda"), verbose=False)
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
            mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = mosaic

    if frame_rois:
        roi_data["frames"][f"frame_{frame_idx:05d}"] = frame_rois

    out.write(frame)

cap.release()
out.release()

# ==== JSON 저장 ====
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(roi_data, f, indent=2, ensure_ascii=False)

print(f"✅ 완료! 모자이크 영상: {output_video}")
print(f"✅ ROI JSON 저장: {output_json}")
