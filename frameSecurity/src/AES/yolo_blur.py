from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
import torch
import uuid
import json
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import numpy as np

# ==== 경로 설정 ====
MODEL_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//model.pt")
VIDEO_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//pd//input2.mp4")
OUTPUT_VIDEO = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//AES//yolo_blur2.mp4")
OUTPUT_JSON = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//AES//yolo_blur_rois.json")

# ==== 모델 로딩 ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

model = YOLO(MODEL_PATH)
model.to(device)

# ==== 비디오 설정 ====
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError("❌ VideoWriter 초기화 실패")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ==== JSON 로그 구조 ====
roi_data = {}

# ==== 프레임 처리 ====
for frame_idx in tqdm(range(total_frames), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.25, device=0, half=True, verbose=False)

    frame_rois = []  # 이 프레임의 ROI 목록

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 좌표 유효성 체크
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # ========== 1️⃣ ROI 암호화 ==========
            success, roi_bytes = cv2.imencode(".png", roi)
            if not success:
                continue
            roi_bytes = roi_bytes.tobytes()

            # AES-GCM 암호화
            key = os.urandom(32)  # 256-bit key
            aesgcm = AESGCM(key)
            iv = os.urandom(12)   # 96-bit nonce
            encrypted_roi = aesgcm.encrypt(iv, roi_bytes, None)

            # UUID 생성
            obj_uuid = str(uuid.uuid4())

            # JSON에 저장 (Base64 변환)
            frame_rois.append({
                "uuid": obj_uuid,
                "bbox": [x1, y1, x2, y2],
                "key": base64.b64encode(key).decode(),
                "iv": base64.b64encode(iv).decode(),
                "encrypted_roi": base64.b64encode(encrypted_roi).decode()
            })

            # ========== 2️⃣ 모자이크 처리 ==========
            mosaic_size = 10
            small = cv2.resize(roi, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = mosaic

    # JSON 기록
    if frame_rois:
        roi_data[f"frame_{frame_idx:05d}"] = frame_rois

    out.write(frame)

cap.release()
out.release()

# ==== JSON 저장 ====
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(roi_data, f, indent=2, ensure_ascii=False)

print(f"✅ 완료! 모자이크 영상: {OUTPUT_VIDEO}")
print(f"✅ ROI JSON 저장: {OUTPUT_JSON}")
