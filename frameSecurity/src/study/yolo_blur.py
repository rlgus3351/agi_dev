from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
import torch

# ==== 경로 설정 ====
MODEL_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//model.pt")
VIDEO_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//input1.mp4")
OUTPUT_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//yolo_blur.mp4")

# ==== 모델 로딩 및 GPU로 이동 ====
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
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError("❌ VideoWriter 초기화 실패")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ==== 프레임 처리 루프 ====
for _ in tqdm(range(total_frames), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 탐지 (GPU 사용)
    results = model.predict(source=frame, conf=0.25, device=0, half=True, verbose=False)

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

            # 모자이크 처리
            mosaic_size = 10
            small = cv2.resize(roi, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = mosaic

    out.write(frame)

cap.release()
out.release()

print(f"✅ 완료! 모자이크 처리된 영상 저장됨: {OUTPUT_PATH}")
