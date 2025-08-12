from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

MODEL_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//model.pt")
VIDEO_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//input1.mp4")
OUTPUT_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//output1.mp4")

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError("❌ VideoWriter 초기화 실패")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for _ in tqdm(range(total_frames), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.25, verbose=False)

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()

print(f"✅ 결과 영상 저장 완료: {OUTPUT_PATH}")
