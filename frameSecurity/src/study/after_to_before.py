import os
import cv2
import time
import torch
from tqdm import tqdm
from ultralytics import YOLO

# ==== 경로 설정 ====
BASE_DIR = os.path.abspath("C://Users//user//Desktop//frameSecurity//data")
FOLDERS = ["melancholia", "pd"]  # 처리할 하위 폴더
LOG_FILE = os.path.join(BASE_DIR, "process_log.txt")

MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

# ==== 모델 로딩 ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

model = YOLO(MODEL_PATH)
model.to(device)

# ==== 로그 초기화 ====
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== YOLO Mosaic Processing Log ===\n\n")

# ==== 전체 통계 ====
grand_total_duration = 0.0
grand_total_time = 0.0

# ==== 폴더별 처리 ====
for folder in FOLDERS:
    folder_path = os.path.join(BASE_DIR, folder)
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".avi"))]

    folder_total_duration = 0.0
    folder_total_time = 0.0

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n====================\n[폴더] {folder}\n====================\n")

    for video_file in video_files:
        input_path = os.path.join(folder_path, video_file)
        output_path = os.path.join(folder_path, f"blurred_{video_file}")

        # === 비디오 읽기 ===
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        folder_total_duration += video_duration
        grand_total_duration += video_duration

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"❌ VideoWriter 초기화 실패: {output_path}")
            continue

        # === 프레임 처리 ===
        start_time = time.time()
        for _ in tqdm(range(total_frames), desc=f"{folder}/{video_file}", ncols=100):
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 탐지
            results = model.predict(source=frame, conf=0.25, device=0, half=True, verbose=False)

            # 탐지 영역 모자이크 처리
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    mosaic_size = 10
                    small = cv2.resize(roi, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
                    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = mosaic

            out.write(frame)

        cap.release()
        out.release()

        # === 처리 시간 계산 ===
        elapsed_time = time.time() - start_time
        folder_total_time += elapsed_time
        grand_total_time += elapsed_time

        # === 로그 기록 (영상별) ===
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"  - {video_file}: {video_duration:.2f} sec / 처리 {elapsed_time:.2f} sec\n")

    # === 폴더 요약 로그 ===
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"▶ [폴더 합계] {folder} 총 영상 길이: {folder_total_duration:.2f} sec / 총 처리 시간: {folder_total_time:.2f} sec\n\n")

# 전체 요약
with open(LOG_FILE, "a", encoding="utf-8") as log:
    log.write(f"\n=== 전체 통계 ===\n")
    log.write(f"총 영상 길이: {grand_total_duration:.2f} sec\n")
    log.write(f"총 처리 시간: {grand_total_time:.2f} sec\n")

print(f"✅ 모든 영상 처리 완료! 로그 파일: {LOG_FILE}")
