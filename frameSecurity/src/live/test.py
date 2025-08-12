import os
import cv2
import time
import torch
from ultralytics import YOLO

# ==== 경로/모델 ====
BASE_DIR = os.path.abspath("C://Users//user//Desktop//frameSecurity//data")
FOLDER   = "250811"
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")
model = YOLO(MODEL_PATH)
model.to(device)
if device == "cuda":
    model.model.half()  # half precision (CUDA only)

# ==== 파라미터 ====
CONF_TH = 0.25
DISPLAY_HEIGHT = 640   # 미리보기 윈도우 높이
DEFAULT_SAVE = True    # 시작 시 저장 여부
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
MOSAIC_SIZE = 10

def mosaic_region(frame, x1, y1, x2, y2, block=MOSAIC_SIZE):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    small = cv2.resize(roi, (block, block), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic

def process_and_preview(input_path, save=DEFAULT_SAVE):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0

    output_path = os.path.splitext(input_path)[0] + "_blurred.mp4"
    writer = cv2.VideoWriter(output_path, FOURCC, fps, (width, height)) if save else None

    print(f"▶ {os.path.basename(input_path)} "
          f"({width}x{height}@{fps:.2f}fps, {duration:.2f}s) | 저장: {save}")

    paused = False
    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                orig = frame.copy()

                # YOLO 추론 (predict 대신 __call__)
                with torch.no_grad():
                    if device == "cuda":
                        # half precision은 입력 변환 없이 자동 처리됨
                        results = model(frame, conf=CONF_TH, verbose=False)
                    else:
                        results = model(frame, conf=CONF_TH, verbose=False)

                # 결과 1장만 사용
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy  # (N,4)
                    for b in xyxy:
                        x1, y1, x2, y2 = map(int, b.tolist())
                        mosaic_region(frame, x1, y1, x2, y2, block=MOSAIC_SIZE)

                # 저장
                if writer is not None:
                    writer.write(frame)

                # ======= 실시간 미리보기 (좌: 원본 | 우: 모자이크) =======
                # 미리보기 해상도 축소
                scale = DISPLAY_HEIGHT / height
                disp_w = int(width * scale)
                disp_h = DISPLAY_HEIGHT

                left = cv2.resize(orig, (disp_w, disp_h))
                right = cv2.resize(frame, (disp_w, disp_h))
                combined = cv2.hconcat([left, right])

                # 정보 오버레이
                elapsed = time.time() - t0
                cur_fps = (frame_idx + 1) / max(elapsed, 1e-6)
                info = f"{os.path.basename(input_path)} | {frame_idx+1}/{total_frames} "\
                       f"| {cur_fps:.1f} FPS | save:{'ON' if writer else 'OFF'}"
                cv2.putText(combined, info, (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("Before | After (q:종료, p:일시정지, s:저장 토글)", combined)
                frame_idx += 1

            # 키보드 제어
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('s'):
                # 저장 토글
                if writer is None:
                    writer = cv2.VideoWriter(output_path, FOURCC, fps, (width, height))
                    print("💾 저장 ON")
                else:
                    writer.release()
                    writer = None
                    print("🛑 저장 OFF")

    except KeyboardInterrupt:
        print("⏹ Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

def run_folder(folder_name):
    folder_path = os.path.join(BASE_DIR, folder_name)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".avi"))]
    for f in files:
        process_and_preview(os.path.join(folder_path, f), save=DEFAULT_SAVE)

if __name__ == "__main__":
    run_folder(FOLDER)
