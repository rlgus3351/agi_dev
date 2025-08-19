from ultralytics import YOLO
import cv2, os, torch, json
from tqdm import tqdm
from collections import deque
import numpy as np

# =====================[ 설정 ]=====================
MODEL_PATH   = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//model.pt")
VIDEO_PATH   = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//250811//input2.mp4")
OUTPUT_VIDEO = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//250811//yolo_blur2.mp4")
OUTPUT_JSON  = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//250811//yolo_blur_rois.json")

# 로그/결과 저장 토글
WRITE_FINAL_JSON = True       # 처리 후 프레임별 ROI 메타데이터 JSON 저장
WRITE_NDJSON     = True       # 실시간 NDJSON 로그(파일 1개, append) 사용
OUTPUT_NDJSON    = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//250811//roi_stream.ndjson")

# 라이브 tail UI
TAIL_LINES = 10               # 라이브 창에 표시할 최근 로그 줄 수

# 디렉터리 준비
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
if WRITE_NDJSON:
    open(OUTPUT_NDJSON, "w", encoding="utf-8").close()

# =====================[ 모델 ]=====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")
model = YOLO(MODEL_PATH)
model.to(device)
if device == 'cuda':
    # half precision으로 속도/메모리 최적화 (지원되는 경우)
    try:
        model.model.half()
    except Exception:
        pass

# =====================[ 비디오 IO ]=====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Video 열기 실패")

fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError("VideoWriter 초기화 실패")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

# =====================[ 유틸 ]=====================
def mosaic_region(frame, x1, y1, x2, y2, block=10):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    small  = cv2.resize(roi, (max(1, block), max(1, block)), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic

def append_ndjson_line(obj):
    if not WRITE_NDJSON:
        return
    with open(OUTPUT_NDJSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

# =====================[ 라이브 로그 tail 뷰 ]=====================
tail_buf = deque(maxlen=TAIL_LINES)
if WRITE_NDJSON:
    ndjson_fp = open(OUTPUT_NDJSON, "r", encoding="utf-8")

    def poll_ndjson_lines():
        while True:
            pos = ndjson_fp.tell()
            line = ndjson_fp.readline()
            if not line:
                ndjson_fp.seek(pos)
                break
            tail_buf.append(line.rstrip("\n"))

    def draw_tail_window():
        rows = max(TAIL_LINES, 6)
        W, H = 900, 22*rows + 40
        img = np.zeros((H, W, 3), dtype=np.uint8)
        y = 28
        cv2.putText(img, "ROI Stream (live) - last lines",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA)
        y += 22
        for s in list(tail_buf)[-rows:]:
            shown = (s[:120] + " ...") if len(s) > 120 else s
            cv2.putText(img, shown, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            y += 22
        cv2.imshow("ROI Stream (live)", img)
else:
    def poll_ndjson_lines(): pass
    def draw_tail_window(): pass

# =====================[ 처리 루프 ]=====================
roi_data = {}   # 최종 JSON 용(프레임별 ROI 메타데이터)
roi_total = 0

for frame_idx in tqdm(range(total_frames or 10**9), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    # YOLO 추론
    results = model.predict(
        source=frame,
        conf=0.25,
        device=0 if device == 'cuda' else None,
        half=(device == 'cuda'),
        verbose=False
    )

    frame_rois = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # ROI 메타만 기록(암호화/파일 저장 없음)
            frame_rois.append({"bbox": [x1, y1, x2, y2]})
            roi_total += 1

            # 모자이크 적용
            mosaic_region(frame, x1, y1, x2, y2, block=10)

            # 실시간 NDJSON (가벼운 메타만)
            append_ndjson_line({
                "frame": frame_idx,
                "bbox": [x1, y1, x2, y2]
            })

    # 프레임 메타 누적
    if frame_rois:
        roi_data[f"frame_{frame_idx:05d}"] = frame_rois

    # Before | After 프리뷰
    scale  = 640 / max(1, height)
    disp_w, disp_h = int(width * scale), 640
    left  = cv2.resize(orig,  (disp_w, disp_h))
    right = cv2.resize(frame, (disp_w, disp_h))
    combo = cv2.hconcat([left, right])
    overlay = f"frame:{frame_idx+1}/{total_frames}  roi_total:{roi_total}"
    cv2.putText(combo, overlay, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Before | After (q:exit)", combo)

    # NDJSON 라이브 tail UI
    poll_ndjson_lines()
    draw_tail_window()

    # 결과 비디오 저장
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================[ 마무리 ]=====================
cap.release()
out.release()
if WRITE_NDJSON:
    ndjson_fp.close()
cv2.destroyAllWindows()

if WRITE_FINAL_JSON:
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(roi_data, f, indent=2, ensure_ascii=False)

print(f"✅ 완료! 모자이크 비디오: {OUTPUT_VIDEO}")
if WRITE_FINAL_JSON:
    print(f"✅ 최종 JSON: {OUTPUT_JSON}")
if WRITE_NDJSON:
    print(f"✅ NDJSON(라이브 로그): {OUTPUT_NDJSON}")
