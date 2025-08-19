from ultralytics import YOLO
import cv2, os, torch, uuid, json, base64, hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from tqdm import tqdm
from collections import deque
import numpy as np

# ==== 경로 ====
MODEL_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//model.pt")
VIDEO_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//250811//input2.mp4")
OUTPUT_VIDEO = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//250811//yolo_blur2.mp4")
OUTPUT_JSON  = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//250811//yolo_blur_rois.json")

# ✅ 실시간 로그(라인 단위) 파일 + blob 폴더
OUTPUT_NDJSON = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//250811//roi_stream.ndjson")
ROI_BIN_DIR   = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//250811//roi_blobs")
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
os.makedirs(ROI_BIN_DIR, exist_ok=True)

# ==== 모델 ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")
model = YOLO(MODEL_PATH)
model.to(device)
if device == 'cuda':
    model.model.half()

# ==== 비디오 ====
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError("❌ VideoWriter 초기화 실패")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ==== JSON(완료본) ====
roi_data = {}

# ==== NDJSON 초기화 ====
open(OUTPUT_NDJSON, "w", encoding="utf-8").close()

def append_ndjson_line(obj):
    with open(OUTPUT_NDJSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

# ✅ 라이브 뷰: NDJSON 최근 N줄 tail
TAIL_LINES = 10
tail_buf = deque(maxlen=TAIL_LINES)
ndjson_fp = open(OUTPUT_NDJSON, "r", encoding="utf-8")  # 계속 열어두고 신규 라인만 읽음

def poll_ndjson_lines():
    """새로 추가된 줄만 읽어서 tail_buf에 축적"""
    while True:
        pos = ndjson_fp.tell()
        line = ndjson_fp.readline()
        if not line:
            ndjson_fp.seek(pos)  # 더 이상 없음
            break
        tail_buf.append(line.rstrip("\n"))

def draw_tail_window():
    """tail_buf 내용을 OpenCV 창으로 렌더"""
    rows = max(TAIL_LINES, 6)
    W, H = 900, 22*rows + 40
    img = np.zeros((H, W, 3), dtype=np.uint8)
    y = 28
    cv2.putText(img, "ROI Stream (live) - last lines", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA)
    y += 22
    for s in list(tail_buf)[-rows:]:
        # 너무 길면 조금 잘라 표시
        shown = (s[:120] + " ...") if len(s) > 120 else s
        cv2.putText(img, shown, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        y += 22
    cv2.imshow("ROI Stream (live)", img)

# ==== 프레임 루프 ====
def mosaic_region(frame, x1, y1, x2, y2, block=10):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    small = cv2.resize(roi, (block, block), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic

roi_total = 0
for frame_idx in tqdm(range(total_frames), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    results = model.predict(source=frame, conf=0.25,
                            device=0 if device=='cuda' else None,
                            half=(device=='cuda'), verbose=False)

    frame_rois = []

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

            # --- 암호화 ---
            ok, buf = cv2.imencode(".png", roi)
            if not ok:
                continue
            roi_bytes = buf.tobytes()
            key = os.urandom(32)
            iv  = os.urandom(12)
            aes = AESGCM(key)
            cipher = aes.encrypt(iv, roi_bytes, None)

            # blob 저장(무거운 본문은 파일로)
            obj_uuid = str(uuid.uuid4())
            blob_path = os.path.join(ROI_BIN_DIR, f"{obj_uuid}.bin")
            with open(blob_path, "wb") as bf:
                bf.write(cipher)

            sha256 = hashlib.sha256(cipher).hexdigest()

            # 최종 JSON(완료본)
            frame_rois.append({
                "uuid": obj_uuid,
                "bbox": [x1, y1, x2, y2],
                "key": base64.b64encode(key).decode(),
                "iv": base64.b64encode(iv).decode(),
                "encrypted_roi": base64.b64encode(cipher).decode()
            })

            # ✅ 실시간 NDJSON 한 줄 append
            append_ndjson_line({
                "frame": frame_idx,
                "uuid": obj_uuid,
                "bbox": [x1, y1, x2, y2],
                "blob_path": blob_path,
                "cipher_len": len(cipher),
                "cipher_sha256": sha256
            })
            roi_total += 1

            # 모자이크
            mosaic_region(frame, x1, y1, x2, y2, block=10)

    if frame_rois:
        roi_data[f"frame_{frame_idx:05d}"] = frame_rois

    # ====== 미리보기: Before | After ======
    scale = 640 / height
    disp_w, disp_h = int(width*scale), 640
    left  = cv2.resize(orig,  (disp_w, disp_h))
    right = cv2.resize(frame, (disp_w, disp_h))
    combo = cv2.hconcat([left, right])
    overlay = f"frame:{frame_idx+1}/{total_frames}  roi_total:{roi_total}"
    cv2.putText(combo, overlay, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Before | After (q:exit)", combo)

    # ✅ NDJSON 새 줄 읽어와 별도 창으로 표시
    poll_ndjson_lines()
    draw_tail_window()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
ndjson_fp.close()
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(roi_data, f, indent=2, ensure_ascii=False)
cv2.destroyAllWindows()

print(f"✅ 완료! 모자이크: {OUTPUT_VIDEO}")
print(f"✅ 최종 JSON: {OUTPUT_JSON}")
print(f"✅ NDJSON(라이브 로그): {OUTPUT_NDJSON}")
print(f"✅ blobs: {ROI_BIN_DIR}")
