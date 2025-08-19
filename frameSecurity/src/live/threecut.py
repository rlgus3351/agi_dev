from ultralytics import YOLO
import cv2, os, torch, json, base64, hashlib
from tqdm import tqdm
from collections import deque
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image, ImageDraw, ImageFont
import numpy as np


KR_FONT_PATH = r"C:/Windows/Fonts/malgun.ttf"   # 환경에 맞게 수정
# =====================[ 경로 / 설정 ]=====================
BASE = "C://Users//user//Desktop//frameSecurity//data//250811"
MODEL_PATH   = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//model.pt")
VIDEO_PATH   = os.path.abspath(f"{BASE}//input2.mp4")
OUTPUT_VIDEO = os.path.abspath(f"{BASE}//yolo_blur2.mp4")
OUTPUT_JSON  = os.path.abspath(f"{BASE}//yolo_blur_rois.json")

# 로그 파일(2개)
DETECT_NDJSON = os.path.abspath(f"{BASE}//detect_stream.ndjson")   # bbox 검출 로그
ENCRYPT_NDJSON = os.path.abspath(f"{BASE}//encrypt_stream.ndjson") # 모자이크 직전 암호화 로그

# 토글
WRITE_FINAL_JSON = True
WRITE_DETECT_NDJSON = True
WRITE_ENCRYPT_NDJSON = True
INCLUDE_KEYS_IN_FINAL_JSON = False    # 최종 JSON에 key/iv를 포함할지(보안상 기본 False)

# 라이브 tail UI
TAIL_LINES = 10

# 디렉터리 준비/로그 초기화
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
if WRITE_DETECT_NDJSON:
    open(DETECT_NDJSON, "w", encoding="utf-8").close()
if WRITE_ENCRYPT_NDJSON:
    open(ENCRYPT_NDJSON, "w", encoding="utf-8").close()

# =====================[ 모델 ]=====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")
model = YOLO(MODEL_PATH)
model.to(device)
if device == 'cuda':
    try:
        model.model.half()
    except Exception:
        pass

# =====================[ 비디오 IO ]=====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Video 열기 실패")

fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError("❌ VideoWriter 초기화 실패")

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

def append_ndjson_line(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

# =====================[ 라이브 tail 뷰(2개) ]=====================
tail_detect = deque(maxlen=TAIL_LINES)
tail_encrypt = deque(maxlen=TAIL_LINES)

if WRITE_DETECT_NDJSON:
    fp_detect = open(DETECT_NDJSON, "r", encoding="utf-8")
if WRITE_ENCRYPT_NDJSON:
    fp_encrypt = open(ENCRYPT_NDJSON, "r", encoding="utf-8")

def poll_tail(fp, buf):
    while True:
        pos = fp.tell()
        line = fp.readline()
        if not line:
            fp.seek(pos)
            break
        buf.append(line.rstrip("\n"))

def draw_tail_window(title, buf):
    rows = max(TAIL_LINES, 6)
    W, H = 1000, 22*rows + 40
    img = np.zeros((H, W, 3), dtype=np.uint8)
    y = 28
    cv2.putText(img, title, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA)
    y += 22
    for s in list(buf)[-rows:]:
        shown = (s[:130] + " ...") if len(s) > 130 else s
        cv2.putText(img, shown, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        y += 22
    cv2.imshow(title, img)

def draw_label_kr_bgr(img_bgr,
                      text,
                      org=(12, 32),
                      font_path=KR_FONT_PATH,
                      font_size=28,
                      text_color=(255,255,255),
                      bg_color=(0,0,0),
                      bg_alpha=0.35,
                      padding=10,
                      thickness=2,
                      rounded=True):
    """
    OpenCV(BGR) 이미지에 한글 라벨을 반투명 박스로 렌더(Pillow 사용).
    반환: 라벨이 그려진 BGR 이미지 (in-place 아님)
    """
    h, w = img_bgr.shape[:2]
    # BGR -> RGBA (Pillow는 RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # 폰트
    font = ImageFont.truetype(font_path, font_size)

    # 텍스트 bbox 계산
    # Pillow 10+: textbbox 사용, 9-: textsize fallback
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)

    x, y = org
    box_tl = (x - padding, y - th - padding)
    box_br = (x + tw + padding, y + padding)

    # 반투명 배경 박스
    bg = (*bg_color, int(255 * bg_alpha))
    if rounded:
      r = 8
      # 사각형 4모서리 라운드
      draw.rounded_rectangle([box_tl, box_br], radius=r, fill=bg)
    else:
      draw.rectangle([box_tl, box_br], fill=bg)

    # 텍스트(살짝 굵게 보이게 테두리 흉내)
    # 테두리 효과 원치 않으면 offsets 없애세요
    oxs = [(-1,0),(1,0),(0,-1),(0,1)]
    for ox, oy in oxs:
        draw.text((x+ox, y-th+oy), text, font=font, fill=(0,0,0,180))
    draw.text((x, y-th), text, font=font, fill=(*text_color, 255))

    # overlay 합성
    out = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

# =====================[ 처리 루프 ]=====================
roi_data = {}   # 최종 JSON 용
roi_total = 0

for frame_idx in tqdm(range(total_frames or 10**9), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    # 3뷰 준비
    view_orig = frame.copy()     # 원본
    view_det  = frame.copy()     # 감지(박스)
    view_mos  = frame.copy()     # 모자이크

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
            # 클래스/점수 추출(가능하면)
            cls_id = int(getattr(box, "cls", [0])[0]) if hasattr(box, "cls") else 0
            conf_v = float(getattr(box, "conf", [0.0])[0]) if hasattr(box, "conf") else 0.0

            # 1) 검출 뷰: 박스 그리기
            cv2.rectangle(view_det, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(view_det, f"id:{cls_id} conf:{conf_v:.2f}", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            # 2) 모자이크 뷰: 모자이크 적용
            mosaic_region(view_mos, x1, y1, x2, y2, block=10)

            # 3) (로그용) 암호화: 원본 ROI를 PNG로 인코딩 → AES-GCM 암호화(메모리만, 파일 저장 X)
            x1_, y1_ = max(0, x1), max(0, y1)
            x2_, y2_ = min(frame.shape[1], x2), min(frame.shape[0], y2)
            roi = view_orig[y1_:y2_, x1_:x2_]
            cipher_len = None
            cipher_sha256 = None
            key_b64 = iv_b64 = None

            if roi.size > 0:
                ok, buf = cv2.imencode(".png", roi)
                if ok:
                    roi_bytes = buf.tobytes()
                    key = os.urandom(32)
                    iv  = os.urandom(12)
                    aes = AESGCM(key)
                    cipher = aes.encrypt(iv, roi_bytes, None)
                    cipher_len = len(cipher)
                    cipher_sha256 = hashlib.sha256(cipher).hexdigest()
                    if INCLUDE_KEYS_IN_FINAL_JSON:
                        key_b64 = base64.b64encode(key).decode()
                        iv_b64  = base64.b64encode(iv).decode()

            # 검출 로그(ndjson1)
            if WRITE_DETECT_NDJSON:
                append_ndjson_line(DETECT_NDJSON, {
                    "frame": frame_idx,
                    "bbox": [x1, y1, x2, y2],
                    "cls_id": cls_id,
                    "conf": round(conf_v, 4)
                })

            # 암호화 로그(ndjson2)
            if WRITE_ENCRYPT_NDJSON and cipher_len is not None:
                append_ndjson_line(ENCRYPT_NDJSON, {
                    "frame": frame_idx,
                    "bbox": [x1, y1, x2, y2],
                    "cipher_len": cipher_len,
                    "cipher_sha256": cipher_sha256
                })

            # 최종 JSON 메타 누적
            entry = {
                "bbox": [x1, y1, x2, y2],
                "cls_id": cls_id,
                "conf": conf_v,
            }
            if cipher_len is not None:
                entry.update({
                    "cipher_len": cipher_len,
                    "cipher_sha256": cipher_sha256
                })
                if INCLUDE_KEYS_IN_FINAL_JSON:
                    entry.update({"key_b64": key_b64, "iv_b64": iv_b64})

            frame_rois.append(entry)
            roi_total += 1

    if frame_rois:
        roi_data[f"frame_{frame_idx:05d}"] = frame_rois

    # ====== 3분할 뷰(원본 | 감지 | 모자이크) ======
    target_h = 720
    scale  = target_h / max(1, height)
    disp_w = int(width * scale)
    disp_h = target_h

    a = cv2.resize(view_orig, (disp_w, disp_h))
    b = cv2.resize(view_det,  (disp_w, disp_h))
    c = cv2.resize(view_mos,  (disp_w, disp_h))
    
    a = draw_label_kr_bgr(a, "오리지널")
    b = draw_label_kr_bgr(b, "얼굴 인식")
    c = draw_label_kr_bgr(c, "비식별화")

    panel = cv2.hconcat([a, b, c])

    overlay = f"frame:{frame_idx+1}/{total_frames}  roi_total:{roi_total}"
    (text_w, text_h), _ = cv2.getTextSize(overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    # 오른쪽 하단 위치 (좌하단이 기준이므로 y는 height-여백, x는 width-text_w-여백)
    text_x = panel.shape[1] - text_w - 12   # 12px 오른쪽 여백
    text_y = panel.shape[0] - 12            # 12px 아래 여백
    cv2.putText(panel, overlay, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Original | Detect | Mosaic  (q:exit)", panel)

    # 로그 tail 두 창
    if WRITE_DETECT_NDJSON:
        poll_tail(fp_detect, tail_detect)
        draw_tail_window("Detect Log (bbox)", tail_detect)
    if WRITE_ENCRYPT_NDJSON:
        poll_tail(fp_encrypt, tail_encrypt)
        draw_tail_window("Encrypt Log (mosaic)", tail_encrypt)

    # 결과 비디오: 모자이크 뷰 저장
    out.write(view_mos)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================[ 마무리 ]=====================
cap.release()
out.release()
if WRITE_DETECT_NDJSON:
    fp_detect.close()
if WRITE_ENCRYPT_NDJSON:
    fp_encrypt.close()
cv2.destroyAllWindows()

if WRITE_FINAL_JSON:
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(roi_data, f, indent=2, ensure_ascii=False)

print(f"✅ 완료! 모자이크 비디오: {OUTPUT_VIDEO}")
if WRITE_FINAL_JSON:
    print(f"✅ 최종 JSON: {OUTPUT_JSON}")
if WRITE_DETECT_NDJSON:
    print(f"✅ 검출 NDJSON: {DETECT_NDJSON}")
if WRITE_ENCRYPT_NDJSON:
    print(f"✅ 암호화 NDJSON: {ENCRYPT_NDJSON}")
