# -*- coding: utf-8 -*-
"""
하나의 창에 다음 레이아웃으로 표시:
[상단] 원본 | 얼굴 인식 | 비식별화 로그(Encrypt)
[하단] 비식별화 | 식별화 영상 | 식별화 로그(Decrypt)

- 각 영상 셀 높이 = 640px (가로는 원본 비율 유지)
- 로그는 픽셀 폭 기준 줄바꿈 + 줄 수 확장
- 모자이크(비식별화) 영상 파일 저장
- 암/복호화 상세는 NDJSON 로그로 저장
"""

from ultralytics import YOLO
import cv2, os, torch, json, base64, hashlib
from tqdm import tqdm
from collections import deque
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image, ImageDraw, ImageFont

# =====================[ 경로 / 설정 ]=====================
KR_FONT_PATH = r"C:/Windows/Fonts/malgun.ttf"   # 환경에 맞게 수정
BASE         = "C://Users//user//Desktop//frameSecurity//data//250811"
MODEL_PATH   = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//model.pt")
VIDEO_PATH   = os.path.abspath(f"{BASE}//input2.mp4")

OUTPUT_VIDEO = os.path.abspath(f"{BASE}//yolo_blur2.mp4")
OUTPUT_JSON  = os.path.abspath(f"{BASE}//yolo_blur_rois.json")

# 로그 파일
ENCRYPT_NDJSON = os.path.abspath(f"{BASE}//encrypt_stream.ndjson")
DECRYPT_NDJSON = os.path.abspath(f"{BASE}//decrypt_stream.ndjson")

# 토글
WRITE_FINAL_JSON     = True
WRITE_ENCRYPT_NDJSON = True
WRITE_DECRYPT_NDJSON = True
INCLUDE_KEYS_IN_FINAL_JSON = True   # 보안상 False 권장

# 로그 표시 길이/폭
TAIL_LINES = 35        # ✅ 더 많은 줄
LOG_COL_SCALE = 1.75   # 로그 폭 = cell_w * 이 값 (값 키우면 더 넓어짐)

# 디렉터리/로그 초기화
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
if WRITE_ENCRYPT_NDJSON: open(ENCRYPT_NDJSON, "w", encoding="utf-8").close()
if WRITE_DECRYPT_NDJSON: open(DECRYPT_NDJSON, "w", encoding="utf-8").close()

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

def paste_roi(dest, x1, y1, x2, y2, roi_img):
    h, w = dest.shape[:2]
    x1_, y1_ = max(0, x1), max(0, y1)
    x2_, y2_ = min(w, x2), min(h, y2)
    if x2_ <= x1_ or y2_ <= y1_:
        return
    tw, th = x2_ - x1_, y2_ - y1_
    if roi_img is None or roi_img.size == 0:
        return
    if roi_img.shape[1] != tw or roi_img.shape[0] != th:
        roi_img = cv2.resize(roi_img, (tw, th), interpolation=cv2.INTER_LINEAR)
    dest[y1_:y2_, x1_:x2_] = roi_img

_created_windows = set()
def ensure_window(title, w=None, h=None, x=None, y=None):
    if title in _created_windows: return
    flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_GUI_EXPANDED"):
        flags |= cv2.WINDOW_GUI_EXPANDED
    cv2.namedWindow(title, flags)
    if w and h: cv2.resizeWindow(title, w, h)
    if x is not None and y is not None: cv2.moveWindow(title, x, y)
    _created_windows.add(title)

def wrap_by_pixel(text, max_px, font_scale=0.52, thickness=1):
    """픽셀 폭 기준 줄바꿈 (단어 우선, 필요시 문자 단위 분할)"""
    words = text.split(' ')
    lines, cur = [], ""
    def wpx(s): return cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
    for w in words:
        cand = (cur + " " + w).strip()
        if wpx(cand) <= max_px:
            cur = cand
        else:
            if cur:
                lines.append(cur); cur = ""
            # 너무 긴 토큰은 문자 단위로 분할
            if wpx(w) > max_px:
                piece = ""
                for ch in w:
                    if wpx(piece + ch) <= max_px:
                        piece += ch
                    else:
                        if piece: lines.append(piece)
                        piece = ch
                cur = piece
            else:
                cur = w
    if cur: lines.append(cur)
    return lines

def render_log_panel(lines_deque, panel_h, panel_w, title, tail_lines=TAIL_LINES):
    """로그 패널(제목 + 줄바꿈 텍스트) 이미지 생성"""
    font_scale = 0.52
    thickness  = 1
    line_h     = 19
    top_pad    = 30
    left_pad   = 12
    right_pad  = 12
    usable_h   = panel_h - top_pad - 8
    max_rows   = max(8, usable_h // line_h)  # 화면 높이에 맞춰 자동 행수 산정

    # 픽셀 폭 기준 래핑
    max_text_w = panel_w - left_pad - right_pad
    raw_lines  = list(lines_deque)[-tail_lines:]
    wrapped = []
    for s in raw_lines:
        wrapped.extend(wrap_by_pixel(s, max_text_w, font_scale, thickness))
    if len(wrapped) > max_rows:
        wrapped = wrapped[-max_rows:]  # 최근 내용 우선

    img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    cv2.putText(img, title, (left_pad, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,255,200), 2, cv2.LINE_AA)
    y = top_pad
    for line in wrapped:
        cv2.putText(img, line, (left_pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220,220,220),
                    thickness, cv2.LINE_AA)
        y += line_h
    return img

def poll_tail(fp, buf):
    """NDJSON 파일을 tail처럼 읽어 deque에 누적"""
    while True:
        pos = fp.tell()
        line = fp.readline()
        if not line:
            fp.seek(pos)
            break
        buf.append(line.rstrip("\n"))

def draw_label_kr_bgr(img_bgr, text, org=(12, 32),
                      font_path=KR_FONT_PATH, font_size=28,
                      text_color=(255,255,255), bg_color=(0,0,0),
                      bg_alpha=0.35, padding=10):
    """BGR 이미지에 한글 라벨(반투명 박스) 그리기 (Pillow 사용)"""
    h, w = img_bgr.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(font_path, font_size)

    try:
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)

    x, y = org
    box_tl = (x - padding, y - th - padding)
    box_br = (x + tw + padding, y + padding)
    bg = (*bg_color, int(255 * bg_alpha))
    draw.rounded_rectangle([box_tl, box_br], radius=8, fill=bg)

    # 얇은 테두리 느낌
    for ox, oy in [(-1,0),(1,0),(0,-1),(0,1)]:
        draw.text((x+ox, y-th+oy), text, font=font, fill=(0,0,0,180))
    draw.text((x, y-th), text, font=font, fill=(*text_color, 255))

    out = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
def wrap_by_pixel_kr(text: str, max_px: int, font: ImageFont.FreeTypeFont, draw: ImageDraw.ImageDraw):
    """Pillow의 textlength로 픽셀 폭 기준 한글 줄바꿈 (단어 우선, 필요시 문자 분해)"""
    words = text.split(' ')
    lines, cur = [], ""
    def wpx(s): return draw.textlength(s, font=font)

    for w in words:
        cand = (cur + " " + w).strip() if cur else w
        if wpx(cand) <= max_px:
            cur = cand
        else:
            if cur:
                lines.append(cur); cur = ""
            # 토큰 자체가 너무 길면 문자 단위 분해
            if wpx(w) > max_px:
                piece = ""
                for ch in w:
                    if wpx(piece + ch) <= max_px:
                        piece += ch
                    else:
                        if piece: lines.append(piece)
                        piece = ch
                cur = piece
            else:
                cur = w
    if cur: lines.append(cur)
    return lines

def render_log_panel_kr(lines_deque, panel_h, panel_w, title,
                        tail_lines=35, font_path=KR_FONT_PATH,
                        title_size=24, line_size=18, fg=(220,220,220), title_fg=(200,255,200)):
    """한글 지원 로그 패널 생성(Pillow)"""
    # 캔버스
    img = Image.new("RGB", (panel_w, panel_h), (0,0,0))
    draw = ImageDraw.Draw(img)

    # 폰트
    try:
        font_title = ImageFont.truetype(font_path, title_size)
        font_line  = ImageFont.truetype(font_path, line_size)
    except Exception:
        # 폰트 경로가 틀려도 크래시 없이 기본으로
        font_title = ImageFont.load_default()
        font_line  = ImageFont.load_default()

    # 레이아웃
    top_pad   = 30
    left_pad  = 12
    right_pad = 12
    max_text_w = panel_w - left_pad - right_pad

    # 제목
    draw.text((left_pad, 6), title, font=font_title, fill=title_fg)

    # 줄 간격 계산 (bbox 기반)
    _, _, _, th = draw.textbbox((0,0), "Hg", font=font_line)
    line_h = int(th * 1.2)

    usable_h = panel_h - top_pad - 8
    max_rows = max(8, usable_h // line_h)

    # 최근 tail_lines만 가져와 래핑
    raw_lines = list(lines_deque)[-tail_lines:]
    wrapped = []
    for s in raw_lines:
        wrapped.extend(wrap_by_pixel_kr(s, max_text_w, font_line, draw))

    if len(wrapped) > max_rows:
        wrapped = wrapped[-max_rows:]

    # 본문 렌더
    y = top_pad
    for line in wrapped:
        draw.text((left_pad, y), line, font=font_line, fill=fg)
        y += line_h

    # OpenCV용 BGR로 변환
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# =====================[ 로그 준비 ]=====================
tail_encrypt = deque(maxlen=TAIL_LINES)
tail_decrypt = deque(maxlen=TAIL_LINES)
if WRITE_ENCRYPT_NDJSON:
    fp_encrypt = open(ENCRYPT_NDJSON, "r", encoding="utf-8")
if WRITE_DECRYPT_NDJSON:
    fp_decrypt = open(DECRYPT_NDJSON, "r", encoding="utf-8")

# =====================[ 셀/패널 크기 ]=====================
CELL_H = 600                      # ✅ 요구사항: 영상 셀 높이 640px
CELL_W = int(width * (CELL_H / max(1, height)))  # 원본 비율 유지 너비
LOG_COL_W = int(CELL_W * LOG_COL_SCALE)          # 로그 폭
ROW_H = CELL_H                                     # 한 행 높이
VIDEO_PANEL_W = CELL_W * 2
VIDEO_PANEL_H = CELL_H * 2
FINAL_W = VIDEO_PANEL_W + LOG_COL_W
FINAL_H = VIDEO_PANEL_H

# =====================[ 처리 루프 ]=====================
roi_data = {}
roi_total = 0

for frame_idx in tqdm(range(total_frames or 10**9), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    # 뷰 준비
    view_orig = frame.copy()
    view_det  = frame.copy()  # 얼굴 인식 박스 덮어쓸 예정
    view_mos  = frame.copy()
    view_rec  = None          # 모자이크에 복원 ROI를 덮어쓰기

    # YOLO 추론
    results = model.predict(
        source=frame,
        conf=0.25,
        device=0 if device == 'cuda' else None,
        half=(device == 'cuda'),
        verbose=False
    )

    frame_rois = []
    roi_idx = 0

    # 박스 그리기/모자이크/암복호화
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(getattr(box, "cls", [0])[0]) if hasattr(box, "cls") else 0
            conf_v = float(getattr(box, "conf", [0.0])[0]) if hasattr(box, "conf") else 0.0

            # 얼굴 인식(박스)
            cv2.rectangle(view_det, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(view_det, f"id:{cls_id} conf:{conf_v:.2f}",
                        (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)

            # 비식별화(모자이크)
            mosaic_region(view_mos, x1, y1, x2, y2, block=10)

            # 암/복호화(데모: 메모리 내 즉시 복호)
            roi_uid = f"{frame_idx:05d}_{roi_idx}"; roi_idx += 1
            x1_, y1_ = max(0, x1), max(0, y1)
            x2_, y2_ = min(width, x2), min(height, y2)
            roi = view_orig[y1_:y2_, x1_:x2_]

            cipher_len = None
            cipher_sha256 = None
            plain_sha256 = None
            key_b64 = iv_b64 = None

            if roi.size > 0:
                ok, buf = cv2.imencode(".png", roi)
                if ok:
                    roi_bytes = buf.tobytes()
                    plain_sha256 = hashlib.sha256(roi_bytes).hexdigest()

                    key = os.urandom(32)
                    iv  = os.urandom(12)
                    aes = AESGCM(key)
                    cipher = aes.encrypt(iv, roi_bytes, None)
                    cipher_len = len(cipher)
                    cipher_sha256 = hashlib.sha256(cipher).hexdigest()

                    if WRITE_ENCRYPT_NDJSON:
                        append_ndjson_line(ENCRYPT_NDJSON, {
                            "roi_uid": roi_uid,
                            "frame": frame_idx,
                            "bbox": [x1, y1, x2, y2],
                            "cipher_len": cipher_len,
                            "cipher_sha256": cipher_sha256
                        })

                    # 복호화해서 view_rec(=모자이크 복원 뷰)에 덮어쓰기
                    try:
                        plain = aes.decrypt(iv, cipher, None)
                        dec_sha256 = hashlib.sha256(plain).hexdigest()
                        dec_arr = np.frombuffer(plain, dtype=np.uint8)
                        dec_img = cv2.imdecode(dec_arr, cv2.IMREAD_COLOR)

                        if view_rec is None:
                            view_rec = view_mos.copy()
                        paste_roi(view_rec, x1_, y1_, x2_, y2_, dec_img)

                        if WRITE_DECRYPT_NDJSON:
                            append_ndjson_line(DECRYPT_NDJSON, {
                                "roi_uid": roi_uid,
                                "frame": frame_idx,
                                "bbox": [x1, y1, x2, y2],
                                "match_sha256": (dec_sha256 == plain_sha256)
                            })
                    except Exception as e:
                        if WRITE_DECRYPT_NDJSON:
                            append_ndjson_line(DECRYPT_NDJSON, {
                                "roi_uid": roi_uid,
                                "frame": frame_idx,
                                "bbox": [x1, y1, x2, y2],
                                "error": f"{type(e).__name__}: {str(e)}"
                            })

                    if INCLUDE_KEYS_IN_FINAL_JSON:
                        key_b64 = base64.b64encode(key).decode()
                        iv_b64  = base64.b64encode(iv).decode()

            # 최종 JSON 메타
            entry = {
                "roi_uid": roi_uid,
                "bbox": [x1, y1, x2, y2],
                "cls_id": cls_id,
                "conf": conf_v
            }
            if cipher_len is not None:
                entry.update({
                    "cipher_len": cipher_len,
                    "cipher_sha256": cipher_sha256,
                    "plain_sha256": plain_sha256
                })
                if INCLUDE_KEYS_IN_FINAL_JSON:
                    entry.update({"key_b64": key_b64, "iv_b64": iv_b64})

            frame_rois.append(entry)
            roi_total += 1

    if frame_rois:
        roi_data[f"frame_{frame_idx:05d}"] = frame_rois

    # =====================[ 표시 패널 구성 ]=====================
    # 각 영상 셀 리사이즈 + 라벨
    orig_disp = draw_label_kr_bgr(cv2.resize(view_orig, (CELL_W, CELL_H)), "원본")
    det_disp  = draw_label_kr_bgr(cv2.resize(view_det,  (CELL_W, CELL_H)), "얼굴 인식")
    mos_disp  = draw_label_kr_bgr(cv2.resize(view_mos,  (CELL_W, CELL_H)), "비식별화")
    rec_disp  = draw_label_kr_bgr(cv2.resize(view_rec if view_rec is not None else view_mos, (CELL_W, CELL_H)),
                                  "식별화 영상")

    # 2×2 영상 패널 (좌측 2열)
    top_row    = cv2.hconcat([orig_disp, det_disp])   # 상단: 원본 | 얼굴 인식
    bottom_row = cv2.hconcat([mos_disp,  rec_disp])   # 하단: 비식별화 | 식별화 영상
    video_panel = cv2.vconcat([top_row, bottom_row])  # (2*CELL_H, 2*CELL_W)

    # 로그 업데이트
    if WRITE_ENCRYPT_NDJSON: poll_tail(fp_encrypt, tail_encrypt)
    if WRITE_DECRYPT_NDJSON: poll_tail(fp_decrypt, tail_decrypt)

    # 로그 패널 (우측 1열, 상단/하단 분할)
    enc_log_panel = render_log_panel_kr(tail_encrypt, panel_h=ROW_H, panel_w=LOG_COL_W,
                                     title="비식별화 로그 (Encrypt)", tail_lines=TAIL_LINES)
    dec_log_panel = render_log_panel_kr(tail_decrypt, panel_h=ROW_H, panel_w=LOG_COL_W,
                                     title="식별화 로그 (Decrypt)", tail_lines=TAIL_LINES)
    log_col = cv2.vconcat([enc_log_panel, dec_log_panel])  # 높이 = 2*CELL_H

    # 최종 합성: [ (2x2 영상) | (로그 상/하) ]
    final_panel = cv2.hconcat([video_panel, log_col])

    # 오버레이 정보
    overlay = f"frame:{frame_idx+1}/{total_frames}  roi_total:{roi_total}"
    (tw, th), _ = cv2.getTextSize(overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(final_panel, overlay,
                (final_panel.shape[1] - tw - 12, final_panel.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # 창 띄우기(드래그 리사이즈 가능)
    ensure_window("Video+Logs", final_panel.shape[1], final_panel.shape[0], 40, 40)
    cv2.imshow("Video+Logs", final_panel)

    # 결과 비식별화 영상 저장
    out.write(view_mos)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================[ 종료 처리 ]=====================
cap.release()
out.release()
if WRITE_ENCRYPT_NDJSON: fp_encrypt.close()
if WRITE_DECRYPT_NDJSON: fp_decrypt.close()
cv2.destroyAllWindows()

if WRITE_FINAL_JSON:
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(roi_data, f, indent=2, ensure_ascii=False)

print(f"✅ 완료! 모자이크 비디오: {OUTPUT_VIDEO}")
if WRITE_FINAL_JSON:
    print(f"✅ 최종 JSON: {OUTPUT_JSON}")
if WRITE_ENCRYPT_NDJSON:
    print(f"✅ 암호화 NDJSON: {ENCRYPT_NDJSON}")
if WRITE_DECRYPT_NDJSON:
    print(f"✅ 복호화 NDJSON: {DECRYPT_NDJSON}")
