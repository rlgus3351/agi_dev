# -*- coding: utf-8 -*-
from ultralytics import YOLO
import cv2, os, torch, numpy as np
from PIL import Image, ImageDraw, ImageFont

# ========= 설정 =========
MODEL_PATH   = r"C:/Users/user/Desktop/frameSecurity/data/model.pt"
VIDEO_PATH   = r"C:/Users/user/Desktop/frameSecurity/data/250811/input2.mp4"
FRAME_INDEX  = 120
KR_FONT_PATH = r"C:/Windows/Fonts/malgun.ttf"  # 시스템 폰트 경로(윈도우 예: 맑은고딕)

BLOCK_SIZE   = 10
PAD_BASE     = 0.00  # 기본
PAD_MEDIUM   = 0.15  # 조금 크게
PAD_LARGE    = 0.30  # 크게

LABELS = ["오리지널", "기본 모자이크", "조금 크게", "크게"]

# ========= 한글 라벨 유틸 =========
FONT = ImageFont.truetype(KR_FONT_PATH, 28)

def draw_label_kr_bgr(img_bgr, text,
                      font_path=KR_FONT_PATH,
                      font_size=48,   # 글씨 크기
                      text_color=(255,255,255), 
                      bg_color=(0,0,0), 
                      bg_alpha=0.35,
                      padding=10):
    """
    OpenCV BGR 이미지에 반투명 박스 + 중앙정렬 한글 라벨
    """
    h, w = img_bgr.shape[:2]
    font = ImageFont.truetype(font_path, font_size)

    # PIL 변환
    base = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # 텍스트 크기
    bbox = draw.textbbox((0,0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]

    # 박스 위치 (좌상단 기준 20,20에서 시작)
    x, y = 20, 20
    box_tl = (x, y)
    box_br = (x + tw + 2*padding, y + th + 2*padding)

    # 반투명 배경
    bg = (*bg_color, int(255*bg_alpha))
    draw.rounded_rectangle([box_tl, box_br], radius=12, fill=bg)

    # 텍스트 중앙 좌표
    text_x = x + padding
    text_y = y + padding
    draw.text((text_x, text_y), text, font=font, fill=(*text_color,255))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
def mosaic_region(img, x1, y1, x2, y2, block=10):
    h, w = img.shape[:2]
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w,x2), min(h,y2)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return
    small  = cv2.resize(roi, (max(1, block), max(1, block)), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = mosaic

def expand_bbox(x1, y1, x2, y2, pad_ratio, w, h):
    bw, bh = x2-x1, y2-y1
    px, py = int(bw*pad_ratio), int(bh*pad_ratio)
    return max(0,x1-px), max(0,y1-py), min(w,x2+px), min(h,y2+py)

# ========= 모델 로드 =========
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)
if device=='cuda':
    try: model.model.half()
    except: pass

# ========= 프레임 로드 =========
cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
FRAME_INDEX = max(0, min(FRAME_INDEX, total-1))
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError(f"{FRAME_INDEX} 프레임 읽기 실패")

h, w = frame.shape[:2]

# ========= 감지 =========
results = model.predict(source=frame, conf=0.25,
                        device=0 if device=='cuda' else None,
                        half=(device=='cuda'), verbose=False)

bboxes=[]
for r in results:
    if r.boxes is None: continue
    for box in r.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        bboxes.append((x1,y1,x2,y2))

# ========= 4개 이미지 생성 =========
orig   = frame.copy()
base   = frame.copy()
medium = frame.copy()
large  = frame.copy()

for (x1,y1,x2,y2) in bboxes:
    ex1,ey1,ex2,ey2 = expand_bbox(x1,y1,x2,y2,PAD_BASE,w,h)
    mosaic_region(base,ex1,ey1,ex2,ey2,BLOCK_SIZE)

for (x1,y1,x2,y2) in bboxes:
    ex1,ey1,ex2,ey2 = expand_bbox(x1,y1,x2,y2,PAD_MEDIUM,w,h)
    mosaic_region(medium,ex1,ey1,ex2,ey2,BLOCK_SIZE)

for (x1,y1,x2,y2) in bboxes:
    ex1,ey1,ex2,ey2 = expand_bbox(x1,y1,x2,y2,PAD_LARGE,w,h)
    mosaic_region(large,ex1,ey1,ex2,ey2,BLOCK_SIZE)

# ========= 라벨 =========
orig   = draw_label_kr_bgr(orig,   LABELS[0])
base   = draw_label_kr_bgr(base,   LABELS[1])
medium = draw_label_kr_bgr(medium, LABELS[2])
large  = draw_label_kr_bgr(large,  LABELS[3])

# ========= 가로 1행 배치 =========
target_h = 720
scale = target_h / h
new_w = int(w * scale)

A = cv2.resize(orig,   (new_w, target_h))
B = cv2.resize(base,   (new_w, target_h))
C = cv2.resize(medium, (new_w, target_h))
D = cv2.resize(large,  (new_w, target_h))

panel = cv2.hconcat([A, B, C, D])

cv2.imshow("Original | Base | Medium | Large (q=exit)", panel)
cv2.waitKey(0)
cv2.destroyAllWindows()
