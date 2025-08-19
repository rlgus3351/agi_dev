# -*- coding: utf-8 -*-
from ultralytics import YOLO
import cv2, torch, numpy as np
from PIL import Image, ImageDraw, ImageFont

# ========= 경로/설정 =========
MODEL_PATH   = r"C:/Users/user/Desktop/frameSecurity/data/model.pt"
VIDEO_PATH   = r"C:/Users/user/Desktop/frameSecurity/data/250811/input2.mp4"
FRAME_INDEX  = 120

# 크롭 패딩(크롭은 bbox보다 이만큼 더 크게 자름 → bbox가 크롭 내부에 보이도록)
CROP_PAD_RATIO = 0.50     # 예: 0.5면 가/세 각각 50% 확장. 0.0으로 두면 bbox와 동일(이때 초록 박스가 테두리와 겹침)

# 모자이크 강도/영역
BLOCK_SIZE   = 12         # 블록 크기(크면 거칠어짐)
PAD_BASE     = 0.00       # 기본 모자이크: 패딩 0
PAD_MEDIUM   = 0.15       # 조금 크게
PAD_LARGE    = 0.30       # 크게

# 라벨/폰트
KR_FONT_PATH = r"C:/Windows/Fonts/malgun.ttf"
LABELS = ["원본(+bbox)", "기본 모자이크", "조금 크게", "크게"]
def load_font(path, size):
    try:    return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()
FONT = load_font(KR_FONT_PATH, 44)

def label_kr(img_bgr, text, pos=(18,18), pad=10, font=FONT,
             fg=(255,255,255), bg=(0,0,0), alpha=0.38):
    h, w = img_bgr.shape[:2]
    base = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov   = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(ov)
    (tw, th) = draw.textbbox((0,0), text, font=font)[2:]
    x, y = pos
    draw.rounded_rectangle([(x, y), (x+tw+2*pad, y+th+2*pad)],
                           radius=12, fill=(bg[0],bg[1],bg[2], int(255*alpha)))
    draw.text((x+pad, y+pad), text, font=font, fill=(fg[0],fg[1],fg[2],255))
    out = Image.alpha_composite(base, ov).convert("RGB")
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

def mosaic_region(img, x1, y1, x2, y2, block=10):
    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return
    small  = cv2.resize(roi, (max(1, block), max(1, block)), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = mosaic

def expand_bbox(x1, y1, x2, y2, pad_ratio, W, H):
    bw, bh = x2-x1, y2-y1
    px, py = int(bw*pad_ratio), int(bh*pad_ratio)
    return max(0,x1-px), max(0,y1-py), min(W,x2+px), min(H,y2+py)

# ========= YOLO/프레임 =========
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)
if device=='cuda':
    try: model.model.half()
    except: pass

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened(): raise RuntimeError("비디오 열기 실패")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
FRAME_INDEX = max(0, min(FRAME_INDEX, total-1))
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ok, frame = cap.read(); cap.release()
if not ok: raise RuntimeError(f"{FRAME_INDEX} 프레임 읽기 실패")

H, W = frame.shape[:2]

# ========= 감지(프레임) =========
results = model.predict(source=frame, conf=0.25,
                        device=0 if device=='cuda' else None,
                        half=(device=='cuda'), verbose=False)

# 얼굴/관심 객체 bbox 목록 생성
bboxes = []
for r in results:
    if r.boxes is None: continue
    for box in r.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        bboxes.append((x1,y1,x2,y2))

if not bboxes:
    # 감지 없음 안내
    msg = np.zeros((200, 900, 3), np.uint8)
    cv2.putText(msg, "No detections in this frame.", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Crops 1x4", msg); cv2.waitKey(0); cv2.destroyAllWindows(); raise SystemExit

# 한 명만 선택(원하면 face_idx로 바꿔가며 사용)
face_idx = 0
x1,y1,x2,y2 = bboxes[face_idx]

# ========= 크롭(패딩 포함) =========
cx1, cy1, cx2, cy2 = expand_bbox(x1,y1,x2,y2, CROP_PAD_RATIO, W, H)
crop = frame[cy1:cy2, cx1:cx2].copy()
ch, cw = crop.shape[:2]

# 원본 크롭 안에서 초록 bbox를 그리려면, 원래 bbox 좌표를 크롭 좌표계로 변환
gx1, gy1 = x1 - cx1, y1 - cy1
gx2, gy2 = x2 - cx2 + (cx2 - cx1), y2 - cy2 + (cy2 - cy1)  # = x2 - cx1, y2 - cy1
gx2, gy2 = x2 - cx1, y2 - cy1

# ========= 4장 만들기 =========
# 1) 크롭 원본 + 초록 bbox
img1 = crop.copy()
cv2.rectangle(img1, (gx1, gy1), (gx2, gy2), (0,255,0), 3)
img1 = label_kr(img1, LABELS[0])

# 2) 기본 모자이크(패딩 0)
img2 = crop.copy()
ex1,ey1,ex2,ey2 = expand_bbox(gx1, gy1, gx2, gy2, PAD_BASE,  cw, ch)
mosaic_region(img2, ex1, ey1, ex2, ey2, block=BLOCK_SIZE)
img2 = label_kr(img2, LABELS[1])

# 3) 조금 크게 모자이크(패딩 0.15)
img3 = crop.copy()
ex1,ey1,ex2,ey2 = expand_bbox(gx1, gy1, gx2, gy2, PAD_MEDIUM, cw, ch)
mosaic_region(img3, ex1, ey1, ex2, ey2, block=BLOCK_SIZE)
img3 = label_kr(img3, LABELS[2])

# 4) 크게 모자이크(패딩 0.30)
img4 = crop.copy()
ex1,ey1,ex2,ey2 = expand_bbox(gx1, gy1, gx2, gy2, PAD_LARGE,  cw, ch)
mosaic_region(img4, ex1, ey1, ex2, ey2, block=BLOCK_SIZE)
img4 = label_kr(img4, LABELS[3])

# ========= 1행 4분할로 보기 =========
target_h = 480
def resize_h(im, h=target_h):
    ih, iw = im.shape[:2]
    return cv2.resize(im, (int(iw*(h/ih)), h))
row = cv2.hconcat([resize_h(img1), resize_h(img2), resize_h(img3), resize_h(img4)])

cv2.imshow("Crops (1x4)", row)
print("캡처해서 발표 자료에 쓰면 돼요. 's' 키를 누르면 파일로도 저장됩니다.")
k = cv2.waitKey(0)
if k in (ord('s'), ord('S')):
    cv2.imwrite("crops_1x4.png", row)
    print("저장: crops_1x4.png")
cv2.destroyAllWindows()
