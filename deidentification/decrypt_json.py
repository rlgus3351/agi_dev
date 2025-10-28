import os
import json
import base64
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ======================================
# 🔧 고정 설정(원하는 값으로 수정)
# ======================================
JSON_PATH     = r"./data/ID-10001_1.json"  # 👉 단일 JSON 파일
OUTPUT_DIR    = r"C:\Users\user\Desktop\DEV_AGI\parkinson\output\preview_video"    # 결과 폴더

# 🔁 프레임/영상 설정 (JSON에 없으면 이 값 사용)
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
FPS           = 30
FOURCC        = "mp4v"   # Windows 호환성 OK

# 🎨 렌더 옵션
BACKGROUND_RGB = (0, 0, 0)   # 검정 배경
DRAW_BBOX      = False       # bbox 테두리 ON/OFF
BBOX_THICKNESS = 2

# 🔎 디버깅/제한 옵션
LIMIT_FRAMES         = None  # 앞에서부터 N 프레임만 (None=전체)
LIMIT_ROIS_PER_FRAME = None  # 프레임당 N개 ROI만 (None=전체)

# ======================================
# 🧠 유틸
# ======================================
def b64d(s: str) -> bytes:
    return base64.b64decode(s)

def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default

def get_json_video_meta(roi_log: Dict[str, Any]) -> Tuple[int, int, int]:
    """JSON meta가 있으면 사용, 없으면 기본값."""
    meta = roi_log.get("meta", {})
    fw = safe_int(meta.get("frame_width"), FRAME_WIDTH)
    fh = safe_int(meta.get("frame_height"), FRAME_HEIGHT)
    fps = safe_int(meta.get("fps"), FPS)
    return fw, fh, fps

def decrypt_roi_to_image(roi_info: Dict[str, Any]) -> Optional[np.ndarray]:
    """단일 ROI 복호화 → OpenCV 이미지(ndarray)"""
    try:
        if not roi_info.get("restorable", True):
            return None
        if not roi_info.get("key") or not roi_info.get("iv") or not roi_info.get("encrypted_roi"):
            return None

        key = b64d(roi_info["key"])
        iv = b64d(roi_info["iv"])
        ciphertext = b64d(roi_info["encrypted_roi"])
        plain_bytes = AESGCM(key).decrypt(iv, ciphertext, None)

        arr = np.frombuffer(plain_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def paste_roi(canvas: np.ndarray, roi_img: np.ndarray, bbox: List[int]) -> None:
    """roi_img를 bbox(x1,y1,x2,y2)에 붙임(클리핑/리사이즈 포함)."""
    h, w = canvas.shape[:2]
    if not bbox or len(bbox) < 4 or roi_img is None:
        return

    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    # 클리핑
    x1c = max(0, min(w, x1)); x2c = max(0, min(w, x2))
    y1c = max(0, min(h, y1)); y2c = max(0, min(h, y2))
    if x2c <= x1c or y2c <= y1c:
        return

    target_w = x2c - x1c
    target_h = y2c - y1c
    rh, rw = roi_img.shape[:2]
    if rw != target_w or rh != target_h:
        if target_w <= 0 or target_h <= 0:
            return
        roi_img = cv2.resize(roi_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    canvas[y1c:y2c, x1c:x2c] = roi_img

def draw_bbox(canvas: np.ndarray, bbox: List[int], color=(255, 255, 255), thickness=2):
    if not bbox or len(bbox) < 4:
        return
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

def build_video_from_json(json_path: str, output_dir: str) -> str:
    """단일 JSON → 프리뷰 mp4 생성."""
    with open(json_path, "r", encoding="utf-8") as f:
        roi_log = json.load(f)

    frames: Dict[str, List[Dict[str, Any]]] = roi_log.get("frames", {})
    frame_keys = sorted(frames.keys())
    if LIMIT_FRAMES is not None:
        frame_keys = frame_keys[:max(0, LIMIT_FRAMES)]

    fw, fh, fps = get_json_video_meta(roi_log)

    base = os.path.splitext(os.path.basename(json_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base}_preview.mp4")

    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))
    if not writer.isOpened():
        raise RuntimeError("비디오 라이터 초기화 실패: 코덱/경로/권한 확인")

    with tqdm(total=len(frame_keys), desc=f"🎬 {base}", ncols=100) as pbar:
        for fk in frame_keys:
            frame = np.zeros((fh, fw, 3), dtype=np.uint8)
            frame[:] = BACKGROUND_RGB

            rois = frames.get(fk, [])
            if LIMIT_ROIS_PER_FRAME is not None:
                rois = rois[:max(0, LIMIT_ROIS_PER_FRAME)]

            for roi_info in rois:
                roi_img = decrypt_roi_to_image(roi_info)
                bbox = roi_info.get("bbox", [None, None, None, None])
                if roi_img is not None:
                    paste_roi(frame, roi_img, bbox)
                    if DRAW_BBOX:
                        draw_bbox(frame, bbox, (255, 255, 255), BBOX_THICKNESS)

            writer.write(frame)
            pbar.update(1)

    writer.release()
    return out_path

def main():
    if not os.path.isfile(JSON_PATH):
        print(f"❌ JSON 파일이 존재하지 않습니다:\n{JSON_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        out = build_video_from_json(JSON_PATH, OUTPUT_DIR)
        print(f"\n✅ 완료: {os.path.basename(JSON_PATH)} → {out}")
    except Exception as e:
        print(f"\n⚠️ 실패: {e}")

if __name__ == "__main__":
    main()
