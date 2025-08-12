import cv2
import os
import json
import base64
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from tqdm import tqdm

# ==== 경로 설정 ====
INPUT_VIDEO = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//AES//yolo_blur.mp4")
INPUT_JSON = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//AES//yolo_blur_rois.json")
OUTPUT_VIDEO = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//AES//yolo_restored_test.mp4")

# ==== JSON 로드 ====
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    roi_data = json.load(f) 

# ==== 모자이크 영상 로드 ====
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ==== 프레임 처리 ====
for frame_idx in tqdm(range(total_frames), desc="Restoring video"):
    ret, frame = cap.read()
    if not ret:
        break

    key_frame = f"frame_{frame_idx:05d}"
    if key_frame in roi_data:
        for roi_info in roi_data[key_frame]:
            x1, y1, x2, y2 = roi_info["bbox"]

            # Base64 디코딩
            key = base64.b64decode(roi_info["key"])
            iv = base64.b64decode(roi_info["iv"])
            encrypted_roi = base64.b64decode(roi_info["encrypted_roi"])

            # AESGCM 복호화
            aesgcm = AESGCM(key)
            decrypted_bytes = aesgcm.decrypt(iv, encrypted_roi, None)

            # 바이트 → 이미지 복원
            roi_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
            roi_img = cv2.imdecode(roi_array, cv2.IMREAD_COLOR)

            # ROI 복원
            if roi_img is not None and roi_img.size > 0:
                frame[y1:y2, x1:x2] = roi_img

    out.write(frame)

cap.release()
out.release()

print(f"✅ 복원 완료! 영상 저장됨: {OUTPUT_VIDEO}")
