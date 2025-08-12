import tkinter as tk
from PIL import Image, ImageTk
import cv2
import json
import base64
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ==== 경로 설정 ====
VIDEO_PATH = r"C:\Users\user\Desktop\frameSecurity\data\AES\yolo_blur3.mp4"
JSON_PATH = r"C:\Users\user\Desktop\frameSecurity\data\AES\yolo_blur_rois3.json"

# ==== JSON 로드 ====
with open(JSON_PATH, "r", encoding="utf-8") as f:
    roi_data = json.load(f)
frames_data = roi_data["frames"]

# ==== 비디오 설정 ====
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

# ==== Tkinter GUI ====
root = tk.Tk()
root.title("모자이크 vs 복호화 실시간 비교")
label = tk.Label(root)
label.pack(fill=tk.BOTH, expand=True)

def decode_roi(frame_idx, frame):
    """특정 프레임에서 ROI 복호화"""
    key_frame = f"frame_{frame_idx:05d}"
    if key_frame not in frames_data:
        return frame

    for roi_info in frames_data[key_frame]:
        x1, y1, x2, y2 = roi_info["bbox"]
        key = base64.b64decode(roi_info["key"])
        iv = base64.b64decode(roi_info["iv"])
        encrypted_roi = base64.b64decode(roi_info["encrypted_roi"])

        aesgcm = AESGCM(key)
        decrypted_bytes = aesgcm.decrypt(iv, encrypted_roi, None)
        roi_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
        roi_img = cv2.imdecode(roi_array, cv2.IMREAD_COLOR)

        if roi_img is not None and roi_img.size > 0:
            frame[y1:y2, x1:x2] = roi_img

    return frame

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    # 왼쪽: 모자이크 원본
    left_frame = frame.copy()
    # 오른쪽: 복호화 적용
    right_frame = decode_roi(frame_idx, frame.copy())

    # 두 프레임을 같은 높이로 리사이즈
    h = 960
    def resize_keep_aspect(img, target_h):
        h0, w0 = img.shape[:2]
        w = int(w0 * target_h / h0)
        return cv2.resize(img, (w, target_h))
    
    left_frame = resize_keep_aspect(left_frame, h)
    right_frame = resize_keep_aspect(right_frame, h)

    combined = cv2.hconcat([left_frame, right_frame])
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

    img = ImageTk.PhotoImage(image=Image.fromarray(combined_rgb))
    label.config(image=img)
    label.image = img

    root.after(int(1000 // fps), update_frame)

update_frame()
root.mainloop()
cap.release()
