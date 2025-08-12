import cv2
import numpy as np
import uuid
import json
import base64
from Crypto.Cipher import AES
import os
import time
from tqdm import tqdm  # ✅ 추가

# ========== 설정 ==========
MASTER_KEY = b'ThisIsMasterKey!'  # 16바이트 고정
VIDEO_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//input2.mp4")
OUTPUT_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//output2.mp4")
KEY_JSON_PATH = os.path.abspath("video_encrypted_keys.json")

# ========== 유틸 함수 ==========
def pad(data): 
    return data + bytes([16 - len(data) % 16]) * (16 - len(data) % 16)

def unpad(data): 
    return data[:-data[-1]]

def aes_encrypt(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(pad(data))

def aes_decrypt(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)
    return unpad(cipher.decrypt(data))

def encrypt_roi(roi: np.ndarray):
    data = roi.tobytes()
    random_key = uuid.uuid4().hex[:16].encode()  # 16 bytes key
    encrypted_roi = aes_encrypt(data, random_key)
    encrypted_key = aes_encrypt(random_key, MASTER_KEY)
    return encrypted_roi, base64.b64encode(encrypted_key).decode(), roi.shape

# ========== 메인 처리 ==========
def process_video():
    print(f"🎬 입력 파일 경로: {VIDEO_PATH}")
    if not os.path.exists(VIDEO_PATH):
        print(f"[❌] 파일이 존재하지 않습니다: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[❌] 비디오 파일을 열 수 없습니다.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📏 해상도: {width}x{height}, FPS: {fps}, 총 프레임: {total_frames}")

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_idx = 0
    all_keys = []

    # ✅ tqdm으로 진행률 바 생성
    for frame_idx in tqdm(range(total_frames), desc="🔄 영상 처리 중", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for face_num, (x, y, w, h) in enumerate(faces):
            roi = frame[y:y+h, x:x+w]
            encrypted_roi, enc_key, shape = encrypt_roi(roi)

            # 블러 처리된 ROI 삽입
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (23, 23), 30)

            all_keys.append({
                "frame": int(frame_idx),
                "face_id": f"frame_{frame_idx}_face_{face_num}",
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "shape": [int(s) for s in shape],
                "encrypted_key": enc_key,
                "roi_base64": base64.b64encode(encrypted_roi).decode()
            })

        out.write(frame)

    cap.release()
    out.release()

    with open(KEY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_keys, f, indent=2)

    print("\n✅ 처리 완료!")
    print(f"📁 출력 영상: {OUTPUT_PATH}")
    print(f"🔐 키 저장: {KEY_JSON_PATH}")

# ========== 실행 ==========
if __name__ == "__main__":
    process_video()
