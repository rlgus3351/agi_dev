import os
import cv2
import json
import base64
import numpy as np
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def decrypt_rois(json_path: str, video_path: str, output_path: str):
    """익명화된 영상과 JSON을 이용해 복호화된 영상을 생성"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ 비식별화된 영상을 찾을 수 없습니다: {video_path}")

    # ✅ JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        roi_log = json.load(f)

    # ✅ 영상 로드
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("❌ 영상을 열 수 없습니다.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎬 복호화 시작: {os.path.basename(video_path)}")
    print(f"🧾 총 프레임: {frame_count} | FPS: {fps} | 해상도: {width}x{height}\n")

    with tqdm(total=frame_count, desc="🔓 복호화 진행중", ncols=110) as pbar:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_key = f"frame_{frame_index:05d}"
            rois = roi_log.get("frames", {}).get(frame_key, [])
            for roi_info in rois:
                try:
                    if not roi_info.get("restorable", True):
                        continue
                    if not roi_info.get("key") or not roi_info.get("iv") or not roi_info.get("encrypted_roi"):
                        continue

                    x1, y1, x2, y2 = roi_info["bbox"]
                    key = base64.b64decode(roi_info["key"])
                    iv = base64.b64decode(roi_info["iv"])
                    encrypted_data = base64.b64decode(roi_info["encrypted_roi"])

                    aes = AESGCM(key)
                    decrypted_bytes = aes.decrypt(iv, encrypted_data, None)
                    roi_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
                    roi_img = cv2.imdecode(roi_array, cv2.IMREAD_COLOR)

                    if roi_img is not None:
                        frame[y1:y2, x1:x2] = roi_img
                except Exception as e:
                    print(f"⚠️ frame {frame_index} ROI 복호화 실패: {e}")

            out.write(frame)
            frame_index += 1
            pbar.update(1)

    cap.release()
    out.release()
    print(f"\n✅ 복호화 완료 → {output_path}")
    print(f"📊 총 프레임: {frame_count} | 복원된 프레임: {frame_index}")

def main():
    print("===========================================")
    print("     🧩 Face ROI 복호화 프로그램 (AES-GCM)     ")
    print("===========================================\n")

    json_path = "C:\\TeamGit\\agi_dev\\deidentification\\data\\ID-10050_2.json"
    video_path = "C:\\TeamGit\\agi_dev\\deidentification\\data\\f6222d72-ca36-40b0-97be-c4bc81d2b168_2_final.mp4"

    output_dir = os.path.join("output", "restored")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_restored.mp4")

    decrypt_rois(json_path, video_path, output_path)

if __name__ == "__main__":
    main()
