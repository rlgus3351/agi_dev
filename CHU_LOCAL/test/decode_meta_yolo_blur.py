import cv2
import os
import json
import base64
import numpy as np
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def decrypt_rois(json_path, anonymized_video_path, output_video_path):
    """AES-GCM ROI 복호화 (익명화된 영상 → 복원 영상 생성)"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")
    if not os.path.exists(anonymized_video_path):
        raise FileNotFoundError(f"❌ 익명화된 영상을 찾을 수 없습니다: {anonymized_video_path}")

    # ✅ JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        roi_log = json.load(f)

    cap = cv2.VideoCapture(anonymized_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ 비디오를 열 수 없습니다: {anonymized_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("❌ VideoWriter 초기화 실패")

    print(f"🔓 복호화 시작: {os.path.basename(anonymized_video_path)} ({total_frames} frames)")

    restored_count = 0
    with tqdm(total=total_frames, desc="Restoring", ncols=100) as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_key = f"frame_{frame_idx:05d}"
            rois = roi_log.get("frames", {}).get(frame_key, [])

            for roi_info in rois:
                try:
                    key = base64.b64decode(roi_info["key"])
                    iv = base64.b64decode(roi_info["iv"])
                    enc_data = base64.b64decode(roi_info["encrypted_roi"])
                    bbox = roi_info["bbox"]

                    aes = AESGCM(key)
                    decrypted_bytes = aes.decrypt(iv, enc_data, None)
                    roi_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
                    roi_img = cv2.imdecode(roi_array, cv2.IMREAD_COLOR)

                    if roi_img is not None:
                        x1, y1, x2, y2 = bbox
                        frame[y1:y2, x1:x2] = roi_img
                        restored_count += 1
                except Exception as e:
                    print(f"⚠️ frame {frame_idx}: ROI 복호화 실패 - {e}")

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()
    print(f"\n✅ 복호화 완료! 복원된 ROI 수: {restored_count}")
    print(f"📁 출력 파일: {output_video_path}")
    
if __name__ == "__main__":
    JSON_PATH = r"c:\TeamGit\agi_dev\CHU_LOCAL\test\data\json\sample_2_rois.json"
    ANONYMIZED_VIDEO = r"c:\TeamGit\agi_dev\CHU_LOCAL\test\data\output\sample_2_anonymized.mp4"
    RESTORED_VIDEO = r"c:\TeamGit\agi_dev\CHU_LOCAL\test\data\output\restored\sample_2_restored.mp4"

    decrypt_rois(JSON_PATH, ANONYMIZED_VIDEO, RESTORED_VIDEO)