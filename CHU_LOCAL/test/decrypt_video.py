import os
import cv2
import json
import base64
import numpy as np
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib


def decrypt_rois(json_path: str, video_path: str, output_dir: str = None):
    """
    JSON(AES 암호화 ROI)과 비식별화된 영상으로 원본 복원
    (원본 영상 필요 없음)
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ 입력 영상을 찾을 수 없습니다: {video_path}")

    # ✅ 출력 경로
    base_dir = os.path.dirname(os.path.abspath(video_path))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "restored")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(video_path)
    filename_wo_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{filename_wo_ext}_restored.mp4")

    # ✅ JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        roi_log = json.load(f)

    frames_info = roi_log.get("frames", {})
    meta = roi_log.get("meta", {})

    print(f"🎬 복호화 시작: {filename}")
    print(f"📄 모델: {meta.get('model')} | ROI 확장: {meta.get('roi_expand', 0)*100:.0f}%")
    print(f"🧾 프레임 수: {meta.get('frame_count')} | 해상도: {meta.get('resolution')}")

    # ✅ 영상 로드
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ 영상을 열 수 없습니다: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_restored = 0
    frame_idx = 0

    with tqdm(total=total_frames, desc="🔓 복호화 진행중", ncols=110) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_key = f"frame_{frame_idx:05d}"
            rois = frames_info.get(frame_key, [])

            for roi_info in rois:
                try:
                    if not all([roi_info.get(k) for k in ("key", "iv", "encrypted_roi")]):
                        continue

                    x1, y1, x2, y2 = roi_info["bbox"]
                    key = base64.b64decode(roi_info["key"])
                    iv = base64.b64decode(roi_info["iv"])
                    encrypted_data = base64.b64decode(roi_info["encrypted_roi"])

                    # ✅ AES-GCM 복호화
                    aes = AESGCM(key)
                    decrypted_bytes = aes.decrypt(iv, encrypted_data, None)
                    roi_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
                    roi_img = cv2.imdecode(roi_array, cv2.IMREAD_COLOR)

                    if roi_img is None:
                        continue

                    # ✅ ROI 크기 보정
                    target_w, target_h = x2 - x1, y2 - y1
                    if roi_img.shape[1] != target_w or roi_img.shape[0] != target_h:
                        roi_img = cv2.resize(roi_img, (target_w, target_h))

                    # ✅ 무결성 검증
                    decoded_hash = hashlib.sha256(cv2.imencode('.png', roi_img)[1]).hexdigest()
                    expected_hash = roi_info.get("roi_hash")
                    if expected_hash and decoded_hash != expected_hash:
                        print(f"⚠️ frame {frame_idx} ROI 무결성 불일치 (uuid={roi_info.get('uuid')})")

                    # ✅ ROI 복원
                    frame[y1:y2, x1:x2] = roi_img
                    total_restored += 1

                except Exception as e:
                    print(f"⚠️ frame {frame_idx} ROI 복호화 실패: {e}")

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()

    print(f"\n✅ 복호화 완료!")
    print(f"📊 복원된 ROI 개수: {total_restored}")
    print(f"📁 결과 파일: {output_path}")


# ----------------------------------------------------------
# ✅ 단독 실행 (테스트)
# ----------------------------------------------------------
if __name__ == "__main__":
    json_path = r"C:\TeamGit\agi_dev\CHU_LOCAL\test\data\json\sample_2_rois.json"
    video_path = r"C:\TeamGit\agi_dev\CHU_LOCAL\test\data\output\sample_2_anonymized.mp4"

    decrypt_rois(json_path, video_path)
