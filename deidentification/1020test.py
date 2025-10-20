import os, cv2, json, uuid, base64, hashlib, subprocess
import numpy as np
from datetime import datetime
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# =============================
# 🔧 PyTorch 2.6 안전 허용 처리
# =============================
import torch
from torch.nn.modules.container import Sequential
torch.serialization.add_safe_globals([
    Sequential,
    __import__('ultralytics').nn.tasks.DetectionModel,
    __import__('ultralytics').nn.modules.conv.Conv,
    __import__('ultralytics').nn.modules.block.C2f,
])

from ultralytics import YOLO


# ============================================
# ✅ SHA256 해시 계산
# ============================================
def sha256_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ============================================
# ✅ 오디오 병합
# ============================================
def merge_audio_with_video(original_video, anonymized_video):
    output_with_audio = anonymized_video.replace("_anonymized.mp4", "_final.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", anonymized_video,
        "-i", original_video,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0",
        output_with_audio
    ]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode == 0:
        print(f"🎵 오디오 포함 영상 생성 완료 → {output_with_audio}")
    else:
        print(f"⚠️ 오디오 병합 실패: {process.stderr.decode()}")
    return output_with_audio


# ============================================
# ✅ 비식별화 (GPU 자동 감지)
# ============================================
def anonymize_video(video_path: str):
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "model", "model.pt")

    # ✅ GPU 자동 감지
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 YOLO 모델 로드 중... (device={device})")

    model = YOLO(model_path)
    print(f"✅ YOLO 모델 로드 완료 ({device})")

    # ✅ 경로 설정
    filename = os.path.basename(video_path)
    filename_wo_ext = os.path.splitext(filename)[0]
    output_dir = os.path.join(base_dir, "output")
    json_dir = os.path.join(base_dir, "json")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{filename_wo_ext}_anonymized.mp4")
    json_path = os.path.join(json_dir, f"{filename_wo_ext}_rois.json")

    # ✅ 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ 비디오를 열 수 없습니다: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    roi_log = {"video_info": {"fps": fps, "frame_count": frame_count, "sha256": sha256_file(video_path)}, "frames": {}}
    total_detected_frames = total_roi_count = total_encrypted = 0

    print(f"🎞️ 총 프레임 수: {frame_count}")
    with tqdm(total=frame_count, desc=f"🔄 {filename} 비식별화 진행중", ncols=110) as pbar:
        for idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=0.25, device=device, verbose=False)
            rois = []

            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    total_roi_count += 1
                    try:
                        key, iv = os.urandom(32), os.urandom(12)
                        aes = AESGCM(key)
                        success, roi_bytes = cv2.imencode('.png', roi)
                        if not success:
                            continue
                        encrypted = aes.encrypt(iv, roi_bytes.tobytes(), None)
                        total_encrypted += 1

                        # ✅ 모자이크 처리
                        small = cv2.resize(roi, (10, 10))
                        mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                        frame[y1:y2, x1:x2] = mosaic

                        rois.append({
                            "uuid": str(uuid.uuid4()),
                            "bbox": [x1, y1, x2, y2],
                            "key": base64.b64encode(key).decode(),
                            "iv": base64.b64encode(iv).decode(),
                            "encrypted_roi": base64.b64encode(encrypted).decode()
                        })
                    except Exception as e:
                        print(f"⚠️ ROI 암호화 실패 (frame {idx}): {e}")
                        continue

            if rois:
                total_detected_frames += 1
                roi_log["frames"][f"frame_{idx:05d}"] = rois

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()

    # ✅ JSON 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(roi_log, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 비식별화 완료: {output_path}")
    print(f"📊 ROI 프레임 {total_detected_frames} / 전체 {frame_count} / ROI {total_roi_count} / 암호화 성공 {total_encrypted}")

    # ✅ 오디오 병합
    output_final_path = merge_audio_with_video(video_path, output_path)
    print(f"🎬 최종 영상 저장 완료 → {output_final_path}")

    return output_final_path, json_path


# ============================================
# ✅ 메인 실행부
# ============================================
def main():
    print("===========================================")
    print("     🧩 Face ROI 비식별화 프로그램 (GPU 지원)     ")
    print("===========================================\n")

    # 🎯 비식별화할 영상 파일 경로
    video_path = r"C:\TeamGit\agi_dev\deidentification\data\example.mp4"

    try:
        anonymized_video, roi_json = anonymize_video(video_path)
        print("\n✅ 전체 처리 완료!")
        print(f"📁 비식별화 영상: {anonymized_video}")
        print(f"📄 ROI JSON 파일: {roi_json}")
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
