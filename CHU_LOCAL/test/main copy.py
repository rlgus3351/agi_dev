from ultralytics import YOLO
import torch
import cv2, os, json, hashlib, subprocess, time, warnings, base64, uuid
import numpy as np
from tqdm import tqdm
from datetime import datetime
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import urllib3
urllib3.disable_warnings()
warnings.filterwarnings("ignore", category=FutureWarning)

# ✅ GPU / CPU 자동 감지
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 YOLO 모델 로드 중... (device={device})")

# ✅ 기본 경로
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model", "model.pt")

# ✅ YOLO 모델 로드
model = YOLO(model_path)
model.to(device)
print(f"✅ YOLO 모델 로드 완료 (device={device})")

# ----------------------------------------
# ✅ SHA256 해시 계산
# ----------------------------------------
def sha256_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()

# ----------------------------------------
# ✅ 오디오 병합 함수 (원본 오디오 → 익명화 영상에 이식)
# ----------------------------------------
def merge_audio_with_video(original_video, anonymized_video):
    output_with_audio = anonymized_video.replace("_anonymized.mp4", "_final.mp4")
    # 원본 오디오 트랙 그대로 복사(+sync 자동 맞춤)
    cmd = f'ffmpeg -y -i "{anonymized_video}" -i "{original_video}" ' \
          f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a copy -shortest "{output_with_audio}"'
    try:
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        if process.returncode == 0:
            print(f"🎵 오디오 포함 영상 생성 완료 → {output_with_audio}")
        else:
            err_msg = process.stderr.decode(errors='ignore')
            print(f"⚠️ 오디오 병합 실패:\n{err_msg}")
    except Exception as e:
        print(f"❌ ffmpeg 실행 실패: {e}")
    return output_with_audio

# ----------------------------------------
# ✅ ROI 암호화 유틸 (PNG 무손실 인코딩 → AES-GCM)
# ----------------------------------------
def encrypt_roi_png(roi_bgr: np.ndarray) -> dict:
    """
    return: { key, iv, encrypted_roi } (모두 base64 str)
    """
    # PNG로 무손실 인코딩 (복원 정확성)
    ok, roi_bytes = cv2.imencode('.png', roi_bgr)
    if not ok:
        raise ValueError("ROI PNG 인코딩 실패")
    key = os.urandom(32)   # AES-256
    iv = os.urandom(12)    # GCM 표준
    aes = AESGCM(key)
    ct = aes.encrypt(iv, roi_bytes.tobytes(), None)
    return {
        "key": base64.b64encode(key).decode(),
        "iv": base64.b64encode(iv).decode(),
        "encrypted_roi": base64.b64encode(ct).decode()
    }

# ----------------------------------------
# ✅ GPU Batch 기반 비식별화 (+ ROI 암호화 JSON 기록)
# ----------------------------------------
def process_video(input_path: str, batch_size: int = 8, conf_thres: float = 0.25, imgsz: int = 640):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {input_path}")

    filename = os.path.basename(input_path)
    filename_wo_ext = os.path.splitext(filename)[0]

    # ✅ 절대 경로 기반 output/json 폴더
    output_dir = os.path.join(base_dir, "data", "output")
    json_dir = os.path.join(base_dir, "data", "json")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{filename_wo_ext}_anonymized.mp4")
    json_path = os.path.join(json_dir, f"{filename_wo_ext}_rois.json")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ 비디오를 열 수 없습니다: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    roi_log = {
        "video_info": {
            "fps": fps,
            "frame_count": frame_count,
            "sha256": sha256_file(input_path),
            "device": device,
            "model": os.path.basename(model_path),
            "created_at": datetime.utcnow().isoformat() + "Z"
        },
        "roi_frames": [],   # ✅ ROI 존재 프레임 인덱스 캐시 (복원 속도 ↑)
        "frames": {}
    }

    total_detected_frames = 0
    total_roi_count = 0
    start_time = time.time()

    print(f"🚀 GPU Batch 비식별화 시작 (frames={frame_count}, batch={batch_size}, device={device}, conf={conf_thres})")

    frame_buffer, frame_indices = [], []
    with tqdm(total=frame_count, desc="🔄 Processing", ncols=110, dynamic_ncols=True) as pbar:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame)
            frame_indices.append(idx)

            if len(frame_buffer) >= batch_size or idx == frame_count - 1:
                results = model.predict(
                    source=frame_buffer,
                    conf=conf_thres,
                    device=device,
                    verbose=False,
                    imgsz=imgsz,
                    half=(device == "cuda")
                )

                for fi, (frame_bgr, res) in enumerate(zip(frame_buffer, results)):
                    rois = []
                    if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                        for box in res.boxes:
                            # 신뢰도 확인
                            conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                            if conf < conf_thres:
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # 안전 클램프
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)
                            if x2 <= x1 or y2 <= y1:
                                continue

                            roi = frame_bgr[y1:y2, x1:x2]
                            if roi.size == 0:
                                continue

                            # ✅ ROI 암호화 데이터 생성 (원본 없이 복원 가능)
                            try:
                                enc = encrypt_roi_png(roi)
                            except Exception as e:
                                # 암호화 실패시 복원 불가로만 기록
                                enc = {"key": None, "iv": None, "encrypted_roi": None}

                            total_roi_count += 1

                            # ✅ 모자이크 적용 (비식별)
                            roi_w, roi_h = x2 - x1, y2 - y1
                            small = cv2.resize(roi, (max(1, roi_w // 20), max(1, roi_h // 20)))
                            mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                            frame_bgr[y1:y2, x1:x2] = mosaic

                            rois.append({
                                "uuid": str(uuid.uuid4()),
                                "bbox": [x1, y1, x2, y2],
                                "conf": round(conf, 4),
                                "restorable": (enc["key"] is not None),
                                "key": enc["key"],
                                "iv": enc["iv"],
                                "encrypted_roi": enc["encrypted_roi"]
                            })

                    # ✅ ROI가 있었던 프레임만 기록 + 캐시에 인덱스 저장
                    if rois:
                        total_detected_frames += 1
                        fidx = frame_indices[fi]
                        roi_log["frames"][f"frame_{fidx:05d}"] = rois
                        roi_log["roi_frames"].append(fidx)

                    out.write(frame_bgr)
                    pbar.update(1)

                frame_buffer.clear()
                frame_indices.clear()

            idx += 1

    cap.release()
    out.release()

    total_time = time.time() - start_time
    # ✅ 오디오 병합 (원본 오디오 → 익명화 영상에 이식)
    output_final_path = merge_audio_with_video(input_path, output_path)

    # ✅ JSON 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(roi_log, f, indent=2, ensure_ascii=False)

    print(f"\n✅ GPU 비식별화 완료!")
    print(f"📊 ROI 프레임: {total_detected_frames}/{frame_count}")
    print(f"📈 총 ROI 수: {total_roi_count}")
    print(f"🕒 처리시간: {total_time:.2f}초 / 평균 {(frame_count/total_time):.2f} FPS")
    print(f"📁 결과 파일(비식별): {output_path}")
    print(f"📁 결과 파일(오디오 병합): {output_final_path}")
    print(f"📄 ROI JSON: {json_path}")

# ----------------------------------------
# ✅ 단독 실행
# ----------------------------------------
if __name__ == "__main__":
    sample_path = r"C:\Users\user\Desktop\sample\sample_4.mov"
    # 필요시 conf_thres=0.30~0.35로 올리면 오탐 줄어듦, imgsz=768~960으로 올리면 탐지↑(속도↓)
    process_video(sample_path, batch_size=8, conf_thres=0.25, imgsz=640)
