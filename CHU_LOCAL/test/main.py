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

# ------------------------------------------------------------
# ✅ 1. GPU / CPU 자동 감지
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 YOLO 모델 로드 중... (device={device})")

# ------------------------------------------------------------
# ✅ 2. 기본 경로 설정
# ------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model", "model.pt")

# ------------------------------------------------------------
# ✅ 3. YOLO 모델 로드
# ------------------------------------------------------------
model = YOLO(model_path)
model.to(device)
print(f"✅ YOLO 모델 로드 완료 (device={device})")

# ------------------------------------------------------------
# ✅ SHA256 해시 계산
# ------------------------------------------------------------
def sha256_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()

# ------------------------------------------------------------
# ✅ 오디오 병합 (원본 오디오 → 익명화 영상에 추가)
# ------------------------------------------------------------
def merge_audio_with_video(original_video, anonymized_video):
    output_with_audio = anonymized_video.replace("_anonymized.mp4", "_final.mp4")
    cmd = f'ffmpeg -y -i "{anonymized_video}" -i "{original_video}" ' \
          f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a copy -shortest "{output_with_audio}"'
    try:
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        if process.returncode == 0:
            print(f"🎵 오디오 포함 영상 생성 완료 → {output_with_audio}")
        else:
            print(f"⚠️ 오디오 병합 실패:\n{process.stderr.decode(errors='ignore')}")
    except Exception as e:
        print(f"❌ ffmpeg 실행 실패: {e}")
    return output_with_audio

# ------------------------------------------------------------
# ✅ ROI 암호화 유틸 (PNG 무손실 인코딩 → AES-GCM)
# ------------------------------------------------------------
def encrypt_roi_png(roi_bgr: np.ndarray) -> dict:
    ok, roi_bytes = cv2.imencode('.png', roi_bgr)
    if not ok:
        raise ValueError("ROI PNG 인코딩 실패")
    key = os.urandom(32)  # 256-bit key
    iv = os.urandom(12)   # 96-bit IV
    aes = AESGCM(key)
    ct = aes.encrypt(iv, roi_bytes.tobytes(), None)
    return {
        "key": base64.b64encode(key).decode(),
        "iv": base64.b64encode(iv).decode(),
        "encrypted_roi": base64.b64encode(ct).decode()
    }

# ------------------------------------------------------------
# ✅ GPU Batch 기반 비식별화 (ROI 암호화 + 모자이크)
# ------------------------------------------------------------
def process_video(input_path: str, batch_size: int = 8, conf_thres: float = 0.25, imgsz: int = 640, roi_expand: float = 0.25):
    """
    roi_expand: ROI 확장 비율 (0.25 = 25% 확장)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {input_path}")

    filename = os.path.basename(input_path)
    filename_wo_ext = os.path.splitext(filename)[0]

    # ✅ 출력 폴더
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

    # ✅ JSON 구조 개선
    roi_log = {
        "meta": {
            "video_name": filename,
            "resolution": [width, height],
            "fps": fps,
            "frame_count": frame_count,
            "roi_expand": roi_expand,
            "model": os.path.basename(model_path),
            "device": device,
            "sha256": sha256_file(input_path),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "encryption": {"method": "AES-GCM", "key_length": 256},
            "version": "1.0.0"
        },
        "frames": {},
        "summary": {"total_frames": frame_count, "roi_frames": 0, "total_rois": 0}
    }

    total_detected_frames = 0
    total_roi_count = 0
    start_time = time.time()

    print(f"🚀 GPU Batch 비식별화 시작 (frames={frame_count}, batch={batch_size}, device={device})")
    print(f"🔍 ROI 확장 비율: {roi_expand*100:.0f}%")

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
                    if hasattr(res, "boxes") and res.boxes is not None:
                        for box in res.boxes:
                            conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                            if conf < conf_thres:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # ✅ ROI 확장
                            w, h = x2 - x1, y2 - y1
                            pad_w = int(w * roi_expand)
                            pad_h = int(h * roi_expand)
                            x1 = max(0, x1 - pad_w)
                            y1 = max(0, y1 - pad_h)
                            x2 = min(width, x2 + pad_w)
                            y2 = min(height, y2 + pad_h)
                            if x2 <= x1 or y2 <= y1:
                                continue

                            roi = frame_bgr[y1:y2, x1:x2]
                            if roi.size == 0:
                                continue

                            # ✅ ROI 해시 계산
                            ok, roi_bytes = cv2.imencode('.png', roi)
                            if not ok:
                                continue
                            roi_hash = hashlib.sha256(roi_bytes).hexdigest()

                            # ✅ ROI 암호화
                            try:
                                enc = encrypt_roi_png(roi)
                            except Exception as e:
                                print(f"⚠️ ROI 암호화 실패(frame {idx}): {e}")
                                enc = {"key": None, "iv": None, "encrypted_roi": None}

                            total_roi_count += 1

                            # ✅ 모자이크 적용
                            roi_w, roi_h = x2 - x1, y2 - y1
                            small = cv2.resize(roi, (max(1, roi_w // 20), max(1, roi_h // 20)))
                            mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                            frame_bgr[y1:y2, x1:x2] = mosaic

                            rois.append({
                                "uuid": str(uuid.uuid4()),
                                "bbox": [x1, y1, x2, y2],
                                "roi_size": [roi_w, roi_h],
                                "conf": round(conf, 4),
                                "roi_hash": roi_hash,
                                "restorable": (enc["key"] is not None),
                                "key": enc["key"],
                                "iv": enc["iv"],
                                "encrypted_roi": enc["encrypted_roi"]
                            })

                    if rois:
                        total_detected_frames += 1
                        fidx = frame_indices[fi]
                        roi_log["frames"][f"frame_{fidx:05d}"] = rois

                    out.write(frame_bgr)
                    pbar.update(1)

                frame_buffer.clear()
                frame_indices.clear()
            idx += 1

    cap.release()
    out.release()

    total_time = time.time() - start_time
    roi_log["summary"]["roi_frames"] = total_detected_frames
    roi_log["summary"]["total_rois"] = total_roi_count

    # ✅ JSON 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(roi_log, f, indent=2, ensure_ascii=False)

    # ✅ 오디오 병합
    output_final_path = merge_audio_with_video(input_path, output_path)

    print(f"\n✅ GPU 비식별화 완료!")
    print(f"📊 ROI 프레임: {total_detected_frames}/{frame_count}")
    print(f"📈 총 ROI 수: {total_roi_count}")
    print(f"🕒 처리시간: {total_time:.2f}초 / 평균 {(frame_count/total_time):.2f} FPS")
    print(f"📁 결과 파일(비식별): {output_path}")
    print(f"📁 결과 파일(오디오 병합): {output_final_path}")
    print(f"📄 ROI JSON: {json_path}")

# ------------------------------------------------------------
# ✅ 단독 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    sample_path = r"C:\Users\user\Desktop\sample\sample_2.mp4"
    process_video(sample_path, batch_size=8, conf_thres=0.25, imgsz=640, roi_expand=0.25)
    # process_video(sample_path, batch_size=8, conf_thres=0.20, imgsz=640, roi_expand=0.30)