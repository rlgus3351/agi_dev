from ultralytics import YOLO
import cv2, os, json, uuid, base64, hashlib, subprocess
import numpy as np
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import requests
from config import PROCESS_BASE_URL, NAS_URL, USERNAME, PASSWORD
from datetime import datetime
import urllib3
urllib3.disable_warnings()

# ✅ YOLO 모델 로드 (CPU 전용)
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "model", "model.pt")
device = "cpu"
model = YOLO(model_path)
print(f"✅ YOLO 모델 로드 완료 (device={device})")

# -------------------------------
# ✅ NAS 연결 설정
# -------------------------------
def nas_login():
    """NAS 로그인 후 SID 반환"""
    params = {
        "api": "SYNO.API.Auth",
        "version": "7",
        "method": "login",
        "account": USERNAME,
        "passwd": PASSWORD,
        "session": "FileStation",
        "format": "sid"
    }
    r = requests.get(f"{NAS_URL}/webapi/entry.cgi", params=params, verify=False)
    sid = r.json()["data"]["sid"]
    print(f"🔑 NAS 로그인 성공 (SID={sid})")
    return sid


def upload_to_nas(sid, local_path: str, nas_folder="/mAGI/CNU_Data/VIDEO"):
    """NAS에 파일 업로드 (자동 폴더 생성 포함)"""
    url = f"{NAS_URL}/webapi/entry.cgi"
    filename = os.path.basename(local_path)

    params = {
        "api": "SYNO.FileStation.Upload",
        "version": "2",
        "method": "upload",
        "_sid": sid
    }
    data = {
        "path": nas_folder,
        "create_parents": "true",
        "overwrite": "true"
    }
    files = {
        "file": (filename, open(local_path, "rb"), "application/octet-stream")
    }

    print(f"📤 NAS 업로드 중... {filename}")
    r = requests.post(url, params=params, data=data, files=files, verify=False)
    try:
        r.raise_for_status()
        print(f"✅ NAS 업로드 완료 → {nas_folder}/{filename}")
        print(r.json())
    except Exception as e:
        print(f"❌ NAS 업로드 실패: {e}")
        print(r.text)


def nas_logout(sid):
    """NAS 로그아웃"""
    params = {
        "api": "SYNO.API.Auth",
        "version": "7",
        "method": "logout",
        "_sid": sid
    }
    requests.get(f"{NAS_URL}/webapi/entry.cgi", params=params, verify=False)
    print("👋 NAS 로그아웃 완료")


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
# ✅ 오디오 병합 함수
# ----------------------------------------
def merge_audio_with_video(original_video, anonymized_video):
    """
    원본 영상의 오디오를 익명화된 영상에 병합
    """
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


# ----------------------------------------
# ✅ 상태 업데이트 API
# ----------------------------------------
def update_anonymization_status(video_id: int):
    try:
        url = f"{PROCESS_BASE_URL}update"
        payload = {
            "video_metadata_id": video_id,
            "is_anonymized": True,
            "anonymized_ts": datetime.utcnow().isoformat()
        }
        res = requests.put(url, json=payload)
        res.raise_for_status()
        print(f"✅ is_anonymized 업데이트 완료 (video_id={video_id})")
    except Exception as e:
        print(f"❌ 서버 업데이트 실패: {e}")


# ----------------------------------------
# ✅ 실패 로그 저장
# ----------------------------------------
def log_failure(video_id_or_path: str, error_msg: str):
    os.makedirs("logs", exist_ok=True)
    with open("logs/error_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[FAIL] {video_id_or_path} → {error_msg}\n")


# ----------------------------------------
# ✅ 비식별화 메인 함수
# ----------------------------------------
def process_video(meta: dict):
    input_path = meta["file_path"]
    filename = os.path.basename(input_path)
    filename_wo_ext = os.path.splitext(filename)[0]
    video_id = meta.get("video_metadata_id")

    os.makedirs("data/output", exist_ok=True)
    os.makedirs("data/json", exist_ok=True)

    output_path = f"data/output/{filename_wo_ext}_anonymized.mp4"
    json_path = f"data/json/{filename_wo_ext}_rois.json"

    start_time = datetime.utcnow()

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ 비디오를 열 수 없습니다: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(3)), int(cap.get(4))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        roi_log = {"video_info": {"fps": fps, "frame_count": frame_count, "sha256": sha256_file(input_path)}, "frames": {}}
        total_detected_frames = total_roi_count = total_encrypted = 0

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
                            if not success: continue
                            encrypted = aes.encrypt(iv, roi_bytes.tobytes(), None)
                            total_encrypted += 1
                            small = cv2.resize(roi, (10, 10))
                            mosaic = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
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
        end_time = datetime.utcnow()

        # ✅ 오디오 병합
        output_final_path = merge_audio_with_video(input_path, output_path)

        # ✅ ROI JSON 저장
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(roi_log, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 완료: {output_final_path}")
        print(f"📊 ROI 프레임 {total_detected_frames} / 전체 {frame_count} / ROI {total_roi_count} / 암호화 성공 {total_encrypted}")

        if video_id:
            update_anonymization_status(video_id)

        # ✅ NAS 업로드
        sid = None
        try:
            sid = nas_login()
            upload_to_nas(sid, output_final_path, nas_folder="/mAGI/CNU_Data/VIDEO")
            upload_to_nas(sid, json_path, nas_folder="/mAGI/CNU_Data/VIDEO/Json")
        except Exception as e:
            print(f"⚠️ NAS 업로드 중 오류 발생: {e}")
        finally:
            if sid:
                nas_logout(sid)

        # ✅ NAS 경로 기반 DB 저장
        nas_video_path = f"/mAGI/CNU_Data/VIDEO/{os.path.basename(output_final_path)}"
        nas_json_path = f"/mAGI/CNU_Data/VIDEO/Json/{os.path.basename(json_path)}"

        save_preprocessing_result(
            meta, nas_video_path, nas_json_path,
            total_frames=frame_count,
            roi_frames=total_detected_frames,
            roi_count=total_roi_count,
            encrypted_count=total_encrypted,
            start_time=start_time,
            end_time=end_time
        )

        return output_final_path, json_path

    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        log_failure(video_id or input_path, str(e))
        raise


# ----------------------------------------
# ✅ 전처리 결과 저장 API 호출
# ----------------------------------------
def save_preprocessing_result(meta: dict, output_path: str, json_path: str,
                              total_frames: int, roi_frames: int,
                              roi_count: int, encrypted_count: int,
                              start_time: datetime, end_time: datetime):
    try:
        url = f"{PROCESS_BASE_URL}"
        duration_sec = (end_time - start_time).total_seconds()
        success_rate = (encrypted_count / roi_count * 100) if roi_count else 0

        payload = {
            "item_id": meta.get("item_id"),
            "data_category": "video",
            "original_file_path": meta.get("file_path"),
            "json_file_path": json_path,
            "encrypted_file_path": output_path,
            "processing_started_at": start_time.isoformat(),
            "processing_ended_at": end_time.isoformat(),
            "processing_duration_sec": round(duration_sec, 2),
            "total_frames": total_frames,
            "detected_face_frames": roi_frames,
            "encrypted_frames": encrypted_count,
            "success_rate": round(success_rate, 2),
            "preprocessing_type": "face_anonymization + AES-GCM + audio_merge",
            "description": f"총 ROI={roi_count}, 암호화 성공={encrypted_count}"
        }

        res = requests.post(url, json=payload)
        res.raise_for_status()
        print(f"✅ 전처리 결과 DB 저장 완료 (item_id={meta.get('item_id')})")

    except Exception as e:
        print(f"❌ 전처리 결과 저장 실패: {e}")
