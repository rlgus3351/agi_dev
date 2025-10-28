import os
import cv2
import json
import base64
import shutil
import subprocess
import numpy as np
from glob import glob
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ==================================================
# 🔊 FFmpeg / ffprobe 유틸
# ==================================================
def _has_audio_stream(path: str) -> bool:
    """ffprobe로 오디오 스트림 존재 여부 확인"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            path
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return False


def _merge_audio(restored_silent: str, src_with_audio: str, out_path: str) -> bool:
    """무음영상 + 원본 오디오 병합 (2가지 매핑 시도)"""
    common = ["-y", "-c:v", "copy", "-c:a", "aac", "-shortest", "-movflags", "+faststart"]
    attempts = [
        ["-i", restored_silent, "-i", src_with_audio, "-map", "0:v:0", "-map", "1:a:0"] + common + [out_path],
        ["-i", restored_silent, "-i", src_with_audio, "-map", "0:v:0", "-map", "1:a"] + common + [out_path],
    ]
    for args in attempts:
        proc = subprocess.run(["ffmpeg"] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and _has_audio_stream(out_path):
            print(f"🎵 오디오 병합 완료 → {out_path}")
            return True
    print(f"⚠️ 오디오 병합 실패")
    return False


def merge_audio_with_video(restored_silent: str, src_with_audio: str, out_dir: str) -> str:
    """복원 무음영상 + 입력영상 오디오 → 유음영상 생성"""
    if shutil.which("ffmpeg") is None:
        print("⚠️ ffmpeg 미설치: 오디오 병합 스킵")
        return restored_silent
    if not _has_audio_stream(src_with_audio):
        print("ℹ️ 입력 영상에 오디오 스트림이 없음 → 무음 유지")
        return restored_silent

    base = os.path.splitext(os.path.basename(restored_silent))[0]
    out_final = os.path.join(out_dir, base.replace("_restored", "_restored_final") + ".mp4")
    os.makedirs(out_dir, exist_ok=True)
    ok = _merge_audio(restored_silent, src_with_audio, out_final)
    return out_final if ok else restored_silent


# ==================================================
# 🔓 복호화 본체
# ==================================================
def decrypt_rois(json_path: str, video_path: str, output_path_silent: str):
    """익명화된 영상과 JSON을 이용해 복호화된 무음 영상을 생성"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON 파일 없음: {json_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ 입력 영상 없음: {video_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        roi_log = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ 영상 열기 실패: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path_silent, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"🎬 복호화 시작 → {os.path.basename(video_path)} ({width}x{height}, {fps}fps, 총 {frame_count}프레임)")

    with tqdm(total=frame_count, desc="🔓 복호화 진행중", ncols=110) as pbar:
        for frame_index in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            frame_key = f"frame_{frame_index:05d}"
            rois = roi_log.get("frames", {}).get(frame_key, [])
            for roi_info in rois:
                try:
                    if not roi_info.get("restorable", True):
                        continue
                    x1, y1, x2, y2 = roi_info["bbox"]
                    key = base64.b64decode(roi_info["key"])
                    iv = base64.b64decode(roi_info["iv"])
                    enc = base64.b64decode(roi_info["encrypted_roi"])

                    aes = AESGCM(key)
                    dec = aes.decrypt(iv, enc, None)
                    roi_img = cv2.imdecode(np.frombuffer(dec, np.uint8), cv2.IMREAD_COLOR)

                    if roi_img is not None:
                        x1c, y1c = max(0, x1), max(0, y1)
                        x2c, y2c = min(width, x2), min(height, y2)
                        if x2c > x1c and y2c > y1c:
                            roi_img = cv2.resize(roi_img, (x2c - x1c, y2c - y1c))
                            frame[y1c:y2c, x1c:x2c] = roi_img
                except Exception as e:
                    print(f"⚠️ frame {frame_index} ROI 복호화 실패: {e}")
            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"✅ 무음 복원 완료 → {output_path_silent}\n")


# ==================================================
# 🧩 배치 복호화 관리 (UUID 구조 지원)
# ==================================================
def batch_decrypt(base_dir: str, output_root: str):
    """
    data/<uuid>/ 구조를 기준으로, 동일한 구조로 output/restored/<uuid>/ 생성.
    각 폴더 내 JSON/MP4 쌍을 복호화 + 오디오 병합 처리.
    """
    os.makedirs(output_root, exist_ok=True)

    # 1️⃣ 환자 UUID 폴더 탐색
    patient_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not patient_dirs:
        print(f"❌ 환자 폴더가 없습니다: {base_dir}")
        return

    for patient_dir in patient_dirs:
        patient_uuid = os.path.basename(patient_dir)
        print(f"\n🧍 환자 처리 시작: {patient_uuid}")

        # 2️⃣ 동일한 구조로 output 폴더 생성
        output_dir = os.path.join(output_root, patient_uuid)
        os.makedirs(output_dir, exist_ok=True)

        # 3️⃣ 내부 JSON 목록 순회
        json_files = sorted(glob(os.path.join(patient_dir, "*.json")))
        if not json_files:
            print(f"⚠️ JSON 없음 → {patient_uuid} 스킵")
            continue

        for json_path in json_files:
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            seq = base_name.split("_")[-1]

            # 같은 시퀀스 번호의 영상 찾기
            mp4_candidates = glob(os.path.join(patient_dir, f"*_{seq}.mp4"))
            if not mp4_candidates:
                print(f"⚠️ {json_path}에 대응하는 mp4 없음")
                continue

            video_path = mp4_candidates[0]
            base_video_name = os.path.splitext(os.path.basename(video_path))[0]

            silent_out = os.path.join(output_dir, f"{base_video_name}_restored.mp4")
            final_out = os.path.join(output_dir, f"{base_video_name}_restored_final.mp4")

            print("===========================================")
            print(f"🧾 JSON : {os.path.basename(json_path)}")
            print(f"🎬 VIDEO: {os.path.basename(video_path)}")
            print(f"📂 OUTPUT: {output_dir}")
            print("===========================================\n")

            try:
                decrypt_rois(json_path, video_path, silent_out)
                merged = merge_audio_with_video(silent_out, video_path, output_dir)

                if merged != silent_out:
                    os.replace(merged, final_out)
                    print(f"✅ 최종 파일 생성 → {final_out}")
                else:
                    print(f"ℹ️ 오디오 병합 실패, 무음 버전 유지.")
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

        print(f"🏁 {patient_uuid} 처리 완료\n")


# ==================================================
# 🚀 실행부
# ==================================================
def main():
    print("===========================================")
    print("     🧩 Face ROI 복호화 배치 프로그램 (AES-GCM)     ")
    print("===========================================\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "data")                 # ✅ ./data → 절대경로
    output_root = os.path.join(script_dir, "output", "restored") # ✅ ./output/restored

    batch_decrypt(base_dir, output_root)


if __name__ == "__main__":
    main()
