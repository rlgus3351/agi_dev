import os
import cv2
import json
import base64
import shutil
import subprocess
import numpy as np
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# =========================
# 🔊 FFmpeg/ffprobe 유틸
# =========================
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
    """
    restored_silent의 비디오 + src_with_audio의 오디오 병합.
    두 번 시도:
      1) 0:v:0 + 1:a:0
      2) 0:v:0 + 1:a (첫 오디오 자동 선택)
    """
    common = ["-y", "-c:v", "copy", "-c:a", "aac", "-shortest", "-movflags", "+faststart"]
    attempts = [
        ["-i", restored_silent, "-i", src_with_audio, "-map", "0:v:0", "-map", "1:a:0"] + common + [out_path],
        ["-i", restored_silent, "-i", src_with_audio, "-map", "0:v:0", "-map", "1:a"]   + common + [out_path],
    ]
    last_err = ""
    for args in attempts:
        proc = subprocess.run(["ffmpeg"] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            # 최종 확인: 진짜 오디오가 붙었는지
            if _has_audio_stream(out_path):
                print(f"🎵 오디오 병합 완료 → {out_path}")
                return True
        last_err = proc.stderr
    print(f"⚠️ 오디오 병합 실패:\n{last_err}")
    return False

def merge_audio_with_video(restored_silent: str, src_with_audio: str, out_dir: str) -> str:
    """복원 무음영상 + 입력영상 오디오 → 최종 유음영상 경로 반환(실패 시 무음 그대로 반환)"""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("⚠️ ffmpeg/ffprobe 미설치: 오디오 병합을 건너뜁니다.")
        return restored_silent
    if not _has_audio_stream(src_with_audio):
        print("ℹ️ 입력 영상에 오디오 스트림이 없습니다. 무음 영상으로 유지됩니다.")
        return restored_silent

    base = os.path.splitext(os.path.basename(restored_silent))[0]
    out_final = os.path.join(out_dir, base.replace("_restored", "_restored_final") + ".mp4")
    os.makedirs(out_dir, exist_ok=True)
    ok = _merge_audio(restored_silent, src_with_audio, out_final)
    return out_final if ok else restored_silent


# =========================
# 🔓 복호화 본체
# =========================
def decrypt_rois(json_path: str, video_path: str, output_path_silent: str):
    """익명화된 영상과 JSON을 이용해 복호화된 '무음' 영상을 생성"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ 비식별화된(입력) 영상을 찾을 수 없습니다: {video_path}")

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
    out = cv2.VideoWriter(output_path_silent, fourcc, fps, (width, height))

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
                        # bbox가 프레임 경계를 벗어나면 안전하게 클리핑
                        x1c = max(0, min(width,  x1))
                        x2c = max(0, min(width,  x2))
                        y1c = max(0, min(height, y1))
                        y2c = max(0, min(height, y2))
                        if x2c > x1c and y2c > y1c:
                            target_w, target_h = x2c - x1c, y2c - y1c
                            if roi_img.shape[1] != target_w or roi_img.shape[0] != target_h:
                                roi_img = cv2.resize(roi_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                            frame[y1c:y2c, x1c:x2c] = roi_img
                except Exception as e:
                    print(f"⚠️ frame {frame_index} ROI 복호화 실패: {e}")

            out.write(frame)
            frame_index += 1
            pbar.update(1)

    cap.release()
    out.release()
    print(f"\n✅ 복호화(무음) 완료 → {output_path_silent}")
    print(f"📊 총 프레임: {frame_count} | 복원된 프레임: {frame_index}")


def main():
    print("===========================================")
    print("     🧩 Face ROI 복호화 프로그램 (AES-GCM)     ")
    print("===========================================\n")

    # 🔧 경로 설정
    json_path  = r"C:\TeamGit\agi_dev\CHU_LOCAL\data_anonymization_app\data\json\ID-10004_2.json"
    video_path = r"C:\Users\user\Desktop\DEV_AGI\parkinson\output\video\adb687ef-854c-4b8d-ab46-e97ff5cafc31_2_final.mp4"

    # 출력 디렉터리 및 파일명
    output_dir = os.path.join("output", "restored")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    silent_out = os.path.join(output_dir, f"{base_name}_restored.mp4")         # 무음
    final_out  = os.path.join(output_dir, f"{base_name}_restored_final.mp4")   # 유음(목표)

    # 1) 무음 복원
    decrypt_rois(json_path, video_path, silent_out)

    # 2) 오디오 병합 (입력 영상의 오디오 사용)
    merged = merge_audio_with_video(silent_out, video_path, output_dir)
    if merged != silent_out:
        # 성공 시 원하는 최종 파일명으로 정리
        if merged != final_out:
            try:
                if os.path.exists(final_out):
                    os.remove(final_out)
                os.replace(merged, final_out)
                print(f"✅ 최종 파일 정리 → {final_out}")
            except Exception as e:
                print(f"⚠️ 최종 파일명 정리 실패: {e}")
        else:
            print(f"✅ 최종 파일 생성 → {final_out}")
    else:
        print("ℹ️ 오디오 병합이 스킵되거나 실패하여 무음 복원 파일만 생성되었습니다.")

if __name__ == "__main__":
    main()
