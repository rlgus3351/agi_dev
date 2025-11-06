# run_worker.py

import os
import shutil
import time
from datetime import datetime
from typing import Tuple, Optional
from processor import process_video

from api_local.processing_api_local import (
    get_next_media_to_process,
    update_processed_video,
    update_processed_audio,
    insert_preprocessing_record,
    make_preproc_payload_video,
    make_preproc_payload_audio,
)

# test_envelope 재사용 (없으면 안내)
try:
    from test_envelope import (
        encrypt_with_envelope,
        modulate_audio,
        MASTER_PASSWORD as AUDIO_MASTER_PASSWORD,
        SHIFT_SEMITONES as AUDIO_SHIFT_SEMITONES,
    )
except Exception as _e:
    encrypt_with_envelope = modulate_audio = None
    AUDIO_MASTER_PASSWORD = "voicecrypto_master"
    AUDIO_SHIFT_SEMITONES = 4
    print(f"[WARN] test_envelope import 실패: {_e} (AUDIO 처리 시 오류가 날 수 있음)")

# 🔧 출력 루트: OUTPUT_BASE 우선, 없으면 BASE_OUTPUT_DIR 폴백
try:
    from config import OUTPUT_BASE  # 권장: C:\...\MDD\output
except Exception:
    try:
        from config import BASE_OUTPUT_DIR as OUTPUT_BASE
    except Exception:
        OUTPUT_BASE = os.path.abspath(os.path.join(os.getcwd(), "output"))

# 미사용: LOCAL_JSON_DIR (필요시 아래 구조에 맞게 사용)
SLEEP_SECONDS_WHEN_IDLE = 60
DATA_CATEGORY = "MDD"
AUDIO_JSON_LOG = None              # .enc 메타에 포함되므로 별도 json 없음
AUDIO_ENCRYPTED_SUFFIX = ".enc"
AUDIO_MASK_SUFFIX = "_mod.wav"


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------
# 경로 유틸
# ---------------------------
def _patient_subdir(patient_id, media_kind: str) -> str:
    """OUTPUT_BASE/<patient_uuid>/<media_kind>/ 디렉토리 생성 후 반환"""
    d = os.path.join(OUTPUT_BASE, str(patient_id), media_kind.lower())
    os.makedirs(d, exist_ok=True)
    return d


def _mv_into_dir(src: Optional[str], dest_dir: str) -> Optional[str]:
    """src 파일을 dest_dir로 이동(이름 유지). 반환: 이동 후 경로"""
    if not src:
        return src
    os.makedirs(dest_dir, exist_ok=True)
    dst = os.path.join(dest_dir, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dst):
        try:
            shutil.move(src, dst)
        except Exception:
            # 다른 파티션 가능성 → copy2 → remove 폴백
            shutil.copy2(src, dst)
            try:
                os.remove(src)
            except Exception:
                pass
    return dst


def _audio_out_paths(dest_dir: str, src_path: str) -> Tuple[str, str]:
    """환자/audio 폴더에 생성할 .enc / _mod.wav 경로 구성"""
    base = os.path.splitext(os.path.basename(src_path))[0]
    enc_out = os.path.join(dest_dir, base + AUDIO_ENCRYPTED_SUFFIX)
    mod_out = os.path.join(dest_dir, base + AUDIO_MASK_SUFFIX)
    return enc_out, mod_out


def _ensure_wav_input(src: str) -> str:
    """
    wave 모듈로 읽을 수 있도록, 비-WAV이면 ffmpeg로 임시 wav 변환.
    반환: WAV 경로 (원본이 wav면 원본 경로 그대로)
    """
    root, ext = os.path.splitext(src)
    if ext.lower() == ".wav":
        return src

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"ffmpeg 없음: 비-WAV 입력을 처리할 수 없습니다 ({src})")

    tmp_wav = root + ".__tmp__.wav"
    cmd = f'ffmpeg -y -i "{src}" -acodec pcm_s16le -ar 16000 "{tmp_wav}"'
    ret = os.system(cmd)
    if ret != 0 or not os.path.exists(tmp_wav):
        raise RuntimeError(f"ffmpeg wav 변환 실패: {src}")
    return tmp_wav


# ---------------------------
# AUDIO 처리
# ---------------------------
def _process_audio_local_to_dir(file_path: str, dest_dir: str) -> Tuple[str, str]:
    """
    test_envelope 기반으로 환자 폴더에 직접 출력 생성
    반환: (enc_out, mod_out)
    """
    if encrypt_with_envelope is None or modulate_audio is None:
        raise RuntimeError("test_envelope 모듈 로드 실패: AUDIO 처리 불가")

    import wave
    os.makedirs(dest_dir, exist_ok=True)

    enc_out, mod_out = _audio_out_paths(dest_dir, file_path)

    # 비-WAV 입력 대응
    src_for_wave = _ensure_wav_input(file_path)

    with wave.open(src_for_wave, 'rb') as w:
        params = {
            "nchannels": w.getnchannels(),
            "sampwidth": w.getsampwidth(),
            "framerate": w.getframerate(),
            "nframes": w.getnframes(),
            "comptype": w.getcomptype(),
            "compname": w.getcompname(),
        }
        frames = w.readframes(w.getnframes())

    # 임시 wav 정리
    if src_for_wave != file_path:
        try:
            os.remove(src_for_wave)
        except Exception:
            pass

    encrypt_with_envelope(frames, params, enc_out, AUDIO_MASTER_PASSWORD)
    modulate_audio(frames, params, mod_out, AUDIO_SHIFT_SEMITONES)
    return enc_out, mod_out


# ---------------------------
# 메인 파이프라인
# ---------------------------
def run_processing_pipeline() -> bool:
    job = None
    try:
        job = get_next_media_to_process()
    except Exception as e:
        print(f"[{now_str()}] ⚠️ 대기 작업 조회 중 오류: {e}")

    if not job:
        print(f"[{now_str()}] ✅ 처리할 작업이 없습니다.\n")
        return False

    kind = (job.get("media_kind") or "").upper()    # 'VIDEO' or 'AUDIO'
    dtype = (job.get("data_type") or "").upper()    # 'MOBILE'|'WEBCAM'|'VOICE'
    item_id = job.get("item_id")
    patient_id = job.get("patient_id")
    src_path = job.get("file_path")
    meta_cts = job.get("meta_created_ts")

    # 메타데이터 id 키 표준화 (반환 함수의 키 이름 차이 대비)
    video_meta_id = job.get("video_metadata_id") or job.get("metadata_id")
    audio_meta_id = job.get("audio_metadata_id") or job.get("metadata_id")

    print(f"[{now_str()}] 🚀 처리 시작 | kind={kind} type={dtype} item_id={item_id}")
    print(f"   • file_path={src_path}")
    print(f"   • meta_created_ts={meta_cts}")
    print("-" * 60)

    started_at = datetime.now()

    try:
        if kind == "VIDEO":
            # === VIDEO ===
            # 1) 처리 (processor는 기존대로 실행)
            output_path, json_path = process_video({
                "item_id": item_id,
                "video_metadata_id": video_meta_id,
                "file_path": src_path,
            })

            # 2) 환자/video 폴더로 이동 정리
            vdir = _patient_subdir(patient_id, "video")
            output_path = _mv_into_dir(output_path, vdir)
            if json_path:
                json_path = _mv_into_dir(json_path, vdir)

            ended_at = datetime.now()

            # payload = make_preproc_payload_video(
            #     item_id=item_id,
            #     data_category=DATA_CATEGORY,
            #     original_file_path=src_path,
            #     json_file_path=json_path,
            #     encrypted_file_path=output_path,
            #     started_at=started_at,
            #     ended_at=ended_at,
            #     duration_sec=(ended_at - started_at).total_seconds(),
            #     total_frames=None,
            #     encrypted_frames=None,
            #     detected_face_frames=None,
            #     success_rate=100.0,
            #     description=f"Video anonymized from {dtype}",
            # )
            # insert_preprocessing_record(payload)

            # DB 상태 업데이트는 datetime으로 전달(드라이버가 타임스탬프 처리)
            # update_processed_video(video_meta_id, anonymized_ts=ended_at)

            print(f"[{now_str()}] ✅ VIDEO 처리 완료 → {output_path}")
            if json_path:
                print(f"[{now_str()}] 📝 ROI JSON → {json_path}\n")

        elif kind == "AUDIO":
            adir = _patient_subdir(patient_id, "audio")
            enc_out, mod_out = _process_audio_local_to_dir(src_path, adir)
            ended_at = datetime.now()
        
            payload = make_preproc_payload_audio(
                item_id=item_id,
                data_category=DATA_CATEGORY,           # 예: "MDD" 또는 "PD" 프로젝트 규칙대로
                original_file_path=src_path,
                json_file_path=AUDIO_JSON_LOG,         # None OK
                encrypted_file_path=enc_out,
                started_at=started_at,
                ended_at=ended_at,
                duration_sec=(ended_at - started_at).total_seconds(),
                success_rate=100.0,
                description=f"Audio anonymized (+{AUDIO_SHIFT_SEMITONES} semitones)",
            )
        
            ok, new_id = insert_preprocessing_record(payload)
            if not ok:
                print(f"[{now_str()}] ❌ AUDIO DB insert 실패. payload=\n{payload}")
            else:
                print(f"[{now_str()}] ✅ AUDIO DB insert 성공 (preprocessing_id={new_id})")
        
            # 상태 업데이트도 실제 적용
            try:
                update_processed_audio(audio_meta_id, anonymized_ts=ended_at)
            except Exception as e:
                print(f"[{now_str()}] ⚠️ update_processed_audio 실패: {e}")
        
            print(f"[{now_str()}] ✅ AUDIO 처리 완료 → enc={enc_out}, mod={mod_out}\n")

        else:
            print(f"[{now_str()}] ⚠️ 알 수 없는 kind={kind} → 스킵")
            return False

        return True

    except Exception as e:
        print(f"[{now_str()}] ❌ 처리 중 오류: {e}\n")
        return False


if __name__ == "__main__":
    print(f"[{now_str()}] 🛠️ 비식별화 자동 처리 파이프라인 시작 (OUTPUT_BASE={OUTPUT_BASE})")
    while True:
        success = run_processing_pipeline()
        if not success:
            print(f"[{now_str()}] ⏸️ 대기… {SLEEP_SECONDS_WHEN_IDLE//60}분 후 재시작\n")
            time.sleep(SLEEP_SECONDS_WHEN_IDLE)
        else:
            print(f"[{now_str()}] 🔁 다음 작업 확인 중…\n")
