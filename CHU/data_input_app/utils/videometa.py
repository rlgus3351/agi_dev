from pymediainfo import MediaInfo
import cv2
from datetime import datetime

from typing import Union

def normalize_timestamp(ts: str) -> Union[str, None]:
    """
    '2025-07-15 05:15:29 UTC' → '2025-07-15T05:15:29Z'
    또는 None 처리
    """
    if ts and isinstance(ts, str) and "UTC" in ts:
        try:
            # "UTC" 제거하고 ISO 형식으로 변환
            dt = datetime.strptime(ts.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
            return dt.isoformat() + "Z"  # 'T' 포함 + UTC 표시
        except Exception as e:
            print(f"[normalize_timestamp] 변환 실패: {e}")
            return None
    return None

def get_video_metadata(file_path: str) -> dict:
    """
    OpenCV + pymediainfo를 사용하여 비디오 메타데이터를 추출합니다.
    - 해상도, FPS, 길이 (OpenCV)
    - 촬영일자 (없으면 수정일로 fallback), 코덱 (pymediainfo)
    """
    metadata = {}

    # --- 1️⃣ OpenCV로 해상도, FPS, 길이 추출 ---
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {file_path}")
            return {}

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration_seconds = round(frame_count / frame_rate, 2) if frame_rate > 0 else 0.0
        resolution = f"{width}x{height}"

        cap.release()

        metadata.update({
            "duration_seconds": duration_seconds,
            "resolution": resolution,
            "frame_rate": round(frame_rate, 2)
        })

    except Exception as e:
        print(f"[OpenCV] 메타데이터 추출 실패: {e}")

    # --- 2️⃣ pymediainfo로 촬영일자, 코덱 추출 ---
    try:
        media_info = MediaInfo.parse(file_path)
        general_track = next((t for t in media_info.tracks if t.track_type == "General"), None)
        video_track = next((t for t in media_info.tracks if t.track_type == "Video"), None)

        # 🔹 촬영일자: 없으면 수정일자로 fallback → ISO 형식으로 변환
        raw_ts = None
        if general_track:
            raw_ts = (
                general_track.tagged_date or
                general_track.encoded_date or
                general_track.file_last_modification_date__local
            )
        metadata["creation_time"] = normalize_timestamp(raw_ts)

    except Exception as e:
        print(f"[MediaInfo] 메타데이터 추출 실패: {e}")
        metadata.setdefault("creation_time", None)

    return metadata
