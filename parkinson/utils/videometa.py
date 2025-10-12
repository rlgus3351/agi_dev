# videometa.py
import cv2
# ... (기존 import) ...


def get_video_metadata(file_path: str) -> dict:
    """
    OpenCV를 사용하여 비디오 파일에서 해상도, 길이, 프레임 레이트 등의 메타데이터를 추출합니다.
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {file_path}")
            return {}

        # 메타데이터 추출
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 계산
        duration_seconds = round(frame_count / frame_rate, 2) if frame_rate > 0 else 0.0
        resolution = f"{width}x{height}"

        cap.release()
        
        return {
            "duration_seconds": duration_seconds,
            "resolution": resolution,
            "frame_rate": round(frame_rate, 2),
            "codec": "N/A (Extracted via CV2)" # 코덱은 CV2에서 얻기 어려울 수 있습니다.
        }

    except Exception as e:
        print(f"Failed to extract metadata from {file_path}: {e}")
        return {} # 실패 시 빈 딕셔너리 반환