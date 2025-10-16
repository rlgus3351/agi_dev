import numpy as np
import cv2
from ultralytics import YOLO

def find_min_detection_threshold(video_path: str, model_path: str, step: float = 0.05, sample_rate: int = 10):
    """
    YOLO confidence threshold를 0~1 범위로 바꿔가며 얼굴 검출률을 측정
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("❌ 영상 열기 실패")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    test_thresholds = np.arange(0.05, 0.91, step)
    detection_stats = {round(th, 2): 0 for th in test_thresholds}
    total_tested = 0

    print(f"🎯 Threshold 테스트 시작 ({video_path})")
    print(f"총 프레임: {total_frames}, 샘플링 간격: {sample_rate}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            total_tested += 1
            # 한 프레임에 대해 threshold별로 검출 시도
            for th in test_thresholds:
                result = model.predict(frame, imgsz=640, conf=th, verbose=False)[0]
                if len(result.boxes) > 0:
                    detection_stats[round(th, 2)] += 1

        frame_count += 1

    cap.release()

    # 검출률 계산
    detection_rates = {th: (cnt / total_tested * 100) for th, cnt in detection_stats.items()}

    print("\n📊 Threshold별 얼굴 검출률:")
    for th, rate in detection_rates.items():
        bar = "█" * int(rate / 5)
        print(f"conf={th:.2f}: {rate:6.2f}% {bar}")

    # 얼굴이 '거의 안 잡히기 시작하는' 임계값 추정
    sorted_thr = sorted(detection_rates.items(), key=lambda x: x[0])
    drop_point = next((th for th, rate in sorted_thr if rate < 5.0), None)

    print("\n⚙️ 추정된 최소 인식 한계(confidence threshold):", 
          f"{drop_point:.2f}" if drop_point else "모델이 모든 threshold에서 얼굴 검출")

    return detection_rates, drop_point


if __name__ == "__main__":
    video = r"C:\TeamGit\agi_dev\deidentification\data\2.mp4"
    model_path = r"C:\TeamGit\agi_dev\deidentification\model\model.pt"

    rates, limit = find_min_detection_threshold(video, model_path, step=0.05, sample_rate=20)
