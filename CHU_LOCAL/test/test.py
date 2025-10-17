import cv2
import torch
from ultralytics import YOLO
import os

# ------------------------------------------------------------
# ✅ GPU / CPU 자동 감지
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 YOLO 모델 로드 중... (device={device})")

# ------------------------------------------------------------
# ✅ 모델 로드 (기존 model.pt 사용)
# ------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model", "model.pt")

model = YOLO(model_path)
model.to(device)
print(f"✅ YOLO 모델 로드 완료 (device={device})")

# ------------------------------------------------------------
# ✅ 테스트 함수
# ------------------------------------------------------------
def preview_detection(video_path, conf_thres=0.15, imgsz=640):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 비디오를 열 수 없습니다.")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))

    print(f"🎬 영상 정보: {width}x{height}, {fps} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 예측
        results = model.predict(
            source=frame,
            conf=conf_thres,
            device=device,
            imgsz=imgsz,
            verbose=False,
            half=(device == "cuda")
        )

        result_img = frame.copy()
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None:
            for box in res.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 바운딩 박스 & 라벨 그리기
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_img, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 좌/우 병합 (왼쪽: 원본, 오른쪽: 감지결과)
        combined = cv2.hconcat([frame, result_img])

        cv2.imshow("Face Detection Preview (Left: Original | Right: Detection)", combined)
        key = cv2.waitKey(1) & 0xFF

        # ESC 종료
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 미리보기 종료")

# ------------------------------------------------------------
# ✅ 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    sample_path = r"C:\Users\user\Desktop\sample\sample_2.mp4"
