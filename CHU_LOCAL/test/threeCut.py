import cv2
import os
import numpy as np

# ✅ 절대경로로 지정
original_path = r"C:\Users\user\Desktop\sample\sample_2.mp4"
mosaic_path = r"C:\TeamGit\agi_dev\CHU_LOCAL\test\data\output\sample_2_final.mp4"
restore_path = r"C:\TeamGit\agi_dev\CHU_LOCAL\test\data\output\restored\sample_2_final_restored_final.mp4"

# ✅ 파일 존재 확인
for path in [original_path, mosaic_path, restore_path]:
    if not os.path.exists(path):
        print(f"❌ 파일을 찾을 수 없습니다: {path}")
        exit()

# ✅ 영상 로드
cap1 = cv2.VideoCapture(original_path)
cap2 = cv2.VideoCapture(mosaic_path)
cap3 = cv2.VideoCapture(restore_path)

# ✅ 기본 정보
fps = int(cap1.get(cv2.CAP_PROP_FPS))
delay = int(1000 / max(1, fps))
total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

# ✅ 윈도우 생성
cv2.namedWindow("Original | Mosaic | Restored", cv2.WINDOW_NORMAL)

# ✅ 트랙바 초기화
current_frame = 0
paused = False

def on_trackbar(val):
    """트랙바로 프레임 이동"""
    global current_frame
    current_frame = val
    cap1.set(cv2.CAP_PROP_POS_FRAMES, val)
    # ✅ cap1 기준으로 나머지 두 영상 동기화
    frame_idx = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

cv2.createTrackbar("Frame", "Original | Mosaic | Restored", 0, total_frames - 1, on_trackbar)

print("🎬 영상 비교 시작 (Space: 재생/일시정지, ESC: 종료)")

while True:
    if not paused:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        ret3, f3 = cap3.read()

        if not (ret1 and ret2 and ret3):
            break

        # ✅ 프레임 크기 통일
        height = 640
        width = int(f1.shape[1] * (height / f1.shape[0]))
        f1 = cv2.resize(f1, (width, height))
        f2 = cv2.resize(f2, (width, height))
        f3 = cv2.resize(f3, (width, height))

        # ✅ 텍스트 표시
        cv2.putText(f1, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2)
        cv2.putText(f2, "Mosaic", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 255), 2)
        cv2.putText(f3, "Restored", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)

        # ✅ 병합
        combined = cv2.hconcat([f1, f2, f3])
        cv2.imshow("Original | Mosaic | Restored", combined)

        # ✅ 트랙바 위치 업데이트
        current_frame = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos("Frame", "Original | Mosaic | Restored", current_frame)

    key = cv2.waitKey(delay) & 0xFF

    if key == 27:  # ESC 종료
        break
    elif key == 32:  # Space: 재생/일시정지
        paused = not paused
    elif key == ord('a'):  # ← 5프레임 뒤로
        current_frame = max(0, current_frame - 5)
        on_trackbar(current_frame)
    elif key == ord('d'):  # → 5프레임 앞으로
        current_frame = min(total_frames - 1, current_frame + 5)
        on_trackbar(current_frame)

cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()

print("✅ 종료되었습니다.")
