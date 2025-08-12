import cv2
import os



MODEL_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//model.pt")
VIDEO_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//input1.mp4")

input_path = os.path.join(folder_path, video_file)
output_path = os.path.join(folder_path, f"blurred_{video_file}")

        # === 비디오 읽기 ===
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))