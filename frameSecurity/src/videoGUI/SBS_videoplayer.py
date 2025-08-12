import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import threading
import time

class VideoComparatorApp:
    """
    두 개의 비디오를 나란히 비교 재생하는 GUI 애플리케이션
    """
    def __init__(self, root):
        # --- 1. 기본 창 및 상태 변수 설정 ---
        self.root = root
        self.root.title("비디오 비교 플레이어")

        self.root.geometry("1600x900")
        self.root.resizable(True, True)
        self.root.minsize(800, 600)

        # 비디오 경로 및 상태 변수
        self.original_path = None
        self.blurred_path = None
        self.cap1 = None
        self.cap2 = None
        self.total_frames = 0
        self.fps = 30

        # 재생 상태
        self.is_playing = False
        self.is_paused = True
        self.video_thread = None
        self.stop_thread = False

        # UI 요소들을 담을 메인 프레임
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # grid 레이아웃 3행 구조
        main_frame.rowconfigure(1, weight=1)  # 비디오 영역만 확장
        main_frame.columnconfigure(0, weight=1)

    # --- 2. UI 위젯 생성 및 배치 ---

        # (상단) 파일 선택 프레임
        file_frame = tk.Frame(main_frame)
        file_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        tk.Button(file_frame, text="📂 원본 영상 열기", command=lambda: self.load_video('original')).pack(side=tk.LEFT, padx=5)
        self.original_label = tk.Label(file_frame, text="선택된 파일 없음", bg='lightgrey', relief='sunken', width=40)
        self.original_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
        tk.Button(file_frame, text="📂 처리 영상 열기", command=lambda: self.load_video('blurred')).pack(side=tk.LEFT, padx=5)
        self.blurred_label = tk.Label(file_frame, text="선택된 파일 없음", bg='lightgrey', relief='sunken', width=40)
        self.blurred_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
        # (중단) 비디오 라벨
        self.video_label = tk.Label(main_frame, bg="black")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
    
        # (하단) 컨트롤 프레임
        control_frame = tk.Frame(main_frame)
        control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
    
        # 재생/일시정지 버튼
        self.play_pause_button = tk.Button(
            control_frame, text="▶️ 재생", width=12,
            command=self.toggle_play_pause, state=tk.DISABLED
        )
        self.play_pause_button.pack(side=tk.LEFT)
    
        # 타임라인 (Scale 위젯)
        self.timeline_var = tk.DoubleVar()
        self.timeline = ttk.Scale(
            control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.timeline_var, command=self.on_timeline_seek, state=tk.DISABLED
        )
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
    
        # 창 닫기 이벤트
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_video(self, video_type):
        """파일 탐색기를 열어 비디오를 로드하는 함수"""
        path = filedialog.askopenfilename(title=f"{video_type} 비디오 파일 선택", filetypes=[("Video files", "*.mp4 *.avi")])
        if not path:
            return

        if video_type == 'original':
            self.original_path = path
            self.original_label.config(text=path.split('/')[-1])
        else:
            self.blurred_path = path
            self.blurred_label.config(text=path.split('/')[-1])

        # 두 비디오가 모두 로드되면 플레이어를 초기화
        if self.original_path and self.blurred_path:
            self.initialize_player()

    def initialize_player(self):
        """비디오 캡처 객체를 생성하고 타임라인을 설정하는 함수"""
        if self.cap1: self.cap1.release()
        if self.cap2: self.cap2.release()

        self.cap1 = cv2.VideoCapture(self.original_path)
        self.cap2 = cv2.VideoCapture(self.blurred_path)

        if not self.cap1.isOpened() or not self.cap2.isOpened():
            print("Error: 비디오를 열 수 없습니다.")
            return

        # 총 프레임 수를 두 비디오 중 더 긴 것을 기준으로 설정
        frames1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frames2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = max(frames1, frames2)
        
        # FPS를 원본 기준으로 설정
        self.fps = self.cap1.get(cv2.CAP_PROP_FPS)

        # 타임라인 설정
        self.timeline.config(to=self.total_frames, state=tk.NORMAL)
        self.timeline_var.set(0)
        
        # 재생 버튼 활성화
        self.play_pause_button.config(state=tk.NORMAL, text="▶️ 재생")
        self.is_paused = True
        
        # 첫 프레임을 화면에 표시
        self.show_first_frame()

    def show_first_frame(self):
        """플레이어 초기화 시 첫 프레임을 가져와 화면에 표시"""
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        if ret1 and ret2:
            self.update_video_display(frame1, frame2)
        # 프레임 위치를 다시 0으로 돌려놓음
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def toggle_play_pause(self):
        """재생/일시정지 상태를 토글하는 함수"""
        if self.is_paused:
            # 재생 시작
            self.is_paused = False
            self.play_pause_button.config(text="⏸️ 일시정지")
            
            # 비디오 재생 스레드가 없으면 새로 생성하여 시작
            if self.video_thread is None or not self.video_thread.is_alive():
                self.stop_thread = False
                self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
                self.video_thread.start()
        else:
            # 일시정지
            self.is_paused = True
            self.play_pause_button.config(text="▶️ 재생")
            
    def video_loop(self):
        """별도 스레드에서 비디오 프레임을 읽고 화면을 업데이트하는 메인 루프"""
        delay = 1 / self.fps if self.fps > 0 else 0.033
        
        while not self.stop_thread:
            if not self.is_paused:
                start_time = time.time()
                
                # 현재 프레임 위치 가져오기
                current_frame_pos = self.cap1.get(cv2.CAP_PROP_POS_FRAMES)
                
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()

                if not ret1 or not ret2:
                    break # 비디오 끝에 도달

                # GUI 업데이트 (메인 스레드에서 실행되도록 예약)
                self.root.after(0, self.update_video_display, frame1, frame2)
                self.timeline_var.set(current_frame_pos)
                
                # FPS에 맞춰 딜레이 계산
                elapsed = time.time() - start_time
                sleep_time = delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                # 일시정지 상태에서는 루프가 너무 빨리 돌지 않도록 약간의 쉼
                time.sleep(0.05)

        # 루프 종료 후 상태 초기화
        self.is_paused = True
        self.root.after(0, self.play_pause_button.config, {"text": "▶️ 재생"})


    def update_video_display(self, frame1, frame2):
        """두 프레임을 받아 크기 조절 후 GUI에 표시하는 함수"""
        # 비디오 라벨의 현재 크기를 가져옴 (창 크기에 따라 변함)
        canvas_w = self.video_label.winfo_width()
        canvas_h = self.video_label.winfo_height()

        if canvas_w < 50 or canvas_h < 50: return # 창이 너무 작으면 실행하지 않음

        # 두 프레임을 나란히 붙일 때 필요한 최대 너비
        target_w = canvas_w // 2
        
        # 각 프레임의 크기를 너비 우선으로 조절 (비율 유지)
        def resize_frame(frame, width):
            h, w, _ = frame.shape
            ratio = width / w
            height = int(h * ratio)
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        resized_frame1 = resize_frame(frame1, target_w)
        resized_frame2 = resize_frame(frame2, target_w)
        
        # 두 프레임의 높이를 더 작은 쪽에 맞춤
        h1, _, _ = resized_frame1.shape
        h2, _, _ = resized_frame2.shape
        min_h = min(h1, h2)
        
        resized_frame1 = resized_frame1[:min_h, :, :]
        resized_frame2 = resized_frame2[:min_h, :, :]
        
        # 두 프레임을 수평으로 합침
        combined_frame_bgr = cv2.hconcat([resized_frame1, resized_frame2])

        # BGR(OpenCV) -> RGB 변환
        combined_frame_rgb = cv2.cvtColor(combined_frame_bgr, cv2.COLOR_BGR2RGB)
        
        # PIL 이미지로 변환
        pil_img = Image.fromarray(combined_frame_rgb)
        
        # Tkinter PhotoImage로 변환
        photo_img = ImageTk.PhotoImage(image=pil_img)
        
        # 라벨의 이미지를 업데이트
        self.video_label.config(image=photo_img)
        # 참조를 유지하여 이미지가 사라지는 것을 방지 (중요!)
        self.video_label.image = photo_img

    def on_timeline_seek(self, value):
        """타임라인을 드래그했을 때 호출되는 함수"""
        # 일시정지 상태일 때만 프레임 이동을 허용하는 것이 안정적
        if self.is_paused:
            frame_num = int(float(value))
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            # 이동된 위치의 프레임을 화면에 즉시 표시
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            if ret1 and ret2:
                self.update_video_display(frame1, frame2)


    def on_closing(self):
        """창을 닫을 때 리소스를 안전하게 해제하는 함수"""
        print("프로그램을 종료합니다.")
        self.stop_thread = True # 스레드 종료 신호
        # 비디오 캡처 객체 해제
        if self.cap1: self.cap1.release()
        if self.cap2: self.cap2.release()
        # Tkinter 창 파괴
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = VideoComparatorApp(root)
    root.mainloop()