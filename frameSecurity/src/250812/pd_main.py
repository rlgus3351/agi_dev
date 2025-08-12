# app.py
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import json, os, uuid, re, shutil
import cv2

# ===== 설정 =====
USE_YOLO = True            # YOLO 모자이크 사용 여부
CONF_TH = 0.25
MOSAIC_SIZE = 10
MODEL_PATH = os.path.abspath("C://Users//user//Desktop//frameSecurity//data//model.pt")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PATIENT_FILE = os.path.join(DATA_DIR, "patients.json")
SURVEY_FILE  = os.path.join(DATA_DIR, "surveys.json")
VIDEO_FILE   = os.path.join(DATA_DIR, "videos.json")
VIDEO_SAVE_DIR = os.path.join(DATA_DIR, "videos")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)



ITEMS_DEF = [
    {"item_id": 1,  "item_name": "말하기",                     "sides": None},
    {"item_id": 2,  "item_name": "얼굴 표정",                 "sides": None},
    {"item_id": 3,  "item_name": "관절의 뻣뻣함",             "sides": ["Neck","RA","LA","RL","LL"]},  # 목/팔/다리
    {"item_id": 4,  "item_name": "손가락 부딪치기",           "sides": ["R","L"]},
    {"item_id": 5,  "item_name": "손 동작",                   "sides": ["R","L"]},
    {"item_id": 6,  "item_name": "손 내전/외전 움직임",       "sides": ["R","L"]},
    {"item_id": 7,  "item_name": "발가락으로 두드리기",       "sides": ["R","L"]},
    {"item_id": 8,  "item_name": "다리 민첩성",               "sides": ["R","L"]},
    {"item_id": 9,  "item_name": "의자에서 일어나기",         "sides": None},
    {"item_id": 10, "item_name": "걷는 자세",                 "sides": None},
    {"item_id": 11, "item_name": "걷는 중 몸의 굳어짐",       "sides": None},
    {"item_id": 12, "item_name": "자세의 안정",               "sides": None},
    {"item_id": 13, "item_name": "자세",                      "sides": None},
    {"item_id": 14, "item_name": "움직임에서 전반적인 자연스러움", "sides": None},
    {"item_id": 15, "item_name": "자세 유지시 손의 떨림",     "sides": ["R","L"]},
    {"item_id": 16, "item_name": "움직일 때 손의 떨림",       "sides": ["R","L"]},
    {"item_id": 17, "item_name": "가만 있을 때 떨림의 폭",    "sides": ["RA","LA","RL","LL","LJ"]},   # 입술/턱 = LJ
    {"item_id": 18, "item_name": "가만 있을 때 떨림의 지속시간", "sides": None},
]




# ===== 유틸 =====
def load_json(file):
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

patients = load_json(PATIENT_FILE)
surveys  = load_json(SURVEY_FILE)
videos   = load_json(VIDEO_FILE)
selected_patient_id = None

# ---- Hoehn & Yahr 진행 단계 표 ----
HY_STAGES = {
    "0":   "질병의 증후가 없음",
    "1":   "일측성 상하지 장애",
    "1.5": "일측성 상하지 장애와 체간 장애가 있음",
    "2":   "양측성 장애이나 균형장애는 전혀 없음",
    "2.5": "양측성 장애이며, 몸을 잡아당기는 검사에서 균형을 잡을 수는 있음",
    "3":   "경도 및 중등도의 양측성 장애, 균형이 불안정, 그러나 독립적인 활동 가능",
    "4":   "걷고 서기는 할 수 있으나 심각한 무능력 상태",
    "5":   "휠체어를 타거나 침대에 누워 있어야만 하는 상태",
}

def show_hy_info():
    win = tk.Toplevel(root)
    win.title("Hoehn & Yahr 진행 단계 안내")
    win.geometry("700x380")
    head = tk.Frame(win); head.pack(fill="x", padx=10, pady=10)
    tk.Label(head, text="Hoehn & Yahr 파킨슨병 진행 단계", font=("Arial", 12, "bold")).pack(side="left")
    body = tk.Frame(win); body.pack(fill="both", expand=True, padx=10, pady=(0,10))
    tk.Label(body, text="단계",  width=8,  bg="#eeeeee").grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
    tk.Label(body, text="설명",  width=70, bg="#eeeeee").grid(row=0, column=1, sticky="nsew", padx=1, pady=1)
    for r,(k,v) in enumerate(HY_STAGES.items(), start=1):
        tk.Label(body, text=k, anchor="w").grid(row=r, column=0, sticky="nsew", padx=1, pady=1)
        tk.Label(body, text=v, anchor="w", wraplength=560, justify="left").grid(row=r, column=1, sticky="nsew", padx=1, pady=1)
    body.grid_columnconfigure(1, weight=1)

# ===== YOLO 모자이크 래퍼 =====
class YoloMosaic:
    def __init__(self, model_path, conf=0.25, block=10, use=True):
        self.enabled = use
        self.conf = conf
        self.block = block
        self.model = None
        self.device = "cpu"
        if not self.enabled:
            return
        try:
            import torch
            from ultralytics import YOLO
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(model_path)
            self.model.to(self.device)
            if self.device == "cuda":
                self.model.model.half()
        except Exception as e:
            messagebox.showwarning("YOLO 비활성화", f"모델 로드 실패: {e}\n모자이크 없이 재생합니다.")
            self.enabled = False

    def mosaic_region(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return
        small = cv2.resize(roi, (self.block, self.block), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = mosaic

    def process(self, frame_bgr):
        if not self.enabled or self.model is None:
            return frame_bgr
        try:
            results = self.model(frame_bgr, conf=self.conf, verbose=False)
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy
                for b in xyxy:
                    x1, y1, x2, y2 = map(int, b.tolist())
                    self.mosaic_region(frame_bgr, x1, y1, x2, y2)
        except Exception as e:
            print(f"[YOLO] 추론 오류: {e} → 모자이크 비활성화")
            self.enabled = False
        return frame_bgr

yolo = YoloMosaic(MODEL_PATH, conf=CONF_TH, block=MOSAIC_SIZE, use=USE_YOLO)

# ===== 수평 스크롤 프레임 (설문 카드용) =====
class HScrollFrame(tk.Frame):
    def __init__(self, parent, height=240, bg="#f7f7f7"):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0, bg=bg)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.hbar.set)

        self.inner = tk.Frame(self.canvas, bg=bg)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.pack(fill="both", expand=True, side="top")
        self.hbar.pack(fill="x", side="bottom")

        self.inner.bind("<Configure>", self._on_configure)

    def _on_configure(self, _e):
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)

# ===== Before | After 플레이어 (원본 크기 + 스크롤) =====
class BeforeAfterPlayer(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bd=1, relief="sunken")

        # Canvas + 스크롤바
        self.canvas = tk.Canvas(self, bg="#202020", highlightthickness=0)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(self, orient="vertical",   command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        ctr = tk.Frame(self); ctr.grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)
        self.btn_play  = tk.Button(ctr, text="재생", command=self.play)
        self.btn_pause = tk.Button(ctr, text="일시정지", command=self.toggle_pause, state="disabled")
        self.btn_stop  = tk.Button(ctr, text="정지", command=self.stop, state="disabled")
        self.btn_save  = tk.Button(ctr, text="저장 OFF", command=self.toggle_save, state="disabled")
        self.lbl_info  = tk.Label(ctr, text="대기", anchor="w")
        self.btn_play.pack(side="left", padx=4)
        self.btn_pause.pack(side="left", padx=4)
        self.btn_stop.pack(side="left", padx=4)
        self.btn_save.pack(side="left", padx=4)
        self.lbl_info.pack(side="left", padx=12)

        # 상태
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.frame_idx = 0
        self.current_path = None
        self.out_path = None
        self.writer = None
        self.running = False
        self.paused = False
        self._imgtk_cache = None

        self.canvas.bind("<Configure>", lambda e: None)  # scrollregion은 draw_frame에서 갱신

    def open(self, video_path, save_dir=None):
        self.close_handles()
        self.current_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.current_path = None
            messagebox.showerror("오류", "영상을 열 수 없습니다.")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if save_dir:
            base, _ = os.path.splitext(os.path.basename(video_path))
            self.out_path = os.path.join(save_dir, f"{base}_blurred.mp4")
        else:
            self.out_path = None
        self.btn_play.config(state="normal")
        self.btn_pause.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.btn_save.config(state="normal" if self.out_path else "disabled")
        self.update_info("대기")

    def play(self):
        if self.cap is None or self.running: return
        self.running = True
        self.paused = False
        self.btn_pause.config(state="normal", text="일시정지")
        self.btn_stop.config(state="normal")
        self.update_loop()

    def toggle_pause(self):
        if not self.running: return
        self.paused = not self.paused
        self.btn_pause.config(text="재생" if self.paused else "일시정지")

    def stop(self):
        self.running = False
        self.paused = False
        self.btn_pause.config(state="disabled", text="일시정지")
        self.btn_stop.config(state="disabled")
        self.update_info("정지")
        self.close_writer()

    def toggle_save(self):
        if self.out_path is None: return
        if self.writer is None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))
            self.btn_save.config(text="저장 ON"); self.update_info("저장 ON")
        else:
            self.close_writer()
            self.btn_save.config(text="저장 OFF"); self.update_info("저장 OFF")

    def update_loop(self):
        if not self.running: return
        if self.paused:
            self.after(50, self.update_loop); return

        ret, frame = self.cap.read()
        if not ret:
            self.stop(); return

        self.frame_idx += 1
        orig = frame.copy()
        after = yolo.process(frame)  # 모자이크(옵션)

        if self.writer is not None:
            self.writer.write(after)

        combined = self.make_side_by_side_original(orig, after)  # 원본 크기 그대로
        self.draw_frame(combined)

        info = f"{os.path.basename(self.current_path)} | {self.frame_idx}/{self.total_frames} | {'저장ON' if self.writer else '저장OFF'}"
        self.update_info(info)

        delay = int(1000 / max(self.fps, 1))
        self.after(max(1, delay // 2), self.update_loop)

    def make_side_by_side_original(self, left_bgr, right_bgr):
        lh, lw = left_bgr.shape[:2]
        rh, rw = right_bgr.shape[:2]
        H = max(lh, rh)

        def pad_to_h(img, H):
            h, w = img.shape[:2]
            if h == H: return img
            pad = H - h
            top = pad // 2
            bottom = pad - top
            return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

        left_p  = pad_to_h(left_bgr,  H)
        right_p = pad_to_h(right_bgr, H)
        return cv2.hconcat([left_p, right_p])

    def draw_frame(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._imgtk_cache = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._imgtk_cache, anchor="nw")
        w, h = img.size
        self.canvas.config(scrollregion=(0, 0, w, h))

    def update_info(self, text):
        self.lbl_info.config(text=text)

    def close_writer(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def close_handles(self):
        self.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.close_writer()

    def destroy(self):
        self.close_handles()
        super().destroy()

# ===== 메인 UI =====
root = tk.Tk()
root.title("환자/설문/영상 관리 (입력/내역 탭) + Before/After")
root.geometry("1700x980")

# 전체 그리드
root.grid_rowconfigure(0, weight=0)  # 헤더
root.grid_rowconfigure(1, weight=1)  # 본문
root.grid_columnconfigure(0, weight=0)
root.grid_columnconfigure(1, weight=1)

# 헤더
# header = tk.Frame(root, bg="#111111")
# header.grid(row=0, column=0, columnspan=2, sticky="ew")
# for i, txt in enumerate(["환자목록", "입력", "설문 내역(가로)", "영상 내역"]):
#     tk.Label(header, text=txt, fg="#ffffff", bg="#111111",
#              font=("Arial", 12, "bold"), padx=12, pady=8).pack(side="left")
#     if i < 3:
#         tk.Label(header, text="|", fg="#888888", bg="#111111",
#                  font=("Arial", 12)).pack(side="left", padx=6)

# 좌: 환자 목록
frame_patient = tk.LabelFrame(root, text="환자 목록", padx=5, pady=5)
frame_patient.grid(row=1, column=0, sticky="ns", padx=5, pady=5)
frame_patient.grid_propagate(False)
frame_patient.config(width=340)
patient_listbox = tk.Listbox(frame_patient, height=36, width=30)
patient_listbox.pack(fill="both", expand=True)
btns = tk.Frame(frame_patient); btns.pack(pady=6)

def add_patient():
    popup = tk.Toplevel(root); popup.title("환자 추가"); popup.geometry("300x200"); popup.grab_set()
    tk.Label(popup, text="이니셜:").pack(pady=5); initials_entry = tk.Entry(popup); initials_entry.pack()
    tk.Label(popup, text="생년월일 (YYYYMMDD):").pack(pady=5); dob_entry = tk.Entry(popup); dob_entry.pack()
    tk.Label(popup, text="성별 (남/여):").pack(pady=5); gender_entry = tk.Entry(popup); gender_entry.pack()
    def save_patient():
        initials, dob, gender = initials_entry.get().strip(), dob_entry.get().strip(), gender_entry.get().strip()
        if not initials or not dob or not gender:
            return messagebox.showwarning("입력 오류", "모든 항목을 입력하세요.")
        if not re.fullmatch(r"\d{8}", dob):
            return messagebox.showwarning("입력 오류", "생년월일은 YYYYMMDD 형식으로 입력.")
        pid = str(uuid.uuid4())
        patients[pid] = {"이니셜": initials, "생년월일": dob, "성별": gender}
        save_json(PATIENT_FILE, patients); update_patient_list(); popup.destroy()
    tk.Button(popup, text="저장", command=save_patient).pack(pady=10)

def delete_patient():
    global selected_patient_id
    if not selected_patient_id:
        return messagebox.showwarning("선택 오류", "삭제할 환자를 먼저 선택하세요.")
    info = patients[selected_patient_id]
    if not messagebox.askyesno("확인", f"정말 {info['이니셜']} 환자와 관련 데이터(설문/영상)를 삭제할까요?"):
        return
    patients.pop(selected_patient_id, None)
    surveys.pop(selected_patient_id, None)
    v = videos.pop(selected_patient_id, None)
    if v:
        pdir = os.path.join(VIDEO_SAVE_DIR, selected_patient_id)
        if os.path.isdir(pdir):
            try: shutil.rmtree(pdir)
            except: pass
    save_json(PATIENT_FILE, patients)
    save_json(SURVEY_FILE, surveys)
    save_json(VIDEO_FILE, videos)
    selected_patient_id = None
    update_patient_list()
    clear_input_status()
    populate_survey_cards()
    populate_video_list()
    player.close_handles()
    messagebox.showinfo("완료", "삭제되었습니다.")

tk.Button(btns, text="환자 추가", command=add_patient).pack(side="left", padx=5)
tk.Button(btns, text="환자 삭제", command=delete_patient).pack(side="left", padx=5)

# 우: 탭 컨테이너
right = tk.Frame(root)
right.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
right.grid_rowconfigure(0, weight=1)
right.grid_columnconfigure(0, weight=1)

notebook = ttk.Notebook(right)
notebook.grid(row=0, column=0, sticky="nsew")
tab_input  = tk.Frame(notebook)
tab_survey = tk.Frame(notebook)
tab_video  = tk.Frame(notebook)
notebook.add(tab_input,  text="입력")
notebook.add(tab_survey, text="설문 내역(가로)")
notebook.add(tab_video,  text="영상 내역")

# ============================================================================
# ===== 입력 탭 (가로 배치: 약물 | Dyskinesia+H&Y | 점수입력) =================
# ============================================================================
lbl_input_title = tk.Label(tab_input, text="선택된 환자: 없음", font=("Arial", 12, "bold"))
lbl_input_title.pack(pady=4, anchor="w")

# 한 줄 컨테이너(그리드 3열)
input_row = tk.Frame(tab_input)
input_row.pack(fill="both", expand=True, padx=6, pady=6)

# 좌/중은 고정폭, 우측(점수입력)은 남는 공간 확장
input_row.grid_rowconfigure(0, weight=1)
input_row.grid_columnconfigure(0, weight=0)   # 약물
input_row.grid_columnconfigure(1, weight=0)   # Dyskinesia + H&Y
input_row.grid_columnconfigure(2, weight=1)   # 점수입력(확장)

# ---- 약물(Medication) ----
med_frame = tk.LabelFrame(input_row, text="약물(Medication)", padx=6, pady=6)
med_frame.grid(row=0, column=0, sticky="n", padx=(0, 8))

tk.Label(med_frame, text="a. 파킨슨 증상 치료 목적 약 복용 여부?").pack(anchor="w")
med_on_var = tk.IntVar(value=0)
tk.Radiobutton(med_frame, text="예", variable=med_on_var, value=1).pack(anchor="w")
tk.Radiobutton(med_frame, text="아니오", variable=med_on_var, value=0).pack(anchor="w")

tk.Label(med_frame, text="b. 임상적 상태 (긍정/부정)").pack(anchor="w")
med_effect_var = tk.StringVar(value="")
tk.Radiobutton(med_frame, text="긍정적 효과", variable=med_effect_var, value="positive").pack(anchor="w")
tk.Radiobutton(med_frame, text="부정적 효과", variable=med_effect_var, value="negative").pack(anchor="w")

tk.Label(med_frame, text="c. Levodopa 복용 여부?").pack(anchor="w")
med_levodopa_var = tk.IntVar(value=0)
tk.Radiobutton(med_frame, text="예", variable=med_levodopa_var, value=1).pack(anchor="w")
tk.Radiobutton(med_frame, text="아니오", variable=med_levodopa_var, value=0).pack(anchor="w")

levodopa_row = tk.Frame(med_frame); levodopa_row.pack(anchor="w", pady=2)
tk.Label(levodopa_row, text="마지막 복용 후 경과 시간(분):").pack(side="left")
med_minutes_entry = tk.Entry(levodopa_row, width=6); med_minutes_entry.pack(side="left", padx=5)

# ---- Dyskinesia Impact + H&Y ----
dysk_col = tk.Frame(input_row)
dysk_col.grid(row=0, column=0, sticky="n", padx=(0, 8))

dysk_frame = tk.LabelFrame(dysk_col, text="Dyskinesia Impact", padx=6, pady=6)
dysk_frame.pack(fill="x")
dys_present_var = tk.IntVar(value=0); dys_interfere_var = tk.IntVar(value=0)
tk.Checkbutton(dysk_frame, text="검사 중 Dyskinesia 발생", variable=dys_present_var).pack(anchor="w")
tk.Checkbutton(dysk_frame, text="Dyskinesia가 평가에 영향", variable=dys_interfere_var).pack(anchor="w")

hy_frame = tk.LabelFrame(dysk_col, text="Hoehn & Yahr 진행 단계", padx=6, pady=6)
hy_frame.pack(fill="x", pady=(8,0))
hoehn_yahr_var = tk.StringVar(value="")  # "", "0"~"5"
row_hy = tk.Frame(hy_frame); row_hy.pack(anchor="w")
for o in ["0","1","1.5","2","2.5"]:
    tk.Radiobutton(row_hy, text=o, value=o, variable=hoehn_yahr_var).pack(side="left", padx=2)
row_hy2 = tk.Frame(hy_frame); row_hy2.pack(anchor="w", pady=(4,2))
for o in ["3","4","5"]:
    tk.Radiobutton(row_hy2, text=o, value=o, variable=hoehn_yahr_var).pack(side="left", padx=2)
tk.Button(hy_frame, text="단계 설명 보기", command=show_hy_info).pack(anchor="w", pady=(6,0))

# ---- 점수 입력 (우측, 내부 세로 스크롤) ----
scores_frame = tk.LabelFrame(input_row, text="점수 입력 (문항별 최대4개)", padx=6, pady=6)
scores_frame.grid(row=0, column=2, sticky="nsew")

scores_canvas = tk.Canvas(scores_frame, highlightthickness=0)
scores_scrollbar = ttk.Scrollbar(scores_frame, orient="vertical", command=scores_canvas.yview)
scores_inner = tk.Frame(scores_canvas)

scores_inner.bind("<Configure>", lambda e: scores_canvas.configure(scrollregion=scores_canvas.bbox("all")))
scores_canvas.create_window((0, 0), window=scores_inner, anchor="nw")
scores_canvas.configure(yscrollcommand=scores_scrollbar.set)

scores_canvas.pack(side="left", fill="both", expand=True)
scores_scrollbar.pack(side="right", fill="y")

question_entries = {}

def _mk_score_entry(parent, width=4):
    e = tk.Entry(parent, width=width, justify="center")
    # 0~4만 허용(빈칸 허용)
    def _ok(P):
        if P == "": return True
        return P.isdigit() and 0 <= int(P) <= 4
    e.config(validate="key", validatecommand=(parent.register(_ok), "%P"))
    return e

ROW_HEIGHT = 28  # 행 높이(px) - 원하면 조정 가능

for item in ITEMS_DEF:  # ← 앞서 만든 항목 정의 리스트
    row = tk.Frame(scores_inner, height=ROW_HEIGHT)
    row.pack(fill="x", pady=4, padx=6)
    row.pack_propagate(False)  # 행 높이 고정

    # row 내부는 grid 사용 (3컬럼: 번호, 항목명, 입력영역)
    row.grid_columnconfigure(0, weight=0)  # 번호
    row.grid_columnconfigure(1, weight=1)  # 항목명(가변)
    row.grid_columnconfigure(2, weight=0)  # 입력칸

    # 번호 / 항목명
    tk.Label(row, text=f"{item['item_id']}번", width=6, anchor="w").grid(row=0, column=0, sticky="w")
    tk.Label(row, text=item["item_name"], anchor="w").grid(row=0, column=1, sticky="w")

    # 입력칸 컨테이너
    inputs = tk.Frame(row)
    inputs.grid(row=0, column=2, sticky="e")

    if not item["sides"]:
        ent = _mk_score_entry(inputs)
        ent.pack(side="right")
        question_entries[item["item_id"]] = ent
    else:
        side_entries = {}
        for side in reversed(item["sides"]):
            cell = tk.Frame(inputs)
            cell.pack(side="right", padx=3)
            tk.Label(cell, text=side, fg="#555").pack(side="top")
            ent = _mk_score_entry(cell)
            ent.pack(side="top")
            side_entries[side] = ent
        question_entries[item["item_id"]] = {s: side_entries[s] for s in item["sides"]}

# ===== 설문 저장/초기화/삭제 =====
def submit_survey():
    if not selected_patient_id:
        return messagebox.showwarning("선택 오류", "환자를 먼저 선택하세요.")
    medication_data = {
        "is_on_medication": bool(med_on_var.get()),
        "clinical_effect": med_effect_var.get(),
        "levodopa_taken": bool(med_levodopa_var.get()),
        "levodopa_elapsed_hours": float(med_minutes_entry.get() or 0) / 60
    }
    dyskinesia_data = {
        "dyskinesia_present": bool(dys_present_var.get()),
        "dyskinesia_interfered": bool(dys_interfere_var.get())
    }
    hy_stage = hoehn_yahr_var.get()  # "" 또는 "0"~"5"
    scores_data = {}

    for item in ITEMS_DEF:
        iid = item["item_id"]
        widget = question_entries.get(iid)

        if not item["sides"]:
            # 단일 점수
            val = widget.get().strip() if widget else ""
            if val != "":
                scores_data[str(iid)] = int(val)  # 0~4
            # 미입력 시 저장 생략(원하면 0으로 기본값 설정도 가능)
        else:
            # 부위별 점수
            side_vals = {}
            for s, e in widget.items():
                v = e.get().strip()
                if v != "":
                    side_vals[s] = int(v)  # 0~4
            if side_vals:
                scores_data[str(iid)] = side_vals
    save_json(SURVEY_FILE, surveys)
    update_patient_list()
    populate_survey_cards()
    messagebox.showinfo("완료", "설문이 저장되었습니다.")

def reset_survey_form():
    med_on_var.set(0)
    med_effect_var.set("")
    med_levodopa_var.set(0)
    med_minutes_entry.delete(0, tk.END)
    dys_present_var.set(0)
    dys_interfere_var.set(0)
    hoehn_yahr_var.set("")
    for iid, widget in question_entries.items():
        if isinstance(widget, dict):
            for s, e in widget.items():
                e.delete(0, tk.END)
        else:
            widget.delete(0, tk.END)

def delete_current_patient_survey():
    global selected_patient_id
    if not selected_patient_id:
        return messagebox.showwarning("선택 오류", "환자를 먼저 선택하세요.")
    if selected_patient_id not in surveys:
        return messagebox.showinfo("알림", "삭제할 설문이 없습니다.")
    p = patients.get(selected_patient_id, {"이니셜": "?"})
    if not messagebox.askyesno("확인", f"{p['이니셜']} 환자의 저장된 설문을 삭제할까요?"):
        return
    surveys.pop(selected_patient_id, None)
    save_json(SURVEY_FILE, surveys)
    update_patient_list()
    populate_survey_cards()
    messagebox.showinfo("완료", "설문이 삭제되었습니다.")

btn_bar = tk.Frame(tab_input)
btn_bar.pack(pady=6, anchor="w", padx=6)
tk.Button(btn_bar, text="설문 제출", command=submit_survey).pack(side="left", padx=4)
tk.Button(btn_bar, text="설문지 초기화", command=reset_survey_form).pack(side="left", padx=4)
tk.Button(btn_bar, text="현재 환자 설문 삭제", command=delete_current_patient_survey).pack(side="left", padx=4)

# ===== 영상 선택/업로드 =====
input_video = tk.LabelFrame(tab_input, text="영상 선택/업로드", padx=6, pady=6)
input_video.pack(fill="x", padx=4, pady=6)

video_labels = {}  # 입력 탭에서 상태 표시
video_keys = ["1번영상", "2번영상", "3번영상", "손글씨영상"]

def select_video(video_key):
    if not selected_patient_id:
        return messagebox.showwarning("선택 오류", "환자를 먼저 선택하세요.")
    file_path = filedialog.askopenfilename(
        title=f"{video_key} 선택", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    if file_path:
        videos.setdefault(selected_patient_id, {})[video_key] = {"temp_path": file_path, "final_path": None}
        save_json(VIDEO_FILE, videos)
        update_video_labels()
        populate_video_list()

for key in video_keys:
    row = tk.Frame(input_video); row.pack(fill="x", pady=2)
    tk.Button(row, text=f"{key} 선택", command=lambda k=key: select_video(k)).pack(side="left")
    lbl = tk.Label(row, text="파일 없음", width=60, anchor="w"); lbl.pack(side="left", padx=6)
    video_labels[key] = lbl

def upload_videos():
    if not selected_patient_id:
        return messagebox.showwarning("선택 오류", "환자를 먼저 선택하세요.")
    patient_videos = videos.get(selected_patient_id, {})
    if not patient_videos:
        return messagebox.showwarning("영상 없음", "선택된 영상이 없습니다.")
    patient_dir = os.path.join(VIDEO_SAVE_DIR, selected_patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    for key, paths in patient_videos.items():
        temp_path = paths.get("temp_path")
        if temp_path and os.path.exists(temp_path):
            ext = os.path.splitext(temp_path)[1]
            save_path = os.path.join(patient_dir, f"{key}{ext}")
            shutil.copy(temp_path, save_path)
            videos[selected_patient_id][key] = {"temp_path": temp_path, "final_path": save_path}
    save_json(VIDEO_FILE, videos)
    update_video_labels()
    populate_video_list()
    messagebox.showinfo("업로드 완료", "선택된 영상이 로컬에 업로드되었습니다.")

tk.Button(input_video, text="영상 업로드", command=upload_videos).pack(pady=6, anchor="w")

# ===== 설문 내역 탭 (가로 카드 전용) =====
survey_bar = tk.Frame(tab_survey)
survey_bar.pack(fill="x", padx=6, pady=(8, 0))
tk.Label(survey_bar, text="설문 내역(가로 보기)", font=("Arial", 12, "bold")).pack(side="left")

hscroll = HScrollFrame(tab_survey, height=260)
hscroll.pack(fill="both", expand=True, padx=6, pady=8)

def make_survey_card(parent, pid, pinfo, total_score, status_text, hy_stage=""):
    card = tk.Frame(parent, bg="white", bd=1, relief="solid")
    card.config(width=220, height=200)
    card.pack_propagate(False)

    tk.Label(card, text=pinfo.get("이니셜","?"),
             font=("Arial", 14, "bold"), bg="white").pack(anchor="w", padx=12, pady=(10, 0))
    sub = f"{pinfo.get('성별','')} · {pinfo.get('생년월일','')}"
    tk.Label(card, text=sub, font=("Arial", 10), fg="#666", bg="white").pack(anchor="w", padx=12, pady=(0, 6))

    hy_txt = f"H&Y: {hy_stage}" if hy_stage else "H&Y: -"
    tk.Label(card, text=hy_txt, font=("Arial", 10), fg="#444", bg="white").pack(anchor="w", padx=12, pady=(0, 6))

    tk.Label(card, text=f"총점 {total_score}", font=("Arial", 16, "bold"), bg="white").pack(pady=(0, 6))

    badge_bg = "#16a34a" if status_text == "완료" else "#a3a3a3"
    tk.Label(card, text=status_text, bg=badge_bg, fg="white", font=("Arial", 9), padx=8, pady=2).pack()

    tk.Button(card, text="열기", command=lambda: open_patient_in_input(pid)).pack(pady=8)
    card.bind("<Button-1>", lambda _e: open_patient_in_input(pid))
    return card

def open_patient_in_input(pid):
    keys = list(patients.keys())
    if pid in keys:
        idx = keys.index(pid)
        patient_listbox.selection_clear(0, "end")
        patient_listbox.selection_set(idx)
        patient_listbox.event_generate("<<ListboxSelect>>")
    notebook.select(tab_input)

def populate_survey_cards():
    for w in hscroll.inner.winfo_children():
        w.destroy()
    xpad, ypad = 8, 8
    for pid, data in surveys.items():
        p = patients.get(pid, {"이니셜":"?", "성별":"", "생년월일":""})
        scores = data.get("scores", {})
        def score_sum(sc):
            if isinstance(sc, list):
                return sum(int(x) for x in sc if str(x).isdigit())
            try:
                return int(sc)
            except:
                return 0
        total = sum(score_sum(v) for v in scores.values())
        status = "완료"
        hy_stage = str(data.get("hoehn_yahr_stage", "") or "")
        card = make_survey_card(hscroll.inner, pid, p, total, status, hy_stage)
        card.pack(side="left", padx=xpad, pady=ypad)

# ===== 영상 내역 탭 =====
tab_video.grid_rowconfigure(0, weight=1)
tab_video.grid_rowconfigure(1, weight=0)
tab_video.grid_columnconfigure(0, weight=1)

video_tree = ttk.Treeview(tab_video, columns=("vkey","fname","path"), show="headings", height=10)
video_tree.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
video_tree.heading("vkey", text="구분")
video_tree.heading("fname", text="파일명")
video_tree.heading("path", text="경로")
video_tree.column("vkey", width=120)
video_tree.column("fname", width=280)
video_tree.column("path", width=640)

player_frame = tk.Frame(tab_video)
player_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
player = BeforeAfterPlayer(player_frame)
player.pack(fill="x", expand=False)

def populate_video_list():
    video_tree.delete(*video_tree.get_children())
    if not selected_patient_id: return
    for k, v in videos.get(selected_patient_id, {}).items():
        fpath = v.get("final_path") or v.get("temp_path") or ""
        fname = os.path.basename(fpath) if fpath else "-"
        video_tree.insert("", "end", values=(k, fname, fpath))

def on_video_dblclick(_e):
    item = video_tree.focus()
    if not item: return
    vkey, fname, path = video_tree.item(item, "values")
    if path and os.path.exists(path):
        save_dir = os.path.join(VIDEO_SAVE_DIR, selected_patient_id) if selected_patient_id else None
        player.open(path, save_dir=save_dir)
        # player.play()  # 자동재생 원하면 주석 해제

video_tree.bind("<Double-1>", on_video_dblclick)

# ===== 좌측 환자 목록 로직 =====
def update_patient_list():
    patient_listbox.delete(0, tk.END)
    for pid, info in patients.items():
        idx = patient_listbox.size()
        patient_listbox.insert(tk.END, f"{info['이니셜']} / {info['생년월일']} / {info['성별']}")
        patient_listbox.itemconfig(idx, {'fg': 'green' if pid in surveys else 'red'})

def clear_input_status():
    lbl_input_title.config(text="선택된 환자: 없음")
    for _, lbl in video_labels.items():
        lbl.config(text="파일 없음")

def update_video_labels():
    if not selected_patient_id:
        clear_input_status(); return
    pv = videos.get(selected_patient_id, {})
    for key, lbl in video_labels.items():
        paths = pv.get(key)
        if paths:
            if paths.get("final_path"):
                lbl.config(text=f"{os.path.basename(paths['final_path'])} (업로드 완료)")
            else:
                lbl.config(text=f"{os.path.basename(paths['temp_path'])} (대기중)")
        else:
            lbl.config(text="파일 없음")

def on_patient_select(_e):
    global selected_patient_id
    idx = patient_listbox.curselection()
    if not idx: return
    selected_patient_id = list(patients.keys())[idx[0]]
    info = patients[selected_patient_id]
    lbl_input_title.config(text=f"선택된 환자: {info['이니셜']}")
    update_video_labels()
    populate_survey_cards()
    populate_video_list()

patient_listbox.bind("<<ListboxSelect>>", on_patient_select)

# 종료 처리
def on_close():
    try:
        player.destroy()
    except:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# 초기 로드
update_patient_list()
populate_survey_cards()
root.mainloop()
