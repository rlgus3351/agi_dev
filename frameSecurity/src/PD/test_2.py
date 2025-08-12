import tkinter as tk
from tkinter import messagebox, filedialog
import json, os, uuid, re
import cv2, shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PATIENT_FILE = os.path.join(DATA_DIR, "patients.json")
SURVEY_FILE = os.path.join(DATA_DIR, "surveys.json")
VIDEO_FILE = os.path.join(DATA_DIR, "videos.json")
VIDEO_SAVE_DIR = os.path.join(DATA_DIR, "videos")
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

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
surveys = load_json(SURVEY_FILE)
videos = load_json(VIDEO_FILE)
selected_patient_id = None

# ==================== 기능 함수 ====================
def update_patient_list():
    patient_listbox.delete(0, tk.END)
    for pid, info in patients.items():
        idx = patient_listbox.size()
        patient_listbox.insert(tk.END, f"{info['이니셜']} / {info['생년월일']} / {info['성별']}")
        patient_listbox.itemconfig(idx, {'fg': 'green' if pid in surveys else 'red'})

def add_patient():
    popup = tk.Toplevel(root)
    popup.title("환자 추가")
    popup.geometry("300x200")
    popup.grab_set()

    tk.Label(popup, text="이니셜:").pack(pady=5)
    initials_entry = tk.Entry(popup)
    initials_entry.pack()
    tk.Label(popup, text="생년월일 (YYYYMMDD):").pack(pady=5)
    dob_entry = tk.Entry(popup)
    dob_entry.pack()
    tk.Label(popup, text="성별 (남/여):").pack(pady=5)
    gender_entry = tk.Entry(popup)
    gender_entry.pack()

    def save_patient():
        initials, dob, gender = initials_entry.get().strip(), dob_entry.get().strip(), gender_entry.get().strip()
        if not initials or not dob or not gender:
            return messagebox.showwarning("입력 오류", "모든 항목을 입력하세요.")
        if not re.fullmatch(r"\d{8}", dob):
            return messagebox.showwarning("입력 오류", "생년월일은 YYYYMMDD 형식으로 입력하세요.")
        pid = str(uuid.uuid4())
        patients[pid] = {"이니셜": initials, "생년월일": dob, "성별": gender}
        save_json(PATIENT_FILE, patients)
        update_patient_list()
        popup.destroy()

    tk.Button(popup, text="저장", command=save_patient).pack(pady=10)

def delete_patient():
    global selected_patient_id
    if not selected_patient_id:
        return messagebox.showwarning("선택 오류", "삭제할 환자를 먼저 선택하세요.")
    info = patients[selected_patient_id]
    confirm = messagebox.askyesno("확인", f"정말 {info['이니셜']} 환자와 관련된 설문/영상 데이터를 모두 삭제할까요?")
    if not confirm: return

    patients.pop(selected_patient_id, None)
    surveys.pop(selected_patient_id, None)
    videos.pop(selected_patient_id, None)
    save_json(PATIENT_FILE, patients)
    save_json(SURVEY_FILE, surveys)
    save_json(VIDEO_FILE, videos)

    selected_patient_id = None
    lbl_survey_title.config(text="설문 내역 - 선택 없음")
    lbl_video_title.config(text="영상 내역 - 선택 없음")
    for lbl in video_labels.values():
        lbl.config(text="파일 없음")
    survey_frame.pack_forget()
    btn_start_survey.config(state="disabled")
    update_patient_list()
    messagebox.showinfo("완료", "환자와 관련된 모든 데이터가 삭제되었습니다.")

def on_patient_select(event):
    global selected_patient_id
    idx = patient_listbox.curselection()
    if not idx: return
    selected_patient_id = list(patients.keys())[idx[0]]
    info = patients[selected_patient_id]

    lbl_survey_title.config(text=f"설문 내역 - {info['이니셜']}")
    lbl_video_title.config(text=f"영상 내역 - {info['이니셜']}")

    if selected_patient_id in surveys:
        survey_frame.pack(fill="both", expand=True)
        btn_start_survey.config(state="disabled")
    else:
        survey_frame.pack_forget()
        btn_start_survey.config(state="normal")

    update_video_labels()

def show_survey_form():
    survey_frame.pack(fill="both", expand=True)
    btn_start_survey.config(state="disabled")

def select_video(video_key):
    if not selected_patient_id:
        return messagebox.showwarning("선택 오류", "환자를 먼저 선택하세요.")
    file_path = filedialog.askopenfilename(
        title=f"{video_key} 선택", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    if file_path:
        videos.setdefault(selected_patient_id, {})[video_key] = {
            "temp_path": file_path,
            "final_path": None
        }
        save_json(VIDEO_FILE, videos)
        update_video_labels()

def get_video_info(file_path):
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        cap.release()
        return f"{os.path.basename(file_path)} ({size_mb:.1f}MB / {minutes:02}:{seconds:02})"
    except:
        return os.path.basename(file_path)

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
            videos[selected_patient_id][key] = {
                "temp_path": temp_path,
                "final_path": save_path
            }

    save_json(VIDEO_FILE, videos)
    update_video_labels()
    messagebox.showinfo("업로드 완료", "선택된 영상이 로컬에 업로드되었습니다.")

def update_video_labels():
    if not selected_patient_id: return
    for key, lbl in video_labels.items():
        paths = videos.get(selected_patient_id, {}).get(key)
        if paths:
            if paths.get("final_path"):
                lbl.config(text=f"{os.path.basename(paths['final_path'])} (업로드 완료)")
            else:
                lbl.config(text=f"{os.path.basename(paths['temp_path'])} (대기중)")
        else:
            lbl.config(text="파일 없음")

# ==================== GUI ====================
root = tk.Tk()
root.title("환자/설문/영상 관리")
root.geometry("1600x900")

# 전체 2행 2열 레이아웃 (grid 사용)
root.grid_rowconfigure(0, weight=1)   # 설문 내역 확장
root.grid_rowconfigure(1, weight=0)   # 영상 내역은 고정 높이
root.grid_columnconfigure(0, weight=0)  # 환자 목록 고정폭
root.grid_columnconfigure(1, weight=1)  # 설문/영상 넓게

# ---------------- [환자 목록] 좌측 ----------------
frame_patient = tk.LabelFrame(root, text="환자 목록", padx=5, pady=5)
frame_patient.grid(row=0, column=0, rowspan=2, sticky="ns", padx=5, pady=5)

patient_listbox = tk.Listbox(frame_patient, height=30, width=30)
patient_listbox.pack(fill="y", expand=True)
patient_listbox.bind("<<ListboxSelect>>", on_patient_select)

btn_frame = tk.Frame(frame_patient)
btn_frame.pack(pady=5)
tk.Button(btn_frame, text="환자 추가", command=add_patient).pack(side="left", padx=5)
tk.Button(btn_frame, text="환자 삭제", command=delete_patient).pack(side="left", padx=5)

# ---------------- [설문 내역] 우측 상단 ----------------
frame_survey = tk.LabelFrame(root, text="설문 내역", padx=5, pady=5)
frame_survey.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

lbl_survey_title = tk.Label(frame_survey, text="설문 내역 - 선택 없음", font=("Arial", 12, "bold"))
lbl_survey_title.pack(pady=5)

btn_start_survey = tk.Button(frame_survey, text="설문 시작", command=show_survey_form)
btn_start_survey.pack(pady=5)

survey_frame = tk.Frame(frame_survey)
survey_frame.pack_forget()  # 설문 시작 전에는 숨김

# ----- Medication 영역 -----
med_frame = tk.LabelFrame(survey_frame, text="약물(Medication)", padx=5, pady=5)
med_frame.pack(fill="x", pady=5)

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

levodopa_frame = tk.Frame(med_frame)
levodopa_frame.pack(anchor="w", pady=2)
tk.Label(levodopa_frame, text="마지막 복용 후 경과 시간(분):").pack(side="left")
med_minutes_entry = tk.Entry(levodopa_frame, width=5)
med_minutes_entry.pack(side="left", padx=5)

# ----- Dyskinesia 영역 -----
dysk_frame = tk.LabelFrame(survey_frame, text="Dyskinesia Impact", padx=5, pady=5)
dysk_frame.pack(fill="x", pady=5)
dys_present_var = tk.IntVar(value=0)
dys_interfere_var = tk.IntVar(value=0)
tk.Checkbutton(dysk_frame, text="검사 중 Dyskinesia 발생", variable=dys_present_var).pack(anchor="w")
tk.Checkbutton(dysk_frame, text="Dyskinesia가 평가에 영향", variable=dys_interfere_var).pack(anchor="w")

# ----- 18문항 점수 입력 -----
question_entries = []
scores_frame = tk.LabelFrame(survey_frame, text="점수 입력 (최대4개)", padx=5, pady=5)
scores_frame.pack(fill="x", pady=5)

for i in range(1, 19):
    row = tk.Frame(scores_frame)
    row.pack(fill="x", pady=2)
    tk.Label(row, text=f"{i}번 문항").pack(side="left")
    entries = []
    for j in range(4):
        e = tk.Entry(row, width=5)
        e.pack(side="left", padx=2)
        entries.append(e)
    question_entries.append(entries)

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
    scores_data = {}
    for idx, entries in enumerate(question_entries, start=1):
        values = [int(e.get()) for e in entries if e.get().strip() != ""]
        scores_data[str(idx)] = values[0] if len(values) == 1 else values

    surveys[selected_patient_id] = {
        "medication": medication_data,
        "dyskinesia_impact": dyskinesia_data,
        "scores": scores_data
    }
    save_json(SURVEY_FILE, surveys)
    update_patient_list()
    messagebox.showinfo("완료", "설문이 저장되었습니다.")

tk.Button(survey_frame, text="설문 제출", command=submit_survey).pack(pady=5)

# ---------------- [영상 내역] 우측 하단 ----------------
frame_video = tk.LabelFrame(root, text="영상 내역", padx=5, pady=5)
frame_video.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

lbl_video_title = tk.Label(frame_video, text="영상 내역 - 선택 없음", font=("Arial", 12, "bold"))
lbl_video_title.pack(pady=5)

video_labels = {}
video_keys = ["1번영상", "2번영상", "3번영상", "손글씨영상"]

for key in video_keys:
    row = tk.Frame(frame_video)
    row.pack(fill="x", pady=3)
    tk.Button(row, text=f"{key} 선택", command=lambda k=key: select_video(k)).pack(side="left")
    lbl = tk.Label(row, text="파일 없음", width=80, anchor="w")
    lbl.pack(side="left", padx=5)
    video_labels[key] = lbl

tk.Button(frame_video, text="영상 업로드", command=upload_videos).pack(pady=10)


update_patient_list()
root.mainloop()
