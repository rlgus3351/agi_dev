import tkinter as tk
from tkinter import messagebox, filedialog
import json, os, uuid, re
import cv2
import shutil

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
    """환자 목록 갱신"""
    patient_listbox.delete(0, tk.END)
    for pid, info in patients.items():
        idx = patient_listbox.size()
        patient_listbox.insert(tk.END, f"{info['이니셜']} / {info['생년월일']} / {info['성별']}")
        patient_listbox.itemconfig(idx, {'fg': 'green' if pid in surveys else 'red'})

def add_patient():
    """환자 추가 팝업"""
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
    confirm = messagebox.askyesno(
        "확인",
        f"정말 {info['이니셜']} 환자와 관련된 설문/영상 데이터를 모두 삭제할까요?"
    )
    if not confirm:
        return

    # 1) 환자 삭제
    patients.pop(selected_patient_id, None)
    # 2) 설문 삭제
    surveys.pop(selected_patient_id, None)
    # 3) 영상 삭제
    videos.pop(selected_patient_id, None)

    # JSON 저장
    save_json(PATIENT_FILE, patients)
    save_json(SURVEY_FILE, surveys)
    save_json(VIDEO_FILE, videos)

    # 선택 초기화 및 목록 갱신
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
    """환자 선택 시 정보 갱신"""
    global selected_patient_id
    idx = patient_listbox.curselection()
    if not idx: return
    selected_patient_id = list(patients.keys())[idx[0]]
    info = patients[selected_patient_id]

    lbl_survey_title.config(text=f"설문 내역 - {info['이니셜']}")
    lbl_video_title.config(text=f"영상 내역 - {info['이니셜']}")

    # 설문 상태
    if selected_patient_id in surveys:
        mood_var.set(surveys[selected_patient_id]["기분"])
        pain_scale.set(surveys[selected_patient_id]["통증"])
        survey_frame.pack(fill="both", expand=True)
        btn_start_survey.config(state="disabled")
    else:
        mood_var.set("보통")
        pain_scale.set(0)
        survey_frame.pack_forget()
        btn_start_survey.config(state="normal")

    # 영상 상태 갱신
    update_video_labels()

def show_survey_form():
    survey_frame.pack(fill="both", expand=True)
    btn_start_survey.config(state="disabled")

def submit_survey():
    """설문 제출"""
    if not selected_patient_id:
        return messagebox.showwarning("선택 오류", "환자를 먼저 선택하세요.")
    surveys[selected_patient_id] = {"기분": mood_var.get(), "통증": pain_scale.get()}
    save_json(SURVEY_FILE, surveys)
    update_patient_list()
    messagebox.showinfo("완료", "설문이 저장되었습니다.")

def select_video(video_key):
    if not selected_patient_id:
        return messagebox.showwarning("선택 오류", "환자를 먼저 선택하세요.")
    
    file_path = filedialog.askopenfilename(
        title=f"{video_key} 선택", 
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    if file_path:
        videos.setdefault(selected_patient_id, {})[video_key] = {
            "temp_path": file_path,
            "final_path": None
        }
        save_json(VIDEO_FILE, videos)
        update_video_labels()


def get_video_info(file_path):
    """파일명 + 크기(MB) + 길이(mm:ss)"""
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
            # JSON 갱신
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
root.geometry("1200x500")

# [환자 목록] 섹션
frame_patient = tk.LabelFrame(root, text="환자 목록", padx=5, pady=5)
frame_patient.pack(side="left", fill="y", padx=5, pady=5)

patient_listbox = tk.Listbox(frame_patient, height=20, width=30)
patient_listbox.pack(fill="y", expand=True)
patient_listbox.bind("<<ListboxSelect>>", on_patient_select)
btn_frame = tk.Frame(frame_patient)
btn_frame.pack(pady=5)

tk.Button(btn_frame, text="환자 추가", command=add_patient).pack(side="left", padx=5)
tk.Button(btn_frame, text="환자 삭제", command=delete_patient).pack(side="left", padx=5)


# [설문 내역] 섹션
frame_survey = tk.LabelFrame(root, text="설문 내역", padx=5, pady=5)
frame_survey.pack(side="left", fill="both", expand=True, padx=5, pady=5)

lbl_survey_title = tk.Label(frame_survey, text="설문 내역 - 선택 없음", font=("Arial", 12, "bold"))
lbl_survey_title.pack(pady=5)

btn_start_survey = tk.Button(frame_survey, text="설문 시작", command=show_survey_form)
btn_start_survey.pack(pady=5)









survey_frame = tk.Frame(frame_survey)
survey_frame.pack_forget()

tk.Label(survey_frame, text="문항1: 현재 기분은?").pack(anchor="w")
mood_var = tk.StringVar(value="보통")
for val in ["좋음", "보통", "나쁨"]:
    tk.Radiobutton(survey_frame, text=val, variable=mood_var, value=val).pack(anchor="w")

tk.Label(survey_frame, text="문항2: 통증 정도 (0~10)").pack(anchor="w")
pain_scale = tk.Scale(survey_frame, from_=0, to=10, orient="horizontal")
pain_scale.pack(anchor="w")
tk.Button(survey_frame, text="설문 제출", command=submit_survey).pack(pady=5)

# [영상 내역] 섹션
frame_video = tk.LabelFrame(root, text="영상 내역", padx=5, pady=5)
frame_video.pack(side="left", fill="both", expand=True, padx=5, pady=5)

lbl_video_title = tk.Label(frame_video, text="영상 내역 - 선택 없음", font=("Arial", 12, "bold"))
lbl_video_title.pack(pady=5)

video_labels = {}
video_keys = ["1번영상", "2번영상", "3번영상", "손글씨영상"]

for key in video_keys:
    row = tk.Frame(frame_video)
    row.pack(fill="x", pady=3)
    tk.Button(row, text=f"{key} 선택", command=lambda k=key: select_video(k)).pack(side="left")
    lbl = tk.Label(row, text="파일 없음", width=50, anchor="w")
    lbl.pack(side="left", padx=5)
    video_labels[key] = lbl

tk.Button(frame_video, text="영상 업로드", command=upload_videos).pack(pady=10)

update_patient_list()
root.mainloop()
