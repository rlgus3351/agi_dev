import customtkinter as ctk
import tkinter as tk
from form import HealthSurveyForm
from test import GenericSurveyForm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.patient_api import load_patients, add_patient, delete_patient


# ... CTk UI 생성 이후





ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("우울증 입력 프로그램")
root.geometry("1000x700")

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=0)
root.grid_columnconfigure(1, weight=1)

# ---------------- [환자 목록] ----------------
frame_patient = ctk.CTkFrame(root)
frame_patient.grid(row=0, column=0, rowspan=2, sticky="ns", padx=5, pady=5)

ctk.CTkLabel(frame_patient, text="환자 목록", font=("", 14, "normal")).pack(pady=5)
patient_listbox = tk.Listbox(frame_patient, height=30, width=30)
patient_listbox.pack(fill="y", expand=True)
load_patients(patient_listbox)

btn_frame = ctk.CTkFrame(frame_patient)
btn_frame.pack(pady=5)
ctk.CTkButton(btn_frame, text="환자 추가", font=("", 14, "normal"),
              command=lambda: open_add_patient()).pack(side="left", padx=5)
ctk.CTkButton(btn_frame, text="환자 삭제", font=("", 14, "normal"),command=lambda:on_delete_patient()).pack(side="left", padx=5)

# ---------------- [설문 내역] ----------------
frame_survey = ctk.CTkFrame(root)
frame_survey.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

lbl_survey_title = ctk.CTkLabel(frame_survey, text="설문 내역", font=("", 16, "bold"))
lbl_survey_title.pack(pady=10)

# 설문 점수 요약 표시 프레임
score_frame = ctk.CTkFrame(frame_survey)
score_frame.pack(fill="x", pady=5)


def open_add_patient():
    modal = ctk.CTkToplevel(root)
    modal.title("환자 추가")
    modal.geometry("400x400")
    modal.transient(root)
    modal.grab_set()

    initials_var = tk.StringVar()
    birth_var = tk.StringVar()
    institution_var = tk.StringVar()
    gender_var = tk.StringVar(value="남성")  # 기본값

    ctk.CTkLabel(modal, text="이니셜").pack(pady=5)
    ctk.CTkEntry(modal, textvariable=initials_var).pack(pady=5)

    ctk.CTkLabel(modal, text="생년월일 (YYYY-MM-DD)").pack(pady=5)
    ctk.CTkEntry(modal, textvariable=birth_var).pack(pady=5)

    ctk.CTkLabel(modal, text="성별").pack(pady=5)
    gender_frame = ctk.CTkFrame(modal)
    gender_frame.pack(pady=5)

    ctk.CTkRadioButton(gender_frame, text="남성", variable=gender_var, value="f").pack(side="left", padx=5)
    ctk.CTkRadioButton(gender_frame, text="여성", variable=gender_var, value="m").pack(side="left", padx=5)
    ctk.CTkRadioButton(gender_frame, text="기타", variable=gender_var, value="기타").pack(side="left", padx=5)

    def submit_patient():
        payload = {
            "patient_initials": initials_var.get(),
            "birth_date": birth_var.get() or None,
            "institution": "조선대학교 병원",
            "gender": gender_var.get(),
            "is_data_complete": False,
            "completion_date": None
        }
        add_patient(payload, patient_listbox)  # api 함수 호출
        modal.destroy()

    ctk.CTkButton(modal, text="등록", command=submit_patient).pack(pady=20)

def on_delete_patient():
    selection = patient_listbox.curselection()
    if not selection:
        tk.messagebox.showwarning("경고", "삭제할 환자를 선택하세요.")
        return

    idx = selection[0]
    patient_id = patient_listbox.patient_map.get(idx)

    if patient_id:
        delete_patient(patient_id, patient_listbox)


# 예시: 점수 없으면 버튼 3개 표시
def show_survey_buttons():
    for widget in score_frame.winfo_children():
        widget.destroy()
    ctk.CTkButton(score_frame, text="기초 평가", command=lambda: open_survey_modal("기초 평가")).pack(pady=5)
    ctk.CTkButton(score_frame, text="정서 설문지 입력", command=lambda: open_phq9("정서 설문지")).pack(pady=5)
    ctk.CTkButton(score_frame, text="수면 설문지 입력", command=lambda: open_madras("수면 설문지")).pack(pady=5)

def show_survey_scores(scores: dict):
    for widget in score_frame.winfo_children():
        widget.destroy()
    for name, val in scores.items():
        ctk.CTkLabel(score_frame, text=f"{name}: {val} 점", font=("", 14)).pack(anchor="w", padx=10, pady=2)

def open_survey_modal(title):
    modal = ctk.CTkToplevel(root)
    modal.title(title)
    modal.geometry("700x800")
    modal.transient(root)
    modal.grab_set()
    modal.focus()

    form = HealthSurveyForm(modal)   # ✅ 프레임 embed
    form.pack(fill="both", expand=True, padx=10, pady=10)

def open_phq9(title="PHQ-9"):
    modal = ctk.CTkToplevel(root)
    modal.title(title)
    modal.geometry("1000x600")
    modal.transient(root)
    modal.grab_set()
    modal.focus()

    form = GenericSurveyForm(modal, json_file="C:/TeamGit/agi_dev/dataInputProgram/src/depression/form/MADRS.json")  # ✅ 프레임 embed
    form.pack(fill="both", expand=True, padx=20, pady=20)

def open_madras(title="Madras"):
    modal = ctk.CTkToplevel(root)
    modal.title(title)
    modal.geometry("1000x600")
    modal.transient(root)
    modal.grab_set()
    modal.focus()

    form = GenericSurveyForm(modal, json_file="C:/TeamGit/agi_dev/dataInputProgram/src/depression/form/sleep_form/PSQI.json")  # ✅ 프레임 embed
    form.pack(fill="both", expand=True, padx=20, pady=20)
# 처음에는 버튼 3개 보이게
show_survey_buttons()

# ---------------- [영상 / 파일 업로드] ----------------
frame_video = ctk.CTkFrame(root)                                      
frame_video.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

lbl_video_title = ctk.CTkLabel(frame_video, text="파일 업로드", font=("", 16, "bold"))
lbl_video_title.pack(pady=10)

upload_frame = ctk.CTkFrame(frame_video)
upload_frame.pack(pady=10)

ctk.CTkButton(upload_frame, text="영상 업로드", width=150).pack(side="left", padx=10)
ctk.CTkButton(upload_frame, text="기타 파일 업로드", width=150).pack(side="left", padx=10)

# ---------------- 프로그램 시작 ----------------
root.mainloop()
