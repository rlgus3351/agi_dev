import customtkinter as ctk
import tkinter as tk
from form import HealthSurveyForm
from test import GenericSurveyForm
import sys
import os
from datetime import datetime
from tkinter import messagebox
import requests
from CTkMessagebox import CTkMessagebox

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.patient_api import add_patient, delete_patient, fetch_patients  # ✅ import

API_URL = "http://localhost:50001"

def check_server_status():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            r2 = requests.get(f"{API_URL}/health/db", timeout=5)
            if r2.status_code == 200:
                return "OK"
            else:
                return "DB_FAIL"
        else:
            return "API_FAIL"
    except Exception as e:
        return "API_FAIL"

def run_check():
    status = check_server_status()
    if status == "OK":
        CTkMessagebox(title="확인", message="서버 연결 성공", icon="check")
    elif status == "DB_FAIL":
        CTkMessagebox(title="오류", message="DB 연결 오류. 관리자에게 문의하세요.", icon="cancel")
    else:
        CTkMessagebox(title="오류", message="API 서버 연결 실패. 네트워크/방화벽 확인 필요", icon="cancel")












# ---------------- [CTk 기본 설정] ----------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("우울증 입력 프로그램")
root.geometry("1100x700")

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=0)
root.grid_columnconfigure(1, weight=1)

# ---------------- 선택 상태 ----------------
selected_patient = None
selected_row = None  # 현재 선택된 행 Frame 저장

# ---------------- [환자 목록 - 테이블] ----------------
frame_patient = ctk.CTkFrame(root)
frame_patient.grid(row=0, column=0, rowspan=2, sticky="ns", padx=5, pady=5)

ctk.CTkLabel(frame_patient, text="환자 목록", font=("", 14, "bold")).pack(pady=5)

# 테이블 헤더
header_frame = ctk.CTkFrame(frame_patient)
header_frame.pack(fill="x")

headers = ["이니셜", "생년월일", "성별", "등록일시", "관리"]
widths = [80, 100, 60, 150, 60]  # 각 컬럼 너비

for i, (h, w) in enumerate(zip(headers, widths)):
    lbl = ctk.CTkLabel(header_frame, text=h, font=("", 13, "bold"), width=w, anchor="w")
    lbl.grid(row=0, column=i, padx=5, pady=5, sticky="w")

# 테이블 데이터 영역
table_frame = ctk.CTkFrame(frame_patient)
table_frame.pack(fill="both", expand=True)


# ---------------- 선택 상태 ----------------

def on_select_patient(patient, row_frame):
    global selected_patient, selected_row
    print(selected_patient)
    print(patient)
    # 이전 선택된 행 색상 원복
    if selected_row:
        selected_row.configure(fg_color="transparent")

    # 새 행 하이라이트
    row_frame.configure(fg_color="#D0E8FF")

    selected_patient = patient
    selected_row = row_frame
    show_selected_patient()
    show_survey_buttons()  # 버튼 활성화

def load_patients_table():
    for widget in table_frame.winfo_children():
        widget.destroy()

    patients = fetch_patients()
    for r, patient in enumerate(patients):
        initials = patient.get("patient_initials") or "이니셜 없음"
        birth = patient.get("birth_date") or "생년월일 없음"
        gender = patient.get("gender") or "?"
        created_ts = patient.get("created_ts")

        # 날짜 포맷 변환
        if birth != "생년월일 없음":
            try:
                birth = datetime.strptime(birth, "%Y-%m-%d").strftime("%Y-%m-%d")
            except Exception:
                pass

        if created_ts:
            try:
                created_ts = datetime.strptime(created_ts, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d %H:%M")
            except Exception:
                try:
                    created_ts = datetime.strptime(created_ts, "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass
        else:
            created_ts = "?"

        # 선택된 환자면 배경 파란색
        is_selected = selected_patient and selected_patient["patient_id"] == patient["patient_id"]
        bg_color = "#D0E8FF" if is_selected else "transparent"

        # 한 행 프레임
        row_frame = ctk.CTkFrame(table_frame, fg_color=bg_color)
        row_frame.grid(row=r, column=0, columnspan=5, sticky="ew", pady=1)

        # ✅ 행 전체 클릭 이벤트 (Label에는 이벤트 없음)
        row_frame.bind("<Button-1>", lambda e, p=patient, rf=row_frame: on_select_patient(p, rf))

        # 컬럼 표시
        ctk.CTkLabel(row_frame, text=initials, width=widths[0], anchor="w").grid(row=0, column=0, padx=5, pady=3)
        ctk.CTkLabel(row_frame, text=birth, width=widths[1], anchor="w").grid(row=0, column=1, padx=5, pady=3)
        ctk.CTkLabel(row_frame, text=gender, width=widths[2], anchor="w").grid(row=0, column=2, padx=5, pady=3)
        ctk.CTkLabel(row_frame, text=created_ts, width=widths[3], anchor="w").grid(row=0, column=3, padx=5, pady=3)

        # 삭제 버튼
        def make_delete_func(pid):
            return lambda: (delete_patient(pid), load_patients_table())

        ctk.CTkButton(
            row_frame,
            text="🗑",
            fg_color="transparent",
            hover_color="#FFCCCC",
            text_color="red",
            width=widths[4],
            command=make_delete_func(patient["patient_id"])
        ).grid(row=0, column=4, padx=5, pady=3)

# 첫 로드
load_patients_table()

# 버튼 영역
btn_frame = ctk.CTkFrame(frame_patient)
btn_frame.pack(pady=5)
ctk.CTkButton(btn_frame, text="환자 추가", font=("", 14),
              command=lambda: open_add_patient()).pack(side="left", padx=5)

# ---------------- [설문 내역] ----------------
frame_survey = ctk.CTkFrame(root)
frame_survey.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

lbl_survey_title = ctk.CTkLabel(frame_survey, text="설문 내역", font=("", 16, "bold"))
lbl_survey_title.pack(pady=10)

# 선택된 환자 정보 표시
selected_info_label = ctk.CTkLabel(frame_survey, text="환자를 선택해주세요.", font=("", 14, "italic"), text_color="gray")
selected_info_label.pack(pady=5)

# 설문 점수 요약 표시 프레임
score_frame = ctk.CTkFrame(frame_survey)
score_frame.pack(fill="x", pady=5)

def show_selected_patient():
    """오른쪽 상단에 선택된 환자 정보 표시"""
    if not selected_patient:
        selected_info_label.configure(text="환자를 선택해주세요.", font=("", 14, "italic"), text_color="gray")
    else:
        initials = selected_patient.get("patient_initials", "?")
        birth = selected_patient.get("birth_date", "?")
        gender = selected_patient.get("gender", "?")
        selected_info_label.configure(
            text=f"선택된 환자: {initials} / {birth} / {gender}",
            font=("", 14, "bold"),
            text_color="black"
        )

def show_empty_state():
    for widget in score_frame.winfo_children():
        widget.destroy()
    ctk.CTkLabel(score_frame, text="환자를 선택해주세요.", font=("", 14, "italic"), text_color="gray").pack(pady=20)

def show_survey_buttons():
    for widget in score_frame.winfo_children():
        widget.destroy()
    ctk.CTkButton(score_frame, text="기초 평가", command=lambda: open_survey_modal("기초 평가")).pack(pady=5)
    ctk.CTkButton(score_frame, text="정서 설문지 입력", command=lambda: open_phq9("정서 설문지")).pack(pady=5)
    ctk.CTkButton(score_frame, text="수면 설문지 입력", command=lambda: open_madras("수면 설문지")).pack(pady=5)

# ---------------- [환자 추가 팝업] ----------------
def open_add_patient():
    modal = ctk.CTkToplevel(root)
    modal.title("환자 추가")
    modal.geometry("400x400")
    modal.transient(root)
    modal.grab_set()

    initials_var = tk.StringVar()
    birth_var = tk.StringVar()
    gender_var = tk.StringVar(value="남성")  # 기본값

    ctk.CTkLabel(modal, text="이니셜").pack(pady=5)
    ctk.CTkEntry(modal, textvariable=initials_var).pack(pady=5)

    ctk.CTkLabel(modal, text="생년월일 (YYYY-MM-DD)").pack(pady=5)
    ctk.CTkEntry(modal, textvariable=birth_var).pack(pady=5)

    ctk.CTkLabel(modal, text="성별").pack(pady=5)
    gender_frame = ctk.CTkFrame(modal)
    gender_frame.pack(pady=5)

    ctk.CTkRadioButton(gender_frame, text="남성", variable=gender_var, value="남성").pack(side="left", padx=5)
    ctk.CTkRadioButton(gender_frame, text="여성", variable=gender_var, value="여성").pack(side="left", padx=5)
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
        add_patient(payload)
        load_patients_table()
        modal.destroy()

    ctk.CTkButton(modal, text="등록", command=submit_patient).pack(pady=20)

# ---------------- [설문 입력 모달들] ----------------
def open_survey_modal(title):
    if not selected_patient:
        messagebox.showwarning("경고", "먼저 환자를 선택해주세요.")
        return
    modal = ctk.CTkToplevel(root)
    modal.title(title)
    modal.geometry("700x800")
    modal.transient(root)
    modal.grab_set()
    modal.focus()
    form = HealthSurveyForm(modal)
    form.pack(fill="both", expand=True, padx=10, pady=10)

def open_phq9(title="PHQ-9"):
    if not selected_patient:
        messagebox.showwarning("경고", "먼저 환자를 선택해주세요.")
        return
    modal = ctk.CTkToplevel(root)
    modal.title(title)
    modal.geometry("1000x600")
    modal.transient(root)
    modal.grab_set()
    modal.focus()
    form = GenericSurveyForm(modal, json_file="C:/TeamGit/agi_dev/dataInputProgram/src/depression/form/MADRS.json")
    form.pack(fill="both", expand=True, padx=20, pady=20)

def open_madras(title="Madras"):
    if not selected_patient:
        messagebox.showwarning("경고", "먼저 환자를 선택해주세요.")
        return
    modal = ctk.CTkToplevel(root)
    modal.title(title)
    modal.geometry("1000x600")
    modal.transient(root)
    modal.grab_set()
    modal.focus()
    form = GenericSurveyForm(modal, json_file="C:/TeamGit/agi_dev/dataInputProgram/src/depression/form/sleep_form/PSQI.json")
    form.pack(fill="both", expand=True, padx=20, pady=20)

# ---------------- 초기 상태 ----------------
show_empty_state()

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
