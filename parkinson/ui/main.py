
import customtkinter as ctk
import tkinter as tk
import sys
import os
from datetime import datetime
from tkinter import messagebox,filedialog
import requests
from CTkMessagebox import CTkMessagebox
import threading
import time



# ✅ sys.path 수정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from api.patient_api import add_patient, delete_patient, fetch_patients  # ✅ import
from api.item_api import fetch_items  # ✅ 꼭 추가!
from utils.loader import run_with_loading,run_with_loading_popup  # 로딩 유틸 추가 import
from api.health_api import check_server_status
from config import HEALTH_URL, INSTITUTION  # config에서 가져옴
from form.survey import HealthSurveyForm


JSON_FILE = os.path.join(CURRENT_DIR, '..', 'form', 'mobility.json')
JSON_FILE = os.path.abspath(JSON_FILE)  # ← 절대경로로 변환 (안전)

API_URL = "http://localhost:30000"
INSTITUTION = "CNU"
items_cache = {}  # 환자별 수집 항목 캐시


# ---------------- 서버 체크 ----------------
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
    except Exception:
        return "API_FAIL"

def show_loading_popup(parent):
    """로딩창 생성"""
    popup = ctk.CTkToplevel(parent)
    popup.title("서버 확인 중...")
    popup.geometry("300x100")
    popup.attributes("-topmost", True)  # 항상 위
    popup.resizable(False, False)
    ctk.CTkLabel(popup, text="서버 상태 확인 중...\n잠시만 기다려주세요.").pack(pady=20)
    return popup

def run_check(parent):
    """서버 상태 확인 후 메시지박스 표시 (로딩 포함)"""
    loading = show_loading_popup(parent)

    def background_check():
        time.sleep(3)  # 로딩 효과용 대기 시간
        status = check_server_status()

        def show_result():
            loading.destroy()
            if status == "OK":
                CTkMessagebox(title="확인", message="서버 연결 성공", icon="check")
                load_patients_table()
            elif status == "DB_FAIL":
                CTkMessagebox(title="오류", message="DB 연결 오류. 관리자에게 문의하세요.", icon="cancel")
                show_server_error()
            else:
                CTkMessagebox(title="오류", message="API 서버 연결 실패. 네트워크/방화벽 확인 필요", icon="cancel")
                show_server_error()

        parent.after(0, show_result)

    threading.Thread(target=background_check, daemon=True).start()

def init_program():
    """프로그램 첫 실행 시 서버 상태 확인 (로딩 오버레이 포함)"""
    def after_check(result):
        if result == "OK":
            CTkMessagebox(title="성공", message="서버 연결 성공", icon="check")
            load_patients_table()
        elif result == "DB_FAIL":
            CTkMessagebox(title="오류", message="DB 연결 실패. 관리자에게 문의하세요.", icon="cancel")
            show_server_error()
        else:
            CTkMessagebox(title="오류", message="API 서버 연결 실패. 네트워크 확인 필요", icon="cancel")
            show_server_error()

    run_with_loading(
        parent_frame=root,
        fetch_function=check_server_status,
        callback=after_check,
        loading_text="서버 상태 확인 중입니다..."
    )


def show_server_error():
    """서버 연결 실패 시 테이블 영역에 표시"""
    for widget in table_frame.winfo_children():
        widget.destroy()
    ctk.CTkLabel(table_frame, text="서버와 연결할 수 없습니다.",
                 font=("", 14, "italic"), text_color="red").pack(pady=20)

# ---------------- [CTk 기본 설정] ----------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("파킨슨병 입력 프로그램")
root.geometry("1100x700")

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=0)
root.grid_columnconfigure(1, weight=1)

selected_patient = None
selected_row = None

frame_patient = ctk.CTkFrame(root)
frame_patient.grid(row=0, column=0, rowspan=2, sticky="ns", padx=5, pady=5)

ctk.CTkLabel(frame_patient, text="환자 목록", font=("", 14, "bold")).pack(pady=5)

header_frame = ctk.CTkFrame(frame_patient)
header_frame.pack(fill="x")

headers = ["이니셜", "생년월일", "성별", "등록일시", "관리"]
widths = [80, 100, 60, 150, 60]

for i, (h, w) in enumerate(zip(headers, widths)):
    lbl = ctk.CTkLabel(header_frame, text=h, font=("", 13, "bold"), width=w, anchor="w")
    lbl.grid(row=0, column=i, padx=5, pady=5, sticky="w")

table_frame = ctk.CTkFrame(frame_patient)
table_frame.pack(fill="both", expand=True)

def handle_items_loaded(result):
    if isinstance(result, Exception):
        messagebox.showerror("에러", f"수집 항목 로딩 실패:\n{result}")
        return

    print(f"✅ 수집 항목 {len(result)}개 불러옴")

    for widget in score_frame.winfo_children():
        widget.destroy()

    if not result:
        messagebox.showinfo("안내", "해당 환자는 아직 수집된 데이터가 없습니다.")
        ctk.CTkLabel(
            score_frame,
            text="수집된 데이터가 없습니다.",
            font=("", 13, "italic"),
            text_color="gray"
        ).pack(pady=20)
         # 설문지 입력 버튼
        ctk.CTkButton(
            score_frame,
            text="📋 설문지 데이터 입력",
            command=open_survey_input,  # 아래에 정의할 함수
            fg_color="#4A90E2",
            hover_color="#357ABD",
            text_color="white"
        ).pack(pady=10)
        return

    # 항목 있는 경우 출력
    ctk.CTkLabel(
        score_frame,
        text="📦 수집 항목 목록",
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(pady=(5, 0))

    for item in result:
        label_text = f"[{item['data_category']}] {item['data_type']} (순서: {item['seq']})"
        ctk.CTkLabel(
            score_frame,
            text=label_text,
            anchor="w",
            justify="left"
        ).pack(fill="x", padx=10, pady=2)


def handle_files_loaded(result):
    for widget in upload_frame.winfo_children():
        widget.destroy()

    if isinstance(result, Exception):
        ctk.CTkLabel(upload_frame, text="파일 불러오기 실패", text_color="red").pack(pady=10)
        return

    if not result:
        ctk.CTkLabel(upload_frame, text="업로드된 파일이 없습니다.", text_color="gray").pack(pady=10)
        ctk.CTkButton(upload_frame, text="📤 파일 업로드", command=open_upload_modal).pack(pady=5)
        return

    ctk.CTkLabel(upload_frame, text="🎥 업로드된 영상", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(5, 0))

    for idx, file in enumerate(result[:4]):
        label_text = f"{idx+1}번 영상: {file.get('filename', 'Unnamed')}"
        ctk.CTkLabel(upload_frame, text=label_text, anchor="w").pack(fill="x", padx=10, pady=2)


def open_upload_modal():
    if not selected_patient:
        messagebox.showwarning("환자 선택 필요", "먼저 환자를 선택해주세요.")
        return

    modal = ctk.CTkToplevel()
    modal.title("파일 업로드")
    modal.geometry("400x200")
    modal.grab_set()

    ctk.CTkLabel(modal, text="업로드할 영상 파일을 선택하세요:").pack(pady=10)

    file_path_var = ctk.StringVar()

    file_entry = ctk.CTkEntry(modal, textvariable=file_path_var, width=250)
    file_entry.pack(pady=5)

    def browse_file():
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")]
        )
        if file_path:
            file_path_var.set(file_path)

    ctk.CTkButton(modal, text="파일 찾기", command=browse_file).pack(pady=5)

    def upload_file():
        file_path = file_path_var.get()
        if not file_path or not os.path.isfile(file_path):
            messagebox.showerror("오류", "유효한 파일을 선택하세요.")
            return

        try:
            files = {"file": open(file_path, "rb")}
            url = f"http://localhost:8000/files/{selected_patient['patient_id']}/upload"
            response = requests.post(url, files=files)
            if response.status_code == 200:
                messagebox.showinfo("성공", "파일 업로드가 완료되었습니다.")
                modal.destroy()
                # 필요 시 업로드 후 파일 목록 새로고침 추가 가능
            else:
                messagebox.showerror("실패", f"업로드 실패: {response.status_code}\n{response.text}")
        except Exception as e:
            messagebox.showerror("에러", f"업로드 중 오류 발생: {e}")

    ctk.CTkButton(modal, text="업로드", command=upload_file).pack(pady=10)


def on_select_patient(patient, row_frame):
    global selected_patient, selected_row

    # 기존 선택 해제
    if selected_row:
        selected_row.configure(fg_color="transparent")

    # 새 선택 표시
    row_frame.configure(fg_color="#D0E8FF")
    selected_patient = patient
    selected_row = row_frame
    show_selected_patient()

    pid = patient["patient_id"]

    # ✅ 항목이 이미 캐시에 있으면 바로 처리
    if pid in items_cache:
        handle_items_loaded(items_cache[pid])
        return

    # ✅ 스레드로 비동기 요청
    def fetch_in_background():
        try:
            items = fetch_items(pid)
        except Exception as e:
            items = e  # 예외 전달
        # UI 업데이트는 main thread에서
        def update_ui():
            if not isinstance(items, Exception):
                items_cache[pid] = items
            handle_items_loaded(items)
        root.after(0, update_ui)

    threading.Thread(target=fetch_in_background, daemon=True).start()


def load_patients_table():
    for widget in table_frame.winfo_children():
        widget.destroy()

    try:
        patients = fetch_patients(INSTITUTION)
    except Exception as e:
        show_server_error()
        return

    def on_row_click(event, p, rf):
        # 🗑 버튼 클릭은 무시
        if event.widget.cget("text") == "🗑":
            return
        on_select_patient(p, rf)

    for r, patient in enumerate(patients):
        initials = patient.get("patient_initials") or "이니셜 없음"
        birth = patient.get("birth_date") or "생년월일 없음"
        gender = patient.get("gender") or "?"
        created_ts = patient.get("created_ts")

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

        is_selected = selected_patient and selected_patient["patient_id"] == patient["patient_id"]
        bg_color = "#C4E1FF" if is_selected else "transparent"
        text_weight = "bold" if is_selected else "normal"
        text_color = "black" if is_selected else "gray20"
        border_width = 2 if is_selected else 0

        row_frame = ctk.CTkFrame(table_frame, fg_color=bg_color, corner_radius=8, border_width=border_width)
        row_frame.grid(row=r, column=0, columnspan=5, sticky="ew", pady=3, padx=5)
        row_frame.configure(cursor="hand2")
        row_frame.bind("<Button-1>", lambda e, p=patient, rf=row_frame: on_row_click(e, p, rf))

        # 라벨 생성 + 클릭 바인딩
        def create_label(col, text, width):
            lbl = ctk.CTkLabel(row_frame, text=text, width=width, anchor="w",
                               font=("", 13, text_weight), text_color=text_color)
            lbl.grid(row=0, column=col, padx=5, pady=3, sticky="w")
            lbl.bind("<Button-1>", lambda e, p=patient, rf=row_frame: on_row_click(e, p, rf))

        create_label(0, initials, widths[0])
        create_label(1, birth, widths[1])
        create_label(2, gender, widths[2])
        create_label(3, created_ts, widths[3])

        def make_delete_func(pid):
            return lambda: (delete_patient(pid, INSTITUTION), load_patients_table())

        ctk.CTkButton(
            row_frame,
            text="🗑",
            fg_color="transparent",
            hover_color="#FFCCCC",
            text_color="red",
            width=widths[4],
            command=make_delete_func(patient["patient_id"])
        ).grid(row=0, column=4, padx=5, pady=3)


btn_frame = ctk.CTkFrame(frame_patient)
btn_frame.pack(pady=5)
ctk.CTkButton(btn_frame, text="환자 추가", font=("", 14),
              command=lambda: open_add_patient()).pack(side="left", padx=5)

frame_survey = ctk.CTkFrame(root)
frame_survey.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

lbl_survey_title = ctk.CTkLabel(frame_survey, text="설문 내역", font=("", 16, "bold"))
lbl_survey_title.pack(pady=10)

selected_info_label = ctk.CTkLabel(frame_survey, text="환자를 선택해주세요.", font=("", 14, "italic"), text_color="gray")
selected_info_label.pack(pady=5)

score_frame = ctk.CTkFrame(frame_survey)
score_frame.pack(fill="x", pady=5)

def show_selected_patient():
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
    ctk.CTkButton(score_frame, text="기초 평가").pack(pady=5)
    ctk.CTkButton(score_frame, text="정서 설문지 입력").pack(pady=5)
    ctk.CTkButton(score_frame, text="수면 설문지 입력").pack(pady=5)

def open_add_patient():
    modal = ctk.CTkToplevel(root)
    modal.title("환자 추가")
    modal.geometry("400x400")
    modal.transient(root)
    modal.grab_set()

    initials_var = tk.StringVar()
    birth_var = tk.StringVar()
    gender_var = tk.StringVar(value="남성")

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
            "institution": INSTITUTION,
            "gender": gender_var.get(),
            "is_data_complete": False,
            "completion_date": None
        }
        add_patient(payload, INSTITUTION)
        load_patients_table()
        modal.destroy()

    ctk.CTkButton(modal, text="등록", command=submit_patient).pack(pady=20)

def open_survey_input():
    if not selected_patient:
        messagebox.showwarning("환자 선택 필요", "먼저 환자를 선택해주세요.")
        return

    modal = ctk.CTkToplevel(root)
    modal.title("운동성 설문지 입력")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

# 화면의 70% 너비, 90% 높이로 설정
    width = int(screen_width * 0.7)
    height = int(screen_height * 0.9)

    modal.geometry(f"{width}x{height}")
    modal.grab_set()

    # 상단 타이틀
    patient_uuid = selected_patient["patient_id"]
    initials = selected_patient.get("patient_initials", "?")

    ctk.CTkLabel(modal, text=f"📝 운동성 설문지 입력 - {initials}", font=("", 16, "bold")).pack(pady=10)

    # 설문 폼 불러오기
    survey_form = HealthSurveyForm(modal, 
                                   json_file=JSON_FILE, 
                                   patient_id=patient_uuid) # ⬅️ UUID 전달
    survey_form.pack(fill="both", expand=True, padx=10, pady=10)


frame_video = ctk.CTkFrame(root)
frame_video.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

lbl_video_title = ctk.CTkLabel(frame_video, text="파일 업로드", font=("", 16, "bold"))
lbl_video_title.pack(pady=10)

upload_frame = ctk.CTkFrame(frame_video)
upload_frame.pack(pady=10)

ctk.CTkButton(upload_frame, text="영상 업로드", width=150).pack(side="left", padx=10)
ctk.CTkButton(upload_frame, text="기타 파일 업로드", width=150).pack(side="left", padx=10)

show_empty_state()
root.after(100, init_program)
root.mainloop()
