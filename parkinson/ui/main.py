import customtkinter as ctk
import tkinter as tk
from tkinter import ttk 
import sys
import os
import shutil
from datetime import datetime
from tkinter import messagebox, filedialog
import requests
from CTkMessagebox import CTkMessagebox
import threading
import time
import subprocess
import platform
from tkvideo import tkvideo

# ✅ sys.path 수정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "video")  # ⬅️ 원하는 위치로 정확히 대응됨

sys.path.append(PROJECT_ROOT)

from api.patient_api import add_patient, delete_patient, fetch_patients
from api.item_api import fetch_items,delete_survey_item
from utils.loader import run_with_loading, run_with_loading_popup
from api.health_api import check_server_status
from api.form_api import fetch_mds_answers
from api.video_api import create_new_item_and_get_id, call_api_to_save_video_metadata,fetch_video_metadata_by_item_id,call_api_to_update_video_metadata
# from config import HEALTH_URL, INSTITUTION  # config에서 가져옴 (현재는 하드코딩 사용)
from form.survey import HealthSurveyForm
from utils.videometa import get_video_metadata

from config import API_URL, INSTITUTION , HEALTH_URL

JSON_FILE = os.path.join(CURRENT_DIR, '..', 'form', 'mobility.json')
JSON_FILE = os.path.abspath(JSON_FILE) # ← 절대경로로 변환 (안전)

items_cache = {} # 환자별 수집 항목 캐시


print(HEALTH_URL)
# ---------------- 서버 체크 ----------------
def check_server_status():
    try:
        r = requests.get(f"{HEALTH_URL}", timeout=5)
        if r.status_code == 200:
            r2 = requests.get(f"{HEALTH_URL}/db", timeout=5)
            if r2.status_code == 200:
                return "OK"
            else:
                return "DB_FAIL"
        else:
            return "API_FAIL"
    except Exception:
        return "API_FAIL"

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

def reload_after_close():
    pid = selected_patient["patient_id"]
    if pid in items_cache:
        del items_cache[pid]  # ✅ 캐시 제거
    on_select_patient(selected_patient, selected_row)  # 설문 + 파일 목록 리로드

# ---------------- [CTk 기본 설정] ----------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("파킨슨병 입력 프로그램")
root.geometry("1100x900")

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


def format_mds_answers(raw_answers: list) -> list:
    """
    원시 MDS 답변 데이터를 generate_summary가 처리할 수 있는 구조로 변환합니다.
    (question_id와 answer_value를 추출하고, 다중 답변을 통합합니다.)
    """
    grouped_answers = {}
    
    for ans in raw_answers:
        q_id = ans['question_id']
        answer_value = ans.get('answer_value', 'N/A')
        answer_comp = ans.get('answer_component')
        
        # 1. 'question_id'를 키로 그룹화
        if q_id not in grouped_answers:
            grouped_answers[q_id] = {
                'question_id': q_id,
                'answers': []
            }
        
        # 2. answer_component가 있으면 '컴포넌트:값' 형태로 저장
        if answer_comp:
            answer_str = f"{answer_comp}:{answer_value}"
        else:
            answer_str = answer_value
            
        grouped_answers[q_id]['answers'].append(answer_str)
        
    # 3. 최종 리스트로 변환 (answers 리스트를 하나의 문자열로 결합)
    formatted_list = []
    for q_id, data in grouped_answers.items():
        formatted_list.append({
            'question_id': q_id,
            # 다중 답변은 "LA:1 | LL:1 | Neck:1" 같은 형태로 합칩니다.
            'answer': " | ".join(data['answers']) 
        })
        
    return formatted_list

# ---------------- [새로운 요약 생성 함수] ----------------
def generate_summary(item_data):
    """항목 데이터에서 question_id 1번부터 8번까지의 답변을 추출하여 요약을 생성합니다."""
    
    # 💡 item_data는 fetch_items의 응답에서 온 항목 데이터라고 가정합니다.
    # 설문 답변 데이터는 item_data['questions'] 리스트에 저장되어 있다고 가정합니다.
    questions = item_data.get('questions')
    if not questions:
        return "데이터 없음"
    
    summary_parts = []
    # question_id가 1부터 8인 항목만 추출하여 답변을 요약합니다.
    for q in questions:
        try:
            q_id = int(q.get('question_id', 0))
            if 1 <= q_id <= 8:
                answer = q.get('answer', 'N/A')
                # 답변이 너무 길면 잘라내거나, 타입에 따라 포맷팅합니다.
                if isinstance(answer, (int, float)):
                    summary_parts.append(f"Q{q_id}:{answer}")
                elif isinstance(answer, str) and len(answer) > 10:
                    summary_parts.append(f"Q{q_id}:{answer[:10]}...")
                else:
                    summary_parts.append(f"Q{q_id}:{answer}")
        except ValueError:
            continue
            
    if not summary_parts:
        return "첫 8개 질문 미응답"
        
    return " | ".join(summary_parts)


# ---------------- 수집 항목 표시 로직 변경 (요약 및 버튼 분리) ----------------

# '수정' 버튼 클릭 시 호출될 함수
def open_survey_edit(item_data):
    # '수정'은 기존 데이터가 있으므로 item_data (questions 포함)를 전달합니다.
    open_survey_form(item_data) 


def open_survey_input(item_data):
    # '입력'은 데이터가 없으므로 item_data (항목 기본 정보만)를 전달합니다.
    open_survey_form(item_data)
    
    
def show_survey_items(survey_items):
    """score_frame에 설문 항목 목록을 표시합니다. 데이터 유무와 요약을 표시합니다."""
    
    # 1️⃣ 기존 위젯 초기화
    for widget in score_frame.winfo_children():
        widget.destroy()

    # 2️⃣ 환자 선택 유무 검사
    if not selected_patient:
        ctk.CTkLabel(score_frame, text="환자를 선택해주세요.", font=("", 14, "italic"), text_color="gray").pack(pady=20)
        return

    # 3️⃣ 설문 항목이 없을 때
    if not survey_items:
        ctk.CTkLabel(
            score_frame,
            text="수집된 설문 데이터가 없습니다.",
            font=("", 13, "italic"),
            text_color="gray"
        ).pack(pady=20)

        # ✨ 설문 항목이 없을 때만 “새 항목 등록” 버튼 표시
        new_item_data = {
            'data_category': 'PD',
            'data_type': 'MDS-UPDRS Part 3',
            'seq': 1
        }
        ctk.CTkButton(
            score_frame,
            text="➕ 새 설문 항목 등록 및 입력",
            font=("", 13),
            command=lambda data=new_item_data: open_survey_form(data),
            fg_color="#007BFF",
            hover_color="#0056b3"
        ).pack(pady=(5, 20))
        return

    # 4️⃣ 설문 항목이 존재할 때
    ctk.CTkLabel(
        score_frame,
        text="📋 설문 항목 목록",
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(pady=(5, 0))

    list_container = ctk.CTkFrame(score_frame, fg_color="transparent")
    list_container.pack(fill="x", padx=10, pady=5)

    for item in survey_items:
        has_data = bool(item.get('questions'))
        row_frame = ctk.CTkFrame(list_container, fg_color="transparent")
        row_frame.pack(fill="x", pady=5)
        row_frame.grid_columnconfigure(0, weight=1)
        row_frame.grid_columnconfigure(1, weight=0)
        row_frame.grid_columnconfigure(2, weight=0)

        # 📅 날짜 포맷 처리
        collected_at_raw = item.get('collected_at')
        formatted_date = ""
        if collected_at_raw:
            try:
                dt_obj = datetime.strptime(collected_at_raw.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                formatted_date = dt_obj.strftime(" (%Y-%m-%d %H:%M:%S)")
            except Exception:
                formatted_date = " (날짜 오류)"

        # 항목 정보 라벨
        item_name = f"[{item['data_category']}] {item['data_type']}\n저장 일자:{formatted_date}"
        item_summary_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
        item_summary_frame.grid(row=0, column=0, sticky="ew", padx=(5, 10))

        ctk.CTkLabel(
            item_summary_frame,
            text=item_name,
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=13, weight="bold" if has_data else "normal")
        ).pack(fill="x", anchor="w")

        # 요약
        summary_text = generate_summary(item) if has_data else "미입력 상태"
        ctk.CTkLabel(
            item_summary_frame,
            text=f"요약: {summary_text}",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="gray"
        ).pack(fill="x", anchor="w")

        # ✏️ 수정 버튼
        edit_color = "#357ABD" if has_data else "#4CAF50"
        edit_text = "수정" if has_data else "입력"
        ctk.CTkButton(
            row_frame,
            text=edit_text,
            command=lambda item_data=item: open_survey_form(item_data),
            fg_color=edit_color,
            hover_color=edit_color,
            width=65,
            height=36
        ).grid(row=0, column=1, padx=5, pady=5)

        # 🗑 삭제 버튼 추가
        def delete_item_action(item_data=item):
            confirm = CTkMessagebox(
                title="삭제 확인",
                message=f"정말로 [{item_data['data_type']}] 설문 항목을 삭제하시겠습니까?",
                icon="warning",
                option_2="삭제",
                option_1="취소"
            ).get()
            if confirm == "삭제":
                success, msg = delete_survey_item(item_data)  # ✅ 백엔드 삭제 API 호출
                if success:
                    CTkMessagebox(title="삭제 완료", message="설문 항목이 삭제되었습니다.", icon="check")
                    # 목록 새로고침
                    reload_after_close();
                else:
                    CTkMessagebox(title="오류", message=f"삭제 실패: {msg}", icon="cancel")

        ctk.CTkButton(
            row_frame,
            text="삭제",
            command=delete_item_action,
            fg_color="#D9534F",  # 붉은색
            hover_color="#C9302C",
            width=55,
            height=36
        ).grid(row=0, column=2, padx=5, pady=5)


def open_survey_form(item_data=None):
    """
    설문지 입력/수정 모달을 엽니다. 모든 설문 관련 동작을 이 함수로 통합합니다.
    """
    if not selected_patient:
        messagebox.showwarning("환자 선택 필요", "먼저 환자를 선택해주세요.")
        return

    # 모달 설정
    modal = ctk.CTkToplevel(root)
    
    # item_data가 비어있지 않고, 'questions' 필드가 있으면 '수정' 모드
    is_edit_mode = bool(item_data and item_data.get('questions'))
    
    # ----------------------------------------------------
    # ✅ [수정] initial_data 변수를 계산하여 NameError를 방지합니다.
    initial_form_data = [] # 변수를 초기화하여 NameError 방지
    
    if is_edit_mode:
        # 수정 모드: 원시 데이터 ('questions_raw')를 우선 사용하여 폼을 초기화
        initial_form_data = item_data.get('questions_raw', []) 
        
        # 안전 장치: 'questions_raw'가 비어있다면, 요약용 데이터라도 시도
        if not initial_form_data:
            initial_form_data = item_data.get('questions', [])
            
        if not initial_form_data:
             messagebox.showerror("오류", "수정할 상세 설문 데이터가 누락되었습니다.")
             modal.destroy()
             return
    # ----------------------------------------------------
    
    modal.title(f"운동성 설문지 {'수정' if is_edit_mode else '입력'}")
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    width = int(screen_width * 0.7)
    height = int(screen_height * 0.9)

    modal.geometry(f"{width}x{height}")
    modal.grab_set()

    # 상단 타이틀
    patient_uuid = selected_patient["patient_id"]
    initials = selected_patient.get("patient_initials", "?")
    item_type = item_data.get('data_type', 'N/A') if item_data else 'N/A'

    ctk.CTkLabel(modal, text=f"📝 운동성 설문지 - {initials} ({item_type})", font=("", 16)).pack(pady=10)

    # ✅ 콜백 전달: 폼이 닫힐 때 환자 데이터 다시 불러오기

    survey_form = HealthSurveyForm(
        modal,
        json_file=JSON_FILE,
        patient_id=patient_uuid,
        initial_data=initial_form_data,
        item_data=item_data,         # 👈 Item 전체 데이터 전달
        on_close_callback=reload_after_close  # ✅ 여기 추가
    )
    survey_form.pack(fill="both", expand=True, padx=10, pady=10)

    # 모달 닫기 시 캐시를 지우고 목록을 새로고침
    def on_modal_close():
        pid = selected_patient["patient_id"]
        if pid in items_cache:
            del items_cache[pid]
        
        modal.destroy()
        
    modal.protocol("WM_DELETE_WINDOW", on_modal_close)


def clear_file_items_area():
    """
    upload_list_frame (파일 항목 목록 영역) 내부의 모든 위젯을 제거하여 초기화합니다.
    """
    # upload_list_frame이 전역 변수인 경우 (클래스 메소드가 아닐 때)
    # global upload_list_frame # 필요하다면 주석 해제
    
    # 🚨 해당 프레임 안에 있는 모든 자식 위젯을 찾아 파괴(destroy)합니다.
    for widget in upload_list_frame.winfo_children():
        widget.destroy()

def show_file_items(parent_frame, file_items):
    """upload_list_frame에 파일 항목 목록을 표시합니다. (설문 항목과 동일한 CTkFrame 기반)"""
    
    # 1. 항목 표시 영역 초기화
    for widget in parent_frame.winfo_children():
        widget.destroy()

    if not selected_patient:
        # 환자 미선택 시 메시지 표시 (필요하다면)
        ctk.CTkLabel(parent_frame, text="환자를 선택해주세요.", font=("", 14, "italic"), text_color="gray").pack(pady=20)
        return
    
    if not file_items:
        ctk.CTkLabel(
            parent_frame,
            text="수집된 파일 데이터(영상 등)가 없습니다.",
            font=("", 13, "italic"),
            text_color="gray"
        ).pack(pady=20)
        return
    else:
        ctk.CTkLabel(
            parent_frame,
            text="📁 수집 파일 항목 목록",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(5, 0))

        list_container = ctk.CTkFrame(parent_frame, fg_color="transparent")
        list_container.pack(fill="x", padx=10, pady=5)
        
        # --- 파일 항목 반복 및 UI 생성 ---
        for item in file_items:
            # 💡 파일 항목은 기본적으로 데이터가 '존재'한다고 간주하며, 버튼은 '보기/다운로드'로 설정합니다.
            
            row_frame = ctk.CTkFrame(list_container, fg_color="transparent")
            row_frame.pack(fill="x", pady=5)
            row_frame.grid_columnconfigure(0, weight=1) # 항목/요약 영역 확장
            row_frame.grid_columnconfigure(1, weight=0) # 버튼 영역 고정

            # 1. 날짜 포맷팅
            collected_at_raw = item.get('collected_at', '')
            formatted_date = ""
            if collected_at_raw:
                try:
                    # ISO 8601 형식 문자열을 datetime 객체로 변환 (datetime 임포트 필요)
                    from datetime import datetime
                    dt_obj = datetime.strptime(collected_at_raw.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                    formatted_date = dt_obj.strftime(" (%Y-%m-%d %H:%M:%S)")
                except Exception:
                    formatted_date = " (날짜 오류)"
            
            # 2. 항목 이름 (첫 번째 줄)
            # data_type이 'VIDEO'인 경우, 상세 설명이나 파일 크기 등을 표시할 수 있습니다.
            item_name = f"[{item['data_category']}]{item['data_type']}\n저장 일자:{formatted_date}"
            
            item_summary_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
            item_summary_frame.grid(row=0, column=0, sticky="ew", padx=(5, 10))

            ctk.CTkLabel(
                item_summary_frame,
                text=item_name,
                anchor="w",
                justify="left",
                font=ctk.CTkFont(size=13, weight="bold"),
                # 파일 항목은 항상 데이터가 있다고 가정하여 굵게 표시
            ).pack(fill="x", anchor="w")
            
            
            # 3. 요약 정보 (두 번째 줄) - 파일 정보 요약
            is_updated_text = "수정됨" if item.get('is_updated', False) else "최초 등록"
            summary_text = f"상태: {is_updated_text} | 설명: {item.get('description', '설명 없음')}"
            
            ctk.CTkLabel(
                item_summary_frame,
                text=f"요약: {summary_text}",
                anchor="w",
                justify="left",
                font=ctk.CTkFont(size=11, slant="italic"),
                text_color="gray"
            ).pack(fill="x", anchor="w")

            # 4. 버튼 영역
            button_text = "보기"
            button_color = "#357ABD" # 파란색
            
            # 🚨 open_file_action 함수는 파일 다운로드, 열기 등을 처리해야 합니다.
            # 이 함수는 별도로 정의되어 있어야 합니다.
            button_command = lambda file_data=item: open_file_action(file_data)
                
            ctk.CTkButton(
                row_frame,
                text=button_text,
                command=button_command,
                fg_color=button_color,
                hover_color=button_color,
                width=100,
                height=40
            ).grid(row=0, column=1, padx=5, pady=5)
            
        # 5. 새 파일 등록 버튼 (선택 사항, 필요하다면)
        ctk.CTkButton(
            parent_frame,
            text="➕ 새 파일 항목 등록 및 업로드", font=("", 13),
            command=lambda: open_upload_modal(), # 🚨 open_file_upload_dialog 함수 정의 필요
            fg_color="#007BFF",
            hover_color="#0056b3"
        ).pack(pady=(5, 20))

# ---------------- 파일 업로드 모달 ----------------
def open_upload_modal():
    if not selected_patient:
        messagebox.showwarning("환자 선택 필요", "먼저 환자를 선택해주세요.")
        return

    # ---------------- 1. 모달 기본 설정 ----------------
    modal = ctk.CTkToplevel(root)
    modal.title(f"{selected_patient.get('patient_initials', '')} 환자 영상 메타데이터 등록/수정")
    modal.geometry("550x480")
    modal.grab_set()

    VIDEO_SLOTS = {
        1: "1번 영상 (A)",
        2: "2번 영상 (B)",
        3: "3번 영상 (C)",
        4: "손글씨 영상 (D)",
    }

    SLOT_TO_SEQ = {1: 1, 2: 2, 3: 3, 4: 4}

    file_paths = {i: ctk.StringVar(value="") for i in VIDEO_SLOTS}
    main_frame = ctk.CTkFrame(modal)
    main_frame.pack(fill="both", expand=True, padx=20, pady=10)

    # ---------------- 기존 Item 정보 조회 ----------------
    current_items = items_cache.get(selected_patient['patient_id'], [])
    seq_to_item = {}
    for item in current_items:
        if item.get("data_type", "").upper() == "VIDEO":
            seq = item.get("seq")
            if seq:
                seq_to_item[seq] = {
                    "item_id": item.get("item_id"),
                    "file_path": item.get("file_path", "")
                }

    # ---------------- 2. 파일 선택 위젯 ----------------
    for i, label_text in VIDEO_SLOTS.items():
        row_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        row_frame.pack(fill="x", pady=5)
        row_frame.grid_columnconfigure(1, weight=1)

        seq_for_slot = SLOT_TO_SEQ.get(i)
        existing_info = seq_to_item.get(seq_for_slot)
        entry_placeholder = existing_info.get("file_path") if existing_info else "선택된 파일 없음 (미등록)"
        file_paths[i].set("")

        label_text_full = f"🎥 (Seq {seq_for_slot}) {label_text}:"
        ctk.CTkLabel(row_frame, text=label_text_full, width=150, anchor="w").grid(row=0, column=0, padx=5, sticky="w")

        entry = ctk.CTkEntry(row_frame, textvariable=file_paths[i], placeholder_text=entry_placeholder, width=230)
        entry.grid(row=0, column=1, padx=5, sticky="ew")

        def create_browse_command(var):
            return lambda: browse_file_to_var(var)

        ctk.CTkButton(row_frame, text="로컬 파일 선택", width=120,
                      command=create_browse_command(file_paths[i])).grid(row=0, column=2, padx=5)

    def browse_file_to_var(var: ctk.StringVar):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")]
        )
        if file_path:
            var.set(file_path)

    # ---------------- 3. 메타데이터 등록/수정 ----------------
    def start_metadata_registration():
        files_to_process_meta = []
        target_patient_id = selected_patient['patient_id']

        for i, var in file_paths.items():
            local_path = var.get().strip()
            seq = SLOT_TO_SEQ.get(i)
            existing_info = seq_to_item.get(seq, {})
            existing_item_id = existing_info.get("item_id")

            # 새 파일도 없고, 기존 Item도 없으면 건너뜀
            if not local_path and not existing_item_id:
                continue

            # 새 파일 선택된 경우
            if local_path and os.path.isfile(local_path):
                file_name = os.path.basename(local_path)
                _, file_ext = os.path.splitext(file_name)
                file_ext = file_ext.lower().lstrip('.')
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                file_size_mb_str = f"{file_size_mb:.2f}"
                video_info = get_video_metadata(local_path)
                simulated_server_path = os.path.join(OUTPUT_DIR, str(target_patient_id), file_name)
            else:
                # 기존 파일 유지
                file_name = os.path.basename(existing_info.get("file_path", f"default_{seq}.mp4"))
                file_ext = file_name.split(".")[-1]
                file_size_mb_str = "0.00"
                video_info = {}
                simulated_server_path = existing_info.get("file_path", "N/A")

            is_anon = True if i in [1, 2, 3] else False

            files_to_process_meta.append({
                "local_source_path": local_path or None,
                "slot_index": i,
                "seq": seq,
                "existing_item_id": existing_item_id,
                "file_name": file_name,
                "file_path": simulated_server_path,
                "file_ext": file_ext,
                "file_size_mb": file_size_mb_str,
                "duration_seconds": int(video_info.get("duration_seconds", 0.0)),
                "resolution": video_info.get("resolution", "N/A"),
                "frame_rate": int(video_info.get("frame_rate", 0.0)),
                "needs_anonymization": is_anon,
                "shooting_ts": video_info.get("creation_time"),
                "data_category": "PD",
            })

        if not files_to_process_meta:
            CTkMessagebox(title="경고", message="선택된 로컬 영상 파일이 없거나 기존 데이터가 없습니다.", icon="warning")
            return

        # ---------------- 등록/수정 실행 ----------------
        def register_video_metadata():
            success_count = 0
            failure_messages = []

            for meta_data in files_to_process_meta:
                slot_index = meta_data.pop('slot_index')
                local_source_path = meta_data.pop('local_source_path')
                seq = meta_data.pop('seq')
                existing_item_id = meta_data.pop('existing_item_id')
                slot_label = VIDEO_SLOTS[slot_index]
                target_file_path = meta_data['file_path']
                action = "등록"
                item_id = None

                if existing_item_id:
                    action = "수정"
                    item_id = existing_item_id
                    video_meta_id_to_update = None

                    meta_list, error_msg = fetch_video_metadata_by_item_id(item_id)
                    if error_msg:
                        failure_messages.append(f"[{slot_label}] ❌ Meta 조회 실패: {error_msg}")
                        continue

                    if meta_list and meta_list[0].get('video_metadata_id'):
                        meta_data['video_metadata_id'] = meta_list[0]['video_metadata_id']
                    else:
                        action = "재등록"

                else:
                    item_id_or_error = create_new_item_and_get_id(target_patient_id, seq)
                    if not isinstance(item_id_or_error, int):
                        failure_messages.append(f"[{slot_label}] ❌ Item 등록 실패: {item_id_or_error[1]}")
                        continue
                    item_id = item_id_or_error

                # 파일 복사/삭제
                try:
                    if local_source_path and os.path.isfile(local_source_path):
                        if existing_item_id:
                            old_path = seq_to_item.get(seq, {}).get("file_path")
                            if old_path and os.path.exists(old_path) and old_path != target_file_path:
                                os.remove(old_path)
                        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                        shutil.copy2(local_source_path, target_file_path)
                except Exception as e:
                    failure_messages.append(f"[{slot_label}] ❌ 파일 복사 실패: {e}")
                    continue

                meta_data['item_id'] = item_id

                if action == "수정":
                    success, msg = call_api_to_update_video_metadata([meta_data])
                else:
                    success, msg = call_api_to_save_video_metadata(item_id, [meta_data])

                if success:
                    success_count += 1
                else:
                    failure_messages.append(f"[{slot_label}] ❌ Meta {action} 실패: {msg}")
                    if os.path.exists(target_file_path):
                        os.remove(target_file_path)

            total = len(files_to_process_meta)
            if success_count == total:
                return True, f"✅ 총 {success_count}개 영상이 성공적으로 처리되었습니다."
            elif success_count > 0:
                return False, f"⚠️ {success_count}/{total} 성공, 일부 실패:\n" + "\n".join(failure_messages)
            else:
                return False, "❌ 모든 영상 등록/수정 실패\n" + "\n".join(failure_messages)

        # ---------------- 결과 처리 ----------------
        def after_registration(result):
            modal.destroy()
            success, message = result
            icon = "check" if success else "cancel"
            CTkMessagebox(title="결과", message=message, icon=icon)

            pid = target_patient_id
            if pid in items_cache:
                del items_cache[pid]
            if selected_patient and selected_row:
                on_select_patient(selected_patient, selected_row)

        run_with_loading_popup(
            parent_frame=root,
            fetch_function=register_video_metadata,
            callback=after_registration,
            loading_text=f"Item 등록/수정, 파일 복사 및 메타데이터 DB 처리 중... (총 {len(files_to_process_meta)}건)"
        )

    # ---------------- 5. 등록 버튼 ----------------
    ctk.CTkButton(
        modal,
        text="💾 메타데이터 DB 등록/수정 실행",
        command=start_metadata_registration,
        fg_color="#34A853",
        hover_color="#2C8E47"
    ).pack(pady=20)



# ---------------- 환자 선택 및 항목 로드 로직 ----------------
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

    # 항목 표시 영역을 초기화하고 로딩 상태 표시
    show_empty_state() # 설문 영역 초기화
    clear_file_items_area()
    # show_file_items([]) # 파일 영역 초기화

    # 로딩 표시 (임시)
    loading_label_survey = ctk.CTkLabel(score_frame, text="수집 항목 로드 중...", text_color="blue")
    loading_label_survey.pack(pady=10)
    loading_label_file = ctk.CTkLabel(upload_list_frame, text="수집 항목 로드 중...", text_color="blue")
    loading_label_file.pack(pady=10)

    # ✅ 항목이 이미 캐시에 있으면 바로 처리
    if pid in items_cache:
        loading_label_survey.destroy()
        loading_label_file.destroy()
        process_items(items_cache[pid])
        return

    # on_select_patient 함수 내부

    # ✅ 스레드로 비동기 요청
    def fetch_in_background():
        try:
            # 1. 메타데이터 (항목 기본 정보) 로드
            items = fetch_items(pid) 

            # 2. 설문 항목에 대해 상세 응답을 추가로 로드 (enrichment)
            enriched_items = []
            for item in items:
                data_type = item.get('data_type', '').upper()
                
                # 설문 항목인지 확인 (process_items의 분류 로직을 따름)
                is_survey_item = (
                    'FORM' in item.get('data_category', '').upper() 
                    or 'MDS-UPDRS' in data_type 
                    or 'SURVEY' in data_type
                )
                
                # 설문 항목인데 questions가 없는 경우에만 상세 조회
                if is_survey_item and not item.get('questions'):
                    item_id = item.get('item_id')

                    if item_id:
                        detailed_answers_raw = fetch_mds_answers(item_id) 
                    
                         # A. 요약 (Summary) 및 데이터 유무 판단용 데이터 (간결/압축)
                        detailed_answers_formatted = format_mds_answers(detailed_answers_raw)
                        
                        # B. 폼 로딩 (Edit)용 원시 데이터 (상세)
                        # 원시 데이터를 'raw_questions'와 같은 별도의 키에 저장합니다.
                        item['questions_raw'] = detailed_answers_raw # <-- 원시 데이터 저장
                        
                        # 'questions' 키에는 요약용 데이터를 저장하여 has_data=True 및 generate_summary가 작동하도록 합니다.
                        item['questions'] = detailed_answers_formatted
                    
                enriched_items.append(item)
            
            # 다음 단계(UI 업데이트)로 넘길 데이터는 enriched_items
            items_to_process = enriched_items 

        except Exception as e:
            items_to_process = e
            
        # UI 업데이트는 main thread에서
        def update_ui():
            loading_label_survey.destroy()
            loading_label_file.destroy()
            
            if not isinstance(items_to_process, Exception):
                items_cache[pid] = items_to_process # 캐시도 상세 내용으로 업데이트
                process_items(items_to_process)
            else:
                messagebox.showerror("에러", f"수집 항목 로딩 실패:\n{items_to_process}")
                show_survey_items([])
                show_file_items(upload_list_frame, [])
                
        root.after(0, update_ui)

    threading.Thread(target=fetch_in_background, daemon=True).start()

def open_file_action(file_data):
    """
    파일 항목의 '보기' 버튼 클릭 시 호출됩니다.
    비디오 항목이면 내부 비디오 재생기로 띄움.
    """
    print(f"DEBUG: open_file_action 호출됨 - {file_data}")
    data_type = file_data.get('data_type', 'FILE')
    item_id = file_data.get('item_id', None)

    if not item_id:
        messagebox.showerror("오류", "item_id가 없습니다.")
        return
    
    # 🎥 비디오 처리
    if 'VIDEO' in data_type.upper():
        print(f"비디오 항목 보기 요청: Item ID {item_id}")

        # 1. 메타데이터 조회
        video_meta_list, error_msg = fetch_video_metadata_by_item_id(item_id)
        if error_msg:
            show_video_player_window(item_id, f"메타데이터 오류: {error_msg}", error=True)
            return
        
        if not video_meta_list:
            show_video_player_window(item_id, f"Item {item_id}에 대한 메타데이터 없음", error=True)
            return
        
        # 2. file_path 추출 및 보정
        video_meta = video_meta_list[0]
        file_path = video_meta.get("file_path")

        if not file_path:
            show_video_player_window(item_id, f"file_path 필드 없음", error=True)
            return

        if not os.path.isabs(file_path):
            project_root = Path(__file__).resolve().parent.parent
            file_path = os.path.join(project_root, file_path)
            file_path = os.path.abspath(file_path)

        print(f"DEBUG: 최종 비디오 경로 = {file_path}")

        # 3. 존재 여부 확인 및 재생
        if not os.path.exists(file_path):
            show_video_player_window(
                item_id,
                f"❌ 비디오 파일을 찾을 수 없습니다.\n경로: {file_path}",
                error=True
            )
        else:
            if os.path.exists(file_path):
                play_video_external(file_path)  # ✅ 외부 플레이어 실
            else:
                show_video_player_window(item_id, f"파일 없음: {file_path}", error=True)

    else:
        print(f"일반 파일 항목 보기 요청: Item ID {item_id}")
        messagebox.showinfo("알림", f"일반 파일 유형입니다.\nItem ID: {item_id}")

def play_video_external(file_path):
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", file_path])  # macOS
        else:
            subprocess.call(["xdg-open", file_path])  # Linux
    except Exception as e:
        messagebox.showerror("실행 오류", f"외부 플레이어 실행 실패\n{e}")      

def show_video_player_window(item_id, file_path, error=False):
    video_window = ctk.CTkToplevel()
    video_window.title(f"비디오 뷰어 - Item ID: {item_id}")
    video_window.geometry("800x600")

    ctk.CTkLabel(video_window, text=f"Item ID: {item_id}", font=("", 16)).pack(pady=10)

    if error:
        ctk.CTkLabel(video_window, text=file_path, text_color="red").pack()
    else:
        label = tk.Label(video_window)  # 일반 tkinter 사용
        label.pack()
        player = tkvideo(file_path, label, loop=1, size=(800, 500))
        player.play()

    ctk.CTkButton(video_window, text="닫기", command=video_window.destroy).pack(pady=10)

def process_items(items):
    """로드된 전체 항목을 설문과 파일로 분류하여 각 영역에 표시합니다."""
    survey_items = []
    file_items = []
    
    # 설문 항목은 data_type이 'MDS-UPDRS PART 3'이거나 'FORM'이 포함된 경우로 가정합니다.
    # 파일 항목은 그 외의 모든 항목으로 간주합니다.
    for item in items:
        # data_category와 data_type을 대문자로 변환하여 비교 준비
        data_category = item.get('data_category', '').upper()
        data_type = item.get('data_type', '').upper()
        
        # 1. 설문 항목 분류 기준 설정:
        # 'FORM', 'MDS-UPDRS', 'SURVEY'를 포함하는 항목은 설문으로 간주
        is_survey_item = (
            'FORM' in data_category 
            or 'MDS-UPDRS' in data_type 
            or 'SURVEY' in data_type
        )
        
        # 2. VIDEO 항목 분류:
        # VIDEO 타입이거나, 설문 항목이 아닌 나머지 항목은 파일 항목으로 간주
        is_file_item = (
            'VIDEO' in data_type 
            or not is_survey_item # 설문 외 모든 항목 (이미지, 기타 파일 등)
        )
        
        if is_survey_item:
            # 설문 항목: score_frame에 표시
            survey_items.append(item)
        elif is_file_item:
            # 파일/미디어 항목 (VIDEO 포함): upload_list_frame에 표시
            print(is_file_item)
            file_items.append(item)
    
    show_survey_items(survey_items)
    show_file_items(upload_list_frame, file_items)  # ✅ 수정

# ---------------- 환자 목록 테이블 로드 ----------------
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
        # 로딩 기능 추가 고려
        try:
            add_patient(payload, INSTITUTION)
            messagebox.showinfo("성공", "환자 등록 완료")
        except Exception as e:
            messagebox.showerror("오류", f"환자 등록 실패: {e}")
        
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

lbl_video_title = ctk.CTkLabel(frame_video, text="파일 관리", font=("", 16, "bold"))
lbl_video_title.pack(pady=10)

# 파일 업로드 버튼을 묶는 프레임
upload_button_frame = ctk.CTkFrame(frame_video)
upload_button_frame.pack(pady=5)
ctk.CTkButton(upload_button_frame, text="📤 파일 업로드",font=("",16) ,width=150, command=open_upload_modal).pack(side="left", padx=10)


# 파일 목록을 표시할 프레임 (show_file_items에서 관리)
upload_list_frame = ctk.CTkFrame(frame_video)
upload_list_frame.pack(fill="x", pady=5)


# 프로그램 시작 시 상태 설정
show_empty_state()
# 파일 영역 초기 상태 설정
# show_file_items([])
root.after(100, init_program)
root.mainloop() 