import customtkinter as ctk
import tkinter as tk
from tkinter import ttk 
import sys
import os
from pathlib import Path
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from utils.survey import format_mds_answers
from typing import Optional, Dict, Any
# ✅ sys.path 수정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "video")  # ⬅️ 원하는 위치로 정확히 대응됨

# sys.path.append(PROJECT_ROOT)

from api_local.patient_api_local import add_patient, delete_patient, fetch_patients
from api_local.item_api_local import fetch_items,delete_survey_item,mark_item_updated_local
from utils.loader import run_with_loading, run_with_loading_popup
from api_local.form_api_local import (
    fetch_mds_answers,
    EMOTION_QMAP_BY_SEQ,
    ANXIETY_DISORDER_QUESTION_MAPPING,
    PHQ9_QUESTION_MAPPING,
    MADRS_QUESTION_MAPPING,
      # ✅ 아래 2줄 추가
    SLEEP_QMAP_BY_SEQ,
    ISI_QUESTION_MAPPING, 
    KESS_QUESTION_MAPPING,
    PSQI_QUESTION_MAPPING, 
    MEQK_QUESTION_MAPPING,
    create_new_item_and_get_id_generic)
from api_local.video_api_local import create_new_item_and_get_id, save_video_metadata,fetch_video_metadata_by_item_id,update_video_metadata
from form.survey import HealthSurveyForm
from utils.videometa import get_video_metadata
from config import INSTITUTION, VIDEO_SAVE_BASE
from utils.db_utils import get_connection, release_connection
from sqlalchemy import text

JSON_FILE = os.path.join(CURRENT_DIR, '..', 'form', 'basic_form', 'basic.json')
JSON_FILE = os.path.abspath(JSON_FILE)  # ← 절대경로로 변환 (안전)

items_cache = {} # 환자별 수집 항목 캐시
EMOTION_TITLES_BY_SEQ = {1: "PHQ-9", 2: "MADRS", 3: "불안증상"}
EMOTION_FORMS_BY_SEQ = {
    1: os.path.join(PROJECT_ROOT,"form","emotion_form","phq9.json"),
    2: os.path.join(PROJECT_ROOT,"form","emotion_form","MADRS.json"),
    3: os.path.join(PROJECT_ROOT,"form","emotion_form","anxiety_disorder.json"),
}


SLEEP_TITLES_BY_SEQ = {
    1: "ISI",     # Insomnia Severity Index
    2: "PSQI",    # Pittsburgh Sleep Quality Index
    3: "KESS",    # Korean Epworth Sleepiness Scale
    4: "MEQ-K",   # Morningness–Eveningness Questionnaire (Korean)
}
SLEEP_FORMS_BY_SEQ = {
    1: os.path.join(PROJECT_ROOT, "form", "sleep_form", "ISI.json"),
    2: os.path.join(PROJECT_ROOT, "form", "sleep_form", "PSQI.json"),
    3: os.path.join(PROJECT_ROOT, "form", "sleep_form", "KESS.json"),
    4: os.path.join(PROJECT_ROOT, "form", "sleep_form", "MEQ_K.json"),
}
# ---------------- 서버 체크 ----------------
def local_db_check():
    """로컬 DB 연결 상태 확인"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        cur.close()
        release_connection(conn)
        if result and result[0] == 1:
            return "OK"
        else:
            return "DB_FAIL"
    except Exception:
        return "CONNECTION_FAIL"
 
def init_program():
    """프로그램 첫 실행 시 로컬 DB 상태 확인 (로딩 오버레이 포함)"""

    def after_check(result):
        if result == "OK":
            CTkMessagebox(title="성공", message="로컬 DB 연결 성공 ✅", icon="check")
            load_patients_table()
        elif result == "DB_FAIL":
            CTkMessagebox(title="오류", message="DB 동작 이상. 관리자에게 문의하세요.", icon="cancel")
            show_server_error()
        else:
            CTkMessagebox(title="오류", message="로컬 DB 연결 실패. 설정을 확인하세요.", icon="cancel")
            show_server_error()

    run_with_loading(
        parent_frame=root,
        fetch_function=local_db_check,  # ✅ 여기서 바로 로컬 DB 확인
        callback=after_check,
        loading_text="로컬 DB 연결 확인 중입니다..."
    )


def show_server_error():
    """DB 연결 실패 시 테이블 영역에 표시"""
    for widget in table_frame.winfo_children():
        widget.destroy()

    ctk.CTkLabel(
        table_frame,
        text="로컬 DB와 연결할 수 없습니다.",
        font=("", 14, "italic"),
        text_color="red"
    ).pack(pady=20)

def reload_after_close():
    pid = selected_patient["patient_id"]
    if pid in items_cache:
        del items_cache[pid]  # ✅ 캐시 제거
    on_select_patient(selected_patient, selected_row)  # 설문 + 파일 목록 리로드

# ---------------- [CTk 기본 설정] ----------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("우울증 입력 프로그램")
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
        has_data = bool(item.get("questions")) or bool(item.get("questions_raw"))
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
                # 🧩 1️⃣ 이미 datetime 객체인 경우
                if isinstance(collected_at_raw, datetime):
                    dt_obj = collected_at_raw
                else:
                    # 🧩 2️⃣ 문자열인 경우 → ISO 형식 or 공백 구분 자동 처리
                    dt_obj = datetime.fromisoformat(str(collected_at_raw).strip())

                formatted_date = dt_obj.strftime(" (%Y-%m-%d %H:%M:%S)")
            except Exception as e:
                print(f"[날짜 파싱 오류] {collected_at_raw} → {e}")
                formatted_date = " (날짜 오류)"
        else:
            formatted_date = ""

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
                    reload_after_close()
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
    if not selected_patient:
        messagebox.showwarning("환자 선택 필요", "먼저 환자를 선택해주세요.")
        return

    modal = ctk.CTkToplevel(root)
    modal.title("설문 입력")
    modal.geometry("1200x900")
    modal.grab_set()


    patient_uuid = item_data['patient_id']
    print(patient_uuid)
    initials = selected_patient.get("patient_initials", "?")
    raw_type = item_data.get('data_type', 'N/A') if item_data else 'N/A'

    # ✅ 사람이 보기 좋은 제목 변환
    type_display_map = {
        "B-SURVEY": "기초 평가",
        "E-SURVEY": "정서 설문지",
        "S-SURVEY": "수면 설문지"
    }
    item_type = type_display_map.get(raw_type.upper(), raw_type)

    # 정서 설문이면 뒤에 세부명(PHQ-9 등) 붙이기
    pretty_suffix = ""
    try:
        if str(raw_type).upper() == "E-SURVEY":
            seq_for_title = item_data.get("seq")
            if seq_for_title in EMOTION_TITLES_BY_SEQ:
                pretty_suffix = f" - {EMOTION_TITLES_BY_SEQ[seq_for_title]}"
    except Exception:
        pass
    ctk.CTkLabel(
        modal,
        text=f"📝 설문지 - {initials} ({item_type})",
        font=("", 16)
    ).pack(pady=10)

    # ✅ JSON 파일 경로 결정
    json_file_path = item_data.get("json_file", JSON_FILE)

    # ✅ JSON 구조 감지
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    is_table_form = any(
        isinstance(v, dict) and ("sections" in v or "body" in v)
        for v in json_data.values()
    )

    # ✅ 폼 종류 분기
    if is_table_form:
        qmap = None
        seq = item_data.get("seq")
        dtype = str(item_data.get("data_type","")).upper()
    
        if dtype == "E-SURVEY":
            # 정서 설문 매핑
            if seq == 1:
                qmap = PHQ9_QUESTION_MAPPING
            elif seq == 2:
                qmap = MADRS_QUESTION_MAPPING
            elif seq == 3:
                qmap = ANXIETY_DISORDER_QUESTION_MAPPING
    
        elif dtype == "S-SURVEY":
            # ✅ 수면 설문 매핑
            if seq == 1:
                qmap = ISI_QUESTION_MAPPING
            elif seq == 2:
                qmap = KESS_QUESTION_MAPPING
            elif seq == 3:
                qmap = PSQI_QUESTION_MAPPING
            elif seq == 4:
                qmap = MEQK_QUESTION_MAPPING
    
        from form.generic_survey import GenericSurveyForm
        form_frame = GenericSurveyForm(
            modal,
            json_file=json_file_path,
            item_data=item_data,
            patient_uuid=patient_uuid,
            qmap=qmap,
            on_close_callback=reload_after_close
        )
    else:
        from form.survey import HealthSurveyForm
        form_frame = HealthSurveyForm(
            modal,
            json_file=json_file_path,
            patient_uuid=patient_uuid,
            item_data=item_data,
            on_close_callback=reload_after_close
        )

    form_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def on_close():
        pid = selected_patient["patient_id"]
        if pid in items_cache:
            del items_cache[pid]
        modal.destroy()

    modal.protocol("WM_DELETE_WINDOW", on_close)


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
    # 1️⃣ 영역 초기화
    for widget in parent_frame.winfo_children():
        widget.destroy()

    # 2️⃣ 환자 미선택 시
    if not selected_patient:
        ctk.CTkLabel(
            parent_frame,
            text="환자를 선택해주세요.",
            font=("", 14, "italic"),
            text_color="gray"
        ).pack(pady=20)
        return

    # 3️⃣ 파일 데이터 없을 때
    if not file_items:
        ctk.CTkLabel(
            parent_frame,
            text="📂 수집된 파일 데이터(영상 등)가 없습니다.",
            font=("", 13, "italic"),
            text_color="gray"
        ).pack(pady=(30, 10))

        # 업로드 버튼 (파일 없을 때도 항상 표시)
        ctk.CTkButton(
            parent_frame,
            text="➕ 새 파일 항목 등록 및 업로드",
            font=("", 13, "bold"),
            command=open_upload_modal,
            fg_color="#007BFF",
            hover_color="#0056b3"
        ).pack(pady=(5, 20))
        return

    # 4️⃣ 파일 데이터가 있을 때
    ctk.CTkLabel(
        parent_frame,
        text="📁 수집 파일 항목 목록",
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(pady=(5, 0))

    list_container = ctk.CTkFrame(parent_frame, fg_color="transparent")
    list_container.pack(fill="x", padx=10, pady=5)

    for item in file_items:
        row_frame = ctk.CTkFrame(list_container, fg_color="transparent")
        row_frame.pack(fill="x", pady=5)
        row_frame.grid_columnconfigure(0, weight=1)
        row_frame.grid_columnconfigure(1, weight=0)

        # 날짜 포맷팅
        collected_at_raw = item.get('collected_at', '')
        formatted_date = ""
        print(collected_at_raw)

        if collected_at_raw:
            try:
                if isinstance(collected_at_raw, datetime):
                    dt_obj = collected_at_raw
                else:
                    # 문자열인 경우 → datetime으로 변환 시도
                    collected_at_str = str(collected_at_raw).strip()
                    # 공백 방지
                    if " " in collected_at_str:
                        # 예: '2025-10-19 21:21:57.618579'
                        dt_obj = datetime.strptime(collected_at_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
                    elif "T" in collected_at_str:
                        # 예: '2025-10-19T21:21:57'
                        dt_obj = datetime.strptime(collected_at_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                    else:
                        # 혹시 몰라 마지막 fallback
                        dt_obj = datetime.fromisoformat(collected_at_str)

                formatted_date = dt_obj.strftime(" (%Y-%m-%d %H:%M:%S)")
            except Exception as e:
                print(f"[날짜 파싱 오류] {collected_at_raw} → {e}")
                formatted_date = " (날짜 오류)"
        else:
            formatted_date = ""

        # 항목 이름 및 요약
        item_name = f"[{item['data_category']}] {item['data_type']}\n저장 일자:{formatted_date}"
        summary_text = f"상태: {'수정됨' if item.get('is_updated', False) else '최초 등록'} | 설명: {item.get('description', '설명 없음')}"

        item_summary_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
        item_summary_frame.grid(row=0, column=0, sticky="ew", padx=(5, 10))

        ctk.CTkLabel(
            item_summary_frame,
            text=item_name,
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(fill="x", anchor="w")

        ctk.CTkLabel(
            item_summary_frame,
            text=f"요약: {summary_text}",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="gray"
        ).pack(fill="x", anchor="w")

        # 보기 버튼
        ctk.CTkButton(
            row_frame,
            text="원본 영상 보기",
            command=lambda file_data=item: open_file_action(file_data),
            fg_color="#357ABD",
            hover_color="#2B5E9E",
            width=100,
            height=36
        ).grid(row=0, column=1, padx=5, pady=5)

    # 5️⃣ 새 파일 등록 버튼 (항상 표시)
    ctk.CTkButton(
        parent_frame,
        text="➕ 새 파일 항목 등록 및 업로드",
        font=("", 13, "bold"),
        command=open_upload_modal,
        fg_color="#007BFF",
        hover_color="#0056b3"
    ).pack(pady=(10, 20))

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
                # 원본 파일 정보 추출
                orig_file_name = os.path.basename(local_path)
                _, file_ext = os.path.splitext(orig_file_name)
                file_ext = file_ext.lower().lstrip('.')
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                file_size_mb_str = f"{file_size_mb:.2f}"
                video_info = get_video_metadata(local_path)
            
                # ✅ 파일 이름을 "환자UUID_seq.확장자" 형식으로 생성
                file_name = f"{target_patient_id}_{seq}.{file_ext}"
            
                # 서버 업로드용 경로 구성
                simulated_server_path = os.path.join(VIDEO_SAVE_BASE, str(target_patient_id), file_name)
            else:
                # 기존 파일 유지
                existing_file_name = os.path.basename(existing_info.get("file_path", f"default_{seq}.mp4"))
                file_ext = existing_file_name.split(".")[-1]
                file_name = f"{target_patient_id}_{seq}.{file_ext}"  # ✅ 기존 것도 동일 형식으로 통일
                file_size_mb_str = "0.00"
                video_info = {}
                simulated_server_path = existing_info.get("file_path", "N/A")

            is_anon = True if i in [1, 2, 3, 4] else False

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
                    success, msg = update_video_metadata([meta_data])
                    if success:
                        # ✅ 수정된 영상일 경우 updated_at 자동 갱신 API 호출
                        mark_item_updated_local(item_id)
                else:
                    success, msg = save_video_metadata(item_id, [meta_data])

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
    """
    환자 선택 시 설문 및 파일 항목 로드
    - 기존 선택된 행 색상 초기화 (존재 체크)
    - 새 선택 행 하이라이트
    - 설문/파일 영역 초기화 후 로딩 표시
    - 비동기 스레드로 항목 불러오기
    """
    global selected_patient, selected_row

    # ✅ 기존 선택 해제 (존재 체크)
    try:
        if selected_row and selected_row.winfo_exists():
            selected_row.configure(fg_color="transparent")
    except Exception as e:
        print(f"[경고] 이전 선택 해제 중 오류: {e}")
        selected_row = None

    # ✅ 새 선택 표시
    try:
        if row_frame and row_frame.winfo_exists():
            row_frame.configure(fg_color="#D0E8FF")
            selected_row = row_frame
    except Exception as e:
        print(f"[경고] 새 선택 표시 오류: {e}")
        selected_row = None

    selected_patient = patient
    pid = patient.get("patient_id")

    # ✅ 설문/파일 영역 초기화 및 로딩 상태 표시
    show_empty_state()
    clear_file_items_area()

    loading_label_survey = ctk.CTkLabel(
        score_frame, text="📋 수집 항목 로드 중...", text_color="blue"
    )
    loading_label_survey.pack(pady=10)
    loading_label_file = ctk.CTkLabel(
        upload_list_frame, text="📂 수집 항목 로드 중...", text_color="blue"
    )
    loading_label_file.pack(pady=10)

    # ✅ 캐시된 데이터가 있으면 즉시 처리
    if pid in items_cache:
        loading_label_survey.destroy()
        loading_label_file.destroy()
        process_items(items_cache[pid])
        return

    # ✅ 비동기 스레드로 DB 접근 (UI 멈춤 방지)
    def fetch_in_background():
        try:
            # 1️⃣ 기본 수집 항목 조회
            items = fetch_items(pid)

            # 2️⃣ 설문 항목이면 응답 데이터까지 불러오기
            enriched_items = []
            for item in items:
                data_type = item.get("data_type", "").upper()
                data_category = item.get("data_category", "").upper()

                # 설문 유형 판별
                is_basic_survey = data_type == "B-SURVEY"
                is_emotion_survey = data_type == "E-SURVEY"
                is_sleep_survey = data_type == "S-SURVEY"
                is_mds_survey = "MDS-UPDRS" in data_type
                is_survey_item = is_basic_survey or is_emotion_survey or is_sleep_survey or is_mds_survey

                # ✅ 설문 데이터만 응답 로드
                if is_survey_item and not item.get("questions"):
                    item_id = item.get("item_id")
                    if item_id:
                        try:
                            detailed_answers_raw = fetch_mds_answers(item_id)

                            # 🧩 각 유형별로 추가 필드 마킹 (UI 분류용)
                            survey_type = None
                            if is_basic_survey:
                                survey_type = "BASIC"
                            elif is_emotion_survey:
                                survey_type = "EMOTION"
                            elif is_sleep_survey:
                                survey_type = "SLEEP"
                            elif is_mds_survey:
                                survey_type = "MDS"
                            # 응답 가공
                            detailed_answers_formatted = format_mds_answers(
                                detailed_answers_raw,
                                survey_type=survey_type
                            )

                            # 원본/요약 저장
                            item["questions_raw"] = detailed_answers_raw
                            item["questions"] = detailed_answers_formatted
                            item["survey_type"] = survey_type  # ✅ UI 구분용

                        except Exception as e:
                            print(f"[경고] item_id={item_id} 설문 상세 로드 실패: {e}")

                enriched_items.append(item)

            items_to_process = enriched_items

        except Exception as e:
            print(f"❌ 항목 로드 실패: {e}")
            items_to_process = e

        # ✅ UI 업데이트는 main thread에서 실행
        def update_ui():
            try:
                loading_label_survey.destroy()
                loading_label_file.destroy()
            except Exception:
                pass

            if not isinstance(items_to_process, Exception):
                items_cache[pid] = items_to_process
                process_items(items_to_process)
            else:
                messagebox.showerror("에러", f"수집 항목 로딩 실패:\n{items_to_process}")
                show_survey_items([])
                show_file_items([])

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
def _find_emotion_item_by_seq(items, seq: int):
    """items 리스트에서 data_type='E-SURVEY' AND seq==X 인 첫 항목 반환"""
    for it in items or []:
        if str(it.get("data_type","")).upper() == "E-SURVEY" and int(it.get("seq") or 0) == int(seq):
            return it
    return None


def open_emotion_survey(seq: int, existing_item: Optional[dict] = None):
    """
    정서 설문(PHQ-9/MADRS/불안척도) 입력/수정 공용 오프너
    - 신규: tb_items 생성하지 않음(저장 시점에 트랜잭션으로 생성)
    - 수정: 기존 item 정보 사용
    """
    if not selected_patient:
        messagebox.showwarning("환자 선택 필요", "먼저 환자를 선택해주세요.")
        return

    patient_uuid = selected_patient["patient_id"]
    
    title = EMOTION_TITLES_BY_SEQ.get(seq, f"SEQ-{seq}")
    json_file_path = EMOTION_FORMS_BY_SEQ.get(seq)
    if not json_file_path:
        messagebox.showerror("오류", f"정서 설문 폼(json) 경로를 찾을 수 없습니다. seq={seq}")
        return

    if existing_item is None:
        item_data = {
            "patient_id": patient_uuid,
            "data_category": "MDD",
            "data_type": "E-SURVEY",
            "seq": seq,
            "title": title,
            "json_file": json_file_path
        }
    else:
        item_data = existing_item.copy()
        item_data.update({
            "patient_id": patient_uuid,           # 혹시 없을 수 있어 보강
            "data_category": existing_item.get("data_category", "MDD"),
            "data_type": "E-SURVEY",
            "seq": existing_item.get("seq", seq),
            "title": title,
            "json_file": json_file_path
        })

    open_survey_form(item_data)



def render_emotion_section(parent, emotion_items: list):
    """
    정서 설문지 3종(PHQ-9/MADRS/불안척도)을 항상 3행으로 보여주고
    각 행에 입력/수정 버튼 제공
    """
    section = ctk.CTkFrame(parent)
    section.pack(fill="x", pady=(10,10), padx=10)

    ctk.CTkLabel(section, text="📋 정서 설문지", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", pady=(5,3))

    rows = ctk.CTkFrame(section, fg_color="transparent")
    rows.pack(fill="x")

    for seq in (1,2,3):
        this_item = _find_emotion_item_by_seq(emotion_items, seq)
        title = EMOTION_TITLES_BY_SEQ.get(seq, f"정서설문-{seq}")

        row = ctk.CTkFrame(rows, fg_color="transparent")
        row.pack(fill="x", pady=4)
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=0)

        status_text = "미입력"
        collected_str = "미입력"
        bold = "normal"

        if this_item:
            has_data = bool(this_item.get("questions") or this_item.get("questions_raw"))
            status_text = "입력됨" if has_data else "생성됨(미입력)"
            bold = "bold" if has_data else "normal"
            collected_at = this_item.get("collected_at")
            try:
                if isinstance(collected_at, datetime):
                    collected_str = collected_at.strftime("%Y-%m-%d %H:%M")
                elif collected_at:
                    collected_str = str(collected_at).split(".")[0].replace("T"," ")
            except Exception:
                collected_str = "(날짜 오류)"

        ctk.CTkLabel(
            row,
            text=f"     {title} | 상태: {status_text} | 입력일시: {collected_str}",
            anchor="w",
            font=ctk.CTkFont(size=13, weight=bold)
        ).grid(row=0, column=0, sticky="w", padx=(10, 0))

        btn_text = "수정" if this_item and (this_item.get("questions") or this_item.get("questions_raw")) else "입력"
        btn_color = "#357ABD" if btn_text=="수정" else "#4CAF50"

        ctk.CTkButton(
            row,
            text=btn_text,
            fg_color=btn_color,
            hover_color=btn_color,
            width=70,
            command=(lambda s=seq, it=this_item: open_emotion_survey(s, it))
        ).grid(row=0, column=1, padx=6, sticky="e")

def _find_sleep_item_by_seq(items, seq: int):
    """items 리스트에서 data_type='S-SURVEY' AND seq==X 인 첫 항목 반환"""
    for it in items or []:
        if str(it.get("data_type","")).upper() == "S-SURVEY" and int(it.get("seq") or 0) == int(seq):
            return it
    return None


def open_sleep_survey(seq: int, existing_item: Optional[dict] = None):
    """
    수면 설문(ISI/KESS/PSQI/MEQ-K) 입력/수정 공용 오프너
    - 신규: tb_items 생성은 제출 시점에 수행(폼에서 처리)
    - 수정: 기존 item 정보 사용
    """
    if not selected_patient:
        messagebox.showwarning("환자 선택 필요", "먼저 환자를 선택해주세요.")
        return

    patient_uuid = selected_patient["patient_id"]

    title = SLEEP_TITLES_BY_SEQ.get(seq, f"SLEEP-{seq}")
    json_file_path = SLEEP_FORMS_BY_SEQ.get(seq)
    if not json_file_path:
        messagebox.showerror("오류", f"수면 설문 폼(json) 경로를 찾을 수 없습니다. seq={seq}")
        return

    if existing_item is None:
        item_data = {
            "patient_id": patient_uuid,
            "data_category": "MDD",
            "data_type": "S-SURVEY",
            "seq": seq,
            "title": title,
            "json_file": json_file_path
        }
    else:
        item_data = existing_item.copy()
        item_data.update({
            "patient_id": patient_uuid,
            "data_category": existing_item.get("data_category", "MDD"),
            "data_type": "S-SURVEY",
            "seq": existing_item.get("seq", seq),
            "title": title,
            "json_file": json_file_path
        })

    open_survey_form(item_data)


def render_sleep_section(parent, sleep_items: list):
    """
    수면 설문지 4종(ISI/KESS/PSQI/MEQ-K)을 항상 4행으로 보여주고
    각 행에 입력/수정 버튼 제공
    """
    section = ctk.CTkFrame(parent)
    section.pack(fill="x", pady=(10,10), padx=10)

    ctk.CTkLabel(section, text="😴 수면 설문지", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", pady=(5,3))

    rows = ctk.CTkFrame(section, fg_color="transparent")
    rows.pack(fill="x")

    for seq in (1,2,3,4):
        this_item = _find_sleep_item_by_seq(sleep_items, seq)
        title = SLEEP_TITLES_BY_SEQ.get(seq, f"수면설문-{seq}")

        row = ctk.CTkFrame(rows, fg_color="transparent")
        row.pack(fill="x", pady=4)
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=0)

        status_text = "미입력"
        collected_str = "미입력"
        bold = "normal"

        if this_item:
            has_data = bool(this_item.get("questions") or this_item.get("questions_raw"))
            status_text = "입력됨" if has_data else "생성됨(미입력)"
            bold = "bold" if has_data else "normal"
            collected_at = this_item.get("collected_at")
            try:
                if isinstance(collected_at, datetime):
                    collected_str = collected_at.strftime("%Y-%m-%d %H:%M")
                elif collected_at:
                    collected_str = str(collected_at).split(".")[0].replace("T"," ")
            except Exception:
                collected_str = "(날짜 오류)"

        ctk.CTkLabel(
            row,
            text=f"     {title} | 상태: {status_text} | 입력일시: {collected_str}",
            anchor="w",
            font=ctk.CTkFont(size=13, weight=bold)
        ).grid(row=0, column=0, sticky="w", padx=(10, 0))

        btn_text = "수정" if this_item and (this_item.get("questions") or this_item.get("questions_raw")) else "입력"
        btn_color = "#357ABD" if btn_text=="수정" else "#4CAF50"

        ctk.CTkButton(
            row,
            text=btn_text,
            fg_color=btn_color,
            hover_color=btn_color,
            width=70,
            command=(lambda s=seq, it=this_item: open_sleep_survey(s, it))
        ).grid(row=0, column=1, padx=6, sticky="e")



def process_items(items):
    """항목을 분류 후 섹션별 표시 (데이터 있으면 수정, 없으면 추가)"""
    basic_survey_items, emotion_survey_items, sleep_survey_items, file_items = [], [], [], []
    for item in items:
        data_type = str(item.get('data_type', '')).upper()
        if data_type == "B-SURVEY":
            basic_survey_items.append(item)
        elif data_type == "E-SURVEY":
            emotion_survey_items.append(item)
        elif data_type == "S-SURVEY":
            sleep_survey_items.append(item)
        elif "VIDEO" in data_type or "FILE" in data_type:
            file_items.append(item)
        else:
            file_items.append(item)

    # 기존 영역 초기화
    for widget in score_frame.winfo_children():
        widget.destroy()

    # 섹션 렌더 함수
    def render_section(title, survey_list, json_paths):
        section = ctk.CTkFrame(score_frame)
        section.pack(fill="x", pady=(10,10), padx=10)

        ctk.CTkLabel(section, text=f"📋 {title}", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", pady=(5,3))

        if survey_list and len(survey_list) > 0:
            # 데이터 있는 경우 → 수정 가능
            for item in survey_list:
                has_data = bool(item.get("questions"))
                collected_at = item.get("collected_at")
                try:
                    collected_str = collected_at.strftime("%Y-%m-%d %H:%M") if isinstance(collected_at, datetime) else str(collected_at).split(".")[0].replace("T", " ")
                except Exception:
                    collected_str = "(날짜 오류)"

                row = ctk.CTkFrame(section, fg_color="transparent")
                row.pack(fill="x", pady=3)
                row.grid_columnconfigure(0, weight=1)

                ctk.CTkLabel(
                    row,
                    text=f"{item['data_type']} | 입력일시: {collected_str if collected_at else '미입력'}",
                    font=ctk.CTkFont(size=13, weight="bold" if has_data else "normal"),
                    anchor="w"
                ).grid(row=0, column=0, sticky="w")
                
                ctk.CTkButton(
                    row,
                    text="수정" if has_data else "입력",
                    fg_color="#357ABD" if has_data else "#4CAF50",
                    hover_color="#285EAD" if has_data else "#3E9B41",
                    width=70,
                    command=lambda d=item: open_survey_form(d)
                ).grid(row=0, column=1, padx=5, sticky="e")
        else:
            # 데이터 없는 경우 → 추가 버튼
            ctk.CTkLabel(section, text=f"{title} 데이터가 없습니다.", font=("",13,"italic"), text_color="gray").pack(pady=(10,5))
            def open_new_form():
                if not json_paths: return messagebox.showerror("오류", f"{title} 폼이 없습니다.")
                json_file = json_paths[0]
                new_item = {
                    "data_category":"MDD",
                    "data_type":title,
                    "seq":1,
                    "json_file":json_file
                }
                open_survey_form(new_item)

            ctk.CTkButton(
                section,
                text=f"➕ {title} 입력하기",
                font=("",13,"bold"),
                fg_color="#007BFF",
                hover_color="#0056b3",
                command=open_new_form
            ).pack(pady=(5,10))

    # 섹션별 출력
    render_section("기초 평가", basic_survey_items, [os.path.join(PROJECT_ROOT,"form","basic_form","basic.json")])
    render_emotion_section(score_frame, emotion_survey_items)
    # render_section("정서 설문지", emotion_survey_items, [
    #     os.path.join(PROJECT_ROOT,"form","emotion_form","phq9.json"),
    #     os.path.join(PROJECT_ROOT,"form","emotion_form","MADRS.json"),
    #     os.path.join(PROJECT_ROOT, "form", "emotion_form", "anxiety_disorder.json")
    # ])
    # render_section("수면 설문지", sleep_survey_items, [
    #     os.path.join(PROJECT_ROOT, "form", "sleep_form", "ISI.json"),
    #     os.path.join(PROJECT_ROOT, "form", "sleep_form", "KESS.json"),
    #     os.path.join(PROJECT_ROOT, "form", "sleep_form", "PSQI.json"),
    #     os.path.join(PROJECT_ROOT, "form", "sleep_form","MEQ_K.json")
    # ])
    render_sleep_section(score_frame, sleep_survey_items)
    show_file_items(upload_list_frame, file_items)

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
        # created_ts = patient.get("created_ts")
        is_data_complete = patient.get("is_data_complete", False)

        if birth != "생년월일 없음":
            try:
                birth = datetime.strptime(birth, "%Y-%m-%d").strftime("%Y-%m-%d")
            except Exception:
                pass

        created_ts_val = patient.get("created_ts")

        if created_ts_val:
            try:
                # 🧩 이미 datetime 객체면 그대로
                if isinstance(created_ts_val, datetime):
                    dt_obj = created_ts_val
                else:
                    # 문자열이면 fromisoformat으로 자동 처리
                    dt_obj = datetime.fromisoformat(str(created_ts_val).split(".")[0])
                created_ts = dt_obj.strftime("%Y-%m-%d %H:%M")
            except Exception as e:
                print(f"[등록일시 파싱 오류] {created_ts_val} → {e}")
                created_ts = "?"
        else:
            created_ts = "?"
        if is_data_complete:
            text_color = "gray60"   # 연한 회색 (뿌옇게)
            text_weight = "normal"
        else:
            text_color = "black"
            text_weight = "normal"

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
        # try:
        #     add_patient(payload, INSTITUTION)
        #     messagebox.showinfo("성공", "환자 등록 완료")
        # except Exception as e:
        #     messagebox.showerror("오류", f"환자 등록 실패: {e}")
        
        # load_patients_table()
        # modal.destroy()
        def after_register(_):
            CTkMessagebox(title="성공", message="환자 등록 완료", icon="check")
            load_patients_table()
            modal.destroy()

        def on_error(e):
            CTkMessagebox(title="오류", message=f"환자 등록 실패: {e}", icon="cancel")

        run_with_loading(
            parent_frame=modal,
            fetch_function=lambda: add_patient(payload, INSTITUTION),
            callback=after_register,
            loading_text="환자 등록 중입니다..."
        )

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
# upload_button_frame = ctk.CTkFrame(frame_video)
# upload_button_frame.pack(pady=5)
# ctk.CTkButton(upload_button_frame, text="📤 파일 업로드",font=("",16) ,width=150, command=open_upload_modal).pack(side="left", padx=10)

# 파일 목록을 표시할 프레임 (show_file_items에서 관리)
upload_list_frame = ctk.CTkFrame(frame_video)
upload_list_frame.pack(fill="x", pady=5)


# 프로그램 시작 시 상태 설정
show_empty_state()
# 파일 영역 초기 상태 설정
# show_file_items([])
root.after(100, init_program)
root.mainloop() 