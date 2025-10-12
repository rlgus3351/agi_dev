import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json
import os
import tkinter.filedialog as filedialog
from datetime import datetime
import requests
from typing import Union

# JSON 파일 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'form', 'mobility.json'))

# API 기본 URL
API_BASE_URL = "http://127.0.0.1:30000"

MDS_QUESTION_MAPPING = {
    "a": 1, "b": 2, "c": 3, "c1": 4, "d": 5, "d1": 6, "d2": 7, "e": 8,
    "1": 9, "2": 10, "3": 11, "4": 12, "5": 13, "6": 14, "7": 15, "8": 16,
    "9": 17, "10": 18, "11": 19, "12": 20, "13": 21, "14": 22, "15": 23,
    "16": 24, "17": 25, "18": 26,
}


class HealthSurveyForm(ctk.CTkFrame):
    def __init__(self, parent, patient_id, json_file=JSON_FILE, initial_data=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.json_file = json_file
        self.patient_id = patient_id
        self.initial_data = initial_data or []
        self.widgets = {}
        self.data_vars = {}
        self.scrollable_frame = None
        self.vcmd = (self.register(self.validate_number_input), '%P', '%S', '%V', '%W')

        # ✅ 초기값 매핑 테이블 준비
        self.initial_answers_map = {}
        for ans in self.initial_data:
            qid = ans.get("question_id")
            comp = ans.get("answer_component")
            val = ans.get("answer_value")
            if qid is not None:
                key = f"{qid}_{comp}" if comp else str(qid)
                self.initial_answers_map[key] = str(val)

        self.load_data_and_create_widgets()

    def validate_number_input(self, new_value, changed_char, validation_type, widget_name):
        if new_value == "":
            return True
        if changed_char.isdigit():
            try:
                int(new_value)
                return True
            except ValueError:
                return False
        else:
            return False

    def validate_number_input_with_range(self, new_value, min_str, max_str):
        if new_value == "":
            return True
        try:
            num_value = int(new_value)
        except ValueError:
            return False

        min_val = int(min_str) if min_str and min_str != 'None' else None
        max_val = int(max_str) if max_str and max_str != 'None' else None

        if max_val is not None and num_value > max_val:
            return False
        if min_val is not None and num_value < min_val:
            return False
        return True

    def scroll_to_widget(self, event):
        if not self.scrollable_frame or not self.scrollable_frame.winfo_exists():
            return
        widget = event.widget
        margin = 30
        try:
            canvas = self.scrollable_frame._parent_canvas
        except AttributeError:
            return
        y_pos = widget.winfo_y()
        widget_height = widget.winfo_height()
        frame_height = self.scrollable_frame.winfo_height()
        scroll_region_str = canvas.cget("scrollregion")
        if not scroll_region_str:
            return
        _, y_min, _, y_max = map(int, scroll_region_str.split())
        total_canvas_height = y_max - y_min
        if total_canvas_height <= frame_height:
            return
        y_scroll_ratio = canvas.yview()[0]
        y_current_top_pixel = int(total_canvas_height * y_scroll_ratio)
        y_relative_top = y_pos - y_current_top_pixel
        if y_relative_top < margin:
            target_y_pixel = y_pos - margin
            new_y_ratio = target_y_pixel / total_canvas_height
            canvas.yview_moveto(max(0, new_y_ratio))
        elif y_relative_top + widget_height > frame_height - margin:
            target_y_pixel = y_pos + widget_height - frame_height + margin
            new_y_ratio = target_y_pixel / total_canvas_height
            canvas.yview_moveto(min(1.0, new_y_ratio))

    def load_data_and_create_widgets(self):
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.survey_data = json.load(f)
        except FileNotFoundError:
            ctk.CTkLabel(self, text=f"오류: {self.json_file} 파일을 찾을 수 없습니다.").pack(pady=20)
            return

        scrollable_frame = ctk.CTkScrollableFrame(self, height=700)
        scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        self.scrollable_frame = scrollable_frame

        row = 0
        for section in self.survey_data.get("운동성 검사", {}).get("sections", []):
            header = section.get("header", {})
            body = section.get("body", [])

            header_label = ctk.CTkLabel(
                scrollable_frame,
                text=header.get("title", "") + f"\n{header.get('description', '')}",
                font=ctk.CTkFont(size=16, weight="bold"),
                anchor="center",
                justify="center",
                wraplength=700
            )
            header_label.grid(row=row, column=0, columnspan=2, sticky="new", pady=(20, 15))
            row += 1

            for item in body:
                row = self._create_widget(scrollable_frame, item, row)

        ctk.CTkButton(scrollable_frame, text="데이터 저장", command=self.get_entered_data).grid(
            row=row, column=0, columnspan=2, pady=(20, 5), sticky="ew"
        )
        row += 1

        ctk.CTkButton(
            scrollable_frame,
            text="로컬 파일 재전송 (API 복구)",
            command=self.load_and_resubmit_data,
            fg_color="darkgreen",
            hover_color="green"
        ).grid(row=row, column=0, columnspan=2, pady=(5, 30), sticky="ew")

    def _create_widget(self, parent_frame, config, row):
        item_type = config.get('type')
        question = config.get('question')
        item_id = config.get('id')

        if item_type == "radio":
            var = ctk.StringVar()
            self.data_vars[item_id] = var
            qnum = MDS_QUESTION_MAPPING.get(item_id)
            init_val = self.initial_answers_map.get(str(qnum))
            if init_val:
                var.set(init_val)

            ctk.CTkLabel(parent_frame, text=question, font=('', 14),
                         justify="left", wraplength=450).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            radio_frame = ctk.CTkFrame(parent_frame)
            radio_frame.grid(row=row, column=1, sticky="w", padx=10, pady=10)
            for i, option in enumerate(config.get('options', [])):
                radio_btn = ctk.CTkRadioButton(radio_frame, text=option, variable=var, value=option)
                radio_btn.grid(row=0, column=i, padx=5)
                radio_btn.bind("<FocusIn>", self.scroll_to_widget)
            row += 1

        elif item_type == "input-number":
            var = ctk.StringVar()
            self.data_vars[item_id] = var
            qnum = MDS_QUESTION_MAPPING.get(item_id)
            init_val = self.initial_answers_map.get(str(qnum))
            if init_val:
                var.set(init_val)

            min_val = config.get('min', None)
            max_val = config.get('max', None)
            dynamic_vcmd = (self.register(self.validate_number_input_with_range),
                            '%P', str(min_val), str(max_val))
            ctk.CTkLabel(parent_frame, text=question, font=('', 14),
                         justify="left", wraplength=450).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            entry = ctk.CTkEntry(parent_frame, textvariable=var, width=100,
                                 validate='key', validatecommand=dynamic_vcmd)
            entry.grid(row=row, column=1, sticky="w", padx=10, pady=10)
            entry.bind("<FocusIn>", self.scroll_to_widget)
            row += 1

        elif item_type == "grouped-inputs":
            sides = config.get("sides", [])
            qnum = MDS_QUESTION_MAPPING.get(item_id)
            ctk.CTkLabel(parent_frame, text=question, font=('', 14),
                         justify="left", wraplength=450).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            side_frame = ctk.CTkFrame(parent_frame)
            side_frame.grid(row=row, column=1, sticky="w", padx=10, pady=10)
            row += 1
            for i, side in enumerate(sides):
                sid = f"{item_id}_{side}"
                var = ctk.StringVar()
                self.data_vars[sid] = var
                if qnum:
                    key = f"{qnum}_{side}"
                    init_val = self.initial_answers_map.get(key)
                    if init_val:
                        var.set(init_val)
                ctk.CTkLabel(side_frame, text=side).grid(row=i, column=0, sticky="w", padx=5, pady=2)
                entry = ctk.CTkEntry(side_frame, textvariable=var, width=80,
                                     validate='key', validatecommand=self.vcmd)
                entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
                entry.bind("<FocusIn>", self.scroll_to_widget)
        return row

    # -------------------------------------------------------------
    # 데이터 변환 / API 전송
    # -------------------------------------------------------------
    def transform_to_api_format(self, raw_data: dict) -> list:
        answers = []
        for key, value in raw_data.items():
            value = value.strip()
            if not value:
                continue
            if "_" in key:
                json_id, component = key.split("_", 1)
            else:
                json_id, component = key, None
            question_db_id = MDS_QUESTION_MAPPING.get(json_id)
            if question_db_id is not None:
                answers.append({
                    "question_id": question_db_id,
                    "answer_component": component if component else None,
                    "answer_value": int(value) if value.isdigit() else value
                })
        return answers

    def create_new_item_and_get_id(self, target_patient_id: str) -> Union[int, None]:
        url = f"{API_BASE_URL}/items/{target_patient_id}/item"
        payload = {
            "patient_id": target_patient_id,
            "data_category": "PD",
            "data_type": "MDS-UPDRS Part 3",
            "seq": 1,
            "description": "MDS-UPDRS Part 3 설문 응답",
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            return response.json().get("item_id")
        except requests.exceptions.RequestException as e:
            CTkMessagebox(title="API 오류 (1단계)",
                          message=f"수집 항목 등록 실패: {e}", icon="cancel")
            return None

    def call_api_to_save_data(self, item_id: int, answers_list: list) -> bool:
        url = f"{API_BASE_URL}/mds/{item_id}"
        try:
            response = requests.post(url, json={"answers": answers_list}, timeout=10)
            response.raise_for_status()
            CTkMessagebox(title="API 저장 성공",
                          message=f"설문 응답이 서버에 등록되었습니다. (Item ID: {item_id})", icon="check")
            if self.master and hasattr(self.master, 'destroy') and not self.is_resubmitting:
                self.master.destroy()
            return True
        except requests.exceptions.RequestException as e:
            CTkMessagebox(title="API 오류 (2단계)", message=str(e), icon="cancel")
            return False

    def get_entered_data(self):
        self.is_resubmitting = False
        if not self.patient_id or len(self.patient_id) < 16:
            CTkMessagebox(title="오류", message="유효한 환자 정보(UUID)가 없습니다.", icon="cancel")
            return
        raw_data = {key: var.get() for key, var in self.data_vars.items()}
        answers_list = self.transform_to_api_format(raw_data)
        if not answers_list:
            CTkMessagebox(title="경고", message="입력된 응답 데이터가 없습니다.", icon="warning")
            return
        item_id = self.create_new_item_and_get_id(self.patient_id)
        if item_id is not None:
            success = self.call_api_to_save_data(item_id, answers_list)
        else:
            success = False
        if not success:
            self.save_to_json_file({
                "metadata": {
                    "patient_id": self.patient_id,
                    "item_id": item_id,
                    "created_at": datetime.now().isoformat()
                },
                "answers": answers_list
            }, prompt_save=True)

    def save_to_json_file(self, data, prompt_save=False):
        try:
            pid_prefix = data['metadata'].get('patient_id', '')[:8]
            default_filename = f"MDS_UPDRS_Part3_{pid_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                     initialfile=default_filename,
                                                     filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
            if not file_path:
                return
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            if prompt_save:
                CTkMessagebox(title="로컬 저장 완료",
                              message=f"설문 응답이 저장되었습니다:\n{file_path}", icon="check")
        except Exception as e:
            CTkMessagebox(title="저장 오류", message=str(e), icon="cancel")

    def load_and_resubmit_data(self):
        self.is_resubmitting = True
        file_path = filedialog.askopenfilename(defaultextension=".json",
                                               filetypes=[("JSON files", "*.json")])
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                recovery = json.load(f)
            metadata = recovery.get('metadata', {})
            answers = recovery.get('answers', [])
            pid = metadata.get('patient_id')
            item_id = metadata.get('item_id')
            if not pid or len(pid) < 16:
                CTkMessagebox(title="오류", message="유효한 환자 ID가 없습니다.", icon="cancel")
                return
            if item_id is None:
                item_id = self.create_new_item_and_get_id(pid)
            if item_id:
                ok = self.call_api_to_save_data(item_id, answers)
                if ok:
                    os.remove(file_path)
                    CTkMessagebox(title="재전송 성공", message="로컬 백업 파일이 삭제되었습니다.", icon="check")
        except Exception as e:
            CTkMessagebox(title="재전송 오류", message=str(e), icon="cancel")
        self.is_resubmitting = False
