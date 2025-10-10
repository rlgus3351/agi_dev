import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json
import os
import tkinter.filedialog as filedialog
from datetime import datetime

# JSON 파일 경로 설정 (절대경로 권장)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'form', 'mobility.json'))


MDS_QUESTION_MAPPING = {
    # DB 삽입 순서: 1~8번 (기초 정보)
    "a": 1, "b": 2, "c": 3, "c1": 4, "d": 5, "d1": 6, "d2": 7, "e": 8,
    # DB 삽입 순서: 9~26번 (운동 항목별 평가)
    "1": 9, "2": 10, "3": 11, "4": 12, "5": 13, "6": 14, "7": 15, "8": 16,
    "9": 17, "10": 18, "11": 19, "12": 20, "13": 21, "14": 22, "15": 23, 
    "16": 24, "17": 25, "18": 26,
}


class HealthSurveyForm(ctk.CTkFrame):
    def __init__(self, parent, json_file=JSON_FILE, patient_id=None):
        super().__init__(parent)
        self.json_file = json_file
        self.patient_id = patient_id  # 환자 UUID
        self.widgets = {}
        self.data_vars = {}
        self.scrollable_frame = None  # 스크롤 프레임 초기화

        self.vcmd = (self.register(self.validate_number_input), '%P', '%S', '%V', '%W')
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

    def scroll_to_widget(self, event):
        """
        포커스를 받은 위젯이 뷰포트 영역을 벗어났을 때, 최소한의 스크롤로 보이게 조정합니다.
        """
        if not self.scrollable_frame or not self.scrollable_frame.winfo_exists():
            return
            
        widget = event.widget
        margin = 30 # 위젯 주변에 둘 여유 공간 (픽셀)
        
        try:
            # CTkScrollableFrame의 내부 캔버스에 접근합니다.
            canvas = self.scrollable_frame._parent_canvas 
        except AttributeError:
            return
        
        # 1. 위젯의 y 좌표 및 크기
        y_pos = widget.winfo_y()
        widget_height = widget.winfo_height()
        frame_height = self.scrollable_frame.winfo_height() # 뷰포트(스크롤 프레임)의 높이

        # 2. 캔버스 총 높이 및 현재 스크롤 위치 계산
        scroll_region_str = canvas.cget("scrollregion")
        if not scroll_region_str:
            return
            
        _, y_min, _, y_max = map(int, scroll_region_str.split())
        total_canvas_height = y_max - y_min
        
        if total_canvas_height <= frame_height: # 스크롤이 필요 없으면 종료
            return
            
        y_scroll_ratio = canvas.yview()[0] 
        y_current_top_pixel = int(total_canvas_height * y_scroll_ratio)
        
        # 3. 위젯의 위치 (뷰포트 상단 기준 상대 좌표)
        y_relative_top = y_pos - y_current_top_pixel 

        # --- 스크롤 조정 로직 ---

        # Case 1: 위젯이 뷰포트 상단 위로 가려졌을 경우 (위로 스크롤)
        if y_relative_top < margin: 
            target_y_pixel = y_pos - margin
            new_y_ratio = target_y_pixel / total_canvas_height
            canvas.yview_moveto(max(0, new_y_ratio))

        # Case 2: 위젯이 뷰포트 하단 아래로 가려졌을 경우 (아래로 스크롤)
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
        self.scrollable_frame = scrollable_frame  # ⬅️ 스크롤 프레임 저장

        row = 0
        for section in self.survey_data.get("운동성 검사", {}).get("sections", []):
            header = section.get("header", {})
            body = section.get("body", [])

            header_label = ctk.CTkLabel(
                scrollable_frame,
                text=header.get("title", "") + f"\n{header.get('description', '')}",
                font=ctk.CTkFont(size=16, weight="bold"),
                anchor="w",             # 왼쪽 정렬
                justify="center",         # 왼쪽 정렬
                wraplength=700          # 자동 줄 바꿈
            )
            header_label.grid(row=row, column=0, columnspan=2, sticky="nsw", pady=(20, 15))
            row += 1

            for item in body:
                row = self._create_widget(scrollable_frame, item, row)

        submit_button = ctk.CTkButton(scrollable_frame, text="데이터 저장", command=self.get_entered_data)
        submit_button.grid(row=row, column=0, columnspan=2, pady=(20, 30), sticky="ew")

    def _create_widget(self, parent_frame, config, row):
        item_type = config.get('type')
        question = config.get('question')
        item_id = config.get('id')

        if item_type == "radio":
            var = ctk.StringVar()
            self.data_vars[item_id] = var

            ctk.CTkLabel(parent_frame, text=question, font=('', 14), justify="left", wraplength=450).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            radio_frame = ctk.CTkFrame(parent_frame)
            radio_frame.grid(row=row, column=1, sticky="w", padx=10, pady=10)

            for i, option in enumerate(config.get('options', [])):
                radio_btn = ctk.CTkRadioButton(
                    radio_frame,
                    text=option,
                    variable=var,
                    value=option
                )
                radio_btn.grid(row=0, column=i, padx=5)
                # ⬅️ 이벤트 바인딩 추가
                radio_btn.bind("<FocusIn>", self.scroll_to_widget) 
            row += 1

        elif item_type == "input-number":
            var = ctk.StringVar()
            self.data_vars[item_id] = var

            ctk.CTkLabel(parent_frame, text=question, font=('', 14), justify="left", wraplength=450).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            entry = ctk.CTkEntry(
                parent_frame,
                textvariable=var,
                width=100,
                validate='key',
                validatecommand=self.vcmd
            )
            entry.grid(row=row, column=1, sticky="w", padx=10, pady=10)
            # ⬅️ 이벤트 바인딩 추가
            entry.bind("<FocusIn>", self.scroll_to_widget)
            row += 1

        elif item_type == "grouped-inputs":
            sides = config.get("sides", [])

            ctk.CTkLabel(parent_frame, text=question, font=('', 14), justify="left", wraplength=450).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            side_frame = ctk.CTkFrame(parent_frame)
            side_frame.grid(row=row, column=1, sticky="w", padx=10, pady=10)
            row += 1

            for i, side in enumerate(sides):
                sid = f"{item_id}_{side}"
                var = ctk.StringVar()
                self.data_vars[sid] = var

                ctk.CTkLabel(side_frame, text=side).grid(row=i, column=0, sticky="w", padx=5, pady=2)
                entry = ctk.CTkEntry(
                    side_frame,
                    textvariable=var,
                    width=80,
                    validate='key',
                    validatecommand=self.vcmd
                )
                entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
                # ⬅️ 이벤트 바인딩 추가
                entry.bind("<FocusIn>", self.scroll_to_widget)

        return row
    
    # ... (데이터 처리 함수 유지) ...

    def transform_to_api_format(self, raw_data: dict) -> list:
        answers = []
        for key, value in raw_data.items():
            value = value.strip()
            if not value: 
                continue

            if "_" in key:
                json_id, component = key.split("_", 1)
            else:
                json_id = key
                component = None
            
            question_db_id = MDS_QUESTION_MAPPING.get(json_id)
            
            if question_db_id is not None:
                answer = {
                    "question_id": question_db_id,
                    "answer_component": component if component else None,
                    "answer_value": value 
                }
                answers.append(answer)
        return answers


    def get_entered_data(self):
        if not self.patient_id:
            CTkMessagebox(title="오류", message="환자 정보(UUID)가 없습니다.", icon="cancel")
            return
            
        raw_data = {}
        for key, var in self.data_vars.items():
            raw_data[key] = var.get()

        answers_list = self.transform_to_api_format(raw_data)
        
        if not answers_list:
            CTkMessagebox(title="경고", message="입력된 응답 데이터가 없습니다.", icon="warning")
            return

        submission_data = {
            "metadata": {
                "patient_id": self.patient_id, 
                "survey_type": "운동성 검사 (MDS-UPDRS Part III)",
                "created_at": datetime.now().isoformat()
            },
            "answers": answers_list
        }
        
        self.save_to_json_file(submission_data)

    def save_to_json_file(self, data):
        try:
            pid_prefix = self.patient_id[:8] if self.patient_id else "NoPID"
            default_filename = f"MDS_UPDRS_Part3_{pid_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile=default_filename,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not file_path:
                return

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            CTkMessagebox(title="저장 완료", message=f"설문 응답이 다음 파일에 저장되었습니다:\n{file_path}", icon="check")
            
        except Exception as e:
            CTkMessagebox(title="저장 오류", message=f"JSON 파일 저장 중 오류 발생: {e}", icon="cancel")