import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json
import os

# JSON 파일 경로 설정 (절대경로 권장)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'form', 'mobility.json'))

class HealthSurveyForm(ctk.CTkFrame):
    def __init__(self, parent, json_file=JSON_FILE):
        super().__init__(parent)
        self.json_file = json_file

        self.widgets = {}
        self.data_vars = {}

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

        row = 0
        for section in self.survey_data.get("운동성 검사", {}).get("sections", []):
            header = section.get("header", {})
            body = section.get("body", [])

            header_label = ctk.CTkLabel(
                scrollable_frame,
                text=header.get("title", "") + f"\n{header.get('description', '')}",
                font=ctk.CTkFont(size=16, weight="bold"),
                anchor="center",
                justify="center"
            )
            header_label.grid(row=row, column=0, columnspan=2, sticky="n", pady=(20, 15))
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

            ctk.CTkLabel(parent_frame, text=question, font=('', 14)).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            radio_frame = ctk.CTkFrame(parent_frame)
            radio_frame.grid(row=row, column=1, sticky="w", padx=10, pady=10)

            for i, option in enumerate(config.get('options', [])):
                ctk.CTkRadioButton(
                    radio_frame,
                    text=option,
                    variable=var,
                    value=option
                ).grid(row=0, column=i, padx=5)
            row += 1

        elif item_type == "input-number":
            var = ctk.StringVar()
            self.data_vars[item_id] = var

            ctk.CTkLabel(parent_frame, text=question, font=('', 14)).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            entry = ctk.CTkEntry(
                parent_frame,
                textvariable=var,
                width=100,
                validate='key',
                validatecommand=self.vcmd
            )
            entry.grid(row=row, column=1, sticky="w", padx=10, pady=10)
            row += 1

        elif item_type == "grouped-inputs":
            sides = config.get("sides", [])

            ctk.CTkLabel(parent_frame, text=question, font=('', 14)).grid(row=row, column=0, sticky="w", padx=10, pady=10)
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

        return row

    def get_entered_data(self):
        result = {}
        for key, var in self.data_vars.items():
            result[key] = var.get()

        print("--- 입력된 데이터 ---")
        print(json.dumps(result, ensure_ascii=False, indent=4))
        CTkMessagebox(title="성공", message="데이터가 성공적으로 저장되었습니다.", icon="check")
