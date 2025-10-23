import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json, os
import psycopg2
from datetime import datetime
from api_local.schemas import Item
from api_local.form_api_local import (
    BASIC_QUESTION_MAPPING,
    transform_to_api_format,
    create_new_item_and_get_id,
    save_mds_answers,
    update_mds_answers
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'form', 'basic_form', 'basic.json'))


class HealthSurveyForm(ctk.CTkFrame):
    """기초 평가 (Basic Health Form) UI"""

    def __init__(self, parent, patient_id: str, json_file=DEFAULT_JSON, item_data=None, on_close_callback=None):
        super().__init__(parent)
        self.json_file = json_file
        self.patient_id = patient_id
        self.item_data = item_data
        self.on_close_callback = on_close_callback

        self.widgets = {}
        self.data_vars = {}
        self.option_subs = {}

        # 숫자 입력 검증
        self.vcmd = (self.register(self.validate_number_input), '%P', '%S', '%V', '%W')

        # 폼 구성
        self.load_data_and_create_widgets()

    # ----------------------------
    # ✅ 숫자 입력 검증
    # ----------------------------
    def validate_number_input(self, new_value, changed_char, validation_type, widget_name):
        if new_value == "":
            return True
        return changed_char.isdigit()

    # ----------------------------
    # ✅ 폼 구성 로드
    # ----------------------------
    def load_data_and_create_widgets(self):
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.survey_data = json.load(f)
        except FileNotFoundError:
            ctk.CTkLabel(self, text=f"오류: {self.json_file} 파일을 찾을 수 없습니다.").pack(pady=20)
            return

        scrollable_frame = ctk.CTkScrollableFrame(self, height=650)
        scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)

        row_num = 0
        for section, items in self.survey_data.items():
            ctk.CTkLabel(
                scrollable_frame,
                text=section.replace('_', ' ').capitalize(),
                font=ctk.CTkFont(size=18, weight="bold")
            ).grid(row=row_num, column=0, columnspan=2, pady=(10, 5), sticky="w")
            row_num += 1

            for key, config in items.items():
                row_num = self._create_widget(scrollable_frame, key, config, row_num)

        ctk.CTkButton(self, text="데이터 저장", command=self.get_entered_data).pack(pady=10)

    # ----------------------------
    # ✅ 위젯 생성 (재귀)
    # ----------------------------
    def _create_widget(self, parent_frame, key, config, row):
        item_type = config.get('type')
        label_text = config.get('label') or key

        # 라디오
        if item_type == "radio":
            radio_var = ctk.StringVar()
            self.data_vars[key] = radio_var

            ctk.CTkLabel(parent_frame, text=label_text, font=("", 14)).grid(row=row, column=0, sticky="w", padx=10, pady=(10, 0))
            radio_frame = ctk.CTkFrame(parent_frame)
            radio_frame.grid(row=row, column=1, sticky="w", padx=10, pady=(10, 0))

            self.option_subs[key] = {}
            col = 0

            for option in config.get("options", []):
                if isinstance(option, str):
                    ctk.CTkRadioButton(radio_frame, text=option, variable=radio_var, value=option)\
                        .grid(row=0, column=col, sticky="w", padx=5)
                    col += 1
                elif isinstance(option, dict):
                    opt_label = option.get("label", "")
                    opt_sub = option.get("sub", {})
                    sub_frame = ctk.CTkFrame(parent_frame)
                    sub_frame.grid(row=row + 1, column=0, columnspan=2, sticky="w", padx=30, pady=(5, 10))
                    sub_frame.grid_remove()

                    sub_row = 0
                    for sub_key, sub_cfg in opt_sub.items():
                        sub_row = self._create_widget(sub_frame, sub_key, sub_cfg, sub_row)

                    ctk.CTkRadioButton(
                        radio_frame, text=opt_label, variable=radio_var, value=opt_label,
                        command=lambda k=key, v=opt_label: self._toggle_radio_sub(k, v)
                    ).grid(row=0, column=col, sticky="w", padx=5)
                    col += 1

                    self.option_subs[key][opt_label] = sub_frame

            radio_var.trace_add("write", lambda *a, k=key: self._toggle_radio_sub(k, radio_var.get()))
            separator = ctk.CTkFrame(parent_frame, height=4, fg_color="#CCCCCC")
            separator.grid(row=row + 2, column=0, columnspan=2, sticky="ew", pady=(10, 5))
            return row + 3

        # 체크박스
        elif item_type == "checkbox":
            ctk.CTkLabel(parent_frame, text=label_text, font=("", 14)).grid(row=row, column=0, sticky="w", padx=10, pady=(10, 0))
            box_frame = ctk.CTkFrame(parent_frame)
            box_frame.grid(row=row, column=1, sticky="w", padx=10, pady=(10, 0))

            for i, opt in enumerate(config.get("options", [])):
                if isinstance(opt, str):
                    var = ctk.BooleanVar()
                    ctk.CTkCheckBox(box_frame, text=opt, variable=var).grid(row=i, column=0, sticky="w", pady=2)
                    self.data_vars[f"{key}_{opt}"] = var
                elif isinstance(opt, dict):
                    opt_label = opt.get("label")
                    sub_opt = opt.get("sub", {})
                    var = ctk.BooleanVar()
                    ctk.CTkCheckBox(box_frame, text=opt_label, variable=var).grid(row=i, column=0, sticky="w", pady=2)
                    self.data_vars[f"{key}_{opt_label}"] = var

                    sub_frame = ctk.CTkFrame(box_frame)
                    sub_frame.grid(row=i + 1, column=0, sticky="w", padx=30, pady=(3, 6))
                    sub_frame.grid_remove()
                    sub_row = 0
                    for sub_key, sub_cfg in sub_opt.items():
                        sub_row = self._create_widget(sub_frame, sub_key, sub_cfg, sub_row)

                    def toggle_sub(*args, v=var, frame=sub_frame):
                        frame.grid() if v.get() else frame.grid_remove()
                    var.trace_add("write", toggle_sub)

            separator = ctk.CTkFrame(parent_frame, height=4, fg_color="#CCCCCC")
            separator.grid(row=row + 2, column=0, columnspan=2, sticky="ew", pady=(10, 5))
            return row + 3

        # 숫자 입력
        elif item_type == "input-number":
            input_var = ctk.StringVar()
            self.data_vars[key] = input_var
            ctk.CTkLabel(parent_frame, text=label_text, font=("", 14)).grid(row=row, column=0, sticky="w", padx=10, pady=(10, 0))
            ctk.CTkEntry(
                parent_frame, textvariable=input_var, width=120,
                validate='key', validatecommand=self.vcmd,
                placeholder_text=config.get("placeholder", "")
            ).grid(row=row, column=1, sticky="w", padx=10, pady=(10, 0))
            separator = ctk.CTkFrame(parent_frame, height=4, fg_color="#CCCCCC")
            separator.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(10, 5))
            return row + 2

        # 텍스트 입력
        elif item_type == "input-text":
            input_var = ctk.StringVar()
            self.data_vars[key] = input_var
            ctk.CTkLabel(parent_frame, text=label_text, font=("", 14)).grid(row=row, column=0, sticky="w", padx=10, pady=(10, 0))
            ctk.CTkEntry(parent_frame, textvariable=input_var, width=300,
                         placeholder_text=config.get("placeholder", "")
            ).grid(row=row, column=1, sticky="w", padx=10, pady=(10, 0))
            separator = ctk.CTkFrame(parent_frame, height=4, fg_color="#CCCCCC")
            separator.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(10, 5))
            return row + 2

        return row + 1

    # ----------------------------
    # ✅ 라디오 하위 토글
    # ----------------------------
    def _toggle_radio_sub(self, key, value):
        if key in self.option_subs:
            for label, frame in self.option_subs[key].items():
                frame.grid() if label == value else frame.grid_remove()

    # ----------------------------
    # ✅ 데이터 저장
    # ----------------------------
    def get_entered_data(self):
        result = {}
        answers = []

        for key, var in self.data_vars.items():
            if isinstance(var, (ctk.StringVar, ctk.BooleanVar)):
                value = var.get()
                result[key] = value

                if key in BASIC_QUESTION_MAPPING:
                    qid = BASIC_QUESTION_MAPPING[key]
                    answers.append({
                        "question_id": qid,
                        "answer_value": value,
                        "answer_component": None
                    })

        result["patient_id"] = self.patient_id
        print("\n--- 입력된 데이터 ---")
        print(json.dumps(result, ensure_ascii=False, indent=4))
        print("\n--- DB 저장용 answers ---")
        print(json.dumps(answers, ensure_ascii=False, indent=4))

        # DB 저장
        self.save_answers_to_db(answers)
        CTkMessagebox(title="성공", message="DB에 저장되었습니다.", icon="check")

        if self.on_close_callback:
            self.on_close_callback()

    def save_answers_to_db(self, answers):
        """DB 저장 로직"""
        if not answers:
            print("❌ 저장할 데이터가 없습니다.")
            return

        conn = psycopg2.connect(
            host="121.178.59.41",
            port="45432",
            dbname="agi_dev",
            user="kkh",
            password="Rkskekfk1!"
        )
        cur = conn.cursor()

        insert_sql = """
            INSERT INTO dev_kkh.tb_questionnaire_answers
                (item_id, question_id, answer_component, answer_value, submission_datetime)
            VALUES (%s, %s, %s, %s, %s)
        """

        now = datetime.now()
        item_id = getattr(self.item_data, "item_id", 0) if self.item_data else 0

        for ans in answers:
            cur.execute(insert_sql, (
                item_id,
                ans["question_id"],
                ans.get("answer_component"),
                ans["answer_value"],
                now
            ))

        conn.commit()
        conn.close()
        print(f"✅ {len(answers)}개 답변 저장 완료")
