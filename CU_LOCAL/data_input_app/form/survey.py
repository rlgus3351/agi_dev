import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json, os
from api_local.form_api_local import (
    BASIC_QUESTION_MAPPING,
    create_new_item_and_get_id_generic,
    save_answers,
    mark_item_updated
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

        self.vcmd = (self.register(self.validate_number_input), '%P', '%S', '%V', '%W')
        self.load_data_and_create_widgets()

    # ✅ 숫자 입력 검증
    def validate_number_input(self, new_value, changed_char, validation_type, widget_name):
        if new_value == "":
            return True
        return changed_char.isdigit()

    # ✅ 폼 로드 및 UI 생성
    def load_data_and_create_widgets(self):
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.survey_data = json.load(f)
        except FileNotFoundError:
            ctk.CTkLabel(self, text=f"❌ {self.json_file} 파일을 찾을 수 없습니다.").pack(pady=20)
            return

        scrollable_frame = ctk.CTkScrollableFrame(self, height=650)
        scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)

        row_num = 0
        for section, items in self.survey_data.items():
            ctk.CTkLabel(
                scrollable_frame,
                text=section,
                font=ctk.CTkFont(size=18, weight="bold")
            ).grid(row=row_num, column=0, columnspan=2, pady=(10, 5), sticky="w")
            row_num += 1

            for key, config in items.items():
                row_num = self._create_widget(scrollable_frame, key, config, row_num)

        ctk.CTkButton(self, text="DB 저장", command=self.get_entered_data).pack(pady=10)

    # ✅ 위젯 생성 (재귀 + 동적 sub 토글)
    def _create_widget(self, parent_frame, key, config, row):
        item_type = config.get('type')
        label_text = config.get('label') or key

        # 🔹 라디오 버튼
        if item_type == "radio":
            radio_var = ctk.StringVar()
            self.data_vars[key] = radio_var

            # 상위 질문 라벨
            ctk.CTkLabel(parent_frame, text=label_text, font=("", 14)).grid(
                row=row, column=0, sticky="w", padx=10, pady=(10, 0)
            )
            radio_frame = ctk.CTkFrame(parent_frame)
            radio_frame.grid(row=row, column=1, sticky="w", padx=10, pady=(10, 0))

            self.option_subs[key] = {}
            col = 0
            for option in config.get("options", []):
                if isinstance(option, str):
                    ctk.CTkRadioButton(
                        radio_frame,
                        text=option,
                        variable=radio_var,
                        value=option,
                        command=lambda k=key, v=option: self._toggle_radio_sub(k, v)
                    ).grid(row=0, column=col, sticky="w", padx=5)
                    col += 1
                elif isinstance(option, dict):
                    opt_label = option.get("label", "")
                    opt_sub = option.get("sub", {})

                    # ✅ 중앙 정렬된 하위 프레임 (카드형)
                    sub_frame = ctk.CTkFrame(
                        parent_frame,
                        fg_color="#E8E8E8",
                        corner_radius=10,
                        border_color="#CCCCCC",
                        border_width=1
                    )
                    sub_frame.grid(
                        row=row + 1,
                        column=0,
                        columnspan=2,
                        sticky="ew",
                        padx=100,      # 좌우 여백 (중앙 위치)
                        pady=(10, 15)  # 위아래 여백
                    )
                    sub_frame.grid_remove()

                    # 내부 컨테이너 (padding)
                    inner = ctk.CTkFrame(sub_frame, fg_color="transparent")
                    inner.pack(fill="both", expand=True, padx=20, pady=15)

                    # 하위 위젯 생성
                    sub_row = 0
                    for sub_key, sub_cfg in opt_sub.items():
                        sub_row = self._create_widget(inner, sub_key, sub_cfg, sub_row)

                    # 상위 라디오 버튼
                    ctk.CTkRadioButton(
                        radio_frame,
                        text=opt_label,
                        variable=radio_var,
                        value=opt_label,
                        command=lambda k=key, v=opt_label: self._toggle_radio_sub(k, v)
                    ).grid(row=0, column=col, sticky="w", padx=5)
                    col += 1
                    self.option_subs[key][opt_label] = sub_frame

            # ✅ 하위 프레임 토글 감시
            radio_var.trace_add("write", lambda *a, k=key: self._toggle_radio_sub(k, radio_var.get()))

            separator = ctk.CTkFrame(parent_frame, height=2, fg_color="#CCCCCC")
            separator.grid(row=row + 3, column=0, columnspan=2, sticky="ew", pady=(5, 5))
            return row + 4

        # 🔹 체크박스
        elif item_type == "checkbox":
            ctk.CTkLabel(parent_frame, text=label_text, font=("", 14)).grid(
                row=row, column=0, sticky="w", padx=10, pady=(10, 0)
            )
            box_frame = ctk.CTkFrame(parent_frame)
            box_frame.grid(row=row, column=1, sticky="w", padx=10, pady=(10, 0))

            for i, opt in enumerate(config.get("options", [])):
                if isinstance(opt, str):
                    var = ctk.BooleanVar()
                    ctk.CTkCheckBox(box_frame, text=opt, variable=var).grid(row=i, column=0, sticky="w", pady=2)
                    self.data_vars[f"{key}_{opt}"] = var
                elif isinstance(opt, dict):
                    opt_label = opt.get("label", "")
                    sub_opt = opt.get("sub", {})

                    var = ctk.BooleanVar()
                    ctk.CTkCheckBox(box_frame, text=opt_label, variable=var).grid(row=i, column=0, sticky="w", pady=2)
                    self.data_vars[f"{key}_{opt_label}"] = var

                    # ✅ 중앙 정렬된 하위 프레임 (카드형)
                    sub_frame = ctk.CTkFrame(
                        parent_frame,
                        fg_color="#E8E8E8",
                        corner_radius=10,
                        border_color="#CCCCCC",
                        border_width=1
                    )
                    sub_frame.grid(
                        row=row + 1,
                        column=0,
                        columnspan=2,
                        sticky="ew",
                        padx=100,
                        pady=(10, 15)
                    )
                    sub_frame.grid_remove()

                    inner = ctk.CTkFrame(sub_frame, fg_color="transparent")
                    inner.pack(fill="both", expand=True, padx=20, pady=10)

                    sub_row = 0
                    for sub_key, sub_cfg in sub_opt.items():
                        sub_row = self._create_widget(inner, sub_key, sub_cfg, sub_row)

                    def toggle_checkbox_sub(v=var, f=sub_frame):
                        f.grid() if v.get() else f.grid_remove()

                    var.trace_add("write", lambda *a, v=var, f=sub_frame: toggle_checkbox_sub(v, f))

            separator = ctk.CTkFrame(parent_frame, height=2, fg_color="#CCCCCC")
            separator.grid(row=row + 3, column=0, columnspan=2, sticky="ew", pady=(5, 5))
            return row + 4

        # 🔹 숫자 입력
        elif item_type == "input-number":
            # ✅ 문자열 변수 초기화 (빈 문자열로 시작해야 placeholder 표시됨)
            var = ctk.StringVar(value="")
            self.data_vars[key] = var
        
            # ✅ 숫자만 남기기 (trace 기반 필터링)
            def only_digits(*args, v=var):
                value = v.get()
                # 숫자와 빈 문자열만 허용
                if value != "" and not value.isdigit():
                    # 숫자가 아닌 문자는 제거
                    v.set("".join([c for c in value if c.isdigit()]))
        
            var.trace_add("write", only_digits)
        
            # ✅ 라벨
            ctk.CTkLabel(
                parent_frame,
                text=label_text,
                font=("", 14)
            ).grid(row=row, column=0, sticky="w", padx=10, pady=(10, 0))
        
            # ✅ 입력창 (placeholder 표시)
            entry = ctk.CTkEntry(
                parent_frame,
                textvariable=var,
                width=120,
                placeholder_text=config.get("placeholder", ""),
                fg_color="white"  # 다크모드에서도 placeholder 가시성 확보
            )
            entry.grid(row=row, column=1, sticky="w", padx=10)
        
            return row + 1
        
        # 🔹 텍스트 입력
        elif item_type == "input-text":
            var = ctk.StringVar()
            self.data_vars[key] = var
            ctk.CTkLabel(parent_frame, text=label_text, font=("", 14)).grid(
                row=row, column=0, sticky="w", padx=10, pady=(10, 0)
            )
            ctk.CTkEntry(parent_frame, textvariable=var, width=300,
                         placeholder_text=config.get("placeholder", "")).grid(row=row, column=1, sticky="w", padx=10)
            return row + 1

        return row + 1

    # ✅ 라디오 하위 프레임 토글
    def _toggle_radio_sub(self, key, value):
        if key not in self.option_subs:
            return
        for label, frame in self.option_subs[key].items():
            frame.grid() if label == value else frame.grid_remove()

    # ✅ DB 저장
    def get_entered_data(self):
        result, answers = {}, []
        for key, var in self.data_vars.items():
            if isinstance(var, (ctk.StringVar, ctk.BooleanVar)):
                value = str(var.get()).strip()
                result[key] = value
                qid = BASIC_QUESTION_MAPPING.get(key)
                if qid:
                    answers.append({
                        "question_id": qid,
                        "answer_component": None,
                        "answer_value": value
                    })

        print("\n--- 입력 데이터 ---")
        print(json.dumps(result, ensure_ascii=False, indent=4))
        print("\n--- DB 저장용 ---")
        print(json.dumps(answers, ensure_ascii=False, indent=4))

        try:
            item_id = getattr(self.item_data, "item_id", None)
            if not item_id:
                item_id = create_new_item_and_get_id_generic(
                    self.patient_id, data_category="MDD", data_type="B-SURVEY", seq=1, description="기초평가"
                )
                if not item_id:
                    CTkMessagebox(title="오류", message="❌ Item 생성 실패", icon="cancel")
                    return

            ok, err = save_answers(item_id, answers)
            if not ok:
                CTkMessagebox(title="오류", message=f"❌ 저장 실패: {err}", icon="cancel")
                return

            mark_item_updated(item_id)
            CTkMessagebox(title="성공", message=f"✅ DB 저장 완료\n(item_id={item_id})", icon="check")

            if self.on_close_callback:
                self.on_close_callback()

        except Exception as e:
            CTkMessagebox(title="오류", message=f"예외 발생: {e}", icon="cancel")
            raise
