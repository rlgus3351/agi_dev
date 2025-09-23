import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json

# JSON 파일 경로
JSON_FILE = './dataInputProgram/src/depression/form/basic.json'

class HealthSurveyForm(ctk.CTkFrame):   # ✅ CTk → CTkFrame
    def __init__(self, parent, json_file=JSON_FILE):
        super().__init__(parent)        # ✅ parent 전달받음
        self.json_file = json_file

        self.widgets = {}
        self.data_vars = {}
        self.option_subs = {}  

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

    def _validate_and_calculate_bmi(self, event=None):
        pass
            
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
                font=ctk.CTkFont(size=18, weight="normal")
            ).grid(row=row_num, column=0, columnspan=2, pady=(10, 5), sticky="w")
            row_num += 1
            for key, config in items.items():
                row_num = self._create_widget(scrollable_frame, key, config, row_num)
        
        submit_button = ctk.CTkButton(self, text="데이터 저장", command=self.get_entered_data)
        submit_button.pack(pady=10)

    def _create_widget(self, parent_frame, key, config, row):
        item_type = config.get('type')
        label_text = config.get('label', key)
        
        if item_type == "radio":
            radio_var = ctk.StringVar()
            self.data_vars[key] = radio_var
            label = ctk.CTkLabel(parent_frame, text=label_text, font=("", 14, "normal"))
            label.grid(row=row, column=0, sticky="w", padx=10)
            
            main_radio_frame = ctk.CTkFrame(parent_frame)
            main_radio_frame.grid(row=row, column=1, sticky="w", padx=10)
            
            self.option_subs[key] = {}
            col_num = 0
            
            for option in config['options']:
                if isinstance(option, str):
                    ctk.CTkRadioButton(
                        main_radio_frame, text=option,
                        variable=radio_var, value=option,
                        font=("", 14, "normal")
                    ).grid(row=0, column=col_num, sticky="w", padx=5)
                    col_num += 1
                elif isinstance(option, dict):
                    option_label = option.get("label")
                    
                    sub_frame = ctk.CTkFrame(main_radio_frame)
                    sub_frame.grid(row=1, column=0, columnspan=10, sticky="w", padx=10, pady=5)
                    sub_frame.grid_remove()

                    self.option_subs[key][option_label] = sub_frame

                    sub_items = option.get("sub", {})
                    sub_row = 0
                    for sub_key, sub_cfg in sub_items.items():
                        sub_row = self._create_widget(sub_frame, sub_key, sub_cfg, sub_row)

                    ctk.CTkRadioButton(
                        main_radio_frame, text=option_label,
                        variable=radio_var, value=option_label,
                        command=lambda: self._toggle_radio_sub(key, radio_var.get()),
                        font=("", 14, "normal")
                    ).grid(row=0, column=col_num, sticky="w", padx=5)
                    col_num += 1
            
            radio_var.trace_add("write", lambda *args: self._toggle_radio_sub(key, radio_var.get()))
            
        elif item_type == "input-number":
            input_var = ctk.StringVar(value="")
            self.data_vars[key] = input_var
            label = ctk.CTkLabel(parent_frame, text=label_text, font=("", 14, "normal"))
            label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            validate_cmd = (self.register(self.validate_number_input), '%P', '%S', '%V', '%W', key)
            entry = ctk.CTkEntry(
                parent_frame, textvariable=input_var, width=300,
                validate='key', validatecommand=validate_cmd,
                font=("", 14, "normal"),
                placeholder_text=config.get("placeholder", "")
            )
            entry.grid(row=row, column=1, sticky="w", padx=10, pady=5)
            self.widgets[key] = entry
            self.data_vars[key + "_min"] = config.get("min")
            self.data_vars[key + "_max"] = config.get("max")

        elif item_type == "checkbox":
            label = ctk.CTkLabel(parent_frame, text=label_text, font=("", 14, "normal"))
            label.grid(row=row, column=0, sticky="w", padx=10, pady=5)

            option_frame = ctk.CTkFrame(parent_frame)
            option_frame.grid(row=row, column=1, sticky="w", padx=10, pady=5)

            opt_row = 0
            for opt in config.get("options", []):
                if isinstance(opt, str):
                    var = ctk.BooleanVar()
                    cb = ctk.CTkCheckBox(option_frame, text=opt, variable=var, font=("", 14, "normal"))
                    cb.grid(row=opt_row, column=0, sticky="w", padx=5, pady=2)
                    self.data_vars[f"{key}_{opt}"] = var
                    opt_row += 1

                elif isinstance(opt, dict):
                    label_text = opt.get("label")
                    var = ctk.BooleanVar()
                    cb = ctk.CTkCheckBox(option_frame, text=label_text, variable=var, font=("", 14, "normal"))
                    cb.grid(row=opt_row, column=0, sticky="w", padx=5, pady=2)
                    self.data_vars[f"{key}_{label_text}"] = var

                    sub_frame = ctk.CTkFrame(option_frame)
                    sub_frame.grid(row=opt_row+1, column=0, sticky="w", padx=20, pady=5)
                    sub_frame.grid_remove()

                    def toggle_sub(*args, v=var, frame=sub_frame):
                        if v.get():
                            frame.grid()
                        else:
                            frame.grid_remove()
                    var.trace_add("write", toggle_sub)

                    sub_items = opt.get("sub", {})
                    sub_row = 0
                    for sub_key, sub_cfg in sub_items.items():
                        sub_row = self._create_widget(sub_frame, sub_key, sub_cfg, sub_row)

                    opt_row += 2
    
        elif item_type == "input-text":
            input_var = ctk.StringVar(value="")
            self.data_vars[key] = input_var
            label = ctk.CTkLabel(parent_frame, text=label_text, font=("", 14, "normal"))
            label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            entry = ctk.CTkEntry(
                parent_frame, textvariable=input_var, width=300,
                font=("", 14, "normal"),
                placeholder_text=config.get("placeholder", "")
            )
            entry.grid(row=row, column=1, sticky="w", padx=10, pady=5)
            self.widgets[key] = entry

            if key == "신장/몸무게":
                entry.bind('<KeyRelease>', self._validate_and_calculate_bmi)

        return row + 1

    def _toggle_radio_sub(self, key, value):
        if key in self.option_subs:
            for label, frame in self.option_subs[key].items():
                if label == value:
                    frame.grid()
                else:
                    frame.grid_remove()

    def get_entered_data(self):
        result = {}
        for key, var in self.data_vars.items():
            if key.endswith("_min") or key.endswith("_max"):
                continue
            value = var.get()
            result[key] = value

        print("--- 입력된 데이터 ---")
        print(json.dumps(result, ensure_ascii=False, indent=4))
        CTkMessagebox(title="성공", message="데이터가 성공적으로 저장되었습니다.", icon="check")


# ✅ 이제 __main__ 제거
# if __name__ == "__main__":
#     app = HealthSurveyForm()
#     app.mainloop()
