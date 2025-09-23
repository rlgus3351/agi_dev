import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json

class GenericSurveyForm(ctk.CTkFrame):
    def __init__(self, parent, json_file):
        super().__init__(parent)
        with open(json_file, "r", encoding="utf-8") as f:
            self.survey_data = json.load(f)

        self.vars = {}  # 문항별 값 저장

        for survey_name, survey in self.survey_data.items():
            self.build_header(survey.get("header"))
            self.build_body(survey.get("body"))

        ctk.CTkButton(self, text="제출", command=self.calculate_score).pack(pady=10)

    def build_header(self, header):
        if not header:
            return
    
        # 제목 + 설명
        ctk.CTkLabel(self, text=header.get("title", ""), font=("", 16, "bold")).pack(pady=(10, 5))
        ctk.CTkLabel(self, text=header.get("description", ""), font=("", 12)).pack(pady=(0, 10))
    
        # 테이블 헤더
        table_header = ctk.CTkFrame(self)
        table_header.pack(fill="x", pady=5)
    
        # 열 비율: 번호(0), 질문(넓게=4), 선택지들(1씩)
        table_header.grid_columnconfigure(0, weight=0)
        table_header.grid_columnconfigure(1, weight=4)
        for i in range(2, len(header.get("columns", []))):
            table_header.grid_columnconfigure(i, weight=1)
    
        for col, text in enumerate(header.get("columns", [])):
            ctk.CTkLabel(
                table_header, text=text,
                font=("", 12, "bold"), anchor="center"
            ).grid(row=0, column=col, padx=5, pady=5, sticky="nsew")


    def build_body(self, body):
        table = ctk.CTkFrame(self)
        table.pack(fill="x", pady=5)

        # 열 비율 조정 → 문항은 넓게(4), 선택지는 균등(1)
        table.grid_columnconfigure(0, weight=0)   # 번호
        table.grid_columnconfigure(1, weight=4)   # 질문 (넓게)
        for i in range(2, 6):                    # 선택지 4개
            table.grid_columnconfigure(i, weight=1)

        for item in body:
            qid = item["id"]
            question = item["question"]
            qtype = item["type"]
            options = item.get("options", [])

            # 번호
            ctk.CTkLabel(table, text=str(qid), width=30, anchor="center").grid(row=qid, column=0, padx=5, pady=5)

            # 질문 (wraplength로 줄바꿈)
            ctk.CTkLabel(
                table, text=question, anchor="w",
                justify="left", wraplength=400  # ✅ 글자 길면 줄바꿈
            ).grid(row=qid, column=1, sticky="w", padx=5, pady=5)

            if qtype == "radio":
                var = ctk.StringVar(value="")
                self.vars[qid] = var
                for i, opt in enumerate(options):
                    ctk.CTkRadioButton(
                        table, text=str(opt), variable=var, value=str(opt)
                    ).grid(row=qid, column=i+2, padx=5, pady=5, sticky="nsew")

    def calculate_score(self):
        total = 0
        for qid, var in self.vars.items():
            try:
                total += int(var.get())
            except ValueError:
                pass
        print("총점:", total)
