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
            self.build_table(survey.get("header"), survey.get("body"))

        ctk.CTkButton(self, text="제출", command=self.calculate_score).pack(pady=10)

    def _fallback_columns(self, body):
        """header가 없을 때 기본 컬럼 구성"""
        n_opts = 0
        if body and isinstance(body, list):
            first = body[0]
            n_opts = len(first.get("options", []))
        cols = ["번호", "문항"] + [f"선택{i+1}" for i in range(n_opts)]
        return cols

    def build_table(self, header, body):
        # ----- 1) 헤더(제목/설명) 카드 -----
        title = (header or {}).get("title", "")
        desc  = (header or {}).get("description", "")

        header_card = ctk.CTkFrame(self)
        header_card.pack(fill="x", padx=6, pady=(8, 4))

        if title:
            ctk.CTkLabel(header_card, text=title, font=("", 16, "bold")).pack(pady=(8, 2))
        if desc:
            ctk.CTkLabel(header_card, text=desc, font=("", 12)).pack(pady=(0, 8))

        # ----- 2) 테이블(표 헤더 + 본문)을 같은 grid 컨테이너에 -----
        columns = (header or {}).get("columns") or self._fallback_columns(body)

        table = ctk.CTkFrame(self)
        table.pack(fill="x", pady=5)

        # 열 비율: 번호(작게), 문항(넓게), 선택지(균등)
        table.grid_columnconfigure(0, weight=0, minsize=40)    # 번호
        table.grid_columnconfigure(1, weight=5, minsize=400)   # 문항
        for i in range(2, len(columns)):
            table.grid_columnconfigure(i, weight=1, minsize=100)

        # 표 헤더
        for col, text in enumerate(columns):
            ctk.CTkLabel(
                table, text=text, font=("", 12, "bold"), anchor="center"
            ).grid(row=0, column=col, padx=5, pady=5, sticky="nsew")

        # 본문
        if not body:
            return

        for r, item in enumerate(body, start=1):
            qid = item["id"]
            question = item["question"]
            qtype = item.get("type", "radio")
            options = item.get("options", [])
            min_val = item.get("min", 0)
            max_val = item.get("max", 100)

            # 번호
            ctk.CTkLabel(table, text=str(qid), anchor="center").grid(
                row=r, column=0, padx=5, pady=5, sticky="nsew"
            )

            # 질문(길면 줄바꿈 → 세로 확장)
            # ctk.CTkLabel(
            #     table, text=question, anchor="w", justify="left", wraplength=400
            # ).grid(row=r, column=1, padx=5, pady=5, sticky="w")
            ctk.CTkLabel(
                table, text=question.replace("\\n","\n"), anchor="w", justify="left", wraplength=400
            ).grid(row=r, column=1, padx=5, pady=5, sticky="w")
            # 선택지
            if qtype == "radio":
                var = ctk.StringVar(value="")
                self.vars[qid] = var
                for i, opt in enumerate(options):
                    ctk.CTkRadioButton(
                        table, text=str(opt), variable=var, value=str(opt)
                    ).grid(row=r, column=i+2, padx=5, pady=5, sticky="nsew")
            elif qtype == "input-number":
                var = ctk.StringVar()
                self.vars[qid] = var
                entry = ctk.CTkEntry(table, textvariable=var, width=80)
                entry.grid(row=r, column=2, padx=5, pady=5, sticky="w")

                # 숫자 범위 표시
                range_text = f"(최소: {min_val}, 최대: {max_val})"
                ctk.CTkLabel(table, text=range_text, font=("", 10), text_color="gray").grid(
                    row=r, column=3, padx=5, pady=5, sticky="w"
                )

                # 유효성 검사: 숫자 only
                def validate_numeric_input(*_):
                    val = var.get()
                    if val and not val.isdigit():
                        var.set('')

                var.trace_add("write", validate_numeric_input)

    # def calculate_score(self):
    #     total = 0
    #     for qid, var in self.vars.items():
    #         try:
    #             total += int(var.get())
    #         except ValueError:
    #             pass
    #     print("총점:", total)
    # def calculate_score(self):
    #     total = 0
    #     unanswered = []

    #     for qid, var in self.vars.items():
    #         value = var.get()
    #         if value == "":
    #             unanswered.append(qid)
    #         else:
    #             try:
    #                 total += int(value)
    #             except ValueError:
    #                 pass

    #     if unanswered:
    #         CTkMessagebox(
    #             title="입력 누락",
    #             message="모든 항목을 입력해주세요.",
    #             icon="warning",
    #             option_1="확인"
    #         )
    #         return

    #     print("총점:", total)
    #     CTkMessagebox(
    #         title="제출 완료",
    #         message=f"총점은 {total}점입니다.",
    #         icon="check",
    #         option_1="확인"
    #     )

    def calculate_score(self):
        total = 0
        unanswered = []
        for qid, var in self.vars.items():
            value = var.get()
            if value.strip() == "":
                unanswered.append(qid)
                continue
            try:
                total += int(value)
            except ValueError:
                unanswered.append(qid)
    
        if unanswered:
            CTkMessagebox(
                title="입력 누락",
                message="모든 항목을 올바르게 입력해주세요.",
                icon="warning",
                option_1="확인"
            )
            return
    
        print("총점:", total)
        CTkMessagebox(
            title="제출 완료",
            message=f"총점은 {total}점입니다.",
            icon="check",
            option_1="확인"
        )
