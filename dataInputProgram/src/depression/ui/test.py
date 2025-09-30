import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json

class GenericSurveyForm(ctk.CTkFrame):
    def __init__(self, parent, json_file):
        super().__init__(parent)
        with open(json_file, "r", encoding="utf-8") as f:
            self.survey_data = json.load(f)


        self.vars = {}  # 문항별 값 저장
         # ✅ 스크롤 가능한 프레임 생성
        scroll_frame = ctk.CTkScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        # for survey_name, survey in self.survey_data.items():
        #     self.build_table(survey.get("header"), survey.get("body"))
        for survey_name, survey in self.survey_data.items():
            sections = survey.get("sections")
            if sections:
                for section in sections:
                    self.build_table(scroll_frame,section.get("header"), section.get("body"))
            else:
                self.build_table(scroll_frame,survey.get("header"), survey.get("body"))

        ctk.CTkButton(self, text="제출", command=self.calculate_score).pack(pady=10)

    def _fallback_columns(self, body):
        """header가 없을 때 기본 컬럼 구성"""
        n_opts = 0
        if body and isinstance(body, list):
            first = body[0]
            n_opts = len(first.get("options", []))
        cols = ["번호", "문항"] + [f"선택{i+1}" for i in range(n_opts)]
        return cols

    def build_table(self, parent, header, body):
        # ----- 1) 헤더(제목/설명) 카드 -----
        title = (header or {}).get("title", "")
        desc  = (header or {}).get("description", "")

        header_card = ctk.CTkFrame(parent)
        header_card.pack(fill="x", padx=6, pady=(8, 4))

        if title:
            ctk.CTkLabel(header_card, text=title, font=("", 16, "bold")).pack(pady=(8, 2))
        if desc:
            ctk.CTkLabel(header_card, text=desc, font=("", 12)).pack(pady=(0, 8))

        # ----- 2) 테이블(표 헤더 + 본문)을 같은 grid 컨테이너에 -----
        columns = (header or {}).get("columns") or self._fallback_columns(body)

        table = ctk.CTkFrame(parent)
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
                 # ✅ 기타 이유 입력창이 필요한 경우
                if item.get("followup_input"):
                    input_var = ctk.StringVar()
                    self.vars[f"{qid}_etc"] = input_var  # 등재 이름은 `_etc` 붙여 구분
                    ctk.CTkEntry(table, textvariable=input_var, placeholder_text="기타 이유를 적어주세요").grid(
                        row=r+1, column=1, columnspan=5, padx=10, pady=(0, 10), sticky="ew"
                    )
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


    
    def calculate_score(self):
        total = 0
        unanswered = []

        for qid, var in self.vars.items():
            if "_etc" in str(qid):
                continue 
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

        # 해석
        interpretation = "해석 기준 없음"
        for survey in self.survey_data.values():
            scoring = survey.get("scoring")
            if scoring:
                interpretation = self.interpret_score(total, scoring)
                break

        CTkMessagebox(
            title="제출 완료",
            message=f"총점은 {total}점입니다.\n\n{interpretation}",
            icon="check",
            option_1="확인"
        )

    @staticmethod
    def interpret_score(score, scoring_rules: dict):
        for rule, label in scoring_rules.items():
            if "-" in rule:
                min_val, max_val = map(int, rule.split("-"))
                if min_val <= score <= max_val:
                    return label
        return "해석 기준 없음"

