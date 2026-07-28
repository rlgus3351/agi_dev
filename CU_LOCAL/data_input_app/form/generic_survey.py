import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json
from typing import Dict, Any, List, Tuple, Optional, Callable
import tkinter as tk  # ✅ multiline 텍스트용
from utils.psql import compute_psqi
from utils.meqk import compute_meqk,debug_meqk
import re
# -------------------------------------------------
# 텍스트 정규화: 줄바꿈/여러 공백 제거 + trim
# -------------------------------------------------
def _norm(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    return " ".join(text.replace("\n", " ").replace("\r", " ").split()).strip()



class TimeRulerSlider(ctk.CTkFrame):
    """시간 슬라이더 (자정 넘김 지원).
       min_hour=20, max_hour=3 처럼 들어오면 내부 범위를 [20, 27]로 잡고 표시/저장은 24h로 모듈러."""
    def __init__(self, parent, min_hour=5.0, max_hour=12.0,
                 step_minutes=30, width=700, initial=None, show_am_label=True):
        super().__init__(parent)
        self.orig_min = float(min_hour)
        self.orig_max = float(max_hour)
        self.min_h = float(min_hour)
        self.max_h = float(max_hour)
        # ✅ 자정 넘김이면 max에 24 더해 내부 구간 확장
        if self.max_h < self.min_h:
            self.max_h += 24.0

        # step & steps
        self.step_minutes = max(1, int(step_minutes))
        self.step = self.step_minutes / 60.0
        span = max(1e-9, (self.max_h - self.min_h))
        n_steps = max(1, int(round(span / self.step)))  # ZeroDivisionError 방지

        self.width = width
        self.canvas_h = 40

        # 초기값
        if initial is None:
            initial = self.min_h
        else:
            initial = float(initial)
        # ✅ initial이 0~24 표기라면, min_h보다 작고 자정넘김이면 +24해서 구간에 맞춤
        if initial < self.min_h:
            initial += 24.0
        initial = max(self.min_h, min(self.max_h, initial))

        # 값 라벨(슬라이더 위쪽)
        self.value_label = ctk.CTkLabel(self, text=self._fmt(initial),
                                        font=ctk.CTkFont(size=13, weight="bold"))
        self.value_label.pack(padx=4, anchor="e")

        # 슬라이더
        self._var = tk.DoubleVar(value=initial)
        self.slider = ctk.CTkSlider(self, from_=self.min_h, to=self.max_h,
                                    width=self.width, number_of_steps=n_steps,
                                    variable=self._var, command=self._on_slide)
        self.slider.pack(pady=(2, 0), fill="x")

        # 눈금자
        self.canvas = tk.Canvas(self, width=self.width, height=self.canvas_h,
                                highlightthickness=0, bg=self._bg())
        self.canvas.pack(pady=(2, 2), fill="x")
        self.show_am_label = show_am_label
        self._draw_ticks()

        # 마우스 놓을 때 스냅
        self.slider.bind("<ButtonRelease-1>", self._snap)

    # ----- helpers -----
    def _bg(self):
        try:
            return self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        except Exception:
            return self.cget("fg_color") or "#FFFFFF"

    def _fmt(self, h):
        """내부값 h(예: 26.5)를 24h로 모듈러하여 HH:MM 문자열로."""
        h24 = h % 24.0
        hh = int(h24)
        mm = int(round((h24 - hh) * 60)) % 60
        return f"{hh:02d}:{mm:02d}"

    def _x(self, h):
        r = (h - self.min_h) / (self.max_h - self.min_h)
        return int(r * self.width)

    def _draw_ticks(self):
        self.canvas.delete("all")
        y = self.canvas_h - 1
        self.canvas.create_line(0, y, self.width, y)
        # 30분 간격 눈금
        h = self.min_h
        while h <= self.max_h + 1e-9:
            x = self._x(h)
            major = abs((h % 1.0)) < 1e-9  # 정시
            tick_h = 18 if major else 10
            self.canvas.create_line(x, y, x, y - tick_h)
            if major:
                label = f"{int(h % 24)}"
                # 범위 시작에만 "(오전)" 같은 꼬리표 원하면 여기에 조건 추가 가능
                self.canvas.create_text(x, y - tick_h - 10, text=label)
            h += 0.5  # 30분

    def _on_slide(self, _=None):
        self.value_label.configure(text=self._fmt(self._var.get()))

    def _snap(self, _=None):
        v = self._var.get()
        snapped = round((v - self.min_h) / self.step) * self.step + self.min_h
        snapped = max(self.min_h, min(self.max_h, snapped))
        if abs(snapped - v) > 1e-6:
            self._var.set(snapped)
            self.slider.set(snapped)
        self.value_label.configure(text=self._fmt(snapped))

    def get_value(self):
        """(float_hour_mod24, 'HH:MM') 반환. float은 0~24 기준으로 돌려줌."""
        v = self._var.get() % 24.0
        return v, self._fmt(self._var.get())


class GenericSurveyForm(ctk.CTkFrame):
    def __init__(
        self,
        parent,
        json_file: str,
        item_data: Optional[Dict[str, Any]] = None,
        patient_uuid: Optional[str] = None,
        qmap: Optional[Dict[str, int]] = None,   # 질문텍스트 → question_id 매핑
        on_close_callback: Optional[Callable[[], None]] = None,   # ✅ 콜백
    ):
        """
        item_data 예:
        {
          "item_id": 123,                # 없으면 신규 저장 시 생성
          "patient_id": "uuid-string",   # 꼭 필요 (없으면 patient_uuid로 대체)
          "data_category": "MDD",
          "data_type": "E-SURVEY",
          "seq": 1,
          "title": "PHQ-9",
          "questions_raw": [
            {"answer_id": 555, "question_id": 27, "answer_component": None, "answer_value": "2"},
            ...
          ]
        }
        """
        super().__init__(parent)

        # 모달/콜백 참조 저장
        self._modal = self.winfo_toplevel()
        self._on_close_callback = on_close_callback

        # JSON 로드
        with open(json_file, "r", encoding="utf-8") as f:
            self.survey_data = json.load(f)

        # 컨텍스트
        self.item_data: Dict[str, Any] = (item_data or {}).copy()
        self.patient_uuid: Optional[str] = self.item_data.get("patient_id") or patient_uuid

        # qmap 정규화 사전으로 내부 보관
        if qmap:
            self.qmap: Optional[Dict[str, int]] = { _norm(k): int(v) for k, v in qmap.items() }
        else:
            self.qmap = None

        if not self.patient_uuid:
            CTkMessagebox(
                title="오류",
                message="patient_id(=UUID)를 찾지 못했습니다.\n폼을 닫고 다시 시도해주세요.",
                icon="cancel",
                option_1="확인",
            )
            return

        # 기존 답변 인덱스/프리필
        self._existing_index: Dict[Tuple[int, Optional[str]], Dict[str, Any]] = self._build_existing_index()
        self._prefill: Dict[Tuple[int, Optional[str]], str] = self._build_prefill()

        # 현재 폼에서 관리하는 변수들
        self.vars: Dict[Any, Any] = {}  # StringVar, Text, TimeRulerSlider 등 혼합 저장
        # ✅ 점수 계산 대상: int(qid) 또는 ('time', qid)
        self._score_keys: set = set()

        # ✅ 시간 구간 점수 규칙 수집
        self._time_scoring = self._collect_time_scoring()
        self._range_scoring = self._collect_range_scoring()   # ← 이 줄을 추가

        # 스크롤 영역
        scroll_frame = ctk.CTkScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 섹션/테이블 렌더
        for _, survey in self.survey_data.items():
            sections = survey.get("sections")
            if sections:
                for section in sections:
                    self.build_table(scroll_frame, section.get("header"), section.get("body"))
            else:
                self.build_table(scroll_frame, survey.get("header"), survey.get("body"))

        # 하단 버튼바 (제출 / 닫기)
        btn_bar = ctk.CTkFrame(self)
        btn_bar.pack(fill="x", pady=(6, 10))
        ctk.CTkButton(btn_bar, text="닫기", command=self._on_close_clicked).pack(side="right", padx=6)
        ctk.CTkButton(btn_bar, text="제출", command=self._on_submit).pack(side="right", padx=6)

        # 우상단 X도 동일 동작
        try:
            self._modal.protocol("WM_DELETE_WINDOW", self._on_close_clicked)
        except Exception:
            pass
    def _collect_psqi_answers_for_util(self) -> dict:
        """
        현재 폼의 값들을 PSQI util 입력 포맷으로 수집.
        qmap(DB question_id 매핑)을 이용해 self.vars에서 안전하게 가져온다.
        결과 키는 PSQI util 사양: "1","2","3","4","5-a"..."5-j","6","7","8","9".
        """
        if not self.qmap:
            # 안전망: qmap이 없으면 기존 로직(기본 ID)으로 시도
            # (원본 함수 내용을 fall-back으로 남기고 싶으면 여기에 붙여도 OK)
            return {
                "1": "0:00", "2": 0, "3": "0:00", "4": 0.0,
                "5-a": 0, "5-b": 0, "5-c": 0, "5-d": 0, "5-e": 0,
                "5-f": 0, "5-g": 0, "5-h": 0, "5-i": 0, "5-j": 0,
                "6": 0, "7": 0, "8": 0, "9": 0,
            }
    
        def _qid(text: str) -> Optional[int]:
            return self.qmap.get(_norm(text))
    
        def _get_time_by_text(text: str) -> str:
            qid = _qid(text)
            if qid is None:
                return "0:00"
            # slider-time 위젯 우선
            w = self.vars.get(f"{qid}__timeruler")
            if w is not None:
                _, hhmm = w.get_value()
                return hhmm
            # 숫자/문자 → HH:MM 로 가정 가능한 경우
            v = self.vars.get(qid)
            if v is None:
                return "0:00"
            s = str(v.get()).strip()
            if ":" in s:
                return s
            try:
                # 정수 시각(0~23) 들어온 경우 HH:00 로 변환
                h = int(float(s))
                return f"{h:02d}:00"
            except Exception:
                return "0:00"
    
        def _get_int_by_text(text: str) -> int:
            qid = _qid(text)
            if qid is None:
                return 0
            v = self.vars.get(qid)
            if v is None:
                return 0
            try:
                return int(str(v.get()).strip())
            except Exception:
                return 0
    
        def _get_float_by_text(text: str) -> float:
            qid = _qid(text)
            if qid is None:
                return 0.0
            v = self.vars.get(qid)
            if v is None:
                return 0.0
            try:
                return float(str(v.get()).strip())
            except Exception:
                return 0.0
    
        # Q1~Q4
        a1 = _get_time_by_text("보통 몇시에 잠자리에 듭니까?")
        a2 = _get_int_by_text("보통 잠 들 때까지 평균 얼마나 걸립니까?")
        a3 = _get_time_by_text("보통 몇 시에 일어납니까?")
        a4 = _get_float_by_text("당신은 실제로 하루에 몇 시간 잡니까?")
    
        # Q5a~Q5j
        five_map = {
            "5-a": "밤에 30분 이내에 잠들지 못해서",
            "5-b": "중간에 깨거나 너무 일찍 깨서",
            "5-c": "화장실을 다녀오려고 일어나서",
            "5-d": "수면 중 숨을 쉬기가 불편해서",
            "5-e": "기침을 하거나 크게 코를 골아서",
            "5-f": "수면 중 너무 춥다고 느껴서",
            "5-g": "수면 중 너무 덥다고 느껴서",
            "5-h": "나쁜 꿈을 꿔서",
            # ❗ JSON 중복 수정 후: '통증이 있어서'는 5-j로 봅니다.
            "5-j": "통증이 있어서",
            "5-i": "위에 적혀진 이유 외에 잠을 못 잔 다른 이유",
        }
        five_vals = {k: _get_int_by_text(txt) for k, txt in five_map.items()}
    
        # Q6~Q9
        a6 = _get_int_by_text("당신은 잠을 잘 자기 위해 수면제 또는 다른 약물(처방 또는 비처방약물)을 복용 한 적이 얼마나 자주 있었습니까?")
        a7 = _get_int_by_text("당신은 운전 중이거나 식사 중, 또는 기타 사회활동을 하는 동안 깨어있기 힘들 떄가 얼마나 자주 있었습니까?")
        a8 = _get_int_by_text("당신은 일을 해내는 데 충분한 활력을 유지하기가 어려웠습니까?")
        a9 = _get_int_by_text("당신은 전반적인 자신의 수면의 질을 어떻게 평가합니까?")
    
        return {
            "1": a1, "2": a2, "3": a3, "4": a4,
            "5-a": five_vals["5-a"], "5-b": five_vals["5-b"], "5-c": five_vals["5-c"],
            "5-d": five_vals["5-d"], "5-e": five_vals["5-e"], "5-f": five_vals["5-f"],
            "5-g": five_vals["5-g"], "5-h": five_vals["5-h"], "5-i": five_vals["5-i"],
            "5-j": five_vals["5-j"],
            "6": a6, "7": a7, "8": a8, "9": a9,
        }
    # ---------------- 기존 답변 인덱스 ----------------
    def _build_existing_index(self) -> Dict[Tuple[int, Optional[str]], Dict[str, Any]]:
        idx: Dict[Tuple[int, Optional[str]], Dict[str, Any]] = {}
        raws = self.item_data.get("questions_raw", [])
        if isinstance(raws, list):
            for row in raws:
                qid = row.get("question_id")
                comp = row.get("answer_component")  # 라디오/숫자: 대부분 None, '기타' 등 사용 가능
                aid = row.get("answer_id")
                val = row.get("answer_value")
                if qid is not None and aid is not None:
                    try:
                        idx[(int(qid), comp)] = {"answer_id": aid, "answer_value": val}
                    except Exception:
                        pass
        return idx

    def _build_prefill(self) -> Dict[Tuple[int, Optional[str]], str]:
        return {key: str(meta.get("answer_value", "")) for key, meta in self._existing_index.items()}

    # ---------------- 시간 스코어 규칙 ----------------
    def _collect_time_scoring(self) -> Dict[int, list]:
        """JSON 루트(각 설문)에서 time_scoring을 모아 qid->규칙 리스트로 반환"""
        rules: Dict[int, list] = {}
        for survey in self.survey_data.values():
            ts = (survey or {}).get("time_scoring", {})
            if not isinstance(ts, dict):
                continue
            for k, v in ts.items():
                try:
                    qid = int(k)
                    rules[qid] = list(v)  # [["HH:MM","HH:MM",점수], ...]
                except Exception:
                    pass
        return rules
    def _collect_range_scoring(self) -> dict:
        """
        JSON에서 MEQ-K scoring 범주(예: 16-30: 극단적 저녁형)를 추출한다.
        첫 번째 설문 블록만 대상으로 한다.
        """
        try:
            for survey in self.survey_data.values():
                scoring = survey.get("scoring")
                if isinstance(scoring, dict):
                    return scoring
        except Exception:
            pass
        return {}

    @staticmethod
    def _hhmm_to_minutes(hhmm: str) -> int:
        hh, mm = hhmm.split(":")
        return int(hh) * 60 + int(mm)

    def _score_slider_time(self, qid: int, hour_value: float) -> Optional[int]:
        """slider_time 값(시간 float)을 time_scoring 규칙으로 점수 환산"""
        rules = self._time_scoring.get(qid)
        if not rules:
            return 0  # 규칙 없으면 0점(필요시 None으로 바꿔 검증 실패 처리 가능)

        h = int(hour_value)
        m = int(round((hour_value - h) * 60))
        minutes = h * 60 + m

        for i, (start_str, end_str, score) in enumerate(rules):
            try:
                s = self._hhmm_to_minutes(start_str)
                e = self._hhmm_to_minutes(end_str)
            except Exception:
                continue
            # [s, e) 권장, 마지막 구간은 e 포함 허용
            last = (i == len(rules) - 1)
            if (s <= minutes < e) or (last and minutes == e):
                try:
                    return int(score)
                except Exception:
                    return None
        return 0
    def _build_custom_radio(self, table, row_idx, qid, item,
                        opt_col_start, last_col) -> int:
        """
        테이블의 옵션 컬럼(2..last_col)을 그대로 사용해, 각 옵션을 '해당 컬럼 한 칸'에 배치.
        각 칸 내부는 [텍스트 | ●라디오] 2열 소프레임으로 고정 정렬.
        반환: 이 항목이 실제로 사용한 grid row 수(rows_used)
        """
        options = [str(x) for x in item.get("options", [])]
        if not options:
            return 1

        # 값 배열
        values = item.get("values") or list(range(len(options)))
        values = [str(v) for v in values]

        # 프리필 & 점수 대상 등록
        prefill = str(self._prefill.get((qid, None), ""))
        var = ctk.StringVar(value=prefill)
        self.vars[qid] = var
        self._score_keys.add(qid)

        # 사용할 옵션 컬럼 개수(테이블의 2..last_col 구간)
        opt_cols_total = max(1, last_col - opt_col_start + 1)

        # 한 줄에 몇 개 놓을지: 기본적으로 "가능한 모든 옵션 컬럼"을 사용
        # (원하면 JSON에 "columns_per_row"로 덮어쓰기 가능하지만, opt_cols를 넘기지 않게 clamp)
        col_cnt = int(item.get("columns_per_row", opt_cols_total))
        col_cnt = max(1, min(col_cnt, opt_cols_total))

        # 래퍼는 '문항 오른쪽 전폭(옵션 컬럼 전부)'을 정확히 차지
        opt_frame = ctk.CTkFrame(table, fg_color="transparent")
        opt_frame.grid(
            row=row_idx,
            column=opt_col_start,
            columnspan=(last_col - opt_col_start + 1),   # ✅ 기존 코드의 off-by-one 수정
            padx=0, pady=0, sticky="nsew"
        )

        # 외곽 그리드: 옵션 칸 수(col_cnt) 만큼 “같은 폭”으로
        # (테이블에서 옵션 컬럼 minsize=100이므로 여기도 맞춰줌)
        for c in range(col_cnt):
            opt_frame.grid_columnconfigure(c, weight=1, minsize=100, uniform=f"optcols-{qid}")

        # 옵션을 col_cnt로 줄바꿈 배치
        import math
        n_rows = math.ceil(len(options) / col_cnt)

        k = 0
        for r in range(n_rows):
            for c in range(col_cnt):
                if k >= len(options):
                    break

                # 각 옵션은 '해당 옵션 컬럼 하나'를 통째로 점유
                cell = ctk.CTkFrame(opt_frame, fg_color="transparent")
                cell.grid(row=r, column=c, padx=5, pady=(2, 6), sticky="nsew")
                # 셀 내부 2열: [텍스트 | 라디오] — 텍스트가 늘어나고, 라디오는 고정
                cell.grid_columnconfigure(0, weight=1)
                cell.grid_columnconfigure(1, weight=0, minsize=32)  # 라디오 영역 고정폭 → 세로줄 정렬 안정화

                txt = options[k]
                val = values[k]

                # 텍스트(왼쪽) : 다른 행과 줄바꿈 폭 맞춤
                ctk.CTkLabel(
                    cell, text=txt, anchor="w", justify="left", wraplength=320
                ).grid(row=0, column=0, padx=(0, 8), sticky="w")

                # 라디오(오른쪽) : 같은 칸의 오른쪽에 딱 붙여 통일
                ctk.CTkRadioButton(
                    cell, text="", variable=var, value=val
                ).grid(row=0, column=1, padx=(0, 0), sticky="e")

                k += 1

        # 이 문항이 실제로 점유한 테이블 그리드 행 수
        rows_used = 1  # 래퍼(opt_frame) 자체는 테이블에서 1줄만 차지
        return rows_used

    # ---------------- QID 결정(핵심) ----------------
    def _qid_from_item(self, item: Dict[str, Any]) -> int:
        """
        question_id 계산 우선순위:
         1) self.qmap이 있으면: 질문문구 정규화 후 qmap으로 우선 매핑
         2) qmap에 없으면: JSON의 'id' 정수 사용(백업)
         3) 그래도 없으면: 0 반환(이 행은 저장 스킵)
        """
        qtext_norm = _norm(item.get("question", ""))
        if self.qmap:
            mapped = self.qmap.get(qtext_norm)
            if mapped is not None:
                try:
                    return int(mapped)
                except Exception:
                    pass
        # 백업: JSON 'id'
        try:
            return int(item["id"])
        except Exception:
            print(f"[경고] question_id 매핑 실패: '{qtext_norm}' → qmap/JSON id 모두 없음")
            return 0

    # ---------------- UI 빌드 ----------------
    def _fallback_columns(self, body):
        n_opts = 0
        if body and isinstance(body, list) and body:
            first = body[0]
            n_opts = len(first.get("options", []))
        return ["번호", "문항"] + [f"선택{i+1}" for i in range(n_opts)]

    def build_table(self, parent, header, body):
        title = (header or {}).get("title", "")
        desc  = (header or {}).get("description", "")

        header_card = ctk.CTkFrame(parent)
        header_card.pack(fill="x", padx=6, pady=(8, 4))
        if title:
            ctk.CTkLabel(header_card, text=title, font=("", 16, "bold")).pack(pady=(8, 2))
        if desc:
            ctk.CTkLabel(header_card, text=desc, font=("", 12)).pack(pady=(0, 8))

        columns = (header or {}).get("columns") or self._fallback_columns(body)
        table = ctk.CTkFrame(parent)
        table.pack(fill="x", pady=5)

        table.grid_columnconfigure(0, weight=0, minsize=40)
        table.grid_columnconfigure(1, weight=5, minsize=400)
        for i in range(2, len(columns)):
            table.grid_columnconfigure(i, weight=1, minsize=100)

        for col, text in enumerate(columns):
            ctk.CTkLabel(table, text=text, font=("", 12, "bold"), anchor="center")\
                .grid(row=0, column=col, padx=5, pady=5, sticky="nsew")

        if not body:
            return

        # ✅ 줄 충돌 방지: 수동 row 포인터 사용
        row_idx = 1  # 헤더가 0행을 쓰므로 1부터 시작
        opt_col_start = 2
        last_col = max(2, len(columns) - 1)

        for item in body:
            qid = self._qid_from_item(item)
            qid_label = str(qid) if qid else "?"

            question = item.get("question", "")
            qtype = item.get("type", "radio")
            options = item.get("options", [])
            min_val = item.get("min", 0)
            max_val = item.get("max", 100)

            # 공통: 번호/문항 라벨 (항상 첫 줄)
            ctk.CTkLabel(table, text=qid_label, anchor="center")\
                .grid(row=row_idx, column=0, padx=5, pady=5, sticky="nsew")
            ctk.CTkLabel(table, text=str(question).replace("\\n", "\n"),
                         anchor="w", justify="left", wraplength=400)\
                .grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")

            rows_used = 1  # 이 항목이 실제로 차지한 줄 수

            # ---------------- 라디오 ----------------
            if qtype == "radio":
                var = ctk.StringVar(value=str(self._prefill.get((qid, None), "")))
                self.vars[qid] = var
                self._score_keys.add(qid)

                has_follow = bool(item.get("followup_input"))
                pos = str(item.get("followup_position", "below")).lower()  # "above"|"below"|"stack"
                # "above"와 "stack"을 동일하게 취급: 텍스트 먼저, 라디오는 그 아래
                stacked = (pos in ("above", "stack"))

                # (A) 스택형: 텍스트 먼저 → 다음 줄에 라디오
                if has_follow and stacked:
                    # 텍스트 입력 줄
                    ph = item.get("followup_placeholder", "기타 이유를 적어주세요")
                    multiline = bool(item.get("followup_multiline", False))
                    height = int(item.get("followup_height", 3))
                    prefill_etc = str(self._prefill.get((qid, "기타"), ""))

                    ctk.CTkLabel(table, text="사유 입력", anchor="w")\
                        .grid(row=row_idx+1, column=1, padx=10, pady=(0, 6), sticky="w")

                    if multiline:
                        tw = tk.Text(table, height=height, wrap="word")
                        tw.grid(row=row_idx+1, column=2,
                                columnspan=(last_col-1), padx=5, pady=(0, 10), sticky="nsew")
                        if prefill_etc:
                            tw.insert("1.0", prefill_etc)
                        self.vars[f"{qid}_etc__textwidget"] = tw
                    else:
                        etc_var = ctk.StringVar(value=prefill_etc)
                        self.vars[f"{qid}_etc"] = etc_var
                        ctk.CTkEntry(table, textvariable=etc_var, placeholder_text=ph)\
                            .grid(row=row_idx+1, column=2,
                                  columnspan=(last_col-1), padx=5, pady=(0, 10), sticky="ew")

                    # 라디오 줄 (세 번째 줄)
                    ctk.CTkLabel(table, text="", anchor="w")\
                        .grid(row=row_idx+2, column=1, padx=5, pady=5, sticky="w")

                    for i, opt in enumerate(options):
                        ctk.CTkRadioButton(table, text=str(opt), variable=var, value=str(opt))\
                            .grid(row=row_idx+2, column=opt_col_start+i, padx=5, pady=5, sticky="nsew")

                    rows_used = 3

                # (B) 기본형: 라디오 먼저, 그 아래에 텍스트(있다면)
                else:
                    for i, opt in enumerate(options):
                        ctk.CTkRadioButton(table, text=str(opt), variable=var, value=str(opt))\
                            .grid(row=row_idx, column=opt_col_start+i, padx=5, pady=5, sticky="nsew")

                    if has_follow:
                        ph = item.get("followup_placeholder", "기타 이유를 적어주세요")
                        multiline = bool(item.get("followup_multiline", False))
                        height = int(item.get("followup_height", 3))
                        prefill_etc = str(self._prefill.get((qid, "기타"), ""))

                        ctk.CTkLabel(table, text="사유 입력", anchor="w")\
                            .grid(row=row_idx+1, column=1, padx=10, pady=(0, 6), sticky="w")

                        if multiline:
                            tw = tk.Text(table, height=height, wrap="word")
                            tw.grid(row=row_idx+1, column=2,
                                    columnspan=(last_col-1), padx=5, pady=(0, 10), sticky="nsew")
                            if prefill_etc:
                                tw.insert("1.0", prefill_etc)
                            self.vars[f"{qid}_etc__textwidget"] = tw
                        else:
                            etc_var = ctk.StringVar(value=prefill_etc)
                            self.vars[f"{qid}_etc"] = etc_var
                            ctk.CTkEntry(table, textvariable=etc_var, placeholder_text=ph)\
                                .grid(row=row_idx+1, column=2,
                                      columnspan=(last_col-1), padx=5, pady=(0, 10), sticky="ew")

                        rows_used = 2

            # ---------------- 숫자 입력 ----------------
            elif qtype == "input-number":
                var = ctk.StringVar(value=str(self._prefill.get((qid, None), "")))
                self.vars[qid] = var
                self._score_keys.add(qid)
                ctk.CTkEntry(table, textvariable=var, width=80)\
                    .grid(row=row_idx, column=2, padx=5, pady=5, sticky="w")
                ctk.CTkLabel(table, text=f"(최소: {min_val}, 최대: {max_val})",
                             font=("", 10), text_color="gray")\
                    .grid(row=row_idx, column=3, padx=5, pady=5, sticky="w")

                def _validate(*_):
                    v = var.get()
                    if v and not v.isdigit():
                        var.set('')
                var.trace_add("write", _validate)
            elif qtype == "input-float":
                # 기본값(사전에서 가져오기), 문자열 Var 사용
                var = ctk.StringVar(value=str(self._prefill.get((qid, None), "")))
                self.vars[qid] = var
                self._score_keys.add(qid)

                # min/max 파싱 (json에서 전달된 경우)
                try:
                    min_val_f = float(min_val) if min_val is not None else None
                except Exception:
                    min_val_f = None
                try:
                    max_val_f = float(max_val) if max_val is not None else None
                except Exception:
                    max_val_f = None

                entry = ctk.CTkEntry(table, textvariable=var, width=80)
                entry.grid(row=row_idx, column=2, padx=5, pady=5, sticky="w")
                ctk.CTkLabel(table, text=f"(최소: {min_val}, 최대: {max_val})",
                             font=("", 10), text_color="gray")\
                    .grid(row=row_idx, column=3, padx=5, pady=5, sticky="w")

                # 이전 유효값을 기억해두고, 유효하면 갱신, 아니면 되돌림
                prev = var.get()

                # 정규식: 숫자(0-9) 여러개, 선택적 하나의 소수점, 이후 숫자 여러개
                # 허용 형태: ''(빈), '123', '12.34', '.5' (현재는 '.5'를 허용하지 않음)
                float_pattern = re.compile(r'^\d*\.?\d*$')

                def _validate(*_):
                    nonlocal prev
                    v = var.get()
                    # 빈값 허용
                    if v == "":
                        prev = ""
                        return
                    # 패턴 매칭(숫자 + 선택적 하나의 점)
                    if not float_pattern.match(v):
                        # 잘못된 입력이면 이전 유효값으로 복구
                        var.set(prev)
                        return
                    # 숫자 변환 시도
                    try:
                        fv = float(v)
                    except ValueError:
                        var.set(prev)
                        return
                    # 범위 검사 (min/max 가 지정되어 있을 때)
                    if (min_val_f is not None and fv < min_val_f) or (max_val_f is not None and fv > max_val_f):
                        # 범위 밖이면 이전 유효값으로 복구
                        var.set(prev)
                        return
                    # (선택) 소수 자리 제한을 두고 싶다면 여기에 처리 가능
                    # 예: precision = 2 -> fv = round(fv, 2); var.set(str(fv))
                    # 성공하면 prev 갱신
                    prev = var.get()

                # trace 등록
                var.trace_add("write", _validate)
            # ---------------- 자유 텍스트(점수 제외) ----------------
            elif qtype == "input-text":
                placeholder = item.get("placeholder", "텍스트를 입력하세요")
                multiline = bool(item.get("multiline", False))
                prefill_val = str(self._prefill.get((qid, None), ""))

                if multiline:
                    tw = tk.Text(table, height=int(item.get("height", 4)), wrap="word")
                    tw.grid(row=row_idx, column=2, columnspan=(last_col-1),
                            padx=5, pady=5, sticky="nsew")
                    if prefill_val:
                        tw.insert("1.0", prefill_val)
                    self.vars[f"{qid}__textwidget"] = tw
                else:
                    var = ctk.StringVar(value=prefill_val)
                    self.vars[qid] = var
                    ctk.CTkEntry(table, textvariable=var, placeholder_text=placeholder)\
                        .grid(row=row_idx, column=2, columnspan=(last_col-1),
                              padx=5, pady=5, sticky="ew")

            # ---------------- 시간 슬라이더 ----------------
            elif qtype == "slider-time":
                step_minutes = int(item.get("step", item.get("step_minutes", 30)))
                # 프리필 파싱: "HH:MM" 또는 "7.5" 형태를 수용
                prefill_val = str(self._prefill.get((qid, None), "")).strip()
                init = float(min_val)
                if prefill_val:
                    try:
                        if ":" in prefill_val:
                            hh, mm = prefill_val.split(":")
                            init = int(hh) + int(mm) / 60.0
                        else:
                            init = float(prefill_val)
                    except Exception:
                        init = float(min_val)

                # 눈금자 + 슬라이더 위젯
                slider = TimeRulerSlider(
                    table,
                    min_hour=float(min_val), max_hour=float(max_val),
                    step_minutes=step_minutes, width=600, initial=float(init),
                    show_am_label=True
                )
                # 그리드: 문항 오른쪽 전폭 차지
                slider.grid(row=row_idx, column=2, columnspan=(last_col-1),
                            padx=5, pady=5, sticky="ew")

                # 저장/스코어링을 위해 위젯 자체를 등록
                self.vars[f"{qid}__timeruler"] = slider
                self._score_keys.add(("time", qid))

                rows_used = 1
            elif qtype in ("custom-radio", "radio-custom"):
                rows_used = self._build_custom_radio(
                    table=table, row_idx=row_idx, qid=qid, item=item,
                    opt_col_start=opt_col_start, last_col=last_col
                    )


            # 다음 항목을 위한 row 포인터 이동 + 구분선
            row_idx += rows_used

            separator = ctk.CTkFrame(table, height=2, fg_color="#C8CCD2")
            separator.grid(row=row_idx, column=0, columnspan=len(columns),
                           sticky="ew", pady=(6, 6))
            # 프레임이 높이 2px로 유지되도록 (그리드 수축 방지)
            separator.grid_propagate(False)

            row_idx += 1

    # ---------------- 제출/점수 ----------------
    
    def _calc_total_and_check(self) -> Optional[int]:
        """
        총점 계산:
          - self._score_keys 에 포함된 키만 점수로 합산
            * int(qid): 라디오/숫자 입력 → 정수 변환
            * ('time', qid): slider_time → 시간 구간 규칙으로 환산
          - 미입력/형식오류 시 None 리턴(검증 실패)
        """
        total = 0

        for key in self._score_keys:

            # ---------------- slider-time ----------------
            if isinstance(key, tuple) and key[0] == "time":
                qid = key[1]
                widget = self.vars.get(f"{qid}__timeruler")
                if widget is None:
                    return None
                hour_value, _ = widget.get_value()
                score = self._score_slider_time(qid, float(hour_value))
                if score is None:
                    return None
                total += score
                continue

            # ---------------- radio / input-number / input-float ----------------
            qid = key
            var = self.vars.get(qid)
            if var is None:
                return None

            v_str = str(var.get()).strip()
            if v_str == "":
                return None

            # 🔥 float 처리: 소수 포함 여부 확인
            if "." in v_str:
                try:
                    fval = float(v_str)
                    total += fval      # 그대로 총점에 더함
                except ValueError:
                    return None
            else:
                try:
                    total += int(v_str)
                except ValueError:
                    return None

        return total


    def _on_submit(self):
        title_str = str(self.item_data.get("title","")).upper()
        if "MEQ" in title_str:
            # MEQ-K는 여기서 총점을 계산하지 않는다
            total = 0     # 제출창에서 compute_meqk 결과로 대체됨
        else:
            total = self._calc_total_and_check()
        if total is None:
            CTkMessagebox(
                title="입력 누락",
                message="점수 항목(라디오/숫자/시간)을 올바르게 입력해주세요.",
                icon="warning",
                option_1="확인",
            )
            return

        # 해석 규칙이 있으면 간단 표시
        interpretation = "해석 기준 없음"
        for survey in self.survey_data.values():
            scoring = survey.get("scoring")
            if scoring:
                interpretation = self.interpret_score(total, scoring)
                break

        confirm = CTkMessagebox(
            title="제출 확인",
            message=f"총점은 {total}점입니다.\n\n{interpretation}\n\n제출하시겠습니까?",
            icon="question",
            option_1="취소",
            option_2="제출",
        ).get()
        if confirm != "제출":
            return

        ok, err, _ = self.save_to_db(total)
        if ok:
            CTkMessagebox(title="제출 완료", message="저장되었습니다.", icon="check", option_1="확인")
            self._close_with_callback()   # ✅ 저장 성공 → 닫고 콜백 실행
        else:
            CTkMessagebox(title="오류", message=f"저장 실패: {err}", icon="cancel", option_1="확인")

    @staticmethod
    def interpret_score(score, scoring_rules: dict):
        for rule, label in scoring_rules.items():
            if "-" in rule:
                try:
                    lo, hi = map(int, rule.split("-"))
                    if lo <= score <= hi:
                        return label
                except Exception:
                    pass
        return "해석 기준 없음"

    # ---------------- DB 페이로드 ----------------
    def get_db_payload(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        INSERT/UPDATE 페이로드 생성
          - updates: [{"answer_id":..., "answer_value":"..."}]
          - inserts: [{"question_id":..., "answer_component":..., "answer_value":"..."}]
        ※ 라디오는 answer_component=None, followup_input는 comp="기타"
        ※ input-text는 answer_component=None
        ※ slider_time은 answer_value에 "HH:MM" 문자열로 저장
        """
        inserts: List[Dict[str, Any]] = []
        updates: List[Dict[str, Any]] = []

        def add(qid: int, comp: Optional[str], value: str):
            if not qid:
                return
            key = (qid, comp)
            existed = self._existing_index.get(key)
            if existed and existed.get("answer_id") is not None:
                updates.append({"answer_id": existed["answer_id"], "answer_value": value})
            else:
                inserts.append({"question_id": qid, "answer_component": comp, "answer_value": value})

        for key, var in list(self.vars.items()):
            # ✅ multiline Text 위젯
            if isinstance(key, str) and key.endswith("__textwidget"):
                try:
                    qid_str = key.split("__textwidget")[0]
                    qid = int(qid_str)
                except Exception:
                    continue
                widget: tk.Text = var  # type: ignore
                val = widget.get("1.0", "end-1c").strip()
                if val != "":
                    add(qid, None, val)
                continue

            # ✅ 시간 슬라이더 위젯
            if isinstance(key, str) and key.endswith("__timeruler"):
                try:
                    qid_str = key.split("__timeruler")[0]
                    qid = int(qid_str)
                except Exception:
                    continue
                widget: TimeRulerSlider = var  # type: ignore
                _, hhmm = widget.get_value()
                if hhmm:
                    add(qid, None, hhmm)  # HH:MM 형식으로 저장
                continue

            # ✅ follow-up 텍스트
            if isinstance(key, str) and key.endswith("_etc"):
                try:
                    qid = int(key.split("_")[0])
                    val = str(var.get()).strip()
                    if val != "":
                        add(qid, "기타", val)
                except Exception:
                    continue
            else:
                # 일반 qid (radio / input-number / input-text 단일라인)
                if isinstance(key, int):
                    try:
                        qid = key
                        val = str(var.get()).strip()
                        if val != "":
                            add(qid, None, val)
                    except Exception:
                        continue

        return {"inserts": inserts, "updates": updates}

    # ---------------- DB 저장 ----------------
    def save_to_db(self, total_score: int) -> Tuple[bool, Optional[str], Optional[int]]:
        try:
            from api_local.form_api_local import (
                create_new_item_and_get_id_generic,
                save_answers,
                update_mds_answers,
                mark_item_updated,
            )
            try:
                from api_local.item_api_local import update_item_description  # type: ignore
                _has_update_item_desc = True
            except Exception:
                _has_update_item_desc = False

            from utils.db_utils import get_connection, release_connection
            from psycopg2.extras import RealDictCursor
        except Exception as e:
            return False, f"API 로드 실패: {e}", None

        payload = self.get_db_payload()
        inserts = payload["inserts"]
        updates = payload["updates"]

        # 컨텍스트
        patient_uuid  = self.patient_uuid
        data_category = self.item_data.get("data_category", "MDD")
        data_type     = self.item_data.get("data_type", "E-SURVEY")
        seq           = int(self.item_data.get("seq", 1))
        title         = self.item_data.get("title", data_type)

        if not patient_uuid:
            return False, "patient_id(=UUID)가 없습니다.", None

        desc_with_score = f"{title} 총점 {total_score}점"
    
        item_id = self.item_data.get("item_id")
        is_psqi = ("PSQI" in str(title).upper()) or ("PSQI" in str(data_type).upper())
        if is_psqi:
            try:
                psqi_answers = self._collect_psqi_answers_for_util()
                psqi_result = compute_psqi(psqi_answers)
                print(psqi_result)
                # 설명에 GPSQI 및 도메인 점수 간단 표기
                desc_with_score = (
                    f"PSQI 종합점수(GPSQI) {psqi_result.GPSQI}점 "
                )
                # 💡 필요하면 여기서 psqi_result.as_dict()의 derived/domains를
                #     별도 '파생지표' 질문ID에 저장하도록 inserts에 추가하는 것도 가능.
                #     (스키마에 저장용 question_id를 마련했다면 여기에 append)
            except Exception as e:
                # PSQI 계산 실패해도 설문 저장은 진행
                print(f"[경고] PSQI 계산 실패: {e}")

        is_meqk = ("MEQ" in str(title).upper()) or ("MEQ" in str(self.item_data.get("data_type","")).upper())

        if is_meqk:
            try:
                ans = self._collect_meqk_answers_for_util()
                total = compute_meqk(ans)
                
                result = debug_meqk(ans)
                print(result)
                total = result["total"]
                desc_with_score = f"MEQ-K 총점 {total}점"
            except Exception as e:
                print(f"[경고] MEQ-K 계산 실패: {e}")
        item_id = self.item_data.get("item_id") 
        print(desc_with_score)
        # 신규: 제출 시 생성
        if item_id is None:
            new_item_id = create_new_item_and_get_id_generic(
                target_patient_id=patient_uuid,
                data_category=data_category,
                data_type=data_type,
                seq=seq,
                description=desc_with_score,
            )
            if not isinstance(new_item_id, int): 
                return False, "tb_items 생성 실패", None

            if inserts:
                ok, err = save_answers(new_item_id, inserts)
                if not ok:
                    return False, f"설문 답변 저장 실패: {err}", None

            self.item_data["item_id"] = new_item_id
            return True, None, new_item_id

        # 수정: 기존 응답 업데이트 + description 갱신
        if updates:
            ok, err = update_mds_answers(updates)
            if not ok:
                return False, f"설문 답변 수정 실패: {err}", item_id
        if inserts:
            ok, err = save_answers(item_id, inserts)
            if not ok:
                return False, f"추가 답변 저장 실패: {err}", item_id

        if _has_update_item_desc:
            ok = update_item_description(item_id, desc_with_score)
            if not ok:
                return False, f"점수 설명 갱신 실패: {err}", item_id
        else:
            # 폴백: 직접 SQL
            conn = None
            try:
                conn = get_connection()
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        UPDATE dev_kkh.tb_items
                           SET description = %s, updated_at = NOW()
                         WHERE item_id = %s;
                    """, (desc_with_score, item_id))
                conn.commit()
            except Exception as e:
                if conn:
                    conn.rollback()
                return False, f"설명 갱신 실패(SQL): {e}", item_id
            finally:
                if conn:
                    release_connection(conn)

        try:
            mark_item_updated(item_id)
        except Exception:
            pass

        return True, None, item_id

    # ---------------- 닫기 공통 처리 ----------------
    def _close_with_callback(self):
        """모달을 닫고 콜백 호출(있으면)"""
        try:
            if callable(self._on_close_callback):
                self._on_close_callback()
        finally:
            try:
                self._modal.destroy()
            except Exception:
                pass

    def _on_close_clicked(self):
        self._close_with_callback()


    

    def _collect_meqk_answers_for_util(self):
    
        answers = {}
    
        MEQK_VALUE_MAP = {
            3:[4,3,2,1],
            4:[1,2,3,4],
            5:[1,2,3,4],
            6:[1,2,3,4],
            7:[1,2,3,4],
            8:[4,3,2,1],
            9:[4,3,2,1],
            11:[6,4,2,0],
            12:[0,2,3,5],
            13:[4,3,2,1],
            14:[1,2,3,4],
            15:[4,3,2,1],
            16:[1,2,3,4],
            19:[6,4,2,0]
        }
    
        for key in self._score_keys:
        
            # ---------------------------------------
            # 🔹 (1) 슬라이더 타입 ("time", qid)
            # ---------------------------------------
            if isinstance(key, tuple) and key[0] == "time":
                _, qid = key  # <-- 반드시 이렇게 분리해야 한다!!
    
                widget = self.vars.get(f"{qid}__timeruler")
                if widget:
                    hour_value, _ = widget.get_value()
                    answers[str(qid)] = float(hour_value)
                continue
            
            # ---------------------------------------
            # 🔹 (2) 라디오/정수 타입 (qid = int)
            # ---------------------------------------
            qid = key
            var = self.vars.get(qid)
            if var is None:
                continue
            
            raw = str(var.get()).strip()
            if raw == "":
                continue
            
            # 선택 index 기반 값 변환
            if raw.isdigit():
                idx = int(raw) - 1
    
                if qid in MEQK_VALUE_MAP and 0 <= idx < len(MEQK_VALUE_MAP[qid]):
                    answers[str(qid)] = MEQK_VALUE_MAP[qid][idx]
                else:
                    answers[str(qid)] = int(raw)
    
        return answers

