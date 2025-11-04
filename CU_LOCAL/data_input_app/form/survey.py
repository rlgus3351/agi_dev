# -*- coding: utf-8 -*-
"""
HealthSurveyForm (기초 평가 폼)
- DB → UI 자동 역매핑 (questions_raw → BASIC_QUESTION_MAPPING label)
- 저장 시 insert / edit 자동 분기:
  - edit 모드: 기존 answers 매칭되면 answer_id 기반 UPDATE, 없으면 INSERT
  - insert 모드: 전부 INSERT
- 저장 성공 후 on_close_callback 호출 + 모달 자동 닫기
"""

import os
import json
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
from typing import Optional, Callable, Dict, Any, List, Tuple

# 백엔드 API
from api_local.form_api_local import (
    BASIC_QUESTION_MAPPING,
    create_new_item_and_get_id_generic,
    save_answers,
    update_existing_survey_answers_by_id,  # ✅ answer_id 기반 수정 API
    mark_item_updated,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.abspath(os.path.join(CURRENT_DIR, "..", "form", "basic_form", "basic.json"))


class HealthSurveyForm(ctk.CTkFrame):
    """기초 평가 (Basic Health Form) UI"""

    # ====== 옵션 그룹/정규화(표준화) 규칙 ======
    _ALCOHOL_OPTS   = {"소주", "맥주", "기타"}
    _CAFFEINE_OPTS  = {"커피", "홍차", "녹차", "기타"}
    _DISEASE_OPTS   = {"우울증", "불면증", "기타", "발병 시점", "발병시점"}  # 띄어쓰기 혼용 허용
    _COMP_NORMALIZE = {
        "발병시점": "발병 시점",
    }

    def __init__(
        self,
        parent,
        patient_id: str,
        json_file: str = DEFAULT_JSON,
        item_data: Optional[Dict[str, Any]] = None,
        on_close_callback: Optional[Callable] = None,
    ):
        super().__init__(parent)
        self.json_file = json_file
        self.patient_id = patient_id
        self.item_data = item_data or {}
        self.on_close_callback = on_close_callback

        # 🔎 모드 감지: "item_data 글자수" 기준 (요청사항)
        self.mode = self._detect_mode()
        print("mode : "+self.mode)
        
        # ✅ 기존 답변 인덱스 (edit 모드에서 answer_id 매칭용)
        # key: (question_id, answer_component or None) -> {"answer_id": int, "answer_value": str}
        self._existing_answers_index: Dict[Tuple[int, Optional[str]], Dict[str, Any]] = self._build_existing_index()

        # ✅ DB → UI 자동 역매핑 (questions_raw → label)
        self.prefilled_data = self._build_prefilled_data()

        self.widgets: Dict[str, Any] = {}
        self.data_vars: Dict[str, Any] = {}
        self.option_subs: Dict[str, Dict[str, ctk.CTkFrame]] = {}

        # 숫자 입력 검증 콜백
        self.vcmd = (self.register(self.validate_number_input), "%P", "%S", "%V", "%W")

        # 안내 문구
        if self.mode == "edit":
            ctk.CTkLabel(
                self,
                text="✏️ 기존 설문 데이터가 불러와졌습니다. 수정 후 저장 가능합니다.",
                text_color="green",
                font=("", 13, "italic"),
            ).pack(pady=(5, 0))
        else:
            ctk.CTkLabel(
                self,
                text="새로운 설문 데이터를 입력하세요.",
                text_color="gray40",
                font=("", 13, "italic"),
            ).pack(pady=(5, 0))

        # UI 생성
        self.load_data_and_create_widgets()

    # =========================================
    # 유틸: 모드/정규화/부모라벨/불리언 문자열
    # =========================================
    def _detect_mode(self) -> str:
        raw = self.item_data.get("questions_raw", None)
        # raw가 리스트이고, answer_id가 하나라도 존재하면 'edit'
        if isinstance(raw, list) and any((r or {}).get("answer_id") is not None for r in raw):
            return "edit"
        # 그 외엔 'insert'
        return "insert"

    def _normalize_component(self, comp: Optional[str]) -> Optional[str]:
        if comp is None:
            return None
        return self._COMP_NORMALIZE.get(str(comp), str(comp))

    def _parent_label_for_component(self, comp: str) -> Optional[str]:
        c = self._normalize_component(comp)
        if c in self._ALCOHOL_OPTS:
            return "음주 종류"            # BASIC_QUESTION_MAPPING에 존재해야 함
        if c in self._CAFFEINE_OPTS:
            return "카페인 음료 섭취"      # BASIC_QUESTION_MAPPING에 존재해야 함
        if c in self._DISEASE_OPTS:
            return "현병력"               # BASIC_QUESTION_MAPPING에 존재해야 함
        return None

    def _normalize_bool_string(self, v: str) -> str:
        s = str(v).strip().lower()
        if s in ("true", "1", "yes", "y"):
            return "true"
        if s in ("false", "0", "no", "n"):
            return "false"
        return str(v)

    # =========================================
    # 기존 답변 인덱스 (edit 매칭용)
    # =========================================
    def _build_existing_index(self) -> Dict[Tuple[int, Optional[str]], Dict[str, Any]]:
        """
        DB의 기존 answers를 (qid, comp)로 키잉.
        comp는 표준화하여 이후 UI에서 넘어오는 comp와 동일 키로 맞춘다.
        """
        index: Dict[Tuple[int, Optional[str]], Dict[str, Any]] = {}
        raw = self.item_data.get("questions_raw", [])
        if not isinstance(raw, list):
            return index

        for row in raw:
            qid = row.get("question_id")
            comp = self._normalize_component(row.get("answer_component"))
            aid = row.get("answer_id")
            val = row.get("answer_value")
            if qid is not None and aid is not None:
                index[(qid, comp)] = {"answer_id": aid, "answer_value": val}
        return index

    # =========================================
    # Prefill 로직: DB → UI 역매핑
    # =========================================
    def _build_prefilled_data(self) -> Dict[str, Any]:
        """
        questions_raw → BASIC_QUESTION_MAPPING 역매핑하여 UI 프리필 데이터 구성.
        - 단일값(라디오/텍스트/숫자):  {라벨: 값}
        - 체크박스(컴포넌트 존재):    {라벨_옵션: "true"/"false"}
        - 옵션명이 잘못된 부모(qid)로 저장된 경우를 옵션셋으로 보정
        """
        prefilled: Dict[str, Any] = {}

        # (1) 이미 dict로 저장되어 있으면 그대로 사용
        if isinstance(self.item_data.get("questions"), dict):
            return self.item_data["questions"]

        # (2) raw answer에서 역매핑
        raw_answers: List[Dict[str, Any]] = self.item_data.get("questions_raw", [])
        if not raw_answers:
            return prefilled

        reverse_map = {qid: label for label, qid in BASIC_QUESTION_MAPPING.items()}

        for ans in raw_answers:
            qid = ans.get("question_id")
            comp = self._normalize_component(ans.get("answer_component"))  # 표준화
            value = ans.get("answer_value", "")

            # 기본 역매핑
            label = reverse_map.get(qid)

            # ✅ 옵션명으로 부모 라벨 보정 (DB에 잘못 들어간 경우 대비)
            if comp:
                parent = self._parent_label_for_component(comp)
                if parent is not None:
                    label = parent

            if comp:
                # 체크박스류: "라벨_옵션" 평탄화 키
                if label:
                    flat_key = f"{label}_{comp}"
                    prefilled[flat_key] = self._normalize_bool_string(value)
            else:
                # 단일값: 라벨 그대로
                if label:
                    prefilled[label] = str(value)

        return prefilled

    # =========================================
    # UI 생성
    # =========================================
    def validate_number_input(self, new_value, changed_char, validation_type, widget_name):
        if new_value == "":
            return True
        return changed_char.isdigit()

    def load_data_and_create_widgets(self):
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                self.survey_data = json.load(f)
        except FileNotFoundError:
            ctk.CTkLabel(self, text=f"❌ {self.json_file} 파일을 찾을 수 없습니다.").pack(pady=20)
            return

        # 스크롤 영역
        scrollable_frame = ctk.CTkScrollableFrame(self, height=650)
        scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)

        row_num = 0
        for section, items in self.survey_data.items():
            ctk.CTkLabel(
                scrollable_frame,
                text=section,
                font=ctk.CTkFont(size=18, weight="bold"),
            ).grid(row=row_num, column=0, columnspan=2, pady=(10, 5), sticky="w")
            row_num += 1

            for key, config in items.items():
                row_num = self._create_widget(scrollable_frame, key, config, row_num)

        # 저장 버튼
        ctk.CTkButton(self, text="💾 DB 저장", command=self.get_entered_data).pack(pady=10)

    def _create_widget(self, parent_frame, key, config, row):
        item_type = config.get("type")
        label_text = config.get("label") or key
        prefill = str(self.prefilled_data.get(key, "")).strip()

        # ---------------- 라디오 ----------------
        if item_type == "radio":
            radio_var = ctk.StringVar(value=prefill)
            self.data_vars[key] = radio_var

            ctk.CTkLabel(parent_frame, text=label_text, font=("", 14)).grid(
                row=row, column=0, sticky="w", padx=10, pady=(10, 0)
            )
            radio_frame = ctk.CTkFrame(parent_frame)
            radio_frame.grid(row=row, column=1, sticky="w", padx=10, pady=(10, 0))

            self.option_subs[key] = {}
            col = 0
            for option in config.get("options", []):
                # 단순 문자열 옵션
                if isinstance(option, str):
                    ctk.CTkRadioButton(
                        radio_frame,
                        text=option,
                        variable=radio_var,
                        value=option,
                        command=lambda k=key, v=option: self._toggle_radio_sub(k, v),
                    ).grid(row=0, column=col, sticky="w", padx=5)
                    col += 1

                # 하위 입력 포함 옵션
                elif isinstance(option, dict):
                    opt_label = option.get("label", "")
                    opt_sub = option.get("sub", {})

                    # 하위 프레임
                    sub_frame = ctk.CTkFrame(
                        parent_frame,
                        fg_color="#E8E8E8",
                        corner_radius=10,
                        border_color="#CCCCCC",
                        border_width=1,
                    )
                    sub_frame.grid(
                        row=row + 1,
                        column=0,
                        columnspan=2,
                        sticky="ew",
                        padx=100,
                        pady=(10, 15),
                    )
                    sub_frame.grid_remove()

                    inner = ctk.CTkFrame(sub_frame, fg_color="transparent")
                    inner.pack(fill="both", expand=True, padx=20, pady=15)

                    sub_row = 0
                    for sub_key, sub_cfg in opt_sub.items():
                        sub_row = self._create_widget(inner, sub_key, sub_cfg, sub_row)

                    ctk.CTkRadioButton(
                        radio_frame,
                        text=opt_label,
                        variable=radio_var,
                        value=opt_label,
                        command=lambda k=key, v=opt_label: self._toggle_radio_sub(k, v),
                    ).grid(row=0, column=col, sticky="w", padx=5)
                    col += 1
                    self.option_subs[key][opt_label] = sub_frame

            # 초기 표시
            if prefill:
                self._toggle_radio_sub(key, prefill)

            # 값 바뀌면 하위 토글
            radio_var.trace_add(
                "write", lambda *a, k=key: self._toggle_radio_sub(k, self.data_vars[k].get())
            )
            return row + 4

        # ---------------- 체크박스 ----------------
        elif item_type == "checkbox":
            ctk.CTkLabel(parent_frame, text=label_text, font=("", 14)).grid(
                row=row, column=0, sticky="w", padx=10, pady=(10, 0)
            )
            box_frame = ctk.CTkFrame(parent_frame)
            box_frame.grid(row=row, column=1, sticky="w", padx=10, pady=(10, 0))

            for i, opt in enumerate(config.get("options", [])):
                # 현재 구조는 "질문_옵션" 같은 평탄화 키로 저장
                full_key = f"{key}_{opt}" if isinstance(opt, str) else f"{key}_{opt.get('label', '')}"
                checked = str(self.prefilled_data.get(full_key, "")).lower() in ("true", "1", "yes")
                var = ctk.BooleanVar(value=checked)

                text_label = opt if isinstance(opt, str) else opt.get("label", "")
                ctk.CTkCheckBox(box_frame, text=text_label, variable=var).grid(
                    row=i, column=0, sticky="w", pady=2
                )
                self.data_vars[full_key] = var

                # 하위(sub)가 있으면 영역 만들고 토글 (필요 시 확장 가능)
                if isinstance(opt, dict) and opt.get("sub"):
                    sub_frame = ctk.CTkFrame(
                        parent_frame,
                        fg_color="#E8E8E8",
                        corner_radius=10,
                        border_color="#CCCCCC",
                        border_width=1,
                    )
                    sub_frame.grid(
                        row=row + 1,
                        column=0,
                        columnspan=2,
                        sticky="ew",
                        padx=100,
                        pady=(10, 15),
                    )
                    sub_frame.grid_remove()

                    inner = ctk.CTkFrame(sub_frame, fg_color="transparent")
                    inner.pack(fill="both", expand=True, padx=20, pady=10)

                    sub_row = 0
                    for sub_key, sub_cfg in opt.get("sub", {}).items():
                        sub_row = self._create_widget(inner, sub_key, sub_cfg, sub_row)

                    def toggle_checkbox_sub(v=var, f=sub_frame):
                        f.grid() if v.get() else f.grid_remove()

                    var.trace_add("write", lambda *a, v=var, f=sub_frame: toggle_checkbox_sub(v, f))
                    if checked:
                        sub_frame.grid()

            return row + 4

        # ---------------- 숫자 입력 ----------------
        elif item_type == "input-number":
            var = ctk.StringVar(value=prefill)
            self.data_vars[key] = var

            def only_digits(*args, v=var):
                value = v.get()
                if value != "" and not value.isdigit():
                    v.set("".join([c for c in value if c.isdigit()]))

            var.trace_add("write", only_digits)

            ctk.CTkLabel(parent_frame, text=label_text, font=("", 14)).grid(
                row=row, column=0, sticky="w", padx=10, pady=(10, 0)
            )
            ctk.CTkEntry(
                parent_frame,
                textvariable=var,
                width=120,
                placeholder_text=config.get("placeholder", ""),
                fg_color="white",
            ).grid(row=row, column=1, sticky="w", padx=10)
            return row + 1

        # ---------------- 텍스트 입력 ----------------
        elif item_type == "input-text":
            var = ctk.StringVar(value=prefill)
            self.data_vars[key] = var

            ctk.CTkLabel(parent_frame, text=label_text, font=("", 14)).grid(
                row=row, column=0, sticky="w", padx=10, pady=(10, 0)
            )
            ctk.CTkEntry(
                parent_frame,
                textvariable=var,
                width=300,
                placeholder_text=config.get("placeholder", ""),
            ).grid(row=row, column=1, sticky="w", padx=10)
            return row + 1

        # 기타 타입은 스킵
        return row + 1

    # 라디오 하위 토글
    def _toggle_radio_sub(self, key, value):
        if key not in self.option_subs:
            return
        for label, frame in self.option_subs[key].items():
            if label == value:
                frame.grid()
            else:
                frame.grid_remove()

    # =========================================
    # 저장 로직
    # =========================================
    def get_entered_data(self):
        """
        - UI 값 수집 → BASIC_QUESTION_MAPPING 기준으로 question_id 매핑
        - insert / edit 자동 분기:
            * edit(기존 answers 존재): answer_id 매칭 → UPDATE, 없으면 INSERT
            * insert: 전부 INSERT
        - 저장 성공 시 on_close_callback 실행 + 모달 자동 닫기
        """
        # 1) UI 값 수집
        raw_result: Dict[str, Any] = {}
        for key, var in self.data_vars.items():
            if isinstance(var, (ctk.StringVar, ctk.BooleanVar)):
                raw_result[key] = str(var.get()).strip()

        # 2) UI → answers 포맷화
        inserts: List[Dict[str, Any]] = []
        updates: List[Dict[str, Any]] = []

        def append_insert_or_update(qid: int, comp: Optional[str], value: str):
            """existing index를 보고 update 또는 insert 분기 (edit일 때만 update 시도)"""
            norm_comp = self._normalize_component(comp)
            existed = self._existing_answers_index.get((qid, norm_comp))
            if self.mode == "edit" and existed and existed.get("answer_id") is not None:
                updates.append({"answer_id": existed["answer_id"], "answer_value": value})
            else:
                inserts.append({
                    "question_id": qid,
                    "answer_component": norm_comp,
                    "answer_value": value
                })

        # 매핑 및 분류
        for key, value in raw_result.items():
            comp = None
            base_key = key
            if "_" in key:
                # 체크박스 평탄화 키: "질문_옵션"
                base_key, comp = key.split("_", 1)

            qid = BASIC_QUESTION_MAPPING.get(base_key)
            if qid is None:
                continue  # 매핑되지 않은 항목은 스킵

            # Boolean 문자열 정규화 (체크박스일 수 있음)
            if value.lower() in ("true", "false"):
                norm_val = "true" if value.lower() == "true" else "false"
                append_insert_or_update(qid, comp, norm_val)
            else:
                append_insert_or_update(qid, comp, value)

        # 3) item_id 결정 (없으면 생성 → insert 모드)
        item_id = getattr(self.item_data, "item_id", None) or self.item_data.get("item_id")
        try:
            if not item_id:
                # 완전 신규
                item_id = create_new_item_and_get_id_generic(
                    self.patient_id, data_category="MDD", data_type="B-SURVEY", seq=1, description="기초평가"
                )
                if not item_id:
                    CTkMessagebox(title="오류", message="❌ Item 생성 실패", icon="cancel")
                    return

            # 4) 저장 실행
            ok_all = True
            msgs = []

            # ✅ edit 모드면 answer_id로 UPDATE 먼저
            if self.mode == "edit" and updates:
                ok, err = update_existing_survey_answers_by_id(updates)
                ok_all = ok_all and ok
                if not ok:
                    msgs.append(f"업데이트 실패: {err}")

            # 신규 INSERT
            if inserts:
                ok, err = save_answers(item_id, inserts)
                ok_all = ok_all and ok
                if not ok:
                    msgs.append(f"신규 저장 실패: {err}")

            # 상태 마킹
            mark_item_updated(item_id)

            # 결과 알림
            if ok_all:
                CTkMessagebox(title="성공", message=f"✅ 저장 완료 (item_id={item_id})", icon="check")
            else:
                CTkMessagebox(title="부분 실패", message=";\n".join(msgs), icon="warning")

            # 콜백
            if self.on_close_callback:
                self.on_close_callback()

            # 닫기
            parent_window = self.winfo_toplevel()
            if parent_window:
                parent_window.destroy()

        except Exception as e:
            CTkMessagebox(title="오류", message=f"예외 발생: {e}", icon="cancel")
            raise
