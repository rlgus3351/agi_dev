import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json
import os
import tkinter.filedialog as filedialog
from datetime import datetime
from typing import Union
from typing import Optional
from api.schemas import Item

from api.form_api import (
    MDS_QUESTION_MAPPING,
    transform_to_api_format,
    create_new_item_and_get_id,
    call_api_to_save_data,
    call_api_to_update_data
)

# ✅ 기본 JSON 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'form', 'mobility.json'))


class HealthSurveyForm(ctk.CTkFrame):
    def __init__(self, parent, patient_id, json_file=JSON_FILE, initial_data=None, item_data: Union[dict, Item, None] = None, on_close_callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.json_file = json_file
        self.patient_id = patient_id
        self.on_close_callback = on_close_callback
        if initial_data:
            print("초기 데이터:", initial_data)
        self.initial_data = initial_data or []
        self.item_data = item_data
        is_item_updated = False
        
        
        if self.item_data:
            # Item 데이터가 딕셔너리일 경우를 기본으로 가정하고 .get()으로 접근
            if isinstance(self.item_data, dict):
                is_item_updated = self.item_data.get('is_updated', False)
            else:
                # Item이 Pydantic 객체일 경우를 대비하여 getattr() 사용
                is_item_updated = getattr(self.item_data, 'is_updated', False)


        # 🚨 모드 결정 로직: VIEW 모드 진입 조건
        if not self.initial_data:
            self.mode = "INSERT" # 답변 데이터 없음 (새로 Item 생성해야 함)
        elif is_item_updated: 
            self.mode = "VIEW"    # Item은 있으나 is_updated가 True (수정 불가)
        else:
            self.mode = "EDIT"    # Item은 있으나 is_updated가 False/None (최초 등록 후 수정 가능)
        self.widgets = {}
        self.data_vars = {}
        self.scrollable_frame = None
        self.vcmd = (self.register(self.validate_number_input), '%P', '%S', '%V', '%W')
        self.is_resubmitting = False

        # ✅ 초기 데이터 매핑 (question_id + component)
        self.initial_answers_map = {
            f"{ans['question_id']}_{ans['answer_component']}" if ans.get("answer_component")
            else str(ans["question_id"]): str(ans.get("answer_value", ""))
            for ans in self.initial_data
        }

        self.load_data_and_create_widgets()

    # ---------------- 입력 유효성 ----------------
    def validate_number_input(self, new_value, changed_char, validation_type, widget_name):
        if new_value == "":
            return True
        return changed_char.isdigit()

    # ---------------- UI 구성 ----------------
    def load_data_and_create_widgets(self):
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.survey_data = json.load(f)
        except FileNotFoundError:
            ctk.CTkLabel(self, text=f"오류: {self.json_file} 파일을 찾을 수 없습니다.").pack(pady=20)
            return

        frame = ctk.CTkScrollableFrame(self, height=700)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        frame.grid_columnconfigure((0, 1), weight=1)
        self.scrollable_frame = frame

        row = 0
        for section in self.survey_data.get("운동성 검사", {}).get("sections", []):
            header = section.get("header", {})
            body = section.get("body", [])

            header_label = ctk.CTkLabel(
                frame,
                text=f"{header.get('title', '')}\n{header.get('description', '')}",
                font=ctk.CTkFont(size=16, weight="bold"),
                justify="center", anchor="center", wraplength=700
            )
            header_label.grid(row=row, column=0, columnspan=2, sticky="new", pady=(20, 15))
            row += 1

            for item in body:
                row = self._create_widget(frame, item, row)

        # 🔸 모드별 버튼 텍스트 + 색상 구분
        if self.mode == "VIEW":
            # VIEW 모드일 경우, 버튼을 '데이터 보기' 텍스트로 변경하고 비활성화/제거
            btn_text = "데이터 보기 (수정 불가)"
            btn_color = "gray"
            hover_color = "gray"
            state = "disabled" # 버튼 비활성화

        else: # INSERT 또는 EDIT 모드
            btn_text = "데이터 수정" if self.mode == "EDIT" else "데이터 저장"
            btn_color = "orange" if self.mode == "EDIT" else "#1E90FF"
            hover_color = "darkorange" if self.mode == "EDIT" else "#0B61A4"

        # 🔸 버튼 생성
        self.submit_btn = ctk.CTkButton(
            frame,
            text=btn_text,
            command=self.on_submit_click,
            fg_color=btn_color,
            hover_color=hover_color,
            text_color="white",
            corner_radius=10,
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.submit_btn.grid(row=row, column=0, columnspan=2, pady=(20, 5), sticky="ew")

    # ---------------- 항목 위젯 생성 ----------------
    def _create_widget(self, parent, config, row):
        item_type = config.get("type")
        question = config.get("question")
        item_id = config.get("id")
        qnum = MDS_QUESTION_MAPPING.get(item_id)

        # ✅ 라디오 버튼
        if item_type == "radio":
            var = ctk.StringVar(value=self.initial_answers_map.get(str(qnum), ""))
            self.data_vars[item_id] = var

            ctk.CTkLabel(parent, text=question, font=('', 14), justify="left", wraplength=450)\
                .grid(row=row, column=0, sticky="w", padx=10, pady=10)

            frame = ctk.CTkFrame(parent)
            frame.grid(row=row, column=1, sticky="w", padx=10, pady=10)

            for i, opt in enumerate(config.get("options", [])):
                # ✅ value는 반드시 option 문자열 그대로
                ctk.CTkRadioButton(frame, text=opt, variable=var, value=opt)\
                    .grid(row=0, column=i, padx=5)

            return row + 1

        # ✅ 숫자 입력 필드
        elif item_type == "input-number":
            var = ctk.StringVar(value=self.initial_answers_map.get(str(qnum), ""))
            self.data_vars[item_id] = var

            ctk.CTkLabel(parent, text=question, font=('', 14), justify="left", wraplength=450)\
                .grid(row=row, column=0, sticky="w", padx=10, pady=10)
            ctk.CTkEntry(parent, textvariable=var, width=100, validate='key', validatecommand=self.vcmd)\
                .grid(row=row, column=1, sticky="w", padx=10, pady=10)
            return row + 1

        # ✅ 양쪽 입력 필드 (grouped-inputs)
        elif item_type == "grouped-inputs":
            ctk.CTkLabel(parent, text=question, font=('', 14), justify="left", wraplength=450)\
                .grid(row=row, column=0, sticky="w", padx=10, pady=10)
            frame = ctk.CTkFrame(parent)
            frame.grid(row=row, column=1, sticky="w", padx=10, pady=10)
            for i, side in enumerate(config.get("sides", [])):
                sid = f"{item_id}_{side}"
                key = f"{qnum}_{side}"
                var = ctk.StringVar(value=self.initial_answers_map.get(key, ""))
                self.data_vars[sid] = var
                ctk.CTkLabel(frame, text=side).grid(row=i, column=0, padx=5)
                ctk.CTkEntry(frame, textvariable=var, width=80, validate='key', validatecommand=self.vcmd)\
                    .grid(row=i, column=1, padx=5)
            return row + 1

        return row

    # ---------------- 버튼 클릭 시 ----------------
    def on_submit_click(self):
        raw_data = {k: v.get() for k, v in self.data_vars.items()}
        answers = transform_to_api_format(raw_data)

        if not answers:
            CTkMessagebox(title="경고", message="입력된 응답 데이터가 없습니다.", icon="warning")
            return

        if self.mode == "INSERT":
            self._handle_insert(answers)
        else:
            self._handle_edit(answers)

    # ---------------- 신규 삽입 ----------------
    # def _handle_insert(self, answers):
    #     item_id = create_new_item_and_get_id(self.patient_id)
    #     if not item_id:
    #         CTkMessagebox(title="API 오류", message="Item 생성 실패", icon="cancel")
    #         return

    #     success, err = call_api_to_save_data(item_id, answers)
    #     if success:
    #         CTkMessagebox(title="성공", message="새 설문이 등록되었습니다.", icon="check")
    #         if callable(self.on_close_callback):
    #             self.on_close_callback()  # ✅ 신규 입력 후도 리로드
    #     else:
    #         CTkMessagebox(title="오류", message=err, icon="cancel")
    def _handle_insert(self, answers):
        """새로운 설문 데이터를 서버에 저장하는 로직 (간단한 필수값 검증 포함)"""

        # ✅ 중복 클릭 방지 (버튼 잠금)
        self.submit_btn.configure(state="disabled", text="저장 중...")

        try:
            # ✅ 필수값 검증: 모든 값이 비어 있으면 경고
            empty_fields = [k for k, v in self.data_vars.items() if not str(v.get()).strip()]
            if len(empty_fields) == len(self.data_vars):
                CTkMessagebox(
                    title="입력 오류",
                    message="값이 입력되지 않았습니다.\n최소 한 항목 이상 입력해주세요.",
                    icon="warning"
                )
                return

            # ✅ 일부 항목만 입력된 경우 안내 (선택사항)
            if empty_fields:
                msg = f"총 {len(empty_fields)}개 항목이 비어 있습니다.\n그래도 저장하시겠습니까?"
                confirm_box = CTkMessagebox(
                    title="확인",
                    message=msg,
                    icon="question",
                    option_1="예",
                    option_2="아니오"
                )
                if confirm_box.get() == "아니오":
                    return

            # ✅ 모든 입력값을 문자열로 변환
            for a in answers:
                a["answer_value"] = str(a.get("answer_value", "")).strip()

            # ✅ 서버 요청
            item_id = create_new_item_and_get_id(self.patient_id)
            if not item_id:
                CTkMessagebox(title="API 오류", message="Item 생성 실패", icon="cancel")
                return

            success, err = call_api_to_save_data(item_id, answers)
            

            if success:
                # ✅ 성공 메시지 → 창 닫기 여부
                confirm_box = CTkMessagebox(
                    title="성공",
                    message="데이터가 성공적으로 저장되었습니다.",
                    icon="check",
                    option_1="확인"
                )
                confirm_box.get()

                close_box = CTkMessagebox(
                    title="창 닫기",
                    message="폼을 닫을까요?",
                    icon="question",
                    option_1="예",
                    option_2="아니오"
                )
                if close_box.get() == "예":
                    if callable(self.on_close_callback):
                        self.on_close_callback()
                    if self.master and hasattr(self.master, "destroy"):
                        self.master.destroy()
            else:
                CTkMessagebox(title="오류", message=err or "저장 실패", icon="cancel")

        finally:
            # ✅ 저장 완료 후 버튼 다시 활성화
            self.submit_btn.configure(state="normal", text="데이터 저장")


    # ---------------- 수정 모드 ----------------
    def _handle_edit(self, answers):
        """
        설문 수정 모드 처리:
        1️⃣ 변경된 응답만 추출
        2️⃣ 범위 유효성 검사
        3️⃣ 서버에 업데이트 요청
        4️⃣ item의 update_ts, is_updated 갱신
        """
        print(f"[DEBUG] data_vars snapshot: { {k: v.get() for k, v in self.data_vars.items()} }")
        updated = []
    
        # ✅ 중복 클릭 방지
        self.submit_btn.configure(state="disabled", text="수정 중...")
    
        # ✅ item_id 추출
        if isinstance(self.item_data, dict):
            item_id = self.item_data.get("item_id")
        else:
            item_id = getattr(self.item_data, "item_id", None)
    
        if not item_id:
            CTkMessagebox(title="오류", message="item_id가 없습니다.", icon="cancel")
            self.submit_btn.configure(state="normal", text="데이터 수정")
            return
    
        try:
            # ✅ question_id ↔ json_id 역매핑
            reverse_map = {v: k for k, v in MDS_QUESTION_MAPPING.items()}
    
            # ✅ JSON 내 min/max 범위 정보 로드
            range_map = {}
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    for section in json_data.get("운동성 검사", {}).get("sections", []):
                        for item in section.get("body", []):
                            item_key = item.get("id")
                            range_map[item_key] = {
                                "min": item.get("min"),
                                "max": item.get("max"),
                                "type": item.get("type")
                            }
            except Exception as e:
                print(f"[WARN] 범위 정보 로드 실패: {e}")
    
            # ✅ 실제 변경 비교 시작
            for ans in self.initial_data:
                aid = ans.get("answer_id")
                qid = ans.get("question_id")
                comp = ans.get("answer_component")
    
                json_id = reverse_map.get(qid)
                if not json_id:
                    print(f"[WARN] question_id {qid} not found in reverse map → skipped")
                    continue
                
                key = f"{json_id}_{comp}" if comp not in [None, "", "None"] else str(json_id)
                var = self.data_vars.get(key)
                if var is None:
                    print(f"[WARN] key {key} not found in self.data_vars → skipped")
                    continue
                
                new_val = str(var.get()).strip()
                old_val = str(ans.get("answer_value", "")).strip()
    
                # ✅ min/max 검사
                range_info = range_map.get(json_id)
                if range_info and range_info["type"] in ["input-number", "grouped-inputs"]:
                    min_v = range_info.get("min")
                    max_v = range_info.get("max")
    
                    try:
                        num_val = float(new_val)
                        if min_v is not None and num_val < min_v:
                            CTkMessagebox(
                                title="입력 오류",
                                message=f"문항 [{json_id}]의 값이 최소값({min_v})보다 작습니다.",
                                icon="warning"
                            )
                            return
                        if max_v is not None and num_val > max_v:
                            CTkMessagebox(
                                title="입력 오류",
                                message=f"문항 [{json_id}]의 값이 최대값({max_v})보다 큽니다.",
                                icon="warning"
                            )
                            return
                    except ValueError:
                        pass  # 숫자 아님 (radio 등)
                    
                # ✅ 변경된 응답만 수집
                if new_val != old_val:
                    updated.append({
                        "answer_id": aid,
                        "answer_value": new_val
                    })
    
            # ✅ 변경된 내용 없으면 중단
            if not updated:
                CTkMessagebox(title="알림", message="변경된 내용이 없습니다.", icon="info")
                return
    
            # ✅ 서버에 업데이트 요청
            success, err = call_api_to_update_data(updated)
            if success:
                # ✅ item의 update_ts, is_updated 갱신
                try:
                    from api.form_api import mark_item_updated
                    mark_item_updated(item_id)
                except Exception as e:
                    print(f"[WARN] item 업데이트 실패: {e}")
    
                # ✅ 성공 알림
                confirm_box = CTkMessagebox(
                    title="성공",
                    message=f"{len(updated)}개 응답이 수정되었습니다.",
                    icon="check",
                    option_1="확인"
                )
                confirm_box.get()
    
                # ✅ 창 닫기 여부 묻기
                close_box = CTkMessagebox(
                    title="창 닫기",
                    message="폼을 닫을까요?",
                    icon="question",
                    option_1="예",
                    option_2="아니오"
                )
                result = close_box.get()
                if result == "예":
                    if callable(self.on_close_callback):
                        self.on_close_callback()
                    if self.master and hasattr(self.master, "destroy"):
                        self.master.destroy()
            else:
                CTkMessagebox(title="오류", message=err or "수정 실패", icon="cancel")
    
        finally:
            # ✅ 버튼 복구
            self.submit_btn.configure(state="normal", text="데이터 수정")
