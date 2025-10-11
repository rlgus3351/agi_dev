import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json
import os
import tkinter.filedialog as filedialog
from datetime import datetime
import requests # API 통신을 위해 추가
import uuid # item_id 생성을 위해 추가
from typing import Union # ⬅️ 추가

# JSON 파일 경로 설정 (절대경로 권장)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'form', 'mobility.json'))

# ⬅️ API 기본 URL 설정 (FastAPI 서버 주소로 변경 필요)
API_BASE_URL = "http://127.0.0.1:30000" 

MDS_QUESTION_MAPPING = {
    # DB 삽입 순서: 1~8번 (기초 정보)
    "a": 1, "b": 2, "c": 3, "c1": 4, "d": 5, "d1": 6, "d2": 7, "e": 8,
    # DB 삽입 순서: 9~26번 (운동 항목별 평가)
    "1": 9, "2": 10, "3": 11, "4": 12, "5": 13, "6": 14, "7": 15, "8": 16,
    "9": 17, "10": 18, "11": 19, "12": 20, "13": 21, "14": 22, "15": 23, 
    "16": 24, "17": 25, "18": 26,
}


class HealthSurveyForm(ctk.CTkFrame):
    def __init__(self, parent, json_file=JSON_FILE, patient_id=None):
        super().__init__(parent)
        self.json_file = json_file
        # patient_id가 None이면 테스트용 UUID로 초기화
        self.patient_id = patient_id if patient_id else str(uuid.uuid4())
        self.widgets = {}
        self.data_vars = {}
        self.scrollable_frame = None  

        # NOTE: self.vcmd는 generic한 숫자 검증에 사용되거나, 더 이상 사용되지 않습니다.
        # input-number에서는 validate_number_input_with_range를 동적으로 등록합니다.
        self.vcmd = (self.register(self.validate_number_input), '%P', '%S', '%V', '%W')
        self.load_data_and_create_widgets()

    # NOTE: 이 함수는 이제 단순 숫자 포맷 검사만 수행하며, 범위 검사는 validate_number_input_with_range가 담당합니다.
    def validate_number_input(self, new_value, changed_char, validation_type, widget_name):
        if new_value == "":
            return True
        if changed_char.isdigit():
            try:
                # NOTE: 입력 중인 전체 값이 정수인지 확인합니다.
                int(new_value)
                return True
            except ValueError:
                return False
        else:
            return False
            
    # ====================================================================
    # ⬅️ 범위 검증 로직 추가
    # ====================================================================
    def validate_number_input_with_range(self, new_value, min_str, max_str):
        """
        숫자 입력 및 지정된 범위(min, max)를 검사하는 함수.
        %P (new_value), min_val, max_val을 인수로 받습니다.
        """
        
        # 1. 빈 값은 항상 허용 (입력 지우기 허용)
        if new_value == "":
            return True
        
        try:
            # 2. 숫자 형식 검사
            num_value = int(new_value)
        except ValueError:
            # 숫자가 아니면 허용 안 함
            return False
            
        # 3. min/max 범위 검사
        
        # min 값 변환 (None 문자열 처리 포함)
        min_val = int(min_str) if min_str and min_str != 'None' else None
        # max 값 변환 (None 문자열 처리 포함)
        max_val = int(max_str) if max_str and max_str != 'None' else None
        
        # max 값 검사: 입력된 값이 max보다 크면 허용 안 함
        if max_val is not None and num_value > max_val:
            return False
            
        # min 값 검사: 입력된 값이 min보다 작으면 허용 안 함
        if min_val is not None and num_value < min_val:
             return False
             
        # 4. 모든 검사를 통과
        return True
    # ====================================================================
    
    def scroll_to_widget(self, event):
        """
        포커스를 받은 위젯이 뷰포트 영역을 벗어났을 때, 최소한의 스크롤로 보이게 조정합니다.
        """
        if not self.scrollable_frame or not self.scrollable_frame.winfo_exists():
            return
            
        widget = event.widget
        margin = 30 
        
        try:
            canvas = self.scrollable_frame._parent_canvas 
        except AttributeError:
            return
        
        y_pos = widget.winfo_y()
        widget_height = widget.winfo_height()
        frame_height = self.scrollable_frame.winfo_height() 

        scroll_region_str = canvas.cget("scrollregion")
        if not scroll_region_str:
            return
            
        _, y_min, _, y_max = map(int, scroll_region_str.split())
        total_canvas_height = y_max - y_min
        
        if total_canvas_height <= frame_height: 
            return
            
        y_scroll_ratio = canvas.yview()[0] 
        y_current_top_pixel = int(total_canvas_height * y_scroll_ratio)
        
        y_relative_top = y_pos - y_current_top_pixel 

        if y_relative_top < margin: 
            target_y_pixel = y_pos - margin
            new_y_ratio = target_y_pixel / total_canvas_height
            canvas.yview_moveto(max(0, new_y_ratio))

        elif y_relative_top + widget_height > frame_height - margin:
            target_y_pixel = y_pos + widget_height - frame_height + margin
            new_y_ratio = target_y_pixel / total_canvas_height
            canvas.yview_moveto(min(1.0, new_y_ratio))

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
        self.scrollable_frame = scrollable_frame 

        row = 0
        for section in self.survey_data.get("운동성 검사", {}).get("sections", []):
            header = section.get("header", {})
            body = section.get("body", [])

            # ⬅️ 헤더 중앙 정렬 로직
            header_label = ctk.CTkLabel(
                scrollable_frame,
                text=header.get("title", "") + f"\n{header.get('description', '')}",
                font=ctk.CTkFont(size=16, weight="bold"),
                anchor="center",     
                justify="center",    
                wraplength=700       
            )
            header_label.grid(row=row, column=0, columnspan=2, sticky="new", pady=(20, 15))
            row += 1

            for item in body:
                row = self._create_widget(scrollable_frame, item, row)

        # ⬅️ 데이터 저장 버튼
        submit_button = ctk.CTkButton(scrollable_frame, text="데이터 저장", command=self.get_entered_data)
        submit_button.grid(row=row, column=0, columnspan=2, pady=(20, 5), sticky="ew")
        row += 1 

        # ⬅️ 로컬 파일 재전송 버튼 추가
        resubmit_button = ctk.CTkButton(scrollable_frame, 
                                        text="로컬 파일 재전송 (API 복구)", 
                                        command=self.load_and_resubmit_data,
                                        fg_color="darkgreen", # 눈에 띄게 다른 색상
                                        hover_color="green")
        resubmit_button.grid(row=row, column=0, columnspan=2, pady=(5, 30), sticky="ew")


    def _create_widget(self, parent_frame, config, row):
        item_type = config.get('type')
        question = config.get('question')
        item_id = config.get('id')

        if item_type == "radio":
            var = ctk.StringVar()
            self.data_vars[item_id] = var

            ctk.CTkLabel(parent_frame, text=question, font=('', 14), justify="left", wraplength=450).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            radio_frame = ctk.CTkFrame(parent_frame)
            radio_frame.grid(row=row, column=1, sticky="w", padx=10, pady=10)

            for i, option in enumerate(config.get('options', [])):
                radio_btn = ctk.CTkRadioButton(
                    radio_frame,
                    text=option,
                    variable=var,
                    value=option
                )
                radio_btn.grid(row=0, column=i, padx=5)
                radio_btn.bind("<FocusIn>", self.scroll_to_widget) 
            row += 1

        elif item_type == "input-number":
            var = ctk.StringVar()
            self.data_vars[item_id] = var
            
            # ====================================================================
            # ⬅️ min/max 값 추출 및 동적 validatecommand 생성
            min_val = config.get('min', None) 
            max_val = config.get('max', None)
            
            # validatecommand에 %P (현재 입력 값), min_val, max_val 전달
            dynamic_vcmd = (self.register(self.validate_number_input_with_range), 
                            '%P', str(min_val), str(max_val))
            # ====================================================================

            ctk.CTkLabel(parent_frame, text=question, font=('', 14), justify="left", wraplength=450).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            entry = ctk.CTkEntry(
                parent_frame,
                textvariable=var,
                width=100,
                validate='key',
                # ⬅️ 동적으로 생성한 dynamic_vcmd 사용
                validatecommand=dynamic_vcmd 
            )
            entry.grid(row=row, column=1, sticky="w", padx=10, pady=10)
            entry.bind("<FocusIn>", self.scroll_to_widget)
            row += 1

        elif item_type == "grouped-inputs":
            sides = config.get("sides", [])

            ctk.CTkLabel(parent_frame, text=question, font=('', 14), justify="left", wraplength=450).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            side_frame = ctk.CTkFrame(parent_frame)
            side_frame.grid(row=row, column=1, sticky="w", padx=10, pady=10)
            row += 1

            for i, side in enumerate(sides):
                sid = f"{item_id}_{side}"
                var = ctk.StringVar()
                self.data_vars[sid] = var

                # grouped-inputs는 주로 0-4 또는 유사한 척도이며, 별도의 min/max가 JSON에 없으므로
                # generic vcmd 또는 필요시 별도의 dynamic vcmd를 생성해야 합니다.
                # 현재는 generic vcmd (숫자 형식만 검증)를 사용합니다.
                
                ctk.CTkLabel(side_frame, text=side).grid(row=i, column=0, sticky="w", padx=5, pady=2)
                entry = ctk.CTkEntry(
                    side_frame,
                    textvariable=var,
                    width=80,
                    validate='key',
                    validatecommand=self.vcmd # ⬅️ 현재는 generic 숫자 포맷 검증
                )
                entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
                entry.bind("<FocusIn>", self.scroll_to_widget)

        return row
    
    # ... (나머지 함수는 동일) ...
    
    def transform_to_api_format(self, raw_data: dict) -> list:
        answers = []
        for key, value in raw_data.items():
            value = value.strip()
            if not value: 
                continue

            if "_" in key:
                json_id, component = key.split("_", 1)
            else:
                json_id = key
                component = None
            
            question_db_id = MDS_QUESTION_MAPPING.get(json_id)
            
            if question_db_id is not None:
                answer = {
                    "question_id": question_db_id,
                    "answer_component": component if component else None,
                    # API가 숫자형 값을 기대할 수 있으므로, 가능한 경우 int로 변환 시도
                    "answer_value": int(value) if value.isdigit() else value 
                }
                answers.append(answer)
        return answers

# -------------------------------------------------------------
# API 통신 함수: 1단계 - Item 등록
# -------------------------------------------------------------
    # patient_id를 인수로 받도록 수정하여 재전송 시 유연하게 대처
    def create_new_item_and_get_id(self, target_patient_id: str) -> Union[int, None]:
        """
        FastAPI의 /items/{patient_id}/item 엔드포인트를 호출하여 새 수집 항목을 등록하고 item_id를 반환합니다.
        """
        url = f"{API_BASE_URL}/items/{target_patient_id}/item"
        
        # NOTE: patient_id는 URL에서 받으므로 스키마에서 제거했다면 payload에서도 제거해야 합니다.
        # 그러나 API 통신이 안정되지 않은 상태이므로, 서버 스키마가 patient_id를 필수적으로 요구한다고 가정하고 유지합니다.
        item_payload  = {
            "patient_id": target_patient_id,  
            "data_category": "PD", 
            "data_type": "MDS-UPDRS Part 3",
            "seq": 1,
            "description": "MDS-UPDRS Part 3 설문 응답",
        }
        try:
            response = requests.post(url, json=item_payload, timeout=5) 
            response.raise_for_status() 

            item_data = response.json()
            return item_data.get("item_id")

        except requests.exceptions.RequestException as e:
            error_msg = f"수집 항목(Item) 등록 실패: {e}"
            try:
                # 422 오류 등 상세 메시지 추가
                if hasattr(response, 'json'):
                    error_msg += f"\n서버 상세: {response.json().get('detail', '알 수 없음')}"
            except Exception:
                pass
            
            CTkMessagebox(
                title="API 오류 (1단계)", 
                message=error_msg, 
                icon="cancel"
            )
            return None


# -------------------------------------------------------------
# API 통신 함수: 2단계 - 설문 응답 저장
# -------------------------------------------------------------
    def call_api_to_save_data(self, item_id: int, answers_list: list) -> bool:
        """
        FastAPI의 /mds-form-answers/{item_id} 엔드포인트에 설문 응답을 전송합니다.
        """
        url = f"{API_BASE_URL}/mds-form-answers/{item_id}"
        api_payload = {"answers": answers_list}
        
        try:
            response = requests.post(url, json=api_payload, timeout=10) 
            response.raise_for_status() 

            CTkMessagebox(
                title="API 저장 성공", 
                message=f"설문 응답이 서버에 성공적으로 등록되었습니다. (Item ID: {item_id})", 
                icon="check"
            )
            # 부모 창 (모달) 닫기
            if self.master and hasattr(self.master, 'destroy') and not self.is_resubmitting:
                 self.master.destroy() 
            return True

        except requests.exceptions.RequestException as e:
            error_msg = f"설문 응답 등록 실패: {e}"
            try:
                if hasattr(response, 'json'):
                    error_msg += f"\n서버 상세: {response.json().get('detail', '알 수 없음')}"
            except Exception:
                pass
                
            CTkMessagebox(
                title="API 오류 (2단계)", 
                message=error_msg, 
                icon="cancel"
            )
            return False


# -------------------------------------------------------------
# 데이터 수집 및 저장 함수 (2단계 API 우선)
# -------------------------------------------------------------
    # HealthSurveyForm 클래스 내부 (get_entered_data 함수만 수정)

    def get_entered_data(self):
        """
        입력된 데이터를 수집하고, Item 등록 -> 설문 응답 저장 순으로 API 서버에 전송합니다.
        API 전송 실패 시 로컬 JSON 파일로 저장합니다.
        """
        self.is_resubmitting = False # 현재 폼 제출임을 표시
        if not self.patient_id or len(self.patient_id) < 16: 
            CTkMessagebox(title="오류", message="유효한 환자 정보(UUID)가 없습니다.", icon="cancel")
            return
            
        raw_data = {key: var.get() for key, var in self.data_vars.items()}
        
        # ====================================================================
        # 🚨 1. 필수 항목 누락 검증 로직 추가
        # ====================================================================
        missing_questions = []
        
        # self.survey_data에서 모든 질문 항목을 순회합니다.
        for section in self.survey_data.get("운동성 검사", {}).get("sections", []):
            for item in section.get("body", []):
                item_id = item.get('id')
                item_type = item.get('type')
                question = item.get('question')
                is_required = item.get('required', False) # required 속성이 없으면 False로 간주
                
                if is_required:
                    if item_type == "grouped-inputs":
                        # grouped-inputs (예: 좌/우)는 각 구성 요소별로 검사
                        sides = item.get("sides", [])
                        for side in sides:
                            # self.data_vars에 저장된 키 형태: "18_Right"
                            sid = f"{item_id}_{side}" 
                            if not raw_data.get(sid, "").strip():
                                 missing_questions.append(f"{question} ({side})")
                                 
                    elif not raw_data.get(item_id, "").strip():
                        # 일반 항목은 해당 item_id로 검사
                        missing_questions.append(question)
                        
        if missing_questions:
            missing_list = '\n- ' + '\n- '.join(missing_questions[:5])
            if len(missing_questions) > 5:
                 missing_list += f"\n... 외 {len(missing_questions) - 5}개 항목"
                 
            CTkMessagebox(
                title="필수 항목 누락", 
                message=f"다음 필수 설문 항목에 응답하지 않았습니다. 입력을 완료해주세요:{missing_list}", 
                icon="warning"
            )
            return # 저장 및 API 전송 중단
        # ====================================================================
    
        answers_list = self.transform_to_api_format(raw_data)
        
        # 이 부분은 빈 값이 없으므로 사실상 answers_list가 비어있을 일은 없지만, 안전을 위해 유지
        if not answers_list:
            CTkMessagebox(title="경고", message="입력된 응답 데이터가 없습니다.", icon="warning")
            return
    
        # 1. Item 등록 (item_id 발급). 현재 폼의 patient_id 사용
        item_id = self.create_new_item_and_get_id(self.patient_id)
        
        # 2. Item 등록 성공 시, 설문 응답 저장 시도
        if item_id is not None:
            api_success = self.call_api_to_save_data(item_id, answers_list)
        else:
            api_success = False
    
        # 3. API 전송에 실패한 경우 로컬 JSON 저장 실행 (폴백)
        if not api_success:
            submission_data = {
                "metadata": {
                    "patient_id": self.patient_id, 
                    "item_id": item_id, 
                    "survey_type": "운동성 검사 (MDS-UPDRS Part III)",
                    "created_at": datetime.now().isoformat()
                },
                "answers": answers_list
            }
            CTkMessagebox(
                title="API 실패, 로컬 저장", 
                message="서버 전송에 실패했습니다. 데이터를 로컬 JSON 파일로 저장합니다.", 
                icon="warning"
            )
            self.save_to_json_file(submission_data, prompt_save=True)


    def save_to_json_file(self, data, prompt_save=False):
        """
        설문 응답을 로컬 JSON 파일로 저장합니다.
        """
        try:
            pid_prefix = data['metadata']['patient_id'][:8] if data['metadata'].get('patient_id') else "NoPID"
            default_filename = f"MDS_UPDRS_Part3_{pid_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile=default_filename,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not file_path:
                return

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            if prompt_save:
                CTkMessagebox(title="로컬 저장 완료", message=f"설문 응답이 다음 파일에 저장되었습니다:\n{file_path}", icon="check")
            
        except Exception as e:
            CTkMessagebox(title="저장 오류", message=f"JSON 파일 저장 중 오류 발생: {e}", icon="cancel")

# -------------------------------------------------------------
# 로컬 파일 재전송 함수 (Recovery)
# -------------------------------------------------------------
    def load_and_resubmit_data(self):
        """
        로컬 JSON 파일을 선택하여 데이터를 로드하고 API 서버로 재전송을 시도합니다.
        """
        self.is_resubmitting = True # 재전송 중임을 표시
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                recovery_data = json.load(f)
                
            metadata = recovery_data.get('metadata', {})
            answers_list = recovery_data.get('answers', [])
            
            if not answers_list:
                CTkMessagebox(title="오류", message="선택한 JSON 파일에 유효한 응답 데이터가 없습니다.", icon="cancel")
                return

            patient_id_from_file = metadata.get('patient_id')
            item_id = metadata.get('item_id') 
            
            if not patient_id_from_file or len(patient_id_from_file) < 16:
                CTkMessagebox(title="오류", message="파일에 유효한 환자 ID(patient_id)가 없어 처리할 수 없습니다.", icon="cancel")
                return
            
            # 1. Item ID 확보
            if item_id is None:
                 # item_id가 없으면, 새 Item을 등록 시도
                 item_id = self.create_new_item_and_get_id(patient_id_from_file) 

            if item_id is None:
                # Item ID 확보 실패 (API 오류로 인한 item_id 발급 실패)
                CTkMessagebox(title="재전송 실패", message="새 Item ID 발급에 실패하여 재전송을 중단합니다. 네트워크 상태를 확인하세요.", icon="cancel")
                return 

            # 2. 설문 응답 저장 (발급받거나 확보한 item_id 사용)
            CTkMessagebox(
                title="재전송 시작", 
                message=f"Item ID {item_id} (환자: {patient_id_from_file[:8]}...)로 설문 데이터를 서버에 재전송합니다.", 
                icon="info"
            )
            api_success = self.call_api_to_save_data(item_id, answers_list)
            
            # 3. 재전송 성공 시 로컬 파일 삭제 안내
            if api_success:
                 # 성공적으로 전송했다는 메시지 후, 파일 삭제 여부를 묻는 메시지 박스
                 if CTkMessagebox(
                    title="재전송 성공",
                    message=f"성공적으로 서버에 데이터를 전송했습니다.\n로컬 백업 파일 ({os.path.basename(file_path)})을 삭제하시겠습니까?",
                    icon="question",
                    option_2="삭제",
                    option_1="유지"
                 ).get() == "삭제":
                     os.remove(file_path)
                     CTkMessagebox(title="파일 삭제 완료", message="로컬 파일이 삭제되었습니다.", icon="check")

        except FileNotFoundError:
            CTkMessagebox(title="오류", message="파일을 찾을 수 없습니다.", icon="cancel")
        except json.JSONDecodeError:
            CTkMessagebox(title="오류", message="선택한 파일의 JSON 형식이 올바르지 않습니다.", icon="cancel")
        except Exception as e:
            CTkMessagebox(title="처리 오류", message=f"데이터 재전송 중 예상치 못한 오류 발생: {e}", icon="cancel")
        
        self.is_resubmitting = False # 재전송 완료 표시