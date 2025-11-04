import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import json
from typing import Dict, Any, List, Tuple, Optional, Callable
import tkinter as tk  # ✅ multiline 텍스트용

# -------------------------------------------------
# 텍스트 정규화: 줄바꿈/여러 공백 제거 + trim
# -------------------------------------------------
def _norm(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    return " ".join(text.replace("\n", " ").replace("\r", " ").split()).strip()


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
        self.vars: Dict[Any, ctk.StringVar] = {}
        # ✅ 점수 계산 대상(qid 키)만 따로 추적: radio, input-number
        self._score_keys: set = set()

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

                    # 텍스트 라벨은 문항 칸에만 표시 (번호/문항은 첫 줄에 이미 있음)
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

            # 다음 항목을 위한 row 포인터 이동
            row_idx += rows_used
            sep_row = row_idx + rows_used  # 이 항목 아래 줄
            separator = ctk.CTkFrame(table, height=4, fg_color="#CCCCCC")
            separator.grid(row=sep_row, column=0,
                           columnspan=len(columns),  # 전체 컬럼 가로지름
                           sticky="ew", pady=(0, 5))
            row_idx = sep_row + 1   

    # ---------------- 제출/점수 ----------------
    def _calc_total_and_check(self) -> Optional[int]:
        """
        총점 계산:
          - self._score_keys 에 포함된 키(qid)만 점수로 합산
          - 라디오/숫자 입력은 값 미입력 또는 숫자 아님 → None(검증 실패)
          - 텍스트 입력은 점수 제외이므로 미입력이어도 통과
        """
        total = 0

        # 1) 점수 대상(qid)만 체크
        for qid in self._score_keys:
            var = self.vars.get(qid)
            if var is None:
                return None
            v = var.get()
            if v.strip() == "":
                return None
            try:
                total += int(v)
            except ValueError:
                return None

        # 2) 기타 텍스트들은 검증에서 제외
        return total

    def _on_submit(self):
        total = self._calc_total_and_check()
        if total is None:
            CTkMessagebox(
                title="입력 누락",
                message="점수 항목(라디오/숫자)을 올바르게 입력해주세요.",
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
        ※ input-text는 answer_component=None 으로 저장 (점수 제외지만 값은 저장)
        """
        inserts: List[Dict[str, Any]] = []
        updates: List[Dict[str, Any]] = []

        def add(qid: int, comp: Optional[str], value: str):
            if not qid:
                # 매핑 실패한 행은 저장 스킵
                return
            key = (qid, comp)
            existed = self._existing_index.get(key)
            if existed and existed.get("answer_id") is not None:
                updates.append({"answer_id": existed["answer_id"], "answer_value": value})
            else:
                inserts.append({"question_id": qid, "answer_component": comp, "answer_value": value})

        for key, var in list(self.vars.items()):
            # ✅ multiline Text 위젯 처리
            if isinstance(key, str) and key.endswith("__textwidget"):
                try:
                    qid_str = key.split("__textwidget")[0]
                    qid = int(qid_str)
                except Exception:
                    continue
                widget: tk.Text = var  # type: ignore
                val = widget.get("1.0", "end-1c").strip()
                # placeholder인지 체크는 생략(placeholder 넣었다면 포커스아웃 이벤트가 지웠을 것)
                if val != "":
                    add(qid, None, val)
                continue

            # 기타 follow-up 텍스트
            if isinstance(key, str) and key.endswith("_etc"):
                try:
                    qid = int(key.split("_")[0])
                    val = str(var.get()).strip()
                    if val != "":
                        add(qid, "기타", val)   # followup free-text
                except Exception:
                    continue
            else:
                # 일반 qid (radio / input-number / input-text 단일라인)
                try:
                    qid = int(key)
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
            ok, err = update_item_description(item_id, desc_with_score)  # type: ignore
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
