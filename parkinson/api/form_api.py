# form_api.py (새 파일)
import requests
from typing import Union, Tuple
from config import FORM_BASE_URL,ITEMS_BASE_URL

# ⬅️ API 기본 URL 설정 (main.py에서 이동)
API_BASE_URL = "http://127.0.0.1:30000" 

# ⬅️ MDS 질문 매핑 (main.py에서 이동)
MDS_QUESTION_MAPPING = {
    # DB 삽입 순서: 1~8번 (기초 정보)
    "a": 1, "b": 2, "c": 3, "c1": 4, "d": 5, "d1": 6, "d2": 7, "e": 8,
    # DB 삽입 순서: 9~26번 (운동 항목별 평가)
    "1": 9, "2": 10, "3": 11, "4": 12, "5": 13, "6": 14, "7": 15, "8": 16,
    "9": 17, "10": 18, "11": 19, "12": 20, "13": 21, "14": 22, "15": 23, 
    "16": 24, "17": 25, "18": 26,
}

# -------------------------------------------------------------
# API 통신 함수: 1단계 - Item 등록 (create_new_item_and_get_id)
# -------------------------------------------------------------
def create_new_item_and_get_id(target_patient_id: str) -> Union[int, None]:
    """
    FastAPI의 /items/{patient_id}/item 엔드포인트를 호출하여 새 수집 항목을 등록하고 item_id를 반환합니다.
    (CTkMessagebox 호출은 GUI 파일에서 처리하기 위해 API 함수는 순수하게 데이터만 반환하도록 수정하는 것이 좋습니다.)
    """
    url = f"{ITEMS_BASE_URL}{target_patient_id}/item"
    
    item_payload = {
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
        # 오류 처리는 GUI (HealthSurveyForm)에서 담당하도록 예외를 다시 발생시키거나 None 반환
        error_msg = f"수집 항목(Item) 등록 실패: {e}"
        if hasattr(response, 'json'):
            error_msg += f" | 서버 상세: {response.json().get('detail', '알 수 없음')}"
        print(f"API Error (1): {error_msg}")
        # GUI에서 메시지 박스를 띄울 수 있도록 예외와 함께 에러 메시지를 반환합니다.
        return None, error_msg


# -------------------------------------------------------------
# API 통신 함수: 2단계 - 설문 응답 저장 (call_api_to_save_data)
# -------------------------------------------------------------
def call_api_to_save_data(item_id: int, answers_list: list) -> Tuple[bool, Union[str, None]]:
    """
    FastAPI의 /mds-form-answers/{item_id} 엔드포인트에 설문 응답을 전송합니다.
    (성공 여부, 에러 메시지) 튜플을 반환합니다.
    """
    url = f"{FORM_BASE_URL}{item_id}"
    api_payload = {"answers": answers_list}
    
    try:
        response = requests.post(url, json=api_payload, timeout=10) 
        response.raise_for_status() 

        return True, None # 성공

    except requests.exceptions.RequestException as e:
        error_msg = f"설문 응답 등록 실패: {e}"
        if hasattr(response, 'json'):
            error_msg += f" | 서버 상세: {response.json().get('detail', '알 수 없음')}"
        print(f"API Error (2): {error_msg}")
        return False, error_msg
        
        
# -------------------------------------------------------------
# MDS 설문 상세 내역 조회 함수 (fetch_mds_answers)
# -------------------------------------------------------------
def fetch_mds_answers(item_id: str) -> list:
    """
    특정 item_id에 해당하는 MDS 설문 답변 상세 내역을 서버에서 조회합니다.
    """
    # 💡 라우터 정보에 따라 /mds-form-answers/{item_id} 또는 /mds/{item_id} 경로를 사용합니다.
    # 이전 경로를 유지하겠습니다. (API 라우터 prefix: /mds-form-answers)
    url = f"{FORM_BASE_URL}{item_id}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # 2xx가 아니면 예외 발생
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # 404는 데이터가 없는 것으로 간주하고 빈 리스트 반환
            return [] 
        raise Exception(f"설문 응답 상세 조회 실패 (HTTP {e.response.status_code})")
    except Exception as e:
        raise Exception(f"설문 응답 상세 조회 중 오류 발생: {e}")

# -------------------------------------------------------------
# 데이터 변환 함수 (transform_to_api_format)
# -------------------------------------------------------------
def transform_to_api_format(raw_data: dict) -> list:

    """
    CTk StringVar에서 추출한 raw_data를 API 전송 형식으로 변환합니다.
    """
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


def call_api_to_update_data(answers_list: list) -> Tuple[bool, Union[str, None]]:
    """
    FastAPI의 /answers 엔드포인트를 호출하여 기존 설문 응답(answer_id, answer_value)을 수정합니다.
    """
    url = f"{FORM_BASE_URL.rstrip('/')}/answers"
    payload = {"answers": answers_list}

    try:
        response = requests.put(url, json=payload, timeout=10)
        response.raise_for_status()
        return True, None
    except requests.exceptions.RequestException as e:
        error_msg = f"설문 응답 수정 실패: {e}"
        if hasattr(response, "json"):
            error_msg += f" | 서버 상세: {response.json().get('detail', '알 수 없음')}"
        print(f"API Error (3): {error_msg}")
        return False, error_msg