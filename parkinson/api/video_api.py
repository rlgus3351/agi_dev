# form_api.py (새 파일)
import requests
from typing import Union, Tuple, List
from config import ITEMS_BASE_URL, VIDEO_BASE_URL

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
# API 통신 함수: 2단계 - 비디오 메타데이터 다중 등록 (call_api_to_save_video_metadata)
# -------------------------------------------------------------
def call_api_to_save_video_metadata(item_id: int, video_meta_list: List[dict]) -> Tuple[bool, Union[str, None]]:
    """
    FastAPI의 /video/{item_id} 엔드포인트에 비디오 메타데이터 목록을 전송하여 등록합니다.
    (성공 여부, 에러 메시지) 튜플을 반환합니다.

    Args:
        item_id: 연결할 수집 항목 ID.
        video_meta_list: 등록할 비디오 메타데이터 딕셔너리 리스트.

    Returns:
        Tuple[bool, Union[str, None]]: 성공 여부 및 에러 메시지 (성공 시 None).
    """
    # 라우터: POST /video/{item_id}
    url = f"{VIDEO_BASE_URL}{item_id}" 
    api_payload = {"videos": video_meta_list} # video_api.py의 schemas.VideoMetasCreate 형식에 맞춤
    
    try:
        response = requests.post(url, json=api_payload, timeout=30) # 파일 업로드 대신 메타데이터만 등록하므로 10초도 충분
        response.raise_for_status() 

        # 응답 메시지 추출 (등록된 개수 등)
        success_message = response.json().get("message", "비디오 메타데이터 등록 완료.")
        return True, success_message # 성공 시 메시지 반환

    except requests.exceptions.RequestException as e:
        error_msg = f"비디오 메타데이터 등록 실패: {e}"
        # 서버 응답에 상세 내용이 있을 경우 추가
        if hasattr(response, 'json'):
            try:
                server_detail = response.json().get('detail', '알 수 없음')
                error_msg += f" | 서버 상세: {server_detail}"
            except requests.exceptions.JSONDecodeError:
                # JSON 디코딩 실패 시 (예: 서버에서 HTML 오류를 반환한 경우)
                error_msg += f" | 서버 응답: {response.text[:100]}..."
                
        print(f"API Error (Video Meta): {error_msg}")
        return False, error_msg


# -------------------------------------------------------------
# API 통신 함수: 3단계 - 비디오 메타데이터 개별 수정 (call_api_to_update_video_metadata)
# -------------------------------------------------------------
def call_api_to_update_video_metadata(update_meta_list: List[dict]) -> Tuple[bool, Union[str, None]]:
    """
    FastAPI의 PUT /video/update 엔드포인트를 호출하여 기존 비디오 메타데이터를 수정합니다.
    (성공 여부, 에러 메시지) 튜플을 반환합니다.

    Args:
        update_meta_list: 수정할 비디오 메타데이터 딕셔너리 리스트 (반드시 video_metadata_id 포함).

    Returns:
        Tuple[bool, Union[str, None]]: 성공 여부 및 에러 메시지 (성공 시 None).
    """
    # 라우터: PUT /video/update
    url = f"{VIDEO_BASE_URL.rstrip('/')}/update" 
    
    try:
        response = requests.put(url, json=update_meta_list, timeout=10) # List[schemas.VideoMetaUpdate] 형식에 맞춤
        response.raise_for_status() 

        # 응답 메시지 추출
        success_message = response.json().get("message", "비디오 메타데이터 수정 완료.")
        return True, success_message # 성공 시 메시지 반환

    except requests.exceptions.RequestException as e:
        error_msg = f"비디오 메타데이터 수정 실패: {e}"
        if hasattr(response, 'json'):
            try:
                server_detail = response.json().get('detail', '알 수 없음')
                error_msg += f" | 서버 상세: {server_detail}"
            except requests.exceptions.JSONDecodeError:
                error_msg += f" | 서버 응답: {response.text[:100]}..."
                
        print(f"API Error (Video Meta Update): {error_msg}")
        return False, error_msg

# -------------------------------------------------------------
# API 통신 함수: 4단계 - 특정 item_id 비디오 메타데이터 조회 (fetch_video_metadata_by_item_id)
# -------------------------------------------------------------
def fetch_video_metadata_by_item_id(item_id: int) -> Tuple[List[dict], Union[str, None]]:
    """
    특정 item_id에 연결된 비디오 메타데이터 목록을 조회합니다.
    (메타데이터 리스트, 에러 메시지) 튜플을 반환합니다.
    """
    # 라우터: GET /video/{item_id}
    url = f"{VIDEO_BASE_URL}{item_id}" 
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        return response.json(), None # 성공 시 데이터 리스트 반환

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # 404는 데이터가 없는 것으로 간주하고 빈 리스트와 None 반환
            return [], None 
        # 그 외 HTTP 에러는 예외 처리
        error_msg = f"비디오 메타데이터 조회 실패 (HTTP {e.response.status_code}): {e.response.json().get('detail', '알 수 없음')}"
        print(f"API Error (Video Meta Fetch): {error_msg}")
        return [], error_msg

    except Exception as e:
        error_msg = f"비디오 메타데이터 조회 중 오류 발생: {e}"
        print(f"API Error (Video Meta Fetch): {error_msg}")
        return [], error_msg