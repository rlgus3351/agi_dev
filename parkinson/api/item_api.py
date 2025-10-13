import requests
from tkinter import messagebox
from config import ITEMS_BASE_URL
# 아이템 API URL

# ---------------------------------------------------
# 1. 특정 환자의 수집 항목 목록 조회
# ---------------------------------------------------
def fetch_items(patient_id: str):
    """
    특정 환자의 수집 항목 목록 가져오기 (List[dict])
    """
    try:
        url = f"{ITEMS_BASE_URL}{patient_id}"  # GET /items/{patient_id}
        res = requests.get(url)
        res.raise_for_status()
        return res.json()  # List[dict]
    except requests.HTTPError as e:
        if res.status_code == 404:
            # messagebox.showwarning("알림", "해당 환자의 수집 항목이 없습니다.")
            return []
        else:
            messagebox.showerror("에러", f"수집 항목 조회 실패: {e}")
            return []
    except requests.RequestException as e:
        messagebox.showerror("에러", f"네트워크 오류: {e}")
        return []
    
def fetch_files(patient_id: str):
    """
    특정 환자의 수집 항목 파일 목록 가져오기 (List[dict])
    """
    try:
        url = f"{ITEMS_BASE_URL}{patient_id}"  # GET /items/{patient_id}
        res = requests.get(url)
        res.raise_for_status()
        return res.json()  # List[dict]
    except requests.HTTPError as e:
        if res.status_code == 404:
            # messagebox.showwarning("알림", "해당 환자의 수집 항목이 없습니다.")
            return []
        else:
            messagebox.showerror("에러", f"수집 항목 조회 실패: {e}")
            return []
    except requests.RequestException as e:
        messagebox.showerror("에러", f"네트워크 오류: {e}")
        return []

# ---------------------------------------------------
# 2. 특정 환자의 수집 항목 단건 등록
# ---------------------------------------------------
def add_item(patient_id: str, item_data: dict):
    """
    특정 환자의 수집 항목 단건 등록
    item_data 예시:
    {
        "data_category": "카테고리",
        "data_type": "타입",
        "seq": 1,
        "description": "설명"
    }
    """
    try:
        url = f"{ITEMS_BASE_URL}{patient_id}/item"  # POST /items/{patient_id}/item
        res = requests.post(url, json=item_data)
        res.raise_for_status()
        messagebox.showinfo("성공", "항목 등록 완료!")
        return res.json()  # dict(row)
    except requests.RequestException as e:
        messagebox.showerror("에러", f"항목 등록 실패: {e}")
        return None

# ---------------------------------------------------
# 3. 특정 환자의 수집 항목 다중 등록
# ---------------------------------------------------
def add_items(patient_id: str, items: list):
    """
    특정 환자의 수집 항목 다중 등록
    items 예시:
    [
        {"data_category": "카테고리1", "data_type": "타입1", "seq": 1, "description": "설명1"},
        {"data_category": "카테고리2", "data_type": "타입2", "seq": 2, "description": "설명2"},
    ]
    """
    try:
        url = f"{ITEMS_BASE_URL}{patient_id}/items"  # POST /items/{patient_id}/items
        res = requests.post(url, json={"items": items})
        res.raise_for_status()
        messagebox.showinfo("성공", "항목(들) 등록 완료!")
        return True
    except requests.RequestException as e:
        messagebox.showerror("에러", f"항목(들) 등록 실패: {e}")
        return False

# ---------------------------------------------------
# 4. 특정 환자의 수집 항목 삭제 (소프트 삭제)
# ---------------------------------------------------
def delete_survey_item(item_data: dict):
    """
    설문 항목 삭제 API 호출 (FastAPI /items/{item_id} DELETE)
    Soft delete 방식
    """
    try:
        item_id = item_data.get("item_id")
        if not item_id:
            return False, "item_id가 없습니다."

        url = f"{ITEMS_BASE_URL}{item_id}"
        res = requests.delete(url)
        if res.status_code == 404:
            return False, "해당 항목을 찾을 수 없습니다."
        res.raise_for_status()
        return True, "삭제 완료"

    except requests.RequestException as e:
        return False, f"요청 실패: {e}"
    except Exception as e:
        return False, f"에러: {e}"