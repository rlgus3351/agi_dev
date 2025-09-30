# api/patient_api.py
import requests
from tkinter import messagebox

BASE_URL = "http://localhost:8000/patients/"

def load_patients(patient_listbox):
    import tkinter as tk
    from datetime import datetime

    patient_listbox.delete(0, tk.END)
    patient_listbox.patient_map = {}  # 👈 선택된 index → patient_id 매핑

    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()
        data = response.json()

        for idx, patient in enumerate(data):
            initials = patient.get("patient_initials") or "이니셜 없음"
            birth = patient.get("birth_date")
            gender = patient.get("gender") or "?"

            if birth:
                birth = datetime.strptime(birth, "%Y-%m-%d").strftime("%Y-%m-%d")
            else:
                birth = "생년월일 없음"

            display_str = f"{initials} / {birth} / {gender}"
            patient_listbox.insert(tk.END, display_str)

            # 👇 index → uuid 매핑 저장
            patient_listbox.patient_map[idx] = patient["patient_id"]

    except requests.RequestException as e:
        print(f"❌ 환자 목록 불러오기 실패: {e}")


def add_patient(patient_data: dict, patient_listbox=None):
    try:
        res = requests.post(BASE_URL, json=patient_data)
        res.raise_for_status()
        messagebox.showinfo("성공", "환자 등록 완료!")

        if patient_listbox is not None:
            load_patients(patient_listbox)

        return res.json()
    except requests.RequestException as e:
        messagebox.showerror("에러", f"등록 실패: {e}")
        return None


def delete_patient(patient_id: str, patient_listbox=None):
    """
    환자 삭제 API 호출
    :param patient_id: 삭제할 환자 UUID
    """
    try:
        url = f"{BASE_URL}{patient_id}"
        res = requests.delete(url)
        res.raise_for_status()
        messagebox.showinfo("성공", "환자 삭제 완료!")

        if patient_listbox is not None:
            load_patients(patient_listbox)

        return True
    except requests.RequestException as e:
        messagebox.showerror("에러", f"삭제 실패: {e}")
        return False