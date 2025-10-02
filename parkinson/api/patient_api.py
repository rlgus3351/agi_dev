# api/patient_api.py
import requests
from tkinter import messagebox
import tkinter as tk
from datetime import datetime
from config import PATIENTS_BASE_URL



def load_patients(patient_listbox):
    

    patient_listbox.delete(0, tk.END)
    patient_listbox.patient_map = {}  # 👈 선택된 index → patient_id 매핑

    try:
        response = requests.get(PATIENTS_BASE_URL)
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

def fetch_patients(institution: str):
    """지정한 병원의 환자 목록 가져오기 (List[dict])"""
    try:
        params = {"institution": institution}
        response = requests.get(PATIENTS_BASE_URL+"hospital", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"❌ 환자 목록 불러오기 실패: {e}")
        return []

def add_patient(patient_data: dict, institution: str):
    """환자 등록 후 해당 병원 목록 다시 불러오기"""
    try:
        res = requests.post(PATIENTS_BASE_URL, json=patient_data)
        res.raise_for_status()
        messagebox.showinfo("성공", "환자 등록 완료!")
        return fetch_patients(institution)  # 병원 기준 목록 다시 조회
    except requests.RequestException as e:
        messagebox.showerror("에러", f"등록 실패: {e}")
        return None


def delete_patient(patient_id: str, institution: str):
    """환자 삭제 후 해당 병원 목록 다시 불러오기"""
    try:
        url = f"{PATIENTS_BASE_URL}{patient_id}"
        res = requests.delete(url)
        res.raise_for_status()
        messagebox.showinfo("성공", "환자 삭제 완료!")
        return fetch_patients(institution)  # 병원 기준 목록 다시 조회
    except requests.RequestException as e:
        messagebox.showerror("에러", f"삭제 실패: {e}")
        return None

