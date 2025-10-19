import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback


# ============================================================
# 🧩 환자 목록 불러오기--gui
# ============================================================
def load_patients(patient_listbox):
    patient_listbox.delete(0, tk.END)
    patient_listbox.patient_map = {}

    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT patient_id, patient_initials, birth_date, gender
                FROM dev_kkh.tb_patient_info
                WHERE is_deleted = FALSE
                ORDER BY created_ts DESC;
            """)
            data = cur.fetchall()

            for idx, patient in enumerate(data):
                initials = patient.get("patient_initials") or "이니셜 없음"
                birth = patient.get("birth_date")
                gender = patient.get("gender") or "?"

                birth = birth.strftime("%Y-%m-%d") if birth else "생년월일 없음"
                display_str = f"{initials} / {birth} / {gender}"

                patient_listbox.insert(tk.END, display_str)
                patient_listbox.patient_map[idx] = patient["patient_id"]

    except Exception as e:
        print(f"❌ 환자 목록 불러오기 실패: {e}")
        traceback.print_exc()

    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 🧩 병원별 환자 조회
# ============================================================
def fetch_patients(institution: str):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT *
                FROM dev_kkh.tb_patient_info
                WHERE institution = %s AND is_deleted = FALSE
                ORDER BY created_ts DESC;
            """, (institution,))
            return cur.fetchall()

    except Exception as e:
        print(f"❌ 병원 환자 목록 불러오기 실패: {e}")
        return []

    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 🧩 환자 등록
# ============================================================
def add_patient(patient_data: dict, institution: str):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO dev_kkh.tb_patient_info (
                    patient_initials, birth_date, institution,
                    gender, is_data_complete, completion_date
                )
                VALUES (%(patient_initials)s, %(birth_date)s, %(institution)s,
                        %(gender)s, %(is_data_complete)s, %(completion_date)s)
                RETURNING patient_id;
            """, patient_data)
            new_patient = cur.fetchone()
            conn.commit()

        messagebox.showinfo("성공", "환자 등록 완료!")
        return fetch_patients(institution)

    except Exception as e:
        messagebox.showerror("에러", f"등록 실패: {e}")
        traceback.print_exc()
        return None

    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 🧩 환자 삭제
# ============================================================
def delete_patient(patient_id: str, institution: str):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # 1️⃣ 환자 soft delete
            cur.execute("""
                UPDATE dev_kkh.tb_patient_info
                SET is_deleted = TRUE, deleted_ts = NOW()
                WHERE patient_id = %s;
            """, (patient_id,))

            # 2️⃣ 해당 환자의 items soft delete
            # cur.execute("""
            #     UPDATE dev_kkh.tb_items
            #     SET is_deleted = TRUE, deleted_at = NOW()
            #     WHERE patient_id = %s;
            # """, (patient_id,))

            conn.commit()

        messagebox.showinfo("성공", "환자 삭제 완료!")
        return fetch_patients(institution)

    except Exception as e:
        messagebox.showerror("에러", f"삭제 실패: {e}")
        traceback.print_exc()
        return None

    finally:
        if conn:
            release_connection(conn)

def fetch_all_patients():
    """
    dev_kkh.tb_patient_info 에서 삭제되지 않은 환자 목록 조회
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    patient_id, 
                    patient_initials, 
                    birth_date, 
                    gender
                FROM dev_kkh.tb_patient_info
                WHERE is_deleted = FALSE
                ORDER BY created_ts DESC;
            """)
            return cur.fetchall()
    except Exception as e:
        print("❌ fetch_all_patients 실패:", e)
        traceback.print_exc()
        return []
    finally:
        if conn:
            release_connection(conn)