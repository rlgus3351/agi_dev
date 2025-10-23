from tkinter import messagebox
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback

# ============================================================
# 1️⃣ 특정 환자의 수집 항목 목록 조회
# ============================================================
def fetch_items(patient_id: str):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * 
                FROM dev_kkh.tb_items
                WHERE patient_id = %s
                  AND is_deleted = FALSE
                ORDER BY data_category, data_type, seq;
            """, (patient_id,))
            return cur.fetchall()

    except Exception as e:
        print(f"❌ 수집 항목 조회 실패: {e}")
        traceback.print_exc()
        return []

    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 2️⃣ 특정 환자의 수집 항목 파일 목록 조회 (동일 로직)
# ============================================================
def fetch_files(patient_id: str):
    return fetch_items(patient_id)


# ============================================================
# 3️⃣ 특정 환자의 수집 항목 단건 등록
# ============================================================
def add_item(patient_id: str, item_data: dict):
    """
    item_data = {
        "data_category": "카테고리",
        "data_type": "타입",
        "seq": 1,
        "description": "설명"
    }
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO dev_kkh.tb_items (
                    patient_id, data_category, data_type, seq, description, collected_at
                )
                VALUES (%s, %s, %s, %s, %s, NOW())
                RETURNING item_id, patient_id, data_category, data_type, seq, description, collected_at;
            """, (
                patient_id,
                item_data.get("data_category"),
                item_data.get("data_type"),
                item_data.get("seq"),
                item_data.get("description"),
            ))
            result = cur.fetchone()
            conn.commit()

        messagebox.showinfo("성공", "항목 등록 완료!")
        return result

    except Exception as e:
        messagebox.showerror("에러", f"항목 등록 실패: {e}")
        traceback.print_exc()
        return None

    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 4️⃣ 특정 환자의 수집 항목 다중 등록
# ============================================================
def add_items(patient_id: str, items: list):
    """
    items = [
        {"data_category": "카테고리1", "data_type": "타입1", "seq": 1, "description": "설명1"},
        {"data_category": "카테고리2", "data_type": "타입2", "seq": 2, "description": "설명2"},
    ]
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            for item in items:
                cur.execute("""
                    INSERT INTO dev_kkh.tb_items (
                        patient_id, data_category, data_type, seq, description, collected_at
                    )
                    VALUES (%s, %s, %s, %s, %s, NOW());
                """, (
                    patient_id,
                    item.get("data_category"),
                    item.get("data_type"),
                    item.get("seq"),
                    item.get("description"),
                ))
            conn.commit()

        messagebox.showinfo("성공", "항목(들) 등록 완료!")
        return True

    except Exception as e:
        messagebox.showerror("에러", f"항목(들) 등록 실패: {e}")
        traceback.print_exc()
        return False

    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 5️⃣ 수집 항목 삭제 (소프트 삭제)
# ============================================================
def delete_survey_item(item_data: dict):
    conn = None
    try:
        item_id = item_data.get("item_id")
        if not item_id:
            return False, "item_id가 없습니다."

        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE dev_kkh.tb_items
                SET is_deleted = TRUE, deleted_at = NOW()
                WHERE item_id = %s
                RETURNING item_id;
            """, (item_id,))
            result = cur.fetchone()
            conn.commit()

        if not result:
            return False, "해당 항목을 찾을 수 없습니다."

        return True, "삭제 완료"

    except Exception as e:
        return False, f"삭제 실패: {e}"

    finally:
        if conn:
            release_connection(conn)

def mark_item_updated_local(item_id: int) -> bool:
    """
    로컬 DB에서 해당 item_id의 updated_at과 is_updated를 갱신합니다.
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE dev_kkh.tb_items
                SET is_updated = TRUE,
                    updated_at = NOW()
                WHERE item_id = %s;
            """, (item_id,))
            conn.commit()
        print(f"✅ item_id={item_id} updated_at 갱신 완료")
        return True
    except Exception as e:
        print(f"❌ item_id={item_id} updated_at 갱신 실패: {e}")
        return False
    finally:
        if conn:
            release_connection(conn)
