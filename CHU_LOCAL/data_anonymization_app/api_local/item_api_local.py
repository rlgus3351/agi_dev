# item_api_local.py
from typing import Optional, Dict
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback

def get_item_by_id(item_id: int) -> Optional[Dict]:
    """
    특정 item_id로 단일 아이템 조회
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT item_id, patient_id, data_category, data_type, seq, description,
                       collected_at, updated_at
                FROM dev_kkh.tb_items
                WHERE item_id = %s
                LIMIT 1;
            """, (item_id,))
            return cur.fetchone()
    except Exception as e:
        print(f"❌ 아이템 조회 실패: {e}")
        traceback.print_exc()
        return None
    finally:
        if conn:
            release_connection(conn)
