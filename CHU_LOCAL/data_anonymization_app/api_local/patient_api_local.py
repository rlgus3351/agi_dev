# patient_api_local.py
from typing import Optional, Dict
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback
import uuid

def read_patient(patient_id: str) -> Optional[Dict]:
    """
    특정 환자 조회 (patient_id: UUID 문자열)
    """
    conn = None
    try:
        # UUID 형식 검증(옵션)
        _ = uuid.UUID(str(patient_id))

        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT patient_id, display_id, patient_initials, birth_date, gender
                FROM dev_kkh.tb_patient_info
                WHERE patient_id = %s
                LIMIT 1;
            """, (str(patient_id),))
            return cur.fetchone()
    except Exception as e:
        print(f"❌ 환자 조회 실패: {e}")
        traceback.print_exc()
        return None
    finally:
        if conn:
            release_connection(conn)
