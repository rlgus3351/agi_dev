# -*- coding: utf-8 -*-
from typing import Union, Tuple, List
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback

# ============================================================
# 1) 수집 항목(Item) 등록 (TXT 전용)
# ============================================================
def create_new_item_and_get_id_txt(target_patient_id: str, seq: int) -> Union[int, Tuple[None, str]]:
    """
    TXT용 tb_items 행을 생성하고 item_id 반환
    - data_category: PD (필요 시 호출부에서 바꿔도 무방)
    - data_type    : TXT
    - description  : '텍스트 파일 (시퀀스: {seq})'
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
                RETURNING item_id;
            """, (
                target_patient_id,
                "PD",
                "TXT",
                seq,
                f"텍스트 파일 (시퀀스: {seq})"
            ))
            row = cur.fetchone()
            conn.commit()

        return row["item_id"] if row else None

    except Exception as e:
        print(f"❌ TXT 항목(Item) 등록 실패: {e}")
        traceback.print_exc()
        return None, str(e)
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 2) 텍스트 파일 메타데이터 다중 등록
#    (요구사항: 용량, 확장자, 파일명, 등록일시)
#    - 등록일시는 DB DEFAULT now() 사용
# ============================================================
def save_file_metadata(item_id: int, file_meta_list: List[dict]) -> Tuple[bool, Union[str, None]]:
    """
    file_meta_list 예시:
    [
        {
            "file_name": "memo.txt",
            "file_path": "/path/memo.txt",
            "file_size_mb": 0.12,
            "file_ext": "txt",
            "data_category": "PD"
        }
    ]
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            for f in file_meta_list:
                cur.execute("""
                    INSERT INTO dev_kkh.tb_file_metadata (
                        item_id, file_name, file_path, file_size_mb, file_ext, data_category
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s
                    );
                """, (
                    item_id,
                    f.get("file_name"),
                    f.get("file_path"),
                    f.get("file_size_mb"),
                    f.get("file_ext"),
                    f.get("data_category", "PD"),
                ))
            conn.commit()

        print(f"✅ {len(file_meta_list)}개의 파일 메타데이터 등록 완료")
        return True, None

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ 파일 메타데이터 등록 실패: {e}")
        traceback.print_exc()
        return False, str(e)
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 3) 텍스트 파일 메타데이터 수정 (부분 업데이트)
# ============================================================
def update_file_metadata(update_meta_list: List[dict]) -> Tuple[bool, Union[str, None]]:
    """
    update_meta_list 예시:
    [
        {
            "file_metadata_id": 7,
            "file_name": "memo_v2.txt",
            "file_path": "/newpath/memo_v2.txt",
            "file_size_mb": 0.15,
            "file_ext": "txt"
        }
    ]
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            for f in update_meta_list:
                cur.execute("""
                    UPDATE dev_kkh.tb_file_metadata
                    SET
                        file_name   = COALESCE(%s, file_name),
                        file_path   = COALESCE(%s, file_path),
                        file_size_mb= COALESCE(%s, file_size_mb),
                        file_ext    = COALESCE(%s, file_ext)
                    WHERE file_metadata_id = %s;
                """, (
                    f.get("file_name"),
                    f.get("file_path"),
                    f.get("file_size_mb"),
                    f.get("file_ext"),
                    f.get("file_metadata_id"),
                ))
            conn.commit()

        print(f"✅ {len(update_meta_list)}개의 파일 메타데이터 수정 완료")
        return True, None

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ 파일 메타데이터 수정 실패: {e}")
        traceback.print_exc()
        return False, str(e)
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 4) 특정 item_id의 텍스트 파일 메타데이터 조회
# ============================================================
def fetch_file_metadata_by_item_id(item_id: int) -> Tuple[List[dict], Union[str, None]]:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    file_metadata_id, item_id, file_name, file_path, file_ext,
                    file_size_mb, created_ts, data_category
                FROM dev_kkh.tb_file_metadata
                WHERE item_id = %s
                ORDER BY created_ts DESC;
            """, (item_id,))
            data = cur.fetchall()

        return data, None

    except Exception as e:
        print(f"❌ 파일 메타데이터 조회 실패: {e}")
        traceback.print_exc()
        return [], str(e)
    finally:
        if conn:
            release_connection(conn)
