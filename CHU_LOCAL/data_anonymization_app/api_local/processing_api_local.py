#processing_api_local.py
from typing import Union, Tuple, List
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback

def get_next_video_to_process():
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT v.*
                FROM dev_kkh.tb_video_metadata v
                JOIN dev_kkh.tb_data_validation d
                  ON v.item_id = d.item_id
                WHERE v.needs_anonymization = TRUE
                  AND (v.is_anonymized IS NULL OR v.is_anonymized = FALSE)
                  AND v.data_category = 'PD'
                  AND d.validation_description LIKE '%PASS%'
                ORDER BY v.created_ts ASC
                LIMIT 1;
            """)
            return cur.fetchone()
    except Exception as e:
        print(f"❌ 처리 대기 영상 조회 실패: {e}")
        return None
    finally:
        if conn:
            release_connection(conn)


def update_processed_video(video_metadata_id: int, anonymized_ts: str, is_anonymized: bool = True):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE dev_kkh.tb_video_metadata
                SET is_anonymized = %s,
                    anonymized_ts = %s
                WHERE video_metadata_id = %s
            """, (is_anonymized, anonymized_ts, video_metadata_id))
        conn.commit()
        print(f"✅ 업데이트 완료: video_metadata_id={video_metadata_id}")
    except Exception as e:
        print(f"❌ 영상 업데이트 실패: {e}")
    finally:
        if conn:
            release_connection(conn)


def insert_preprocessing_record(payload: dict) -> bool:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO dev_kkh.tb_data_preprocessing (
                    item_id, data_category, original_file_path,
                    json_file_path, encrypted_file_path,
                    processing_started_at, processing_ended_at,
                    processing_duration_sec, total_frames,
                    encrypted_frames, detected_face_frames, success_rate,
                    preprocessing_type, description
                ) VALUES (
                    %(item_id)s, %(data_category)s, %(original_file_path)s,
                    %(json_file_path)s, %(encrypted_file_path)s,
                    %(processing_started_at)s, %(processing_ended_at)s,
                    %(processing_duration_sec)s, %(total_frames)s,
                    %(encrypted_frames)s, %(detected_face_frames)s, %(success_rate)s,
                    %(preprocessing_type)s, %(description)s
                )
                RETURNING preprocessing_id
            """, payload)
            new_id = cur.fetchone()[0]
        conn.commit()
        print(f"✅ 전처리 결과 저장 완료 (preprocessing_id={new_id})")
        return True
    except Exception as e:
        # 원인 파악을 위해 예외 메시지까지 남겨두기
        print(f"❌ 전처리 결과 저장 실패: {e}")
        # 필요하면 로그에 stack trace도 남기기
        # import traceback; traceback.print_exc()
        return False
    finally:
        if conn:
            release_connection(conn)
