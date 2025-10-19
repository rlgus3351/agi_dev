from typing import Union, Tuple, List
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback


# ============================================================
# 1️⃣ 수집 항목(Item) 등록 (기존: POST /items/{patient_id}/item)
# ============================================================
def create_new_item_and_get_id(target_patient_id: str, seq: int) -> Union[int, Tuple[None, str]]:
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
                "VIDEO",
                seq,
                f"운동성 검사 영상 (시퀀스: {seq})"
            ))
            row = cur.fetchone()
            conn.commit()

        return row["item_id"] if row else None

    except Exception as e:
        print(f"❌ 수집 항목(Item) 등록 실패: {e}")
        traceback.print_exc()
        return None, str(e)
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 2️⃣ 비디오 메타데이터 다중 등록 (기존: POST /video/{item_id})
# ============================================================
def save_video_metadata(item_id: int, video_meta_list: List[dict]) -> Tuple[bool, Union[str, None]]:
    """
    video_meta_list = [
        {
            "file_name": "video1.mp4",
            "file_path": "/path/video1.mp4",
            "file_size_mb": 12.3,
            "duration_seconds": 45,
            "resolution": "1920x1080",
            "frame_rate": 30,
            "file_ext": ".mp4",
            "needs_anonymization": True,
            "shooting_ts": "2025-10-19 14:00:00",
            "data_category": "PD"
        }
    ]
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            for video in video_meta_list:
                cur.execute("""
                    INSERT INTO dev_kkh.tb_video_metadata (
                        item_id, file_name, file_path, file_size_mb, file_ext,
                        duration_seconds, resolution, frame_rate,
                        needs_anonymization, shooting_ts, data_category
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s
                    );
                """, (
                    item_id,
                    video.get("file_name"),
                    video.get("file_path"),
                    video.get("file_size_mb"),
                    video.get("file_ext"),
                    video.get("duration_seconds"),
                    video.get("resolution"),
                    video.get("frame_rate"),
                    video.get("needs_anonymization"),
                    video.get("shooting_ts"),
                    video.get("data_category", "PD")
                ))
            conn.commit()

        print(f"✅ {len(video_meta_list)}개의 비디오 메타데이터 등록 완료")
        return True, None

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ 비디오 메타데이터 등록 실패: {e}")
        traceback.print_exc()
        return False, str(e)
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 3️⃣ 비디오 메타데이터 수정 (기존: PUT /video/update)
# ============================================================
def update_video_metadata(update_meta_list: List[dict]) -> Tuple[bool, Union[str, None]]:
    """
    update_meta_list = [
        {
            "video_metadata_id": 10,
            "file_name": "updated.mp4",
            "file_path": "/newpath/video1.mp4",
            "file_size_mb": 15.0
        }
    ]
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            for video in update_meta_list:
                cur.execute("""
                    UPDATE dev_kkh.tb_video_metadata
                    SET 
                        file_name = COALESCE(%s, file_name),
                        file_path = COALESCE(%s, file_path),
                        file_size_mb = COALESCE(%s, file_size_mb),
                        duration_seconds = COALESCE(%s, duration_seconds),
                        resolution = COALESCE(%s, resolution),
                        frame_rate = COALESCE(%s, frame_rate),
                        needs_anonymization = COALESCE(%s, needs_anonymization),
                        file_ext = COALESCE(%s, file_ext),
                        shooting_ts = COALESCE(%s, shooting_ts)
                    WHERE video_metadata_id = %s;
                """, (
                    video.get("file_name"),
                    video.get("file_path"),
                    video.get("file_size_mb"),
                    video.get("duration_seconds"),
                    video.get("resolution"),
                    video.get("frame_rate"),
                    video.get("needs_anonymization"),
                    video.get("file_ext"),
                    video.get("shooting_ts"),
                    video.get("video_metadata_id")
                ))
            conn.commit()

        print(f"✅ {len(update_meta_list)}개의 비디오 메타데이터 수정 완료")
        return True, None

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ 비디오 메타데이터 수정 실패: {e}")
        traceback.print_exc()
        return False, str(e)
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 4️⃣ 특정 item_id 비디오 메타데이터 조회 (기존: GET /video/{item_id})
# ============================================================
def fetch_video_metadata_by_item_id(item_id: int):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    video_metadata_id, item_id, file_name, file_path, file_ext,
                    file_size_mb, duration_seconds, resolution, frame_rate,
                    needs_anonymization, is_anonymized,
                    created_ts, shooting_ts, anonymized_ts, data_category
                FROM dev_kkh.tb_video_metadata
                WHERE item_id = %s
                ORDER BY created_ts DESC;
            """, (item_id,))
            return cur.fetchall()
    except Exception as e:
        print(f"❌ 비디오 메타데이터 조회 실패: {e}")
        traceback.print_exc()
        return []
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 5️⃣ 비식별화 상태 업데이트 (기존: PUT /video/anonymization/update)
# ============================================================
def update_anonymization_status(update_list: List[dict]) -> Tuple[bool, Union[str, None]]:
    """
    update_list = [
        {
            "video_metadata_id": 10,
            "needs_anonymization": False,
            "is_anonymized": True,
            "anonymized_ts": "2025-10-19 14:20:00"
        }
    ]
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            for video in update_list:
                cur.execute("""
                    UPDATE dev_kkh.tb_video_metadata
                    SET 
                        needs_anonymization = COALESCE(%s, needs_anonymization),
                        is_anonymized = COALESCE(%s, is_anonymized),
                        anonymized_ts = COALESCE(%s, anonymized_ts)
                    WHERE video_metadata_id = %s;
                """, (
                    video.get("needs_anonymization"),
                    video.get("is_anonymized"),
                    video.get("anonymized_ts"),
                    video.get("video_metadata_id")
                ))
            conn.commit()

        print(f"✅ {len(update_list)}개의 비식별화 상태 업데이트 완료")
        return True, None

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ 비식별화 상태 업데이트 실패: {e}")
        traceback.print_exc()
        return False, str(e)
    finally:
        if conn:
            release_connection(conn)
