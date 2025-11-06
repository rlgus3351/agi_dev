# -*- coding: utf-8 -*-
from typing import Union, Tuple, List
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback

# ============================================================
# 1) 수집 항목(Item) 등록 (VOICE 전용)
#    - 필요 시 create_new_item_and_get_id_generic을 쓰고 싶다면
#      동일 시그니처로 래핑해둔 이 함수를 사용해도 됨.
# ============================================================
def create_new_item_and_get_id_voice(target_patient_id: str, seq: int) -> Union[int, Tuple[None, str]]:
    """
    VOICE용 tb_items 행을 생성하고 item_id 반환
    - data_category: PD (필요 시 호출부에서 바꿔도 무방)
    - data_type    : VOICE
    - description  : '음성 수집 (시퀀스: {seq})'
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
                "VOICE",
                seq,
                f"음성 수집 (시퀀스: {seq})"
            ))
            row = cur.fetchone()
            conn.commit()

        return row["item_id"] if row else None

    except Exception as e:
        print(f"❌ VOICE 항목(Item) 등록 실패: {e}")
        traceback.print_exc()
        return None, str(e)
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 2) 음성 메타데이터 다중 등록
# ============================================================
def save_audio_metadata(item_id: int, audio_meta_list: List[dict]) -> Tuple[bool, Union[str, None]]:
    """
    audio_meta_list 예시:
    [
        {
            "file_name": "speech.wav",
            "file_path": "/path/speech.wav",
            "file_size_mb": 3.21,
            "duration_seconds": 42,
            "sample_rate_hz": 16000,
            "channels": 1,
            "bit_rate_kbps": 128,
            "codec": "pcm_s16le",
            "file_ext": "wav",
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
            for a in audio_meta_list:
                cur.execute("""
                    INSERT INTO dev_kkh.tb_audio_metadata (
                        item_id, file_name, file_path, file_size_mb, file_ext,
                        duration_seconds, sample_rate_hz, channels, bit_rate_kbps, codec,
                        needs_anonymization,  data_category
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s
                    );
                """, (
                    item_id,
                    a.get("file_name"),
                    a.get("file_path"),
                    a.get("file_size_mb"),
                    a.get("file_ext"),
                    a.get("duration_seconds"),
                    a.get("sample_rate_hz"),
                    a.get("channels"),
                    a.get("bit_rate_kbps"),
                    a.get("codec"),
                    a.get("needs_anonymization","True"),
                    a.get("data_category", "MDD"),
                ))
            conn.commit()

        print(f"✅ {len(audio_meta_list)}개의 음성 메타데이터 등록 완료")
        return True, None

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ 음성 메타데이터 등록 실패: {e}")
        traceback.print_exc()
        return False, str(e)
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 3) 음성 메타데이터 수정 (부분 업데이트)
# ============================================================
def update_audio_metadata(update_meta_list: List[dict]) -> Tuple[bool, Union[str, None]]:
    """
    update_meta_list 예시:
    [
        {
            "audio_metadata_id": 10,
            "file_name": "updated.wav",
            "file_path": "/newpath/updated.wav",
            "file_size_mb": 3.99,
            "duration_seconds": 45,
            "sample_rate_hz": 44100,
            "channels": 2,
            "bit_rate_kbps": 192,
            "codec": "aac",
            "file_ext": "wav",
            "needs_anonymization": False
        }
    ]
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            for a in update_meta_list:
                cur.execute("""
                    UPDATE dev_kkh.tb_audio_metadata
                    SET
                        file_name           = COALESCE(%s, file_name),
                        file_path           = COALESCE(%s, file_path),
                        file_size_mb        = COALESCE(%s, file_size_mb),
                        duration_seconds    = COALESCE(%s, duration_seconds),
                        sample_rate_hz      = COALESCE(%s, sample_rate_hz),
                        channels            = COALESCE(%s, channels),
                        bit_rate_kbps       = COALESCE(%s, bit_rate_kbps),
                        codec               = COALESCE(%s, codec),
                        file_ext            = COALESCE(%s, file_ext),
                        needs_anonymization = COALESCE(%s, needs_anonymization),
                    WHERE audio_metadata_id = %s;
                """, (
                    a.get("file_name"),
                    a.get("file_path"),
                    a.get("file_size_mb"),
                    a.get("duration_seconds"),
                    a.get("sample_rate_hz"),
                    a.get("channels"),
                    a.get("bit_rate_kbps"),
                    a.get("codec"),
                    a.get("file_ext"),
                    a.get("needs_anonymization"),
                    a.get("audio_metadata_id"),
                ))
            conn.commit()

        print(f"✅ {len(update_meta_list)}개의 음성 메타데이터 수정 완료")
        return True, None

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ 음성 메타데이터 수정 실패: {e}")
        traceback.print_exc()
        return False, str(e)
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 4) 특정 item_id의 음성 메타데이터 조회
# ============================================================
def fetch_audio_metadata_by_item_id(item_id: int) -> Tuple[List[dict], Union[str, None]]:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    audio_metadata_id, item_id, file_name, file_path, file_ext,
                    file_size_mb, duration_seconds, sample_rate_hz, channels,
                    bit_rate_kbps, codec,
                    needs_anonymization, is_anonymized,
                    created_ts, anonymized_ts, data_category
                FROM dev_kkh.tb_audio_metadata
                WHERE item_id = %s
                ORDER BY created_ts DESC;
            """, (item_id,))
            data = cur.fetchall()

        return data, None

    except Exception as e:
        print(f"❌ 음성 메타데이터 조회 실패: {e}")
        traceback.print_exc()
        return [], str(e)
    finally:
        if conn:
            release_connection(conn)


# ============================================================
# 5) 비식별화 상태 업데이트 (음성)
# ============================================================
def update_audio_anonymization_status(update_list: List[dict]) -> Tuple[bool, Union[str, None]]:
    """
    update_list 예시:
    [
        {
            "audio_metadata_id": 10,
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
            for a in update_list:
                cur.execute("""
                    UPDATE dev_kkh.tb_audio_metadata
                    SET
                        needs_anonymization = COALESCE(%s, needs_anonymization),
                        is_anonymized       = COALESCE(%s, is_anonymized),
                        anonymized_ts       = COALESCE(%s, anonymized_ts)
                    WHERE audio_metadata_id = %s;
                """, (
                    a.get("needs_anonymization"),
                    a.get("is_anonymized"),
                    a.get("anonymized_ts"),
                    a.get("audio_metadata_id"),
                ))
            conn.commit()

        print(f"✅ {len(update_list)}개의 음성 비식별화 상태 업데이트 완료")
        return True, None

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ 음성 비식별화 상태 업데이트 실패: {e}")
        traceback.print_exc()
        return False, str(e)
    finally:
        if conn:
            release_connection(conn)
