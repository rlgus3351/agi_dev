# processing_api_local.py
from typing import Union, Tuple, List, Optional, Dict, Any
from psycopg2.extras import RealDictCursor
from utils.db_utils import get_connection, release_connection
import traceback

# =========================
# 기존: 영상 1건 조회 (호환용)
# =========================
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
                  AND v.data_category = 'MDD'
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

# =========================
# 기존: 영상 업데이트
# =========================
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

# =========================
# 기존: 전처리 결과 기록 (VIDEO/AUDIO 공용)
# =========================
def insert_preprocessing_record(payload: dict):
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
        return True, new_id
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"❌ 전처리 결과 저장 실패: {e}")
        return False, None
    finally:
        if conn:
            release_connection(conn)

# ============================================================
# ⬇️⬇️ 여기서부터 추가된 기능들 (VIDEO/AUDIO 통합 처리)
# ============================================================

# 1) PD에서 "가장 먼저 올라왔고 비식별 필요" 1건을 tb_items 기준으로 선별하고
#    타입(MOBILE/WEBCAM/VOICE)에 맞는 메타(비디오/오디오) 1행을 붙여 반환
#    - advisory lock으로 동시성 충돌 방지
SQL_PICK_ITEM_PD = """
SELECT
  i.item_id,
  i.patient_id,
  UPPER(REPLACE(i.data_type, '-', '_')) AS data_type_norm,
  COALESCE(i.updated_at, i.collected_at) AS item_ts
FROM dev_kkh.tb_items i
JOIN dev_kkh.tb_data_validation d ON d.item_id = i.item_id
WHERE i.is_deleted = FALSE
  AND UPPER(REPLACE(i.data_category, '-', '_')) = 'MDD'
  AND UPPER(REPLACE(i.data_type, '-', '_')) IN ('MOBILE','WEBCAM','VOICE')
  AND d.validation_description ILIKE '%%PASS%%'
  AND (
    (UPPER(REPLACE(i.data_type, '-', '_')) IN ('MOBILE','WEBCAM') AND EXISTS (
      SELECT 1 FROM dev_kkh.tb_video_metadata v
      WHERE v.item_id = i.item_id
        AND v.needs_anonymization = TRUE
        AND COALESCE(v.is_anonymized, FALSE) = FALSE
        AND UPPER(REPLACE(v.data_category, '-', '_')) = 'MDD'
    ))
    OR
    (UPPER(REPLACE(i.data_type, '-', '_')) = 'VOICE' AND EXISTS (
      SELECT 1 FROM dev_kkh.tb_audio_metadata a
      WHERE a.item_id = i.item_id
        AND a.needs_anonymization = TRUE
        AND COALESCE(a.is_anonymized, FALSE) = FALSE
        AND UPPER(REPLACE(a.data_category, '-', '_')) = 'MDD'
    ))
  )
  AND pg_try_advisory_xact_lock(i.item_id::bigint)
ORDER BY COALESCE(i.updated_at, i.collected_at) ASC, i.item_id ASC
LIMIT 1;
"""

SQL_VIDEO_META_ONE = """
SELECT
  v.video_metadata_id,
  v.item_id,
  v.file_path, v.file_ext, v.file_name,
  v.created_ts, v.needs_anonymization, COALESCE(v.is_anonymized, FALSE) AS is_anonymized,
  v.data_category
FROM dev_kkh.tb_video_metadata v
WHERE v.item_id = %s
  AND v.needs_anonymization = TRUE
  AND COALESCE(v.is_anonymized, FALSE) = FALSE
  AND UPPER(REPLACE(v.data_category, '-', '_')) = 'MDD'
ORDER BY v.created_ts ASC
LIMIT 1;
"""

SQL_AUDIO_META_ONE = """
SELECT
  a.audio_metadata_id,
  a.item_id,
  a.file_path, a.file_ext, a.file_name,
  a.created_ts, a.needs_anonymization, COALESCE(a.is_anonymized, FALSE) AS is_anonymized,
  a.data_category
FROM dev_kkh.tb_audio_metadata a
WHERE a.item_id = %s
  AND a.needs_anonymization = TRUE
  AND COALESCE(a.is_anonymized, FALSE) = FALSE
  AND UPPER(REPLACE(a.data_category, '-', '_')) = 'MDD'
ORDER BY a.created_ts ASC
LIMIT 1;
"""

def get_next_media_to_process() -> Optional[Dict[str, Any]]:
    """
    반환 예:
    {
      'media_kind': 'VIDEO'|'AUDIO',
      'data_type': 'MOBILE'|'WEBCAM'|'VOICE',
      'item_id': int, 'patient_id': UUID,
      'file_path': str, 'file_ext': str, 'file_name': str,
      'meta_created_ts': timestamp,
      'video_metadata_id' or 'audio_metadata_id': int
    }
    """
    conn = None
    try:
        conn = get_connection()
        with conn:  # 트랜잭션 (advisory_xact_lock 유지)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(SQL_PICK_ITEM_PD)
                it = cur.fetchone()
                if not it:
                    return None
                print(it)
                item_id = it["item_id"]
                dtype = (it["data_type_norm"] or "").upper()

                if dtype in ("MOBILE", "WEBCAM"):
                    cur.execute(SQL_VIDEO_META_ONE, (item_id,))
                    meta = cur.fetchone()
                    if not meta:
                        return None
                    return {
                        "media_kind": "VIDEO",
                        "data_type": dtype,
                        "item_id": item_id,
                        "patient_id": it["patient_id"],
                        "file_path": meta["file_path"],
                        "file_ext": meta["file_ext"],
                        "file_name": meta["file_name"],
                        "meta_created_ts": meta["created_ts"],
                        "video_metadata_id": meta["video_metadata_id"],
                    }
                else:  # VOICE
                    cur.execute(SQL_AUDIO_META_ONE, (item_id,))
                    meta = cur.fetchone()
                    if not meta:
                        return None
                    return {
                        "media_kind": "AUDIO",
                        "data_type": dtype,
                        "item_id": item_id,
                        "patient_id": it["patient_id"],
                        "file_path": meta["file_path"],
                        "file_ext": meta["file_ext"],
                        "file_name": meta["file_name"],
                        "meta_created_ts": meta["created_ts"],
                        "audio_metadata_id": meta["audio_metadata_id"],
                    }
    except Exception as e:
        print(f"❌ 처리 대기 미디어 조회 실패: {e}")
        traceback.print_exc()
        return None
    finally:
        if conn:
            release_connection(conn)

# 2) (옵션) MOBILE/WEBCAM/VOICE에서 "각각 1건씩" 뽑기
SQL_PICK_ONE_EACH = """
WITH pending_items AS (
    SELECT
        i.item_id,
        i.patient_id,
        UPPER(REPLACE(i.data_type, '-', '_')) AS data_type_norm,
        COALESCE(i.updated_at, i.collected_at) AS item_ts
    FROM dev_kkh.tb_items i
    JOIN dev_kkh.tb_data_validation d ON d.item_id = i.item_id
    WHERE i.is_deleted = FALSE
      AND UPPER(REPLACE(i.data_category, '-', '_')) = 'PD'
      AND UPPER(REPLACE(i.data_type, '-', '_')) IN ('MOBILE','WEBCAM','VOICE')
      AND d.validation_description ILIKE '%%PASS%%'
),
video_meta AS (
    SELECT v.item_id, v.file_path, v.file_ext, v.file_name, v.created_ts AS meta_created_ts
    FROM dev_kkh.tb_video_metadata v
    WHERE v.needs_anonymization = TRUE
      AND COALESCE(v.is_anonymized, FALSE) = FALSE
      AND UPPER(REPLACE(v.data_category, '-', '_')) = 'PD'
),
audio_meta AS (
    SELECT a.item_id, a.file_path, a.file_ext, a.file_name, a.created_ts AS meta_created_ts
    FROM dev_kkh.tb_audio_metadata a
    WHERE a.needs_anonymization = TRUE
      AND COALESCE(a.is_anonymized, FALSE) = FALSE
      AND UPPER(REPLACE(a.data_category, '-', '_')) = 'PD'
),
merged AS (
    SELECT p.item_id, p.patient_id, p.data_type_norm,
           'VIDEO'::text AS media_kind,
           vm.file_path, vm.file_ext, vm.file_name,
           COALESCE(vm.meta_created_ts, p.item_ts) AS ts
    FROM pending_items p
    JOIN video_meta vm ON vm.item_id = p.item_id
    WHERE p.data_type_norm IN ('MOBILE','WEBCAM')

    UNION ALL

    SELECT p.item_id, p.patient_id, p.data_type_norm,
           'AUDIO'::text AS media_kind,
           am.file_path, am.file_ext, am.file_name,
           COALESCE(am.meta_created_ts, p.item_ts) AS ts
    FROM pending_items p
    JOIN audio_meta am ON am.item_id = p.item_id
    WHERE p.data_type_norm = 'VOICE'
),
ranked AS (
    SELECT m.*,
           ROW_NUMBER() OVER (PARTITION BY m.data_type_norm ORDER BY m.ts ASC, m.item_id ASC) AS rn
    FROM merged m
)
SELECT data_type_norm, media_kind, item_id, patient_id,
       file_path, file_ext, file_name, ts
FROM ranked
WHERE rn = 1
ORDER BY data_type_norm;
"""

def get_one_each_mobile_webcam_voice() -> Dict[str, Dict[str, Any]]:
    """
    MOBILE / WEBCAM / VOICE에서 각각 1건씩 (있으면) 반환.
    {'MOBILE': {...}, 'WEBCAM': {...}, 'VOICE': {...}}
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(SQL_PICK_ONE_EACH)
            rows = cur.fetchall() or []
            return { r["data_type_norm"]: r for r in rows }
    except Exception as e:
        print(f"❌ 타입별 1건 조회 실패: {e}")
        traceback.print_exc()
        return {}
    finally:
        if conn:
            release_connection(conn)

# 3) 음성 업데이트 (VIDEO와 대칭)
def update_processed_audio(audio_metadata_id: int, anonymized_ts: str, is_anonymized: bool = True):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE dev_kkh.tb_audio_metadata
                SET is_anonymized = %s,
                    anonymized_ts = %s
                WHERE audio_metadata_id = %s
            """, (is_anonymized, anonymized_ts, audio_metadata_id))
        conn.commit()
        print(f"✅ 업데이트 완료(음성): audio_metadata_id={audio_metadata_id}")
    except Exception as e:
        print(f"❌ 음성 업데이트 실패: {e}")
        traceback.print_exc()
    finally:
        if conn:
            release_connection(conn)

# 4) (선택) VIDEO/AUDIO 공용 payload 빌더 (프레임 컬럼: 음성은 None)
from datetime import datetime

def make_preproc_payload_video(
    *,
    item_id: int,
    data_category: str,
    original_file_path: str,
    json_file_path: Optional[str],
    encrypted_file_path: Optional[str],
    started_at: datetime,
    ended_at: datetime,
    duration_sec: float,
    total_frames: int,
    encrypted_frames: int,
    detected_face_frames: int,
    success_rate: float,
    description: str = "YOLO face ROI encryption"
) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "data_category": data_category,
        "original_file_path": original_file_path,
        "json_file_path": json_file_path,
        "encrypted_file_path": encrypted_file_path,
        "processing_started_at": started_at,
        "processing_ended_at": ended_at,
        "processing_duration_sec": round(float(duration_sec), 2),
        "total_frames": total_frames,
        "encrypted_frames": encrypted_frames,
        "detected_face_frames": detected_face_frames,
        "success_rate": round(float(success_rate), 2),
        "preprocessing_type": "VIDEO_FACE_ENCRYPT",
        "description": description,
    }

def make_preproc_payload_audio(
    item_id: int,
    data_category: str,
    original_file_path: str,
    json_file_path,
    encrypted_file_path: str,
    started_at,
    ended_at,
    duration_sec: float,
    success_rate: float,
    description: str,
):
    return {
        "item_id": item_id,
        "data_category": data_category,
        "original_file_path": original_file_path,
        "json_file_path": json_file_path,
        "encrypted_file_path": encrypted_file_path,
        "processing_started_at": started_at,
        "processing_ended_at": ended_at,
        "processing_duration_sec": round(duration_sec or 0.0, 2),
        "total_frames": None,                # AUDIO는 NULL 허용
        "encrypted_frames": None,            # AUDIO는 NULL 허용
        "detected_face_frames": None,        # AUDIO는 NULL 허용
        "success_rate": round(success_rate or 0.0, 2),
        "preprocessing_type": "audio_envelope + pitch_shift",  # ✅ 필수!
        "description": description,
    }
