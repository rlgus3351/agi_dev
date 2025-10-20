# run_worker.py (수정본)

import os
import time
from datetime import datetime
from processor import process_video

# ✅ 로컬 DB 연동 함수들
from api_local.processing_api_local import get_next_video_to_process

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def fetch_one_video():
    """비식별화가 필요한 영상 1건 조회 (로컬 DB)"""
    try:
        row = get_next_video_to_process()
        # get_next_video_to_process()가 없으면 None 반환하도록 작성되어 있음
        return row  # dict | None
    except Exception as e:
        print(f"[{now()}] ⚠️ 영상 조회 중 오류 발생: {e}")
        return None

def run_processing_pipeline():
    """단일 영상 처리 실행"""
    video_info = fetch_one_video()
    if not video_info:
        print(f"[{now()}] ✅ 처리할 영상이 없습니다.\n")
        return False  # 처리 없음

    file_path = video_info.get("file_path")
    video_id = video_info.get("video_metadata_id")
    item_id = video_info.get("item_id")

    print(f"[{now()}] 🎬 처리 시작")
    print(f"   • file_path = {file_path}")
    print(f"   • video_metadata_id = {video_id}")
    print(f"   • item_id = {item_id}")
    print("-" * 60)

    try:
        output_path, json_path = process_video(video_info)  # processor.py 내부가 로컬 DB 연동으로 변경됨
        print(f"[{now()}] ✅ 처리 완료: {output_path}")
        print(f"[{now()}] 📝 ROI 저장: {json_path}\n")
        return True  # 처리 성공
    except Exception as e:
        print(f"[{now()}] ❌ 처리 중 오류 발생: {e}\n")
        return False

if __name__ == "__main__":
    print(f"[{now()}] 🚀 비식별화 자동 처리 파이프라인 시작")
    while True:
        success = run_processing_pipeline()
        if not success:
            print(f"[{now()}] ⏸️ 10분 대기 후 재시작...\n")
            time.sleep(600)  # 10분 대기
        else:
            print(f"[{now()}] 🔁 다음 영상 확인 중...\n")
