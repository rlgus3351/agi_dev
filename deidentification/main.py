import os
import requests
from datetime import datetime
from processor import process_video
from config import PROCESS_BASE_URL

def fetch_one_video():
    """비식별화가 필요한 영상 1건 조회"""
    url = f"{PROCESS_BASE_URL}next"
    try:
        res = requests.get(url)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"[{datetime.now()}] ⚠️ 영상 조회 중 오류 발생: {e}")
        return None


def run_processing_pipeline():
    """단일 영상 처리 실행"""
    video_info = fetch_one_video()
    if not video_info:
        print(f"[{datetime.now()}] ✅ 처리할 영상이 없습니다.\n")
        return False  # 처리 없음
    file_path = video_info.get("file_path")
    video_id = video_info.get("video_metadata_id")
    item_id = video_info.get("item_id")

    print(f"[{datetime.now()}] 🎬 처리 시작")
    print(f"   • file_path = {file_path}")
    print(f"   • video_metadata_id = {video_id}")
    print(f"   • item_id = {item_id}")
    print("-" * 60)
    
    file_path = video_info["file_path"]
    print(f"[{datetime.now()}] 🎬 처리 시작: {file_path}")

    try:
        output_path, json_path = process_video(video_info)
        print(f"[{datetime.now()}] ✅ 처리 완료: {output_path}")
        print(f"[{datetime.now()}] 📝 ROI 저장: {json_path}\n")
        return True  # 처리 성공
    except Exception as e:
        print(f"[{datetime.now()}] ❌ 처리 중 오류 발생: {e}\n")
        return False


if __name__ == "__main__":
    print(f"[{datetime.now()}] 🚀 비식별화 자동 처리 파이프라인 시작")

    while True:
        success = run_processing_pipeline()

        if not success:
            print(f"[{datetime.now()}] ⏸️ 10분 대기 후 재시작...\n")
            time.sleep(600)  # 10분 대기
        else:
            print(f"[{datetime.now()}] 🔁 다음 영상 확인 중...\n")
