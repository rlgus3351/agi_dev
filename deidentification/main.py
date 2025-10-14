import os
import requests
from datetime import datetime
from processor import process_video
from config import PROCESS_BASE_URL


def fetch_one_video():
    """비식별화가 필요한 영상 1건 조회"""
    url = f"{PROCESS_BASE_URL}next"
    res = requests.get(url)
    res.raise_for_status()
    return res.json()


# def update_video_status(video_metadata_id: int):
#     """처리 완료 후 서버에 is_anonymized=True로 업데이트"""
#     url = f"{PROCESS_BASE_URL}update"
#     payload = {
#         "video_metadata_id": video_metadata_id,
#         "is_anonymized": True,
#         "anonymized_ts": datetime.now().isoformat()
#     }
#     res = requests.put(url, json=[payload])
#     res.raise_for_status()
#     return res.json()


def run_processing_pipeline():
    """단일 영상 처리 실행"""
    video_info = fetch_one_video()
    if not video_info:
        print("✅ 처리할 영상이 없습니다.")
        return

    file_path = video_info["file_path"]  # 예: "data/input/input1.mp4"
    video_metadata_id = video_info["video_metadata_id"]

    # 처리 시작
    print(f"🎬 처리 시작: {file_path}")
    output_path, json_path = process_video(video_info)

    # 서버 업데이트
    # update_video_status(video_metadata_id)
    print(f"✅ 처리 완료: {output_path}")
    print(f"📝 ROI 저장: {json_path}")


if __name__ == "__main__":
    run_processing_pipeline()
