import requests
import urllib3
urllib3.disable_warnings()

NAS_URL = "https://121.147.253.141:5001"
USERNAME = "ai03"
PASSWORD = "Rkskekfk1!"

def login():
    """로그인 후 SID(세션 토큰) 반환"""
    params = {
        "api": "SYNO.API.Auth",
        "version": "7",
        "method": "login",
        "account": USERNAME,
        "passwd": PASSWORD,
        "session": "FileStation",
        "format": "sid"
    }
    r = requests.get(f"{NAS_URL}/webapi/entry.cgi", params=params, verify=False)
    sid = r.json()["data"]["sid"]
    print(f"✅ 로그인 성공 → SID: {sid}")
    return sid

def upload_text_file(sid, folder="/mAGI/CNU_Data", filename="sample.txt", content="Hello NAS!"):
    """텍스트 파일 생성 후 업로드"""
    # 1️⃣ 임시 파일 생성
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    # 2️⃣ 업로드 API 호출
    url = f"{NAS_URL}/webapi/entry.cgi"
    params = {
        "api": "SYNO.FileStation.Upload",
        "version": "2",
        "method": "upload",
        "_sid": sid
    }
    data = {
        "path": folder,
        "create_parents": "true",
        "overwrite": "true"
    }
    files = {
        "file": (filename, open(filename, "rb"), "text/plain")
    }

    r = requests.post(url, params=params, data=data, files=files, verify=False)
    print("📤 업로드 응답:", r.status_code)
    print(r.json())

def logout(sid):
    params = {
        "api": "SYNO.API.Auth",
        "version": "7",
        "method": "logout",
        "_sid": sid
    }
    requests.get(f"{NAS_URL}/webapi/entry.cgi", params=params, verify=False)
    print("👋 로그아웃 완료")

if __name__ == "__main__":
    sid = login()
    upload_text_file(sid, folder="/mAGI/CNU_Data", filename="note.txt", content="이 파일은 NAS API로 업로드되었습니다 🚀")
    logout(sid)
