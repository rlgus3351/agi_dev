import requests
from config import HEALTH_URL  # 또는 직접 문자열로 대체 가능

def check_server_status():
    try:
        r = requests.get(HEALTH_URL,timeout=5)
        if r.status_code == 200:
            r2 = requests.get(f"{HEALTH_URL}/db", timeout=5)
            if r2.status_code == 200:
                return "OK"
            else:
                return "DB_FAIL"
        else:
            return "API_FAIL"
    except Exception:
        return "API_FAIL"