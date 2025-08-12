import gradio as gr
import json, os, uuid, shutil

# ---------------- 경로/데이터 초기화 ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PATIENT_FILE = os.path.join(DATA_DIR, "patients.json")
SURVEY_FILE = os.path.join(DATA_DIR, "surveys.json")
VIDEO_FILE = os.path.join(DATA_DIR, "videos.json")
VIDEO_SAVE_DIR = os.path.join(DATA_DIR, "videos")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)

def load_json(file):
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 전역 상태
patients = load_json(PATIENT_FILE)
surveys = load_json(SURVEY_FILE)
videos = load_json(VIDEO_FILE)

# ---------------- 설문 템플릿 ----------------
survey_template = [
    (1, "말하기", ["score"]),
    (2, "얼굴 표정", ["score"]),
    (3, "관절의 뻣뻣함", ["Neck", "RA", "LA", "RL", "LL"]),
    (4, "손가락 부딪치기", ["R", "L"]),
    (5, "손 동작", ["R", "L"]),
    (6, "손 내전/외전 움직임", ["R", "L"]),
    (7, "발가락으로 두드리기", ["R", "L"]),
    (8, "다리 민첩성", ["R", "L"]),
    (9, "의자에서 일어나기", ["score"]),
    (10, "걷는 자세", ["score"]),
    (11, "걷는 중 몸의 굳어짐", ["score"]),
    (12, "자세의 안정", ["score"]),
    (13, "자세", ["score"]),
    (14, "움직임에서 전반적인 자연스러움", ["score"]),
    (15, "자세 유지시 손의 떨림", ["R", "L"]),
    (16, "움직일 때 손의 떨림", ["R", "L"]),
    (17, "가만 있을 때 떨림의 폭", ["RA", "LA", "RL", "LL", "LJ"]),
    (18, "가만 있을 때 떨림의 지속시간", ["score"])
]

# ---------------- 유틸 함수 ----------------
def refresh_all():
    """파일에서 다시 읽어와 테이블과 드롭다운 반환"""
    global patients, surveys, videos
    patients = load_json(PATIENT_FILE)
    surveys = load_json(SURVEY_FILE)
    videos = load_json(VIDEO_FILE)
    table = [[v["이니셜"], v["생년월일"], v["성별"]] for v in patients.values()]
    dropdown = [v["이니셜"] for v in patients.values()]
    return table, dropdown

# ---------------- 동작 함수 ----------------
def add_patient(initials, dob, gender):
    if not initials or not dob or not gender:
        table, dropdown = refresh_all()
        return "❌ 모든 항목을 입력하세요.", table, dropdown, dropdown
    pid = str(uuid.uuid4())
    patients[pid] = {"이니셜": initials, "생년월일": dob, "성별": gender}
    save_json(PATIENT_FILE, patients)
    table, dropdown = refresh_all()
    return f"✅ 환자 추가 완료: {initials}", table, dropdown, dropdown

def delete_patient(initials):
    if not initials:
        table, dropdown = refresh_all()
        return "❌ 삭제할 환자를 선택하세요.", table, dropdown, dropdown
    for pid, info in list(patients.items()):
        if info["이니셜"] == initials:
            patients.pop(pid, None)
            surveys.pop(pid, None)
            videos.pop(pid, None)
    save_json(PATIENT_FILE, patients)
    save_json(SURVEY_FILE, surveys)
    save_json(VIDEO_FILE, videos)
    table, dropdown = refresh_all()
    return f"🗑️ 삭제 완료: {initials}", table, dropdown, dropdown

def submit_survey(initials, is_on_med, clinical_effect, levodopa, levodopa_min,
                  dys_present, dys_interfere, *score_inputs):
    if not initials:
        return "❌ 먼저 환자를 선택하세요."
    pid = next((k for k,v in patients.items() if v["이니셜"] == initials), None)
    if not pid:
        return "❌ 해당 환자를 찾을 수 없습니다."

    # 설문 데이터 구성
    medication_data = {
        "is_on_medication": bool(is_on_med),
        "clinical_effect": clinical_effect,
        "levodopa_taken": bool(levodopa),
        "levodopa_elapsed_hours": float(levodopa_min or 0) / 60
    }
    dyskinesia_data = {
        "dyskinesia_present": bool(dys_present),
        "dyskinesia_interfered": bool(dys_interfere)
    }

    items = []
    idx = 0
    for item_id, item_name, parts in survey_template:
        scores = score_inputs[idx: idx+len(parts)]
        idx += len(parts)
        if parts == ["score"]:
            val = int(scores[0]) if scores[0] not in [None,""] else 0
            items.append({"item_id": item_id, "item_name": item_name, "scores": val})
        else:
            part_scores = {p:int(s or 0) for p,s in zip(parts,scores)}
            items.append({"item_id": item_id, "item_name": item_name, "scores": part_scores})

    surveys[pid] = {
        "medication": medication_data,
        "dyskinesia_impact": dyskinesia_data,
        "items": items
    }
    save_json(SURVEY_FILE, surveys)
    return f"✅ 설문 저장 완료: {initials}"

def upload_videos(initials, v1, v2, v3, v4):
    if not initials:
        return "❌ 먼저 환자를 선택하세요."
    if not all([v1,v2,v3,v4]):
        return "❌ 1~3번 및 손글씨 영상 모두 업로드해야 합니다."
    
    pid = initials
    patient_dir = os.path.join(VIDEO_SAVE_DIR, pid)
    os.makedirs(patient_dir, exist_ok=True)

    videos[pid] = {}
    file_paths = []
    for idx, video_file in zip(["1번영상","2번영상","3번영상","손글씨영상"], [v1,v2,v3,v4]):
        save_path = os.path.join(patient_dir, f"{idx}_{os.path.basename(video_file.name)}")
        shutil.copy(video_file.name, save_path)
        videos[pid][idx] = save_path
        file_paths.append(save_path)

    save_json(VIDEO_FILE, videos)
    return "✅ 업로드 완료:\n" + "\n".join(file_paths)

# ---------------- Gradio UI ----------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🏥 환자 / 설문 / 영상 관리 (Gradio 버전)")

    with gr.Tab("환자 관리"):
        patient_table = gr.Dataframe(
            headers=["이니셜","생년월일","성별"],
            value=refresh_all()[0],
            datatype=["str","str","str"],
            interactive=False
        )
        with gr.Row():
            initials = gr.Textbox(label="이니셜")
            dob = gr.Textbox(label="생년월일 (YYYYMMDD)")
            gender = gr.Radio(["남", "여"], label="성별")
            add_btn = gr.Button("➕ 환자 추가")
            add_status = gr.Label()
        with gr.Row():
            delete_dropdown = gr.Dropdown(choices=refresh_all()[1], label="삭제할 환자 선택")
            delete_btn = gr.Button("🗑️ 삭제")
            delete_status = gr.Label()

        add_btn.click(add_patient, inputs=[initials,dob,gender],
                      outputs=[add_status, patient_table, delete_dropdown, delete_dropdown])
        delete_btn.click(delete_patient, inputs=[delete_dropdown],
                         outputs=[delete_status, patient_table, delete_dropdown, delete_dropdown])

    with gr.Tab("설문 입력"):
        patient_select = gr.Dropdown(choices=refresh_all()[1], label="환자 선택")
        is_on_med = gr.Checkbox(label="약 복용 중")
        clinical_effect = gr.Radio(["positive","negative"], label="임상 효과")
        levodopa = gr.Checkbox(label="Levodopa 복용")
        levodopa_min = gr.Number(label="마지막 복용 후 경과 시간(분)")
        dys_present = gr.Checkbox(label="검사 중 Dyskinesia 발생")
        dys_interfere = gr.Checkbox(label="Dyskinesia 평가에 영향")

        # 설문 문항 입력
        score_inputs = []
        for item_id, item_name, parts in survey_template:
            with gr.Row():
                gr.Markdown(f"**{item_id}. {item_name}**")
            with gr.Row():
                for part in parts:
                    with gr.Column():
                        gr.Markdown(f"**{part}**")           # 부위 표시
                        score_inputs.append(gr.Number(label="", precision=0))

        submit_btn = gr.Button("설문 제출")
        submit_status = gr.Label()
        submit_btn.click(submit_survey,
                         inputs=[patient_select,is_on_med,clinical_effect,levodopa,levodopa_min,
                                 dys_present,dys_interfere]+score_inputs,
                         outputs=submit_status)

    with gr.Tab("영상 업로드"):
        patient_video_select = gr.Dropdown(choices=refresh_all()[1], label="환자 선택")
        with gr.Row():
            v1 = gr.File(label="1번 영상", file_types=[".mp4", ".avi", ".mov"])
            v2 = gr.File(label="2번 영상", file_types=[".mp4", ".avi", ".mov"])
        with gr.Row():
            v3 = gr.File(label="3번 영상", file_types=[".mp4", ".avi", ".mov"])
            v4 = gr.File(label="손글씨 영상", file_types=[".mp4", ".avi", ".mov"])
        upload_btn = gr.Button("업로드 실행")
        upload_status = gr.Label()
        upload_btn.click(upload_videos,
                         inputs=[patient_video_select,v1,v2,v3,v4],
                         outputs=upload_status)

demo.launch(inbrowser=True)
