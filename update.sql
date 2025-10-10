-- 시퀀스
CREATE SEQUENCE IF NOT EXISTS seq_patient_display_id START WITH 10001 INCREMENT BY 1;

-- 테이블
CREATE TABLE tb_patient_info (
    patient_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    display_id VARCHAR(20) UNIQUE,
    patient_initials VARCHAR(10),
    gender VARCHAR(10),
    birth_date DATE,
    institution VARCHAR(100),
    is_data_complete BOOLEAN DEFAULT false,
    completion_date TIMESTAMP,
    created_ts TIMESTAMP DEFAULT now(),
    update_ts TIMESTAMP
);


COMMENT ON TABLE tb_patient_info IS '환자 정보 테이블';
COMMENT ON COLUMN dev_kkh.tb_patient_info.patient_id IS '환자 고유 아이디';
COMMENT ON COLUMN dev_kkh.tb_patient_info.display_id IS '외부 공개용 아이디';
COMMENT ON COLUMN dev_kkh.tb_patient_info.patient_initials IS '환자 이니셜';
COMMENT ON COLUMN dev_kkh.tb_patient_info.gender IS '환자 성별';
COMMENT ON COLUMN dev_kkh.tb_patient_info.birth_date IS '환자 생년월일';
COMMENT ON COLUMN dev_kkh.tb_patient_info.institution IS '수집 기관';
COMMENT ON COLUMN dev_kkh.tb_patient_info.is_data_complete IS '데이터 수집 완료 여부';
COMMENT ON COLUMN dev_kkh.tb_patient_info.completion_date IS '데이터 수집 완료 시점';
COMMENT ON COLUMN dev_kkh.tb_patient_info.created_ts IS '데이터 최초 수집 시간';
COMMENT ON COLUMN dev_kkh.tb_patient_info.update_ts IS '데이터 변경 시간';


GRANT DELETE, INSERT, UPDATE, SELECT ON TABLE dev_kkh.tb_items TO pd_dep_collector;

-- 트리거 함수
CREATE OR REPLACE FUNCTION trg_set_display_id()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.display_id IS NULL OR NEW.display_id = '' THEN
        NEW.display_id := 'ID-' || nextval('seq_patient_display_id');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 트리거
CREATE TRIGGER set_display_id_trigger
BEFORE INSERT ON tb_patient_info
FOR EACH ROW
EXECUTE FUNCTION trg_set_display_id();

CREATE TABLE tb_Items (
    item_id SERIAL PRIMARY KEY,               -- 각 수집 항목의 고유 식별자 (PostgreSQL의 자동 증가 기본 키)
    patient_id UUID,                          -- 외래 키: 환자 ID
    data_category VARCHAR(50),                -- 데이터 분류 (예: 우울증, 파킨슨병)
    data_type VARCHAR(50),                    -- 데이터 유형 (예: 설문지, 영상, 생체 데이터)
    seq INT,                                  -- 동일 유형 내의 순번
    description TEXT,                         -- 항목 상세 설명
    collected_at TIMESTAMP DEFAULT now(),     -- 데이터 수집 시점 (PostgreSQL의 표준 날짜/시간 유형)
    is_deleted BOOLEAN DEFAULT FALSE,         -- ✅ 소프트 삭제 여부 (TRUE면 삭제된 상태)
    deleted_at TIMESTAMP NULL,                -- ✅ 실제 삭제된 시각 (NULL이면 아직 살아있음)
    FOREIGN KEY (patient_id) REFERENCES tb_Patient_Info(patient_id)
);

COMMENT ON TABLE tb_items IS '수집 항목 데이터 테이블';
-- 컬럼 설명 추가
COMMENT ON COLUMN dev_kkh.tb_Items.item_id IS '각 수집 항목의 고유 식별자';
COMMENT ON COLUMN dev_kkh.tb_Items.patient_id IS '외래 키: 환자 ID';
COMMENT ON COLUMN dev_kkh.tb_Items.data_category IS '데이터 분류 (예: 우울증, 파킨슨병)';
COMMENT ON COLUMN dev_kkh.tb_Items.data_type IS '데이터 유형 (예: 설문지, 영상, 생체 데이터)';
COMMENT ON COLUMN dev_kkh.tb_Items.seq IS '동일 유형 내의 순번';
COMMENT ON COLUMN dev_kkh.tb_Items.description IS '항목 상세 설명';
COMMENT ON COLUMN dev_kkh.tb_Items.collected_at IS '데이터 수집 시점';
COMMENT ON COLUMN dev_kkh.tb_Items.is_deleted IS '소프트 삭제 여부 (TRUE면 삭제 처리됨)';
COMMENT ON COLUMN dev_kkh.tb_Items.deleted_at IS '소프트 삭제된 시각';

-- 권한 부여
GRANT DELETE, INSERT, UPDATE, SELECT ON TABLE dev_kkh.tb_items TO pd_dep_collector;


CREATE TABLE tb_Data_Validation (
    validation_id SERIAL PRIMARY KEY,     -- 검증 내역 고유 식별자 (기본 키)
    item_id INT,                                      -- 외래 키: 어떤 항목에 대한 검증인지 연결
    validation_method VARCHAR(50),                    -- 검증 방법
    validation_description TEXT,                      -- 검증에 대한 상세 설명
    validation_datetime TIMESTAMP,                     -- 검증이 이루어진 시점
    FOREIGN KEY (item_id) REFERENCES tb_Items(item_id)
);
COMMENT ON TABLE tb_Data_Validation IS '데이터 검증 테이블';
COMMENT ON COLUMN tb_Data_Validation.validation_id IS '검증 인덱스';
COMMENT ON COLUMN tb_Data_Validation.item_id IS '외래 키 : 데이터 항목 식별자';
COMMENT ON COLUMN tb_Data_Validation.validation_method IS '검증 방법';
COMMENT ON COLUMN tb_Data_Validation.validation_description IS '검증에 대한 상세 설명';
COMMENT ON COLUMN tb_Data_Validation.validation_datetime IS '검증 시점';


CREATE TABLE tb_Video_Metadata (
    video_metadata_id SERIAL PRIMARY KEY,
    item_id INT,                                     -- 외래 키: 어떤 항목에 대한 영상인지 연결
    file_path VARCHAR(255),                          -- 실제 영상 파일의 경로
    file_size_mb DECIMAL(10, 2),                     -- 파일 크기 (MB)
    duration_seconds INT,                            -- 영상 길이 (초)
    resolution VARCHAR(20),                          -- 해상도 (예: '1920x1080')
    frame_rate INT,                                  -- 프레임 레이트 (fps)
    is_anonymized INT,                        -- 비식별화 여부 (0:N, 1:Y)
    FOREIGN KEY (item_id) REFERENCES tb_Items(item_id)
);
GRANT DELETE, INSERT, UPDATE, SELECT ON TABLE dev_kkh.tb_Video_Metadata TO pd_dep_collector;
COMMENT ON TABLE tb_Video_Metadata IS '비디오 메타데이터 테이블';
COMMENT ON COLUMN tb_Video_Metadata.video_metadata_id IS '비디오 메타데이터 항목';
COMMENT ON COLUMN tb_Video_Metadata.item_id IS '외래 키 : 데이터 항목 식별자';
COMMENT ON COLUMN tb_Video_Metadata.file_path IS '파일 경로';
COMMENT ON COLUMN tb_Video_Metadata.file_size_mb IS '파일 크기(mb단위)';
COMMENT ON COLUMN tb_Video_Metadata.duration_seconds IS '영상 길이 (초)';
COMMENT ON COLUMN tb_Video_Metadata.resolution IS '해상도 (예 : 1920*1080)';
COMMENT ON COLUMN tb_Video_Metadata.frame_rate IS '프레임 레이트(fps)';
COMMENT ON COLUMN tb_Video_Metadata.is_anonymized IS '비식별화 여부(0:N, 1:Y)';


CREATE TABLE tb_Questionnaire_Questions (
    question_id SERIAL PRIMARY KEY,          -- 질문의 고유 식별자
    questionnaire_type VARCHAR(50),                      -- 설문지 유형 (예: 우울증-A형, 우울증-B형)
    question_number INT,                                 -- 설문지 내 질문 번호
    question_text VARCHAR(500)                           -- 질문 내용
);

COMMENT ON TABLE tb_Questionnaire_Questions IS '설문지 질문 테이블';
COMMENT ON COLUMN tb_Questionnaire_Questions.question_id IS '질문 ID';
COMMENT ON COLUMN tb_Questionnaire_Questions.questionnaire_type IS '설문지 유형';
COMMENT ON COLUMN tb_Questionnaire_Questions.question_number IS '질문 번호';
COMMENT ON COLUMN tb_Questionnaire_Questions.question_text IS '질문 내용';

GRANT DELETE, INSERT, UPDATE, SELECT ON TABLE dev_kkh.tb_Questionnaire_Questions TO pd_dep_collector;

INSERT INTO tb_Questionnaire_Questions (questionnaire_type, question_number, question_text) VALUES
-- 운동성 검사 기초 정보 (question_number는 JSON의 id 값 사용, 문자열 id는 0.x 형태로 변환)
('운동성 검사 (MDS-UPDRS Part III)', 0, '증상 치료 목적 약물 복용 여부'),
('운동성 검사 (MDS-UPDRS Part III)', 0, '환자 임상적 상태 (약물 복용 후)'),
('운동성 검사 (MDS-UPDRS Part III)', 0, '환자 Levodopa 복용 여부'),
('운동성 검사 (MDS-UPDRS Part III)', 0, '마지막 복용 경과 시간 (분)'),
('운동성 검사 (MDS-UPDRS Part III)', 0, 'DYSKINESIA가 검사 결과에 영향을 주었는가?'),
('운동성 검사 (MDS-UPDRS Part III)', 0, '검사 도중 dyskinesia 유무'),
('운동성 검사 (MDS-UPDRS Part III)', 0, '검사 결과에 영향을 주었는지 여부'),
('운동성 검사 (MDS-UPDRS Part III)', 0, 'Hoehn & Yahr 파킨슨병 진행 단계'), -- Hoehn & Yahr 척도
-- 운동 항목별 평가 (question_number는 JSON의 id 값 사용)
('운동성 검사 (MDS-UPDRS Part III)', 1, '말하기'),
('운동성 검사 (MDS-UPDRS Part III)', 2, '얼굴 표정'),
('운동성 검사 (MDS-UPDRS Part III)', 3, '관절의 뻣뻣함 (Neck, RA, LA, RL, LL)'), -- grouped-inputs는 항목명에 부위를 추가
('운동성 검사 (MDS-UPDRS Part III)', 4, '손가락 부딪치기 (R, L)'),
('운동성 검사 (MDS-UPDRS Part III)', 5, '손 동작 (R, L)'),
('운동성 검사 (MDS-UPDRS Part III)', 6, '손 내전/외전 움직임 (R, L)'),
('운동성 검사 (MDS-UPDRS Part III)', 7, '발가락으로 두드리기 (R, L)'),
('운동성 검사 (MDS-UPDRS Part III)', 8, '다리 민첩성 (R, L)'),
('운동성 검사 (MDS-UPDRS Part III)', 9, '의자에서 일어나기'),
('운동성 검사 (MDS-UPDRS Part III)', 10, '걷는 자세'),
('운동성 검사 (MDS-UPDRS Part III)', 11, '걷는 중 몸의 굳어짐'),
('운동성 검사 (MDS-UPDRS Part III)', 12, '자세의 안정'),
('운동성 검사 (MDS-UPDRS Part III)', 13, '자세'),
('운동성 검사 (MDS-UPDRS Part III)', 14, '자연스러운 움직임'),
('운동성 검사 (MDS-UPDRS Part III)', 15, '자세 유지시 손의 떨림 (R, L)'),
('운동성 검사 (MDS-UPDRS Part III)', 16, '움직일 때 손의 떨림 (R, L)'),
('운동성 검사 (MDS-UPDRS Part III)', 17, '가만 있을 때 떨림의 폭 (RA, LA, RL, LL, LJ)'),
('운동성 검사 (MDS-UPDRS Part III)', 18, '가만 있을 때 떨림의 지속시간');