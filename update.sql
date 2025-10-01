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
