-- 시퀀스
CREATE SEQUENCE IF NOT EXISTS seq_patient_display_id START WITH 10001 INCREMENT BY 1;

-- 테이블
CREATE TABLE tb_patient_info (
    patient_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    display_id VARCHAR(20) UNIQUE,
    patient_initials VARCHAR(10),
    gender VARCHAR(10);
    birth_date DATE,
    institution VARCHAR(100),
    is_data_complete BOOLEAN DEFAULT false,
    completion_date TIMESTAMP,
    created_ts TIMESTAMP DEFAULT now(),
    update_ts TIMESTAMP
);

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
